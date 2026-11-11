"""Bounded, resumable source acquisition for IBM Code Engine."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from acquisition.collect import DownloadRateLimited, collect_batch
from acquisition.cos import AtomicCOSClient
from acquisition.discovery import DiscoveryCommit, DiscoveryPage, append_discovery_page
from acquisition.http import PinnedRequestsClient
from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from acquisition.types import SourceAccessRecord, SourcePolicy
from infra.ibm_cos import (
    create_oauth_cos_client,
    verify_no_public_access_policies,
    verify_private_versioned_bucket,
)
from scrapers.goldsmiths import discover_goldsmiths
from scrapers.vsfactoryusa import discover_vsfactoryusa
from scrapers.watchesofswitzerland import discover_watchesofswitzerland
from scrapers.wikimedia import WikimediaAPIError, WikimediaDiscovery, discover_wikimedia

_LOG = logging.getLogger(__name__)
_RATE_LIMITED = "Wikimedia rate limited; resume later"


def run_wikimedia_once(
    cos_client: Any,
    bucket: str,
    prefix: str,
    policy: SourcePolicy,
    access: SourceAccessRecord,
    *,
    now: datetime,
    api_client: Any,
    image_client: Any,
    discover: Callable[..., WikimediaDiscovery] = discover_wikimedia,
) -> dict[str, object]:
    """Collect at most one Commons page and commit its cursor last."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not bucket or not prefix.strip("/"):
        raise ValueError("bucket and prefix must be non-empty")
    if not callable(getattr(cos_client, "put_object_if_absent", None)):
        raise RuntimeError("COS client lacks proven atomic create-if-absent")

    cycle, previous = _active_cycle(
        cos_client,
        bucket,
        prefix,
        now.date().isoformat(),
        "wikimedia",
    )
    ledger_prefix = f"{prefix.strip('/')}/discovery/{cycle}"
    if previous is not None and previous.next_cursor is None:
        return {"cycle": cycle, "source": "wikimedia", "state": "already_complete"}

    try:
        result = discover(api_client, policy, access, now=now, previous=previous)
        accepted: list[str] = []
        for candidate in result.candidates:
            batch_id = "wikimedia-" + hashlib.sha256(
                (
                    f"{candidate.metadata.source_record_id}\0"
                    f"{candidate.rights.rights_record_id}\0{candidate.observed_at.isoformat()}"
                ).encode()
            ).hexdigest()[:24]
            receipt = collect_batch(
                (candidate.metadata,),
                policy,
                candidate.rights,
                access=access,
                http_client=image_client,
                cos_client=cos_client,
                bucket=bucket,
                batch_id=batch_id,
                now=now,
            )
            accepted.extend(receipt.accepted_candidate_ids)

        commit = append_discovery_page(
            cos_client,
            bucket,
            ledger_prefix,
            result.page,
            previous=previous,
        )
        status: dict[str, object] = {
            "accepted_candidate_ids": sorted(accepted),
            "commit_sha256": commit.sha256,
            "cycle": cycle,
            "next_cursor": commit.next_cursor,
            "source": "wikimedia",
            "state": "complete" if commit.next_cursor is None else "page_complete",
        }
    except DownloadRateLimited:
        status = {"cycle": cycle, "source": "wikimedia", "state": "rate_limited"}
    except WikimediaAPIError as exc:
        if str(exc) == _RATE_LIMITED:
            status = {"cycle": cycle, "source": "wikimedia", "state": "rate_limited"}
        else:
            _persist_failed(cos_client, bucket, prefix, cycle, "wikimedia", exc)
            raise
    except Exception as exc:
        _persist_failed(cos_client, bucket, prefix, cycle, "wikimedia", exc)
        raise
    _persist_status(cos_client, bucket, prefix, status)
    _LOG.info("acquisition_terminal %s", _canonical(status).decode())
    return status


def _active_cycle(
    cos_client: Any,
    bucket: str,
    prefix: str,
    today: str,
    source: str,
) -> tuple[str, DiscoveryCommit | None]:
    root = f"{prefix.strip('/')}/discovery/"
    page = cos_client.list_objects_v2(Bucket=bucket, Prefix=root, Delimiter="/")
    if not isinstance(page, dict) or page.get("IsTruncated"):
        raise RuntimeError("discovery cycle inventory is incomplete")
    common_prefixes = page.get("CommonPrefixes", [])
    if not isinstance(common_prefixes, list):
        raise RuntimeError("discovery cycle inventory is invalid")
    cycles: set[str] = set()
    for item in common_prefixes:
        item_prefix = item.get("Prefix") if isinstance(item, dict) else None
        if isinstance(item_prefix, str) and item_prefix.startswith(root):
            cycle = item_prefix[len(root) :].removesuffix("/")
            if len(cycle) == 10 and cycle[4] == "-" and cycle[7] == "-":
                cycles.add(cycle)
    for latest in sorted(cycles, reverse=True):
        previous = _load_latest_commit(
            cos_client,
            bucket,
            f"{root}{latest}",
            source,
        )
        if previous is None:
            continue
        if previous.next_cursor is not None or latest == today:
            return latest, previous
        break
    return today, None


def _load_latest_commit(
    cos_client: Any,
    bucket: str,
    prefix: str,
    source: str,
) -> DiscoveryCommit | None:
    parent = "root"
    previous_sha256: str | None = None
    commit: DiscoveryCommit | None = None
    for _ in range(1000):
        body = _read_if_present(
            cos_client,
            bucket,
            f"{prefix.rstrip('/')}/{source}/children/{parent}.json",
        )
        if body is None:
            return commit
        document = _canonical_document(body)
        if document.get("source") != source or document.get("previous_sha256") != previous_sha256:
            raise RuntimeError("discovery checkpoint chain is invalid")
        record_hashes = document.get("record_hashes")
        if not isinstance(record_hashes, list) or not all(
            isinstance(item, list)
            and len(item) == 2
            and all(isinstance(value, str) for value in item)
            for item in record_hashes
        ):
            raise RuntimeError("discovery checkpoint record hashes are invalid")
        digest = hashlib.sha256(body).hexdigest()
        commit = DiscoveryCommit(
            source=source,
            access_record_id=_required_string(document, "access_record_id"),
            previous_sha256=previous_sha256,
            page_sha256=_sha256_string(document, "page_sha256"),
            next_cursor=_optional_string(document.get("next_cursor")),
            record_hashes=tuple((item[0], item[1]) for item in record_hashes),
            sha256=digest,
        )
        parent = digest
        previous_sha256 = digest
    raise RuntimeError("discovery checkpoint chain exceeds 1000 pages")


def _persist_status(
    cos_client: Any,
    bucket: str,
    prefix: str,
    status: dict[str, object],
) -> str:
    body = _canonical(status)
    digest = hashlib.sha256(body).hexdigest()
    key = f"{prefix.strip('/')}/acquisition-status/{status['cycle']}/{digest}.json"
    existing = _read_if_present(cos_client, bucket, key)
    if existing is None:
        cos_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={"sha256": digest, "length": str(len(body))},
            ContentType="application/json",
        )
    if _read_if_present(cos_client, bucket, key) != body:
        raise RuntimeError("acquisition terminal status readback failed")
    return key


def _persist_failed(
    cos_client: Any,
    bucket: str,
    prefix: str,
    cycle: str,
    source: str,
    error: Exception,
) -> None:
    status: dict[str, object] = {
        "cycle": cycle,
        "error_type": type(error).__name__,
        "source": source,
        "state": "failed",
    }
    _persist_status(cos_client, bucket, prefix, status)
    _LOG.error("acquisition_terminal %s", _canonical(status).decode())


def run_metadata_discovery_once(
    cos_client: Any,
    bucket: str,
    prefix: str,
    source: str,
    *,
    now: datetime,
    discover: Callable[[DiscoveryCommit | None], Any],
) -> dict[str, object]:
    """Commit one metadata-only source page without downloading image bytes."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not bucket or not prefix.strip("/") or not source:
        raise ValueError("bucket, prefix and source must be non-empty")
    cycle, previous = _active_cycle(
        cos_client, bucket, prefix, now.date().isoformat(), source
    )
    if previous is not None and previous.next_cursor is None:
        return {"cycle": cycle, "source": source, "state": "already_complete"}
    try:
        page = discover(previous)
        if not isinstance(page, DiscoveryPage) or page.source != source:
            raise RuntimeError("metadata discovery returned the wrong source")
        commit = append_discovery_page(
            cos_client,
            bucket,
            f"{prefix.strip('/')}/discovery/{cycle}",
            page,
            previous=previous,
        )
        status: dict[str, object] = {
            "commit_sha256": commit.sha256,
            "cycle": cycle,
            "record_count": len(page.records),
            "source": source,
            "state": "complete" if commit.next_cursor is None else "page_complete",
        }
    except Exception as exc:
        _persist_failed(cos_client, bucket, prefix, cycle, source, exc)
        raise
    _persist_status(cos_client, bucket, prefix, status)
    _LOG.info("acquisition_terminal %s", _canonical(status).decode())
    return status


def _canonical(status: Mapping[str, object]) -> bytes:
    return json.dumps(status, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_document(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("discovery checkpoint is not JSON") from exc
    if not isinstance(value, dict) or _canonical(value) != body:
        raise RuntimeError("discovery checkpoint is not canonical JSON")
    return value


def _read_if_present(cos_client: Any, bucket: str, key: str) -> bytes | None:
    try:
        response = cos_client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        error = getattr(exc, "response", {}).get("Error", {})
        if str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    stream = response.get("Body") if isinstance(response, dict) else None
    read = getattr(stream, "read", None)
    body = read() if callable(read) else stream
    if not isinstance(body, bytes):
        raise RuntimeError("COS object body is unavailable")
    return body


def _required_string(document: dict[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"discovery checkpoint {field} is invalid")
    return value


def _sha256_string(document: dict[str, object], field: str) -> str:
    value = _required_string(document, field)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"discovery checkpoint {field} is invalid")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError("discovery checkpoint next_cursor is invalid")
    return value


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> None:
    """Create a credentialless Code Engine client and run one page."""
    from ibm_cloud_sdk_core.authenticators import ContainerAuthenticator

    bucket = _required_env("WATCH_AUTH_COS_BUCKET")
    prefix = os.environ.get("WATCH_AUTH_ACQUISITION_PREFIX", "datasets").strip("/")
    endpoint = _required_env("WATCH_AUTH_COS_ENDPOINT")
    instance_id = _required_env("WATCH_AUTH_COS_SERVICE_INSTANCE_ID")
    profile_id = _required_env("WATCH_AUTH_TRUSTED_PROFILE_ID")
    account_id = _required_env("WATCH_AUTH_ACCOUNT_ID")
    reader_policy_id = _required_env("WATCH_AUTH_IAM_READER_POLICY_ID")
    auth = ContainerAuthenticator(iam_profile_id=profile_id)
    cos = AtomicCOSClient(
        create_oauth_cos_client(
            auth,
            endpoint_url=endpoint,
            service_instance_id=instance_id,
        )
    )
    verify_no_public_access_policies(
        auth,
        account_id=account_id,
        profile_id=profile_id,
        reader_policy_id=reader_policy_id,
    )
    verify_private_versioned_bucket(cos, bucket)
    setattr(cos, "_watch_auth_bucket", bucket)
    setattr(cos, "_watch_auth_prefix", prefix)
    setattr(cos, "_watch_auth_configuration_verified", True)
    root = Path(__file__).resolve().parents[1]
    policies = load_source_policies(root / "config/source_policies.json")
    access_registry = load_source_access_registry(root / "config/source_access.json")
    image_requests_per_minute = access_registry["wikimedia"].requests_per_minute
    if image_requests_per_minute is None or image_requests_per_minute < 2:
        raise RuntimeError("Wikimedia rate limit cannot reserve the API request")
    logging.basicConfig(level=logging.INFO)
    now = datetime.now(timezone.utc)
    run_wikimedia_once(
        cos,
        bucket,
        prefix,
        get_source_policy(policies, "wikimedia"),
        access_registry["wikimedia"],
        now=now,
        api_client=requests.Session(),
        image_client=PinnedRequestsClient(
            requests_per_minute=image_requests_per_minute - 1
        ),
    )
    vsfactory_policy = get_source_policy(policies, "vsfactoryusa")
    vsfactory_access = access_registry["vsfactoryusa"]
    run_metadata_discovery_once(
        cos,
        bucket,
        prefix,
        "vsfactoryusa",
        now=now,
        discover=lambda previous: discover_vsfactoryusa(
            requests.Session(),
            vsfactory_policy,
            vsfactory_access,
            now=now,
            previous=previous,
        ),
    )
    goldsmiths_policy = get_source_policy(policies, "goldsmiths")
    goldsmiths_access = access_registry["goldsmiths"]
    run_metadata_discovery_once(
        cos,
        bucket,
        prefix,
        "goldsmiths",
        now=now,
        discover=lambda previous: discover_goldsmiths(
            requests.Session(),
            goldsmiths_policy,
            goldsmiths_access,
            now=now,
            previous=previous,
        ),
    )
    wos_policy = get_source_policy(policies, "watchesofswitzerland")
    wos_access = access_registry["watchesofswitzerland"]
    run_metadata_discovery_once(
        cos,
        bucket,
        prefix,
        "watchesofswitzerland",
        now=now,
        discover=lambda previous: discover_watchesofswitzerland(
            requests.Session(),
            wos_policy,
            wos_access,
            now=now,
            previous=previous,
        ),
    )


if __name__ == "__main__":
    main()
