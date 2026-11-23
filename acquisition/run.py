"""Bounded, resumable source acquisition for IBM Code Engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import requests

from acquisition.collect import (
    CandidateMetadata,
    DownloadRateLimited,
    UnsafeImage,
    collect_batch,
)
from acquisition.cos import AtomicCOSClient
from acquisition.discovery import DiscoveryCommit, DiscoveryPage, append_discovery_page
from acquisition.http import PinnedRequestsClient
from acquisition.research_census import run_research_census_once
from acquisition.source_policy import (
    authorize_candidate,
    authorize_discovery,
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from acquisition.supercloneluxury_archive import CatalogError
from acquisition.types import (
    CollectionAuthorizationRecord,
    SourceAccessRecord,
    SourcePolicy,
    TrainingRecord,
)
from infra.ibm_cos import (
    IntegrityError,
    create_oauth_cos_client,
    verify_no_public_access_policies,
    verify_private_versioned_bucket,
)
from scrapers.bobswatches import BobsWatchesDiscoveryError, discover_bobswatches
from scrapers.cbp import CBPDiscoveryError, discover_cbp
from scrapers.goldsmiths import GoldsmithsAPIError, discover_goldsmiths
from scrapers.metadata_http import HTTPDeadlineExceeded
from scrapers.precisionwatches import PrecisionWatchesAPIError, discover_precisionwatches
from scrapers.replica_sitemap import ReplicaSitemapError
from scrapers.richardmillereplica import discover_richardmillereplica
from scrapers.richfactoryus import discover_richfactoryus
from scrapers.supercloneluxury import discover_supercloneluxury
from scrapers.superclonerolex import SuperCloneRolexAPIError, discover_superclonerolex
from scrapers.swisswatchexpo import SwissWatchExpoDiscoveryError, discover_swisswatchexpo
from scrapers.vsfactoryusa import discover_vsfactoryusa
from scrapers.vsfwatcheshub import discover_vsfwatcheshub
from scrapers.watchesofswitzerland import (
    WatchesOfSwitzerlandAPIError,
    discover_watchesofswitzerland,
)
from scrapers.wikimedia import WikimediaAPIError, WikimediaDiscovery, discover_wikimedia

if TYPE_CHECKING:
    from acquisition.planner import AcquisitionTarget

_LOG = logging.getLogger(__name__)
_RATE_LIMITED = "Wikimedia rate limited; resume later"
_OWNER_PUBLIC_DOWNLOAD_DIRECTIVE = (
    "2026-11-12 project owner authorization: archive publicly accessible watch "
    "images privately; do not treat download as permission to train or proof of authenticity"
)
_RECOVERABLE_METADATA_ERRORS = (
    BobsWatchesDiscoveryError,
    CBPDiscoveryError,
    GoldsmithsAPIError,
    HTTPDeadlineExceeded,
    PrecisionWatchesAPIError,
    ReplicaSitemapError,
    requests.RequestException,
    SuperCloneRolexAPIError,
    SwissWatchExpoDiscoveryError,
    CatalogError,
    UnicodeDecodeError,
    WatchesOfSwitzerlandAPIError,
)


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
        return _persist_already_complete(cos_client, bucket, prefix, cycle, "wikimedia", now)

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
                rights_evidence=candidate.rights_evidence,
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


def run_wikimedia_lanes(
    cos_client: Any,
    bucket: str,
    prefix: str,
    policy: SourcePolicy,
    access: SourceAccessRecord,
    *,
    now: datetime,
    lanes: tuple[tuple[str, tuple[int, int]], ...],
) -> None:
    """Advance independent Commons lanes within one source request budget."""
    if not lanes or access.requests_per_minute is None or access.requests_per_minute < 4 * len(lanes):
        raise RuntimeError("Wikimedia rate limit cannot cover all lanes")
    lane_limit = access.requests_per_minute // len(lanes)
    lane_access = replace(access, requests_per_minute=lane_limit)
    errors: list[Exception] = []
    for lane, query_range in lanes:
        try:
            run_wikimedia_once(
                cos_client, bucket, f"{prefix}/lanes/{lane}",
                policy, lane_access, now=now, api_client=requests.Session(),
                image_client=PinnedRequestsClient(requests_per_minute=lane_limit - 1),
                discover=lambda *args, _query_range=query_range, **kwargs: discover_wikimedia(
                    *args, **kwargs, query_range=_query_range
                ),
            )
        except Exception as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("Wikimedia lanes failed", errors)


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
    jobrun = os.environ.get("CE_JOBRUN", "").strip()
    if jobrun:
        status["code_engine_jobrun"] = jobrun
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


def _persist_already_complete(
    cos_client: Any, bucket: str, prefix: str, cycle: str, source: str, now: datetime
) -> dict[str, object]:
    status: dict[str, object] = {
        "cycle": cycle,
        "source": source,
        "state": "already_complete",
        "run_started_at_utc": now.astimezone(timezone.utc).isoformat(),
    }
    _persist_status(cos_client, bucket, prefix, status)
    _LOG.info("acquisition_terminal %s", _canonical(status).decode())
    return status


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
        return _persist_already_complete(cos_client, bucket, prefix, cycle, source, now)
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


def run_metadata_discovery_resilient(
    cos_client: Any,
    bucket: str,
    prefix: str,
    source: str,
    *,
    now: datetime,
    discover: Callable[[DiscoveryCommit | None], Any],
) -> dict[str, object]:
    """Keep independent sources running after a recorded provider failure."""
    try:
        return run_metadata_discovery_once(
            cos_client,
            bucket,
            prefix,
            source,
            now=now,
            discover=discover,
        )
    except _RECOVERABLE_METADATA_ERRORS as exc:
        cycle, _previous = _active_cycle(
            cos_client, bucket, prefix, now.date().isoformat(), source
        )
        return {
            "cycle": cycle,
            "error_type": type(exc).__name__,
            "source": source,
            "state": "failed",
        }


def owner_public_download_authorization(
    policy: SourcePolicy,
    access: SourceAccessRecord,
) -> CollectionAuthorizationRecord:
    """Bind the owner's standing private-download directive to one source."""
    return CollectionAuthorizationRecord(
        rights_record_id=f"owner-public-download-20261112-{policy.source}",
        source=policy.source,
        acquisition_method=access.acquisition_method,
        allowed_purposes=("download",),
        authorized_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
        authorized_by="project-owner",
        evidence_sha256=hashlib.sha256(
            _OWNER_PUBLIC_DOWNLOAD_DIRECTIVE.encode()
        ).hexdigest(),
    )


def run_public_image_collection_once(
    cos_client: Any,
    bucket: str,
    prefix: str,
    policy: SourcePolicy,
    access: SourceAccessRecord,
    authorization: CollectionAuthorizationRecord,
    *,
    now: datetime,
    image_client: Any,
    discover: Callable[[DiscoveryCommit | None], Any],
    candidate_limit: int | None = None,
    on_collected: Callable[[tuple[str, ...], str, str, int], None] | None = None,
) -> dict[str, object]:
    """Discover, privately archive every image, then commit the source cursor."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    collection_prefix = f"{prefix.strip('/')}/public-downloads"
    source = policy.source
    cycle, previous = _active_cycle(
        cos_client, bucket, collection_prefix, now.date().isoformat(), source
    )
    if previous is not None and previous.next_cursor is None:
        return _persist_already_complete(
            cos_client, bucket, collection_prefix, cycle, source, now
        )

    page = discover(previous)
    if not isinstance(page, DiscoveryPage) or page.source != source:
        raise RuntimeError("public image discovery returned the wrong source")
    if candidate_limit is not None and (
        isinstance(candidate_limit, bool)
        or not isinstance(candidate_limit, int)
        or candidate_limit < 1
        or sum(len(record.image_urls) for record in page.records) > candidate_limit
    ):
        raise ValueError("discovery page exceeds bounded candidate wave; cursor not advanced")
    candidates: list[CandidateMetadata] = []
    for record in page.records:
        metadata = dict(record.metadata)
        listing_text = (
            "replica"
            if metadata.get("classification") in {
                "replica_candidate_vendor_declared", "vendor_declared_replica"
            }
            else None
        )
        for index, (image_id, image_url) in enumerate(
            zip(record.image_ids, record.image_urls, strict=True), start=1
        ):
            candidates.append(
                CandidateMetadata(
                    source=source,
                    source_record_id=(f"{record.source_record_id}#image-{index}-{image_id[:12]}"),
                    source_url=record.source_url,
                    image_url=image_url,
                    declared_reference=record.declared_reference,
                    listing_text=listing_text,
                    factory=metadata.get("factory_claim"),
                )
            )
    batch_id = (
        "public-"
        + hashlib.sha256(f"{source}\0{cycle}\0{page.cursor or 'root'}".encode()).hexdigest()[:24]
    )
    accepted_candidate_ids: list[str] = []
    rejected_images: list[dict[str, str]] = []
    for candidate in candidates:
        try:
            receipt = collect_batch(
                (candidate,), policy, authorization, access=access,
                http_client=image_client, cos_client=cos_client, bucket=bucket,
                batch_id=batch_id, now=now,
            )
        except UnsafeImage as exc:
            rejected_images.append({
                "source_record_id": candidate.source_record_id,
                "image_url": candidate.image_url,
                "reason": str(exc),
            })
            continue
        accepted_candidate_ids.extend(receipt.accepted_candidate_ids)
    rejection_key = None
    if rejected_images:
        rejection_record: dict[str, object] = {
            "batch_id": batch_id, "cycle": cycle, "source": source,
            "rejected_images": rejected_images,
        }
        rejection_digest = hashlib.sha256(_canonical(rejection_record)).hexdigest()
        rejection_key = (
            f"{collection_prefix}/rejections/{cycle}/{source}/"
            f"{batch_id}-{rejection_digest}.json"
        )
        if _immutable_review_document(cos_client, bucket, rejection_key, rejection_record) != rejection_record:
            raise RuntimeError("public image rejection readback differs")
    if on_collected is not None:
        on_collected(tuple(accepted_candidate_ids), cycle, batch_id, len(candidates))
    commit = append_discovery_page(
        cos_client,
        bucket,
        f"{collection_prefix}/discovery/{cycle}",
        page,
        previous=previous,
    )
    status: dict[str, object] = {
        "accepted_candidate_ids": sorted(accepted_candidate_ids),
        "accepted_count": len(accepted_candidate_ids),
        "rejected_images": rejected_images,
        "rejection_key": rejection_key,
        "commit_sha256": commit.sha256,
        "cycle": cycle,
        "next_cursor": commit.next_cursor,
        "source": source,
        "state": "complete" if commit.next_cursor is None else "page_complete",
    }
    _persist_status(cos_client, bucket, collection_prefix, status)
    _LOG.info("acquisition_terminal %s", _canonical(status).decode())
    return status


def run_planned_public_collection_once(
    cos_client: Any,
    bucket: str,
    prefix: str,
    targets: Sequence[AcquisitionTarget],
    *,
    policies: Mapping[str, SourcePolicy],
    accesses: Mapping[str, SourceAccessRecord],
    authorizations: Mapping[str, CollectionAuthorizationRecord],
    discoveries: Mapping[str, Callable[[DiscoveryCommit | None], DiscoveryPage]],
    now: datetime,
    image_client: Any,
    known_entity_ids: frozenset[str],
    review: Callable[
        [AcquisitionTarget, tuple[str, ...], str],
        tuple[Mapping[str, str], Sequence[TrainingRecord]],
    ],
) -> dict[str, object]:
    """Collect one authorized planned wave; measure only manifest-admitted new entities.

    review(target, candidate_ids, batch_id) must persist/reuse its admission decision
    idempotently by batch_id before returning candidate bindings and admitted rows.
    It may instead read already persisted decisions without admission side effects.
    A failed result write retries the same operation key and pre-admission baseline;
    discovery metadata and private-download receipts never establish eligibility.
    """
    target = next(
        (
            item
            for item in targets
            if item.source in policies
            and item.source in accesses
            and item.source in authorizations
            and item.source in discoveries
            and authorize_discovery(policies[item.source], accesses[item.source], now=now).allowed
            and authorize_candidate(policies[item.source], authorizations[item.source], now=now).may_download
        ),
        None,
    )
    if target is None:
        return {"state": "no_viable_target"}
    source = target.source

    def bounded_discover(previous: DiscoveryCommit | None) -> DiscoveryPage:
        page = discoveries[source](previous)
        if any(record.seller_group_id != target.seller_group_id for record in page.records):
            raise ValueError("planned wave contains an unplanned seller")
        return page

    reviewed_status: dict[str, object] = {}

    def on_collected(candidate_ids: tuple[str, ...], cycle: str, batch_id: str, consumed: int) -> None:
        nonlocal reviewed_status
        root = f"{prefix.strip('/')}/eligible-yields/batches/{batch_id}"
        context = {"source": source, "reference_id": target.reference_id,
                   "output": target.output, "angle": target.angle,
                   "seller_group_id": target.seller_group_id,
                   "candidate_ids": sorted(candidate_ids), "collection_batch_id": batch_id,
                   "candidate_consumption": consumed}
        baseline = _immutable_review_document(
            cos_client, bucket, root + "/baseline.json",
            {"context": context, "baseline_entity_ids": sorted(known_entity_ids)},
        )
        if baseline.get("context") != context:
            raise RuntimeError("collection batch review context conflicts with immutable baseline")
        baseline_ids = baseline.get("baseline_entity_ids")
        if not isinstance(baseline_ids, list) or not all(isinstance(item, str) for item in baseline_ids):
            raise RuntimeError("collection review baseline entity IDs invalid")
        previous = _read_if_present(cos_client, bucket, root + "/review.json")
        if previous is not None:
            reviewed_status = _canonical_document(previous)
            if reviewed_status.get("context") != context:
                raise RuntimeError("immutable collection review context mismatch")
            return
        bindings, admitted = review(target, candidate_ids, batch_id)
        if set(bindings) != set(candidate_ids) or any(not entity for entity in bindings.values()):
            raise ValueError("review must bind every collected candidate to a watch entity")
        new_entities: set[str] = set()
        for row in admitted:
            output = "genuine" if row.label.value == "genuine" else f"fake-{row.grade}"
            if (
                row.candidate_id not in bindings
                or row.watch_entity_id != bindings[row.candidate_id]
                or row.reference_id != target.reference_id
                or row.source_domain != source
                or row.seller_group_id != target.seller_group_id
                or row.angle is None
                or row.angle.value != target.angle
                or output != target.output
                or not row.evidence_tier.training_eligible
                or row.is_augmented
            ):
                raise ValueError("admitted row does not match reviewed planned cell")
            if row.watch_entity_id not in baseline_ids:
                new_entities.add(row.watch_entity_id)
        if len(new_entities) > target.eligible_source_capacity:
            raise ValueError("admitted yield exceeds source/seller capacity")
        reviewed_status = {
            "cycle": cycle,
            "context": context,
            "baseline_entity_ids": baseline_ids,
            "source": source,
            "state": "review_complete",
            "collection_batch_id": batch_id,
            "output": target.output,
            "angle": target.angle,
            "seller_group_id": target.seller_group_id,
            "exploratory": target.exploratory,
            "candidate_limit": target.candidate_limit,
            "missing_images": target.missing_images,
            "missing_entities": target.missing_entities,
            "eligible_source_capacity": target.eligible_source_capacity,
            "eligible_yield": [len(set(bindings.values())), len(new_entities)],
            "candidate_consumption": consumed,
            "reviewed_at": now.isoformat(),
            "evidence_record_ids": sorted({row.evidence_record_id for row in admitted}),
        }
        reviewed_status = _immutable_review_document(
            cos_client, bucket, root + "/review.json", reviewed_status,
        )

    status = run_public_image_collection_once(
        cos_client, bucket, prefix, policies[source], accesses[source], authorizations[source],
        now=now, image_client=image_client, discover=bounded_discover,
        candidate_limit=target.candidate_limit, on_collected=on_collected,
    )
    return {**reviewed_status, **status}


def _immutable_review_document(
    cos_client: Any, bucket: str, key: str, document: dict[str, object],
) -> dict[str, object]:
    body = _canonical(document)
    cos_client.put_object_if_absent(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json",
        Metadata={"sha256": hashlib.sha256(body).hexdigest(), "length": str(len(body))},
    )
    persisted = _read_if_present(cos_client, bucket, key)
    if persisted is None:
        raise RuntimeError("immutable collection review readback failed")
    return _canonical_document(persisted)


def run_public_image_collection_resilient(
    cos_client: Any,
    bucket: str,
    prefix: str,
    policy: SourcePolicy,
    access: SourceAccessRecord,
    *,
    now: datetime,
    discover: Callable[[DiscoveryCommit | None], Any],
) -> dict[str, object]:
    """Keep later public sources running after a recorded provider failure."""
    try:
        requests_per_minute = access.requests_per_minute
        if requests_per_minute is None:
            raise RuntimeError("public image source lacks a rate limit")
        return run_public_image_collection_once(
            cos_client,
            bucket,
            prefix,
            policy,
            access,
            owner_public_download_authorization(policy, access),
            now=now,
            image_client=PinnedRequestsClient(
                requests_per_minute=requests_per_minute
            ),
            discover=discover,
        )
    except _RECOVERABLE_METADATA_ERRORS + (
        DownloadRateLimited,
        IntegrityError,
        PermissionError,
        UnsafeImage,
    ) as exc:
        collection_prefix = f"{prefix.strip('/')}/public-downloads"
        cycle, _previous = _active_cycle(
            cos_client, bucket, collection_prefix, now.date().isoformat(), policy.source
        )
        _persist_failed(
            cos_client, bucket, collection_prefix, cycle, policy.source, exc
        )
        return {
            "cycle": cycle,
            "error_type": type(exc).__name__,
            "source": policy.source,
            "state": "failed",
        }


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


def main(source: str = "all") -> None:
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
    if source not in {"all", "supercloneluxury", "richardmillereplica", "cbp", "goldsmiths", "bobswatches", "swisswatchexpo", "wikimedia-speedmaster", "wikimedia-auckland", "wikimedia-classic-speedmaster", "wikimedia-daytona", "wikimedia-royal-oak", "wikimedia-counterfeit-category", "research-census"}:
        raise ValueError("unsupported acquisition source")
    logging.basicConfig(level=logging.INFO)
    now = datetime.now(timezone.utc)
    job_deadline = time.monotonic() + 420
    if source == "research-census":
        run_research_census_once(cos, bucket, prefix, now=now)
        return
    if source == "supercloneluxury":
        luxury_policy = get_source_policy(policies, source)
        luxury_access = access_registry[source]
        run_public_image_collection_resilient(
            cos,
            bucket,
            prefix,
            luxury_policy,
            luxury_access,
            now=now,
            discover=lambda previous: discover_supercloneluxury(
                requests.Session(),
                luxury_policy,
                luxury_access,
                now=now,
                previous=previous,
                deadline=job_deadline,
            ),
        )
        return
    if source == "richardmillereplica":
        policy = get_source_policy(policies, source)
        access = access_registry[source]
        run_public_image_collection_resilient(
            cos, bucket, prefix, policy, access, now=now,
            discover=lambda previous: discover_richardmillereplica(
                requests.Session(), policy, access, now=now, previous=previous
            ),
        )
        return
    if source == "cbp":
        cbp_policy = get_source_policy(policies, source)
        cbp_access = access_registry[source]
        run_public_image_collection_resilient(
            cos,
            bucket,
            prefix,
            cbp_policy,
            cbp_access,
            now=now,
            discover=lambda previous: discover_cbp(
                requests.Session(),
                cbp_policy,
                cbp_access,
                now=now,
                previous=previous,
                deadline=job_deadline,
            ),
        )
        return
    if source == "goldsmiths":
        goldsmiths_policy = get_source_policy(policies, source)
        goldsmiths_access = access_registry[source]
        run_metadata_discovery_resilient(
            cos,
            bucket,
            prefix,
            source,
            now=now,
            discover=lambda previous: discover_goldsmiths(
                requests.Session(),
                goldsmiths_policy,
                goldsmiths_access,
                now=now,
                previous=previous,
                deadline=job_deadline,
            ),
        )
        return
    if source == "bobswatches":
        bobs_policy = get_source_policy(policies, source)
        bobs_access = access_registry[source]
        run_metadata_discovery_resilient(
            cos, bucket, prefix, source, now=now,
            discover=lambda previous: discover_bobswatches(
                requests.Session(), bobs_policy, bobs_access,
                now=now, previous=previous, deadline=job_deadline,
            ),
        )
        return
    if source == "swisswatchexpo":
        swe_policy = get_source_policy(policies, source)
        swe_access = access_registry[source]
        if swe_access.requests_per_minute is None:
            raise RuntimeError("SwissWatchExpo request limit is missing")
        for _ in range(16):
            result = run_metadata_discovery_resilient(
                cos, bucket, prefix, source, now=now,
                discover=lambda previous: discover_swisswatchexpo(
                    requests.Session(), swe_policy, swe_access,
                    now=now, previous=previous, deadline=job_deadline,
                ),
            )
            if result["state"] in {"complete", "already_complete", "failed"}:
                return
            time.sleep(60 / swe_access.requests_per_minute)
        raise RuntimeError("SwissWatchExpo exceeded bounded sitemap page count")
    if source == "wikimedia-speedmaster":
        access = access_registry["wikimedia"]
        run_wikimedia_lanes(
            cos, bucket, prefix, get_source_policy(policies, "wikimedia"), access,
            now=now,
            lanes=(("speedmaster/replica-research", (6, 7)),
                   ("classic-speedmaster/research", (9, 10)),
                   ("nasa-speedmaster/research", (14, 15))),
        )
        return
    if source == "wikimedia-auckland":
        access = access_registry["wikimedia"]
        if access.requests_per_minute is None or access.requests_per_minute < 4:
            raise RuntimeError("Wikimedia rate limit cannot cover API and image retries")
        run_wikimedia_once(
            cos, bucket, f"{prefix}/lanes/auckland-oyster/research",
            get_source_policy(policies, "wikimedia"), access,
            now=now, api_client=requests.Session(),
            image_client=PinnedRequestsClient(requests_per_minute=access.requests_per_minute - 1),
            discover=lambda *args, **kwargs: discover_wikimedia(*args, **kwargs, query_range=(8, 9)),
        )
        return
    if source == "wikimedia-classic-speedmaster":
        access = access_registry["wikimedia"]
        if access.requests_per_minute is None or access.requests_per_minute < 4:
            raise RuntimeError("Wikimedia rate limit cannot cover API and image retries")
        run_wikimedia_once(
            cos, bucket, f"{prefix}/lanes/classic-speedmaster/research",
            get_source_policy(policies, "wikimedia"), access,
            now=now, api_client=requests.Session(),
            image_client=PinnedRequestsClient(requests_per_minute=access.requests_per_minute - 1),
            discover=lambda *args, **kwargs: discover_wikimedia(*args, **kwargs, query_range=(9, 10)),
        )
        return
    if source == "wikimedia-daytona":
        access = access_registry["wikimedia"]
        run_wikimedia_lanes(
            cos, bucket, prefix, get_source_policy(policies, "wikimedia"), access, now=now,
            lanes=(("daytona/replica-research", (10, 12)),
                   ("datejust/replica-research", (7, 8))),
        )
        return
    if source == "wikimedia-royal-oak":
        access = access_registry["wikimedia"]
        if access.requests_per_minute is None or access.requests_per_minute < 4:
            raise RuntimeError("Wikimedia rate limit cannot cover API and image retries")
        run_wikimedia_once(
            cos, bucket, f"{prefix}/lanes/royal-oak/research",
            get_source_policy(policies, "wikimedia"), access,
            now=now, api_client=requests.Session(),
            image_client=PinnedRequestsClient(requests_per_minute=access.requests_per_minute - 1),
            discover=lambda *args, **kwargs: discover_wikimedia(*args, **kwargs, query_range=(12, 13)),
        )
        return
    if source == "wikimedia-counterfeit-category":
        access = access_registry["wikimedia"]
        run_wikimedia_lanes(
            cos, bucket, prefix, get_source_policy(policies, "wikimedia"), access,
            now=now,
            lanes=(("counterfeit-category/research", (13, 14)),
                   ("counterfeit-broad-category/research", (15, 16))),
        )
        return
    image_requests_per_minute = access_registry["wikimedia"].requests_per_minute
    if image_requests_per_minute is None or image_requests_per_minute < 2:
        raise RuntimeError("Wikimedia rate limit cannot reserve the API request")
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
    run_public_image_collection_resilient(
        cos,
        bucket,
        prefix,
        vsfactory_policy,
        vsfactory_access,
        now=now,
        discover=lambda previous: discover_vsfactoryusa(
            requests.Session(),
            vsfactory_policy,
            vsfactory_access,
            now=now,
            previous=previous,
        ),
    )
    vsfwatcheshub_policy = get_source_policy(policies, "vsfwatcheshub")
    vsfwatcheshub_access = access_registry["vsfwatcheshub"]
    run_public_image_collection_resilient(
        cos,
        bucket,
        prefix,
        vsfwatcheshub_policy,
        vsfwatcheshub_access,
        now=now,
        discover=lambda previous: discover_vsfwatcheshub(
            requests.Session(),
            vsfwatcheshub_policy,
            vsfwatcheshub_access,
            now=now,
            previous=previous,
        ),
    )
    richfactory_policy = get_source_policy(policies, "richfactoryus")
    richfactory_access = access_registry["richfactoryus"]
    run_public_image_collection_resilient(
        cos,
        bucket,
        prefix,
        richfactory_policy,
        richfactory_access,
        now=now,
        discover=lambda previous: discover_richfactoryus(
            requests.Session(),
            richfactory_policy,
            richfactory_access,
            now=now,
            previous=previous,
        ),
    )
    superclone_policy = get_source_policy(policies, "superclonerolex")
    superclone_access = access_registry["superclonerolex"]
    run_public_image_collection_resilient(
        cos,
        bucket,
        prefix,
        superclone_policy,
        superclone_access,
        now=now,
        discover=lambda previous: discover_superclonerolex(
            requests.Session(),
            superclone_policy,
            superclone_access,
            now=now,
            previous=previous,
            deadline=job_deadline,
        ),
    )
    precision_policy = get_source_policy(policies, "precisionwatches")
    precision_access = access_registry["precisionwatches"]
    run_metadata_discovery_resilient(
        cos,
        bucket,
        prefix,
        "precisionwatches",
        now=now,
        discover=lambda previous: discover_precisionwatches(
            requests.Session(),
            precision_policy,
            precision_access,
            now=now,
            previous=previous,
            deadline=job_deadline,
        ),
    )
    wos_policy = get_source_policy(policies, "watchesofswitzerland")
    wos_access = access_registry["watchesofswitzerland"]
    run_metadata_discovery_resilient(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("all", "supercloneluxury", "cbp", "goldsmiths", "bobswatches", "swisswatchexpo", "wikimedia-speedmaster", "wikimedia-auckland", "wikimedia-classic-speedmaster", "wikimedia-daytona", "wikimedia-royal-oak", "wikimedia-counterfeit-category", "research-census"), default="all")
    main(parser.parse_args().source)
