"""Bounded daily census of research-only physical-watch manifests in IBM COS."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import Any

_RECEIPT_HASH = re.compile(r"(?:-|/)([0-9a-f]{64})\.json$")
_MAX_KEYS = 20_000
_MAX_OBJECT_BYTES = 1_000_000
_MAX_OUTPUT_BYTES = _MAX_OBJECT_BYTES
_LOG = logging.getLogger(__name__)
_V1_FIELDS = {
    "schema", "cycle", "manifest_count", "file_count", "probable_watch_count",
    "confirmed_watch_count", "training_admission_count", "manifest_sha256s",
}
_V2_FIELDS = _V1_FIELDS | {"provisional_cells", "file_role_claim_counts"}
_V3_FIELDS = _V2_FIELDS | {"provisional_view_cells"}


def run_research_census_once(cos: Any, bucket: str, prefix: str, *, now: datetime) -> dict[str, object]:
    """Count probable groups, never labels, after verifying every COS receipt."""
    from dataset.research_entities import read_research_entity_manifest

    if now.tzinfo is None or now.utcoffset() is None or not bucket or not prefix.strip("/"):
        raise ValueError("bucket, prefix and timezone-aware time are required")
    root = prefix.strip("/")
    manifest_keys = _list_keys(cos, bucket, f"{root}/research-entities/")
    receipt_keys = _list_keys(cos, bucket, f"{root}/receipts/wikimedia-")
    receipt_index: dict[str, str] = {}
    for key in receipt_keys:
        match = _RECEIPT_HASH.search(key)
        if match is None or match[1] in receipt_index:
            raise ValueError("research receipt inventory is ambiguous")
        receipt_index[match[1]] = key

    candidate_ids: set[str] = set()
    source_record_ids: set[str] = set()
    raw_hashes: set[str] = set()
    manifest_hashes: list[str] = []
    manifest_details: dict[str, tuple[str, str, int, int, tuple[str, ...]]] = {}
    files = 0
    probable = 0
    for key in manifest_keys:
        body = _read_bytes(cos, bucket, key)
        digest = hashlib.sha256(body).hexdigest()
        if not key.endswith(f"/{digest}.json"):
            raise ValueError("research manifest key differs from content")
        document = json.loads(body)
        if not isinstance(document, dict) or not isinstance(document.get("files"), list):
            raise ValueError("research manifest files are invalid")
        receipts: dict[str, bytes] = {}
        for item in document["files"]:
            receipt_hash = item.get("receipt_sha256") if isinstance(item, dict) else None
            if not isinstance(receipt_hash, str) or receipt_hash not in receipt_index:
                raise ValueError("research receipt is missing from COS inventory")
            receipts[receipt_hash] = _read_bytes(cos, bucket, receipt_index[receipt_hash])
        report = read_research_entity_manifest(body, receipts)
        for receipt_body in receipts.values():
            raw = json.loads(receipt_body).get("raw_sha256")
            if raw in raw_hashes:
                raise ValueError("research groups reuse an image")
            raw_hashes.add(raw)
        if candidate_ids.intersection(report.candidate_ids):
            raise ValueError("research groups reuse a candidate")
        if source_record_ids.intersection(report.source_record_ids):
            raise ValueError("research groups reuse a source record")
        candidate_ids.update(report.candidate_ids)
        source_record_ids.update(report.source_record_ids)
        manifest_hashes.append(report.manifest_sha256)
        manifest_details[report.manifest_sha256] = (
            report.source, document["reference"], report.file_count,
            report.probable_watch_count, tuple(item["role"] for item in document["files"]),
        )
        files += report.file_count
        probable += report.probable_watch_count

    cells, roles, views = _provisional_breakdown(manifest_hashes, manifest_details)
    status: dict[str, object] = {
        "schema": "watch-auth-research-census-v3",
        "cycle": now.date().isoformat(),
        "manifest_count": len(manifest_hashes),
        "file_count": files,
        "probable_watch_count": probable,
        "confirmed_watch_count": 0,
        "training_admission_count": 0,
        "manifest_sha256s": sorted(manifest_hashes),
        "provisional_cells": cells,
        "file_role_claim_counts": roles,
        "provisional_view_cells": views,
    }
    payload = json.dumps(status, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > _MAX_OUTPUT_BYTES:
        raise ValueError("research census output exceeds the object size limit")
    digest = hashlib.sha256(payload).hexdigest()
    key = f"{root}/research-census/{now.date().isoformat()}/terminal.json"
    cos.put_object_if_absent(
        Bucket=bucket, Key=key, Body=payload, ContentType="application/json",
        Metadata={"sha256": digest, "length": str(len(payload))},
    )
    terminal = _read_bytes(cos, bucket, key)
    if terminal != payload:
        _validate_prior_terminal(terminal, status, manifest_details)
        revision_key = f"{root}/research-census/{now.date().isoformat()}/revisions/{digest}.json"
        cos.put_object_if_absent(
            Bucket=bucket, Key=revision_key, Body=payload, ContentType="application/json",
            Metadata={"sha256": digest, "length": str(len(payload))},
        )
        if _read_bytes(cos, bucket, revision_key) != payload:
            raise RuntimeError("research census revision differs from current inputs")
        _LOG.info("research_census_revision %s %s", revision_key, payload.decode())
    else:
        _LOG.info("research_census_terminal %s", payload.decode())
    return status


def _validate_prior_terminal(
    body: bytes, current: dict[str, object],
    manifest_details: dict[str, tuple[str, str, int, int, tuple[str, ...]]],
) -> None:
    try:
        prior = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("research census already closed with invalid terminal") from exc
    schema = prior.get("schema") if isinstance(prior, dict) else None
    if (
        not isinstance(prior, dict)
        or schema not in {"watch-auth-research-census-v1", "watch-auth-research-census-v2",
                          "watch-auth-research-census-v3"}
        or set(prior) != {"watch-auth-research-census-v1": _V1_FIELDS,
                          "watch-auth-research-census-v2": _V2_FIELDS,
                          "watch-auth-research-census-v3": _V3_FIELDS}[schema]
        or prior.get("cycle") != current["cycle"]
        or prior.get("confirmed_watch_count") != 0
        or prior.get("training_admission_count") != 0
        or not isinstance(prior.get("manifest_sha256s"), list)
        or not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in prior["manifest_sha256s"])
        or len(set(prior["manifest_sha256s"])) != len(prior["manifest_sha256s"])
        or not set(prior["manifest_sha256s"]).issubset(manifest_details)
        or prior.get("manifest_count") != len(prior["manifest_sha256s"])
        or not isinstance(prior.get("file_count"), int)
        or isinstance(prior["file_count"], bool)
        or not isinstance(prior.get("probable_watch_count"), int)
        or isinstance(prior["probable_watch_count"], bool)
        or prior["file_count"] != sum(manifest_details[digest][2] for digest in prior["manifest_sha256s"])
        or prior["probable_watch_count"] != sum(manifest_details[digest][3] for digest in prior["manifest_sha256s"])
    ):
        raise RuntimeError("research census already closed with incompatible terminal")
    if schema in {"watch-auth-research-census-v2", "watch-auth-research-census-v3"}:
        cells, roles, views = _provisional_breakdown(prior["manifest_sha256s"], manifest_details)
        if _canonical(prior["provisional_cells"], prior["file_role_claim_counts"]) != _canonical(cells, roles):
            raise RuntimeError("research census already closed with incompatible terminal")
        if schema == "watch-auth-research-census-v3" and _canonical(prior["provisional_view_cells"]) != _canonical(views):
            raise RuntimeError("research census already closed with incompatible terminal")


def _canonical(*values: object) -> str:
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _provisional_breakdown(
    digests: list[str],
    details: dict[str, tuple[str, str, int, int, tuple[str, ...]]],
) -> tuple[list[dict[str, object]], dict[str, int], list[dict[str, object]]]:
    cells: dict[tuple[str, str], list[int]] = {}
    view_cells: dict[tuple[str, str, str], list[int]] = {}
    roles: Counter[str] = Counter()
    for digest in digests:
        source, reference, photos, watches, file_roles = details[digest]
        cell = cells.setdefault((source, reference), [0, 0])
        cell[0] += watches
        cell[1] += photos
        roles.update(file_roles)
        for role, count in Counter(file_roles).items():
            view = view_cells.setdefault((source, reference, role), [0, 0])
            view[0] += watches
            view[1] += count
    return (
        [
            {"source": source, "reference_claim": reference,
             "probable_watch_count": counts[0], "file_count": counts[1]}
            for (source, reference), counts in sorted(cells.items())
        ],
        dict(sorted(roles.items())),
        [
            {"source": source, "reference_claim": reference, "view_claim": role,
             "class_status": "unverified", "probable_watch_count": counts[0],
             "file_count": counts[1]}
            for (source, reference, role), counts in sorted(view_cells.items())
        ],
    )


def _list_keys(cos: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    while True:
        params: dict[str, object] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token is not None:
            params["ContinuationToken"] = token
        page = cos.list_objects_v2(**params)
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("Contents", []), list)
            or not isinstance(page.get("IsTruncated"), bool)
        ):
            raise ValueError("research COS inventory is invalid")
        for item in page.get("Contents", []):
            key = item.get("Key") if isinstance(item, dict) else None
            if not isinstance(key, str) or not key.startswith(prefix) or not key.endswith(".json"):
                raise ValueError("research COS key is invalid")
            keys.append(key)
        if len(keys) > _MAX_KEYS or len(set(keys)) != len(keys):
            raise ValueError("research COS inventory is too large or duplicated")
        if page.get("IsTruncated") is not True:
            return keys
        next_token = page.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token in seen_tokens:
            raise ValueError("research COS pagination is invalid")
        seen_tokens.add(next_token)
        token = next_token


def _read_bytes(cos: Any, bucket: str, key: str) -> bytes:
    response = cos.get_object(Bucket=bucket, Key=key)
    body = response.get("Body") if isinstance(response, dict) else None
    reader = getattr(body, "read", None)
    raw = reader(_MAX_OBJECT_BYTES + 1) if callable(reader) else body
    if not isinstance(raw, bytes) or len(raw) > _MAX_OBJECT_BYTES:
        raise ValueError("research COS object body is unavailable")
    return raw
