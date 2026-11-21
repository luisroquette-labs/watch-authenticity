"""Read research-only physical-watch groups without admitting training labels."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ResearchEntityReport:
    manifest_sha256: str
    source: str
    candidate_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    file_count: int
    probable_watch_count: int
    confirmed_watch_count: int


def read_research_entity_manifest(
    body: bytes, receipts_by_sha256: Mapping[str, bytes]
) -> ResearchEntityReport:
    """Verify one provisional group against its immutable acquisition receipts."""
    manifest = json.loads(body)
    if not isinstance(manifest, dict) or manifest.get("schema") != "watch-auth-research-entity-v1":
        raise ValueError("unsupported research entity manifest")
    if manifest.get("training_admission") is not False or manifest.get("test_admission") is not False:
        raise ValueError("research entity manifest cannot admit model data")
    source = manifest.get("source")
    reference = manifest.get("reference")
    files = manifest.get("files")
    probable = manifest.get("physical_watch_count_probable")
    confirmed = manifest.get("physical_watch_count_confirmed")
    if (
        source != "wikimedia"
        or not isinstance(reference, str) or not reference
        or not isinstance(files, list) or not files
        or manifest.get("physical_watch_group") != "probable-one-watch"
        or not isinstance(manifest.get("visual_basis"), str) or not manifest["visual_basis"].strip()
    ):
        raise ValueError("research entity source, reference and files are required")
    if (
        probable != 1 or isinstance(probable, bool)
        or confirmed != 0 or isinstance(confirmed, bool)
        or isinstance(manifest.get("content_file_count"), bool)
        or manifest.get("content_file_count") != len(files)
    ):
        raise ValueError("research entity counts are inconsistent")

    candidate_ids: list[str] = []
    source_record_ids: list[str] = []
    receipt_hashes: set[str] = set()
    for file in files:
        if not isinstance(file, dict):
            raise ValueError("research entity file must be an object")
        if any(not isinstance(file.get(key), str) or not file[key] for key in ("candidate_id", "source_record_id")):
            raise ValueError("research entity candidate identity is required")
        if any(not _is_sha256(file.get(key)) for key in ("raw_sha256", "derivative_sha256")):
            raise ValueError("research entity image hash is invalid")
        digest = file.get("receipt_sha256")
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ValueError("receipt hash is invalid")
        receipt_body = receipts_by_sha256.get(digest)
        if receipt_body is None or hashlib.sha256(receipt_body).hexdigest() != digest:
            raise ValueError("research entity receipt is missing or changed")
        receipt = json.loads(receipt_body)
        if not isinstance(receipt, dict) or any(
            receipt.get(key) != file.get(key)
            for key in ("candidate_id", "source_record_id", "raw_sha256", "derivative_sha256")
        ):
            raise ValueError("research entity receipt lineage differs")
        if receipt.get("proposed_label") is not None or receipt.get("proposed_grade") is not None:
            raise ValueError("research entity receipt contains a model label")
        if receipt.get("declared_reference") != manifest.get("reference"):
            raise ValueError("research entity reference differs from receipt")
        if (
            not isinstance(receipt.get("policy_id"), str)
            or not receipt["policy_id"].startswith("wikimedia-")
            or not isinstance(receipt.get("source_url"), str)
            or urlsplit(receipt["source_url"]).hostname != "commons.wikimedia.org"
        ):
            raise ValueError("research entity receipt source differs")
        if not isinstance(file.get("role"), str) or not file["role"]:
            raise ValueError("research entity file role is required")
        candidate_ids.append(receipt["candidate_id"])
        record_id = receipt["source_record_id"]
        match = re.fullmatch(r"q\d+:(\d+)", record_id)
        source_record_ids.append(match[1] if match else record_id)
        receipt_hashes.add(digest)
    if (len(set(candidate_ids)) != len(files) or len(set(source_record_ids)) != len(files)
            or len(receipt_hashes) != len(files)):
        raise ValueError("research entity files must be distinct")
    return ResearchEntityReport(
        manifest_sha256=hashlib.sha256(body).hexdigest(),
        source=source,
        candidate_ids=tuple(sorted(candidate_ids)),
        source_record_ids=tuple(sorted(source_record_ids)),
        file_count=len(files),
        probable_watch_count=probable,
        confirmed_watch_count=0,
    )


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
