"""Safe JSONL partner-export import under one declared directory root."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any

from acquisition.collect import (
    MAX_IMAGE_BYTES,
    AcquisitionRefused,
    persist_candidate_payload,
    sanitize_image,
)
from acquisition.source_policy import authorize_candidate
from acquisition.types import BatchReceipt, CandidateRecord, RightsRecord, SourcePolicy
from core.rules import propose_source_label
from infra.ibm_cos import upload_verified


@dataclass(frozen=True)
class _PartnerPayload:
    candidate: CandidateRecord
    raw: bytes
    derivative: bytes


def import_partner_export(path: Path, declaration: RightsRecord) -> list[CandidateRecord]:
    """Hash and validate declared files; reject traversal, symlinks and extras."""

    return [payload.candidate for payload in _import_partner_payloads(path, declaration)]


def collect_partner_batch(
    path: Path,
    evidence_path: Path,
    policy: SourcePolicy,
    declaration: RightsRecord | None,
    *,
    cos_client: Any,
    bucket: str,
    batch_id: str,
    now: datetime,
) -> BatchReceipt:
    """Persist a rights-cleared local export through the canonical collector."""
    decision = authorize_candidate(policy, declaration, now=now)
    if not decision.may_download:
        raise AcquisitionRefused(", ".join(decision.reasons))
    assert declaration is not None
    root = path.parent.resolve()
    try:
        evidence_relative = evidence_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("rights evidence must be inside the partner export") from exc
    safe_evidence = _safe_file(root, evidence_relative)
    evidence = _read_regular_file(safe_evidence)
    evidence_sha256 = hashlib.sha256(evidence).hexdigest()
    if evidence_sha256 != declaration.evidence_sha256:
        raise ValueError("rights evidence hash differs from declaration")

    payloads = _import_partner_payloads(
        path,
        declaration,
        extra_allowed=(safe_evidence,),
    )
    if any(payload.candidate.acquisition_batch_id != batch_id for payload in payloads):
        raise ValueError("partner candidate belongs to another batch")
    if any(payload.candidate.source_policy_id != policy.policy_id for payload in payloads):
        raise ValueError("partner candidate belongs to another source policy")
    if any(
        not declaration.effective_at <= payload.candidate.collected_at < declaration.expires_at
        for payload in payloads
    ):
        raise ValueError("partner candidate was collected outside rights validity")

    upload_verified(
        cos_client,
        safe_evidence,
        bucket,
        f"datasets/evidence/{evidence_sha256}",
        evidence_sha256,
    )
    receipt_hashes: list[str] = []
    accepted: list[str] = []
    for payload in payloads:
        receipt_hashes.append(
            persist_candidate_payload(
                cos_client,
                bucket,
                payload.candidate,
                payload.raw,
                payload.derivative,
            )
        )
        accepted.append(payload.candidate.candidate_id)
    batch_sha256 = hashlib.sha256(
        json.dumps(receipt_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    return BatchReceipt(
        batch_id=batch_id,
        batch_sha256=batch_sha256,
        policy_sha256=policy.policy_evidence_sha256,
        rights_sha256=declaration.evidence_sha256,
        accepted_candidate_ids=tuple(accepted),
        refused_candidate_ids=(),
    )


def _import_partner_payloads(
    path: Path,
    declaration: RightsRecord,
    *,
    extra_allowed: tuple[Path, ...] = (),
) -> list[_PartnerPayload]:
    """Read a complete export without persisting partial candidates."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("partner manifest must be a regular non-symlink file")
    root = path.parent.resolve()
    rows = _load_jsonl(path)
    declared: dict[str, Path] = {}
    for row in rows:
        relative = row.get("file")
        if not isinstance(relative, str):
            raise ValueError("file must be a safe relative path")
        if relative in declared:
            raise ValueError("partner export declares the same file twice")
        declared[relative] = _safe_file(root, relative)

    allowed = {
        path.resolve(),
        *(candidate.resolve() for candidate in declared.values()),
        *(candidate.resolve() for candidate in extra_allowed),
    }
    for item in root.rglob("*"):
        if item.is_symlink():
            raise ValueError("partner export contains a symlink")
        if item.is_file() and item.resolve() not in allowed:
            raise ValueError(f"partner export contains undeclared file: {item.relative_to(root)}")

    payloads: list[_PartnerPayload] = []
    for row in rows:
        image_path = declared[str(row["file"])]
        payload = _read_regular_file(image_path)
        derivative, phash = sanitize_image(payload)
        listing_text = row.get("listing_text")
        if listing_text is not None and not isinstance(listing_text, str):
            raise ValueError("listing_text must be a string")
        price = row.get("price_usd")
        payloads.append(
            _PartnerPayload(
                candidate=CandidateRecord(
                candidate_id=_required(row, "candidate_id"),
                acquisition_batch_id=_required(row, "acquisition_batch_id"),
                source_policy_id=_required(row, "source_policy_id"),
                source_record_id=_required(row, "source_record_id"),
                source_url=_required(row, "source_url"),
                collected_at=_timestamp(_required(row, "collected_at")),
                declared_reference=_required(row, "declared_reference"),
                proposed_label=propose_source_label(
                    source=declaration.source,
                    listing_text=listing_text,
                ),
                proposed_grade=None,
                factory=_optional(row, "factory"),
                price_usd=None if price is None else Decimal(str(price)),
                raw_sha256=hashlib.sha256(payload).hexdigest(),
                derivative_sha256=hashlib.sha256(derivative).hexdigest(),
                phash=phash,
                rights_record_id=declaration.rights_record_id,
                ),
                raw=payload,
                derivative=derivative,
            )
        )
    return payloads


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid partner JSONL line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"partner JSONL line {line_number} must be an object")
        rows.append(row)
    return rows


def _safe_file(root: Path, relative: str) -> Path:
    parsed = PurePosixPath(relative)
    if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts:
        raise ValueError("file must be a safe relative path")
    candidate = root.joinpath(*parsed.parts)
    current = root
    for part in parsed.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("partner export file must not be a symlink")
    if not candidate.is_file():
        raise ValueError("declared partner file is missing")
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("file must be a safe relative path") from exc
    return candidate


def _read_regular_file(path: Path) -> bytes:
    try:
        before = path.lstat()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError("declared partner file must remain a non-symlink regular file") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise ValueError("declared partner file must remain a non-symlink regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            descriptor = -1
            payload = source.read(MAX_IMAGE_BYTES + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("declared partner file exceeds 25 MiB")
    return payload


def _required(row: dict[str, object], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional(row: dict[str, object], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("collected_at must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("collected_at must be timezone-aware")
    return parsed
