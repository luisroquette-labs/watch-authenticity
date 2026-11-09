"""Versioned manifest I/O over `core.types.ManifestRecord` (CK-10).

Every scraped photo must pass through here before it is eligible for
training: this module is the single call site into `core.grade_map` for
fake-grade assignment (no scraper or dataset stage may re-derive its own
factory/price -> grade mapping — CK-22), and it persists rows as append-only
newline-delimited JSON so a new dataset version is a new file, never an
in-place rewrite of a prior version's rows (reproducibility/audit trail,
SKILL.md:135).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from acquisition.types import (
    Angle,
    CandidateRecord,
    EvidenceRecord,
    ReviewDecision,
    RightsRecord,
    RightsStatus,
    TrainingRecord,
    validate_contract_bundle,
)
from core.types import Label, ManifestRecord
from dataset.entities import GroupingDecision
from infra.ibm_types import ObjectRef
from labeling.adjudicate import NotTrainingEligible, adjudicate, verify_evidence_record
from labeling.review import ReviewerRecord, candidate_digest

# Each dataset version is its own file, not a flag on a row — this is what
# makes "append a version" structurally unable to mutate an earlier version.
_MANIFEST_FILENAME_TEMPLATE = "manifest_v{version}.jsonl"


def manifest_path(manifest_dir: Path, version: int) -> Path:
    """Returns the file path for one versioned manifest."""
    return manifest_dir / _MANIFEST_FILENAME_TEMPLATE.format(version=version)


def build_fake_row(
    *,
    path: str,
    source: str,
    url: str,
    phash: str,
    scraped_at: date,
    accepted_grade: int,
    blur_score: float | None = None,
) -> ManifestRecord:
    """Legacy row builder requiring an already adjudicated fake grade."""
    if isinstance(accepted_grade, bool) or accepted_grade not in {1, 2, 3, 4}:
        raise ValueError("accepted_grade must be between 1 and 4")
    return ManifestRecord(
        path=path,
        source=source,
        url=url,
        label=Label.FAKE,
        grade=accepted_grade,
        phash=phash,
        scraped_at=scraped_at,
        blur_score=blur_score,
    )


def append(record: ManifestRecord, *, manifest_dir: Path, version: int) -> Path:
    """Appends one row to a versioned manifest file, never overwriting it.

    Creates `manifest_dir` and the version's file on first write. Appending
    rather than rewriting is what keeps every row ever recorded for this
    version on the audit trail (CK-10).
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_path(manifest_dir, version)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_row(record)) + "\n")
    return target


# The Expected Output names both "write_row" and "append" for this
# operation; `append` is the one implementation, `write_row` its alias.
write_row = append


def build_training_record(
    candidate: CandidateRecord,
    evidence: EvidenceRecord,
    grouping: GroupingDecision,
    rights: RightsRecord,
    *,
    object_ref: ObjectRef,
    angle: Angle,
    capture_style: str,
    now: datetime,
    reviews: Sequence[ReviewDecision],
    reviewer_registry: Mapping[str, ReviewerRecord],
    gold_candidate_ids: frozenset[str] = frozenset(),
) -> TrainingRecord:
    """Build a training row only from accepted evidence, never price guesses."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if (
        rights.status is not RightsStatus.ACTIVE
        or "train" not in rights.allowed_purposes
        or rights.rights_record_id != candidate.rights_record_id
        or not rights.effective_at <= now < rights.expires_at
    ):
        raise NotTrainingEligible("active training rights are required")
    verify_evidence_record(evidence)
    if evidence.candidate_sha256 != candidate_digest(candidate):
        raise NotTrainingEligible("reviewed candidate content differs")
    expected_evidence = adjudicate(
        candidate,
        reviews,
        reviewer_registry,
        gold_candidate_ids=gold_candidate_ids,
    )
    if expected_evidence != evidence:
        raise NotTrainingEligible("evidence differs from immutable review decisions")
    if grouping.quarantined or not evidence.tier.training_eligible:
        raise NotTrainingEligible("candidate evidence is not training eligible")
    if (
        evidence.candidate_id != candidate.candidate_id
        or grouping.candidate_id != candidate.candidate_id
        or evidence.accepted_label is None
    ):
        raise NotTrainingEligible("accepted evidence lineage is required")
    if (
        object_ref.sha256 != candidate.derivative_sha256
        or object_ref.key
        != f"datasets/derivatives/{candidate.derivative_sha256}"
    ):
        raise NotTrainingEligible("verified candidate derivative is required")
    domain = urlsplit(candidate.source_url).hostname
    if domain is None:
        raise ValueError("candidate source domain is missing")
    asset_digest = hashlib.sha256(
        f"{candidate.candidate_id}\0{object_ref.sha256}\0{angle.value}".encode()
    ).hexdigest()
    record = TrainingRecord(
        asset_id=f"asset-{asset_digest[:24]}",
        candidate_id=candidate.candidate_id,
        object_ref=object_ref,
        reference_id=evidence.reference_id,
        watch_entity_id=grouping.watch_entity_id,
        content_cluster_id=grouping.content_cluster_id,
        seller_group_id=grouping.seller_group_id,
        acquisition_batch_id=candidate.acquisition_batch_id,
        angle=angle,
        source_domain=domain.rstrip(".").lower(),
        capture_style=capture_style,
        label=evidence.accepted_label,
        grade=evidence.accepted_grade,
        factory=candidate.factory if evidence.accepted_label is Label.FAKE else None,
        evidence_tier=evidence.tier,
        evidence_record_id=evidence.evidence_record_id,
        rights_record_id=rights.rights_record_id,
        image_sha256=object_ref.sha256,
        phash=candidate.phash,
    )
    validate_contract_bundle(candidate, rights, evidence, record)
    return record


def read_manifest(manifest_dir: Path, version: int) -> list[ManifestRecord]:
    """Reads every row of one manifest version back into `ManifestRecord`s.

    Returns an empty list for a version that was never written, rather
    than raising — a not-yet-created version is a normal, not an error,
    state for a dataset that versions by file.
    """
    target = manifest_path(manifest_dir, version)
    if not target.exists():
        return []

    lines = target.read_text(encoding="utf-8").splitlines()
    return [_from_json_row(json.loads(line)) for line in lines if line]


def _to_json_row(record: ManifestRecord) -> dict:
    """Serializes a `ManifestRecord` to a JSON-compatible dict."""
    row = asdict(record)
    row["label"] = record.label.value
    row["scraped_at"] = record.scraped_at.isoformat()
    return row


def _from_json_row(row: dict) -> ManifestRecord:
    """Deserializes one JSON row back into a `ManifestRecord`."""
    return ManifestRecord(
        path=row["path"],
        source=row["source"],
        url=row["url"],
        label=Label(row["label"]),
        grade=row["grade"],
        phash=row["phash"],
        scraped_at=date.fromisoformat(row["scraped_at"]),
        blur_score=row.get("blur_score"),
    )
