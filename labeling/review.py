"""Qualified reviewer registry and immutable review decisions."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from acquisition.types import (
    CandidateRecord,
    EvidenceTier,
    ReviewDecision,
    ReviewStatus,
)
from core.types import Label
from infra.ibm_cos import upload_verified
from infra.ibm_types import ObjectRef

_EVIDENCE_TIERS = {
    "physical_examination": EvidenceTier.A,
    "service_record": EvidenceTier.A,
    "authentication_record": EvidenceTier.A,
    "replica_transaction": EvidenceTier.B,
    "factory_record": EvidenceTier.B,
    "partner_authenticated_original": EvidenceTier.B,
    "source_reputation": EvidenceTier.C,
    "seller_text": EvidenceTier.C,
    "price": EvidenceTier.C,
    "owner_claim": EvidenceTier.C,
}
_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class ReviewerConflict(ValueError):
    pass


@dataclass(frozen=True)
class ReviewerRecord:
    reviewer_id: str
    allowed_evidence_types: tuple[str, ...]
    qualification_artifact_sha256: str
    effective_at: datetime
    expires_at: datetime
    prohibited_contributor_groups: tuple[str, ...]
    adjudicator: bool

    def __post_init__(self) -> None:
        if _SAFE_ID.fullmatch(self.reviewer_id) is None or not self.allowed_evidence_types:
            raise ValueError("reviewer identity and evidence scope are required")
        if len(self.qualification_artifact_sha256) != 64:
            raise ValueError("qualification artifact must be SHA-256")
        if self.effective_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("reviewer timestamps must be timezone-aware")
        if self.expires_at <= self.effective_at:
            raise ValueError("reviewer expiry must follow effective time")
        if len(set(self.allowed_evidence_types)) != len(self.allowed_evidence_types):
            raise ValueError("reviewer evidence types must be unique")
        if not set(self.allowed_evidence_types).issubset(_EVIDENCE_TIERS):
            raise ValueError("reviewer evidence type is unsupported")
        if any(not value.strip() for value in self.prohibited_contributor_groups):
            raise ValueError("prohibited contributor groups must be non-empty")
        if len(set(self.prohibited_contributor_groups)) != len(
            self.prohibited_contributor_groups
        ):
            raise ValueError("prohibited contributor groups must be unique")
        if not isinstance(self.adjudicator, bool):
            raise ValueError("adjudicator must be bool")


@dataclass(frozen=True)
class ReviewProjection:
    status: ReviewStatus
    decisions: tuple[ReviewDecision, ...]


class ReviewStore:
    """Immutable review objects with an in-memory adapter for directed tests."""

    def __init__(self, cos_client: Any | None = None, bucket: str | None = None) -> None:
        if (cos_client is None) != (bucket is None):
            raise ValueError("COS client and bucket must be supplied together")
        self._cos_client = cos_client
        self._bucket = bucket
        self._decisions: dict[str, ReviewDecision] = {}

    def put(self, decision: ReviewDecision) -> None:
        verify_decision(decision)
        if self._cos_client is None:
            self._decisions.setdefault(decision.decision_sha256, decision)
        else:
            assert self._bucket is not None
            persist_review(self._cos_client, self._bucket, decision)

    def list(self) -> tuple[ReviewDecision, ...]:
        if self._cos_client is None:
            return tuple(self._decisions[key] for key in sorted(self._decisions))
        assert self._bucket is not None
        return _list_remote_reviews(self._cos_client, self._bucket)

    def project(self) -> ReviewProjection:
        decisions = self.list()
        reviewers: dict[tuple[str, str], set[str]] = {}
        for decision in decisions:
            reviewers.setdefault(
                (decision.candidate_id, decision.reviewer_id), set()
            ).add(decision.decision_sha256)
        status = (
            ReviewStatus.QUARANTINED
            if any(len(hashes) > 1 for hashes in reviewers.values())
            else ReviewStatus.ACCEPTED
        )
        return ReviewProjection(status=status, decisions=decisions)


def load_reviewer_registry(path: Path) -> dict[str, ReviewerRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("reviewer registry must be a list")
    records: dict[str, ReviewerRecord] = {}
    for item in raw:
        record = ReviewerRecord(
            reviewer_id=item["reviewer_id"],
            allowed_evidence_types=tuple(item["allowed_evidence_types"]),
            qualification_artifact_sha256=item["qualification_artifact_sha256"],
            effective_at=datetime.fromisoformat(item["effective_at"]),
            expires_at=datetime.fromisoformat(item["expires_at"]),
            prohibited_contributor_groups=tuple(item["prohibited_contributor_groups"]),
            adjudicator=item["adjudicator"],
        )
        if record.reviewer_id in records:
            raise ValueError("duplicate reviewer ID")
        records[record.reviewer_id] = record
    return records


def submit_review(
    candidate: CandidateRecord,
    reviewer: ReviewerRecord,
    *,
    proposed_label: Label | None,
    proposed_grade: int | None,
    proposed_tier: EvidenceTier,
    evidence_type: str,
    supporting_artifact_hashes: tuple[str, ...],
    reviewed_at: datetime,
) -> ReviewDecision:
    del proposed_tier
    if not reviewer.effective_at <= reviewed_at < reviewer.expires_at:
        raise ReviewerConflict("reviewer is not active")
    if evidence_type not in reviewer.allowed_evidence_types:
        raise ReviewerConflict("reviewer is not qualified for evidence type")
    if {candidate.acquisition_batch_id, candidate.source_policy_id} & set(
        reviewer.prohibited_contributor_groups
    ):
        raise ReviewerConflict("reviewer cannot approve own contributor or source group")
    tier = evidence_tier_for(evidence_type)
    validate_evidence_proposal(evidence_type, proposed_label, proposed_grade)
    candidate_sha256 = candidate_digest(candidate)
    registry_sha256 = reviewer_digest(reviewer)
    payload = {
        "candidate_id": candidate.candidate_id,
        "reviewer_id": reviewer.reviewer_id,
        "proposed_label": None if proposed_label is None else proposed_label.value,
        "proposed_grade": proposed_grade,
        "proposed_tier": tier.value,
        "evidence_type": evidence_type,
        "supporting_artifact_hashes": list(supporting_artifact_hashes),
        "reviewed_at": reviewed_at.isoformat(),
        "candidate_sha256": candidate_sha256,
        "proposed_reference_id": candidate.declared_reference,
        "reviewer_registry_sha256": registry_sha256,
        "adjudicator": reviewer.adjudicator,
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return ReviewDecision(
        candidate_id=candidate.candidate_id,
        reviewer_id=reviewer.reviewer_id,
        proposed_label=proposed_label,
        proposed_grade=proposed_grade,
        proposed_tier=tier,
        evidence_type=evidence_type,
        supporting_artifact_hashes=supporting_artifact_hashes,
        reviewed_at=reviewed_at,
        decision_sha256=digest,
        candidate_sha256=candidate_sha256,
        proposed_reference_id=candidate.declared_reference,
        reviewer_registry_sha256=registry_sha256,
        adjudicator=reviewer.adjudicator,
    )


def verify_decision(decision: ReviewDecision) -> None:
    payload = {
        "candidate_id": decision.candidate_id,
        "reviewer_id": decision.reviewer_id,
        "proposed_label": None
        if decision.proposed_label is None
        else decision.proposed_label.value,
        "proposed_grade": decision.proposed_grade,
        "proposed_tier": decision.proposed_tier.value,
        "evidence_type": decision.evidence_type,
        "supporting_artifact_hashes": list(decision.supporting_artifact_hashes),
        "reviewed_at": decision.reviewed_at.isoformat(),
        "candidate_sha256": decision.candidate_sha256,
        "proposed_reference_id": decision.proposed_reference_id,
        "reviewer_registry_sha256": decision.reviewer_registry_sha256,
        "adjudicator": decision.adjudicator,
    }
    if hashlib.sha256(_canonical(payload)).hexdigest() != decision.decision_sha256:
        raise ValueError("review decision hash differs from content")


def evidence_tier_for(evidence_type: str) -> EvidenceTier:
    try:
        return _EVIDENCE_TIERS[evidence_type]
    except KeyError as exc:
        raise ValueError("evidence type is unsupported") from exc


def validate_evidence_proposal(
    evidence_type: str, proposed_label: Label | None, proposed_grade: int | None
) -> None:
    if evidence_type in {"replica_transaction", "factory_record"}:
        if proposed_label is not Label.FAKE:
            raise ReviewerConflict("replica evidence requires a fake label")
        if proposed_grade is not None and proposed_grade not in {3, 4}:
            raise ReviewerConflict("factory or order evidence supports only grades 3 and 4")
    if (
        evidence_type == "partner_authenticated_original"
        and proposed_label is not Label.GENUINE
    ):
        raise ReviewerConflict("authenticated-original evidence requires a genuine label")


def persist_review(
    cos_client: Any, bucket: str, decision: ReviewDecision
) -> ObjectRef:
    verify_decision(decision)
    payload = _canonical(_decision_json(decision))
    handle = tempfile.NamedTemporaryFile(prefix="watch-review-", delete=False)
    path = Path(handle.name)
    try:
        handle.write(payload)
        handle.close()
        key = (
            f"datasets/reviews/{decision.candidate_id}/{decision.reviewer_id}/"
            f"{decision.decision_sha256}.json"
        )
        return upload_verified(
            cos_client, path, bucket, key, decision.decision_sha256
        )
    finally:
        if not handle.closed:
            handle.close()
        path.unlink(missing_ok=True)


def _decision_json(decision: ReviewDecision) -> dict[str, object]:
    raw = asdict(decision)
    raw.pop("decision_sha256")
    raw["proposed_label"] = (
        None if decision.proposed_label is None else decision.proposed_label.value
    )
    raw["proposed_tier"] = decision.proposed_tier.value
    raw["reviewed_at"] = decision.reviewed_at.isoformat()
    return raw


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _list_remote_reviews(client: Any, bucket: str) -> tuple[ReviewDecision, ...]:
    prefix = "datasets/reviews/"
    token: str | None = None
    decisions: dict[str, ReviewDecision] = {}
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        if not isinstance(page, dict) or not isinstance(page.get("Contents", []), list):
            raise ValueError("review inventory is malformed")
        for item in page.get("Contents", []):
            key = item.get("Key") if isinstance(item, dict) else None
            if not isinstance(key, str) or not key.startswith(prefix):
                raise ValueError("review inventory key is malformed")
            response = client.get_object(Bucket=bucket, Key=key)
            body = response.get("Body") if isinstance(response, dict) else None
            if body is None or not hasattr(body, "read"):
                raise ValueError("review object body is missing")
            decision_sha256 = key.rsplit("/", 1)[-1].removesuffix(".json")
            decision = _decision_from_json(json.loads(body.read()), decision_sha256)
            verify_decision(decision)
            decisions.setdefault(decision.decision_sha256, decision)
        if page.get("IsTruncated") is not True:
            break
        next_token = page.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise ValueError("review inventory pagination is malformed")
        token = next_token
    return tuple(decisions[key] for key in sorted(decisions))


def _decision_from_json(raw: object, decision_sha256: str) -> ReviewDecision:
    if not isinstance(raw, dict):
        raise ValueError("review object must be a JSON object")
    return ReviewDecision(
        candidate_id=raw["candidate_id"],
        reviewer_id=raw["reviewer_id"],
        proposed_label=(
            None if raw["proposed_label"] is None else Label(raw["proposed_label"])
        ),
        proposed_grade=raw["proposed_grade"],
        proposed_tier=EvidenceTier(raw["proposed_tier"]),
        evidence_type=raw["evidence_type"],
        supporting_artifact_hashes=tuple(raw["supporting_artifact_hashes"]),
        reviewed_at=datetime.fromisoformat(raw["reviewed_at"]),
        decision_sha256=decision_sha256,
        candidate_sha256=raw["candidate_sha256"],
        proposed_reference_id=raw["proposed_reference_id"],
        reviewer_registry_sha256=raw["reviewer_registry_sha256"],
        adjudicator=raw["adjudicator"],
    )


def candidate_digest(candidate: CandidateRecord) -> str:
    payload = asdict(candidate)
    payload["collected_at"] = candidate.collected_at.isoformat()
    payload["proposed_label"] = (
        None if candidate.proposed_label is None else candidate.proposed_label.value
    )
    payload["price_usd"] = (
        None if candidate.price_usd is None else str(candidate.price_usd)
    )
    return hashlib.sha256(_canonical(payload)).hexdigest()


def reviewer_digest(reviewer: ReviewerRecord) -> str:
    payload = asdict(reviewer)
    payload["effective_at"] = reviewer.effective_at.isoformat()
    payload["expires_at"] = reviewer.expires_at.isoformat()
    return hashlib.sha256(_canonical(payload)).hexdigest()
