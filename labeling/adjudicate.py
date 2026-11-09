"""Evidence-backed label adjudication."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Mapping, Sequence

from acquisition.types import CandidateRecord, EvidenceRecord, EvidenceTier, ReviewDecision
from core.types import Label
from labeling.review import (
    ReviewerConflict,
    ReviewerRecord,
    candidate_digest,
    evidence_tier_for,
    reviewer_digest,
    validate_evidence_proposal,
    verify_decision,
)

_ADMITTED_REFERENCES = frozenset({"rolex-submariner-126610ln"})


class EvidenceTampered(ValueError):
    pass


class NotTrainingEligible(ValueError):
    pass


def adjudicate(
    candidate: CandidateRecord,
    reviews: Sequence[ReviewDecision],
    reviewer_registry: Mapping[str, ReviewerRecord],
    *,
    gold_candidate_ids: frozenset[str] = frozenset(),
) -> EvidenceRecord:
    if not reviews:
        raise ValueError("at least one review is required")
    candidate_sha256 = candidate_digest(candidate)
    unique: dict[str, ReviewDecision] = {}
    reviewer_hashes: dict[str, set[str]] = {}
    for review in reviews:
        if review.candidate_id != candidate.candidate_id:
            raise ValueError("review candidate differs")
        reviewer = reviewer_registry.get(review.reviewer_id)
        if reviewer is None or reviewer_digest(reviewer) != review.reviewer_registry_sha256:
            raise ReviewerConflict("reviewer registry binding is missing or changed")
        if not reviewer.effective_at <= review.reviewed_at < reviewer.expires_at:
            raise ReviewerConflict("reviewer was not active at review time")
        if review.evidence_type not in reviewer.allowed_evidence_types:
            raise ReviewerConflict("reviewer is not qualified for evidence type")
        if review.proposed_tier is not evidence_tier_for(review.evidence_type):
            raise ReviewerConflict("review evidence tier differs from the evidence matrix")
        validate_evidence_proposal(
            review.evidence_type, review.proposed_label, review.proposed_grade
        )
        if review.adjudicator is not reviewer.adjudicator:
            raise ReviewerConflict("adjudicator authority differs from the registry")
        if {candidate.acquisition_batch_id, candidate.source_policy_id} & set(
            reviewer.prohibited_contributor_groups
        ):
            raise ReviewerConflict("reviewer cannot approve own contributor or source group")
        if review.candidate_sha256 != candidate_sha256:
            raise EvidenceTampered("candidate binding differs")
        if review.proposed_reference_id != candidate.declared_reference:
            raise EvidenceTampered("review reference differs")
        try:
            verify_decision(review)
        except ValueError as exc:
            raise EvidenceTampered(str(exc)) from exc
        unique.setdefault(review.decision_sha256, review)
        reviewer_hashes.setdefault(review.reviewer_id, set()).add(
            review.decision_sha256
        )
    if any(len(hashes) > 1 for hashes in reviewer_hashes.values()):
        raise ReviewerConflict("same reviewer submitted divergent decisions")
    decisions = tuple(unique.values())
    votes = Counter((item.proposed_label, item.proposed_grade) for item in decisions)
    winner, count = votes.most_common(1)[0]
    if len(votes) > 1:
        adjudicator_votes = [
            (item.proposed_label, item.proposed_grade)
            for item in decisions
            if item.adjudicator
        ]
        conflict = len(adjudicator_votes) != 1 or adjudicator_votes[0] != winner or count < 2
    else:
        conflict = False
    matching = tuple(
        item
        for item in decisions
        if (item.proposed_label, item.proposed_grade) == winner
    )
    tier = min((item.proposed_tier for item in matching), key=_tier_rank)
    needs_two = tier is EvidenceTier.A or candidate.candidate_id in gold_candidate_ids
    tier_votes = sum(item.proposed_tier is tier for item in matching)
    independent_votes = tier_votes if tier is EvidenceTier.A else len(matching)
    unsupported_label = winner[0] is None or winner[0] is Label.EXCLUDED
    unknown_fake_grade = winner[0] is Label.FAKE and winner[1] is None
    unsupported_reference = candidate.declared_reference not in _ADMITTED_REFERENCES
    if (
        conflict
        or unsupported_label
        or unknown_fake_grade
        or unsupported_reference
        or (needs_two and independent_votes < 2)
    ):
        tier = EvidenceTier.INCONCLUSIVE
        accepted_label = None
        accepted_grade = None
    else:
        accepted_label, accepted_grade = winner
    ordered = tuple(sorted(decisions, key=lambda item: item.reviewer_id))
    artifact_hashes = tuple(
        sorted({value for item in ordered for value in item.supporting_artifact_hashes})
    )
    evidence_type = (
        ordered[0].evidence_type
        if len({item.evidence_type for item in ordered}) == 1
        else "adjudicated"
    )
    review_hashes = tuple(item.decision_sha256 for item in ordered)
    aggregate = {
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": candidate_sha256,
        "accepted_grade": accepted_grade,
        "accepted_label": None if accepted_label is None else accepted_label.value,
        "evidence_type": evidence_type,
        "reference_id": candidate.declared_reference,
        "review_decision_sha256s": list(review_hashes),
        "review_timestamps": [item.reviewed_at.isoformat() for item in ordered],
        "reviewer_ids": [item.reviewer_id for item in ordered],
        "supporting_artifact_hashes": list(artifact_hashes),
        "tier": tier.value,
    }
    digest = _digest(aggregate)
    record = EvidenceRecord(
        evidence_record_id=f"evidence-{digest[:24]}",
        candidate_id=candidate.candidate_id,
        candidate_sha256=candidate_sha256,
        accepted_label=accepted_label,
        accepted_grade=accepted_grade,
        reference_id=candidate.declared_reference,
        tier=tier,
        evidence_type=evidence_type,
        reviewer_ids=tuple(item.reviewer_id for item in ordered),
        review_timestamps=tuple(item.reviewed_at for item in ordered),
        supporting_artifact_hashes=artifact_hashes,
        review_decision_sha256s=review_hashes,
        decision_sha256=digest,
    )
    verify_evidence_record(record)
    return record


def verify_evidence_record(evidence: EvidenceRecord) -> None:
    payload = {
        "candidate_id": evidence.candidate_id,
        "candidate_sha256": evidence.candidate_sha256,
        "accepted_grade": evidence.accepted_grade,
        "accepted_label": (
            None if evidence.accepted_label is None else evidence.accepted_label.value
        ),
        "evidence_type": evidence.evidence_type,
        "reference_id": evidence.reference_id,
        "review_decision_sha256s": list(evidence.review_decision_sha256s),
        "review_timestamps": [value.isoformat() for value in evidence.review_timestamps],
        "reviewer_ids": list(evidence.reviewer_ids),
        "supporting_artifact_hashes": list(evidence.supporting_artifact_hashes),
        "tier": evidence.tier.value,
    }
    digest = _digest(payload)
    if (
        digest != evidence.decision_sha256
        or evidence.evidence_record_id != f"evidence-{digest[:24]}"
    ):
        raise EvidenceTampered("evidence record hash differs from content")


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tier_rank(tier: EvidenceTier) -> int:
    return {
        EvidenceTier.A: 0,
        EvidenceTier.B: 1,
        EvidenceTier.C: 2,
        EvidenceTier.INCONCLUSIVE: 3,
    }[tier]
