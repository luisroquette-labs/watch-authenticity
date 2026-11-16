"""Separate, evidence-bound quality assessment of an already confirmed fake.

Rubric fake-quality-v1 (fidelity, never authenticity or model confidence):
Score exterior (0–400), inside construction/finishing (0–300), and mechanical
construction/function (0–300) against the confirmed reference's cited inspection
anchors. For each attribute: 0% = incompatible construction or nonfunctional;
25% = gross readily visible departures; 50% = several clear departures;
75% = fine-detail departures; 100% = no departure found in the documented checks.
Thus 0 = incompatible/nonfunctional throughout, 500 = half of weighted fidelity,
750 = fine-detail departures throughout, 1000 = all documented checks matched.
1000 does not authenticate or prove absence of hidden defects. Interpolate only
between recorded anchors and retain observations/calibration examples in each
attribute's hashed artifact. All three attributes require direct inspection;
inside requires opened-case evidence, mechanical requires movement examination
and functional measurements. Photos of the exterior alone are unassessed.

Two independent qualified reviewers score that same complete inspection under
this rubric. Difference <=100: midpoint rounded half up. Difference >100: a
separate qualified adjudicator supplies the final anchored score. Missing data,
unknown rubric, conflicts, and draft/unqualified scores return None (unassessed).
No conversion from training grade, seller language or confidence is permitted.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping, Sequence

from acquisition.types import CandidateRecord, EvidenceRecord
from core.types import Label
from dataset.entities import GroupingDecision
from labeling.adjudicate import _digest, verify_evidence_record
from labeling.review import candidate_digest

RUBRIC_VERSION = "fake-quality-v1"
RUBRIC_SHA256 = _digest(__doc__)
_ATTRIBUTES = frozenset({"exterior", "inside", "mechanical"})


def _hash(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _groups(values: tuple[str, ...]) -> bool:
    return (
        isinstance(values, tuple)
        and len(set(values)) == len(values)
        and all(isinstance(v, str) and v.strip() for v in values)
    )


@dataclass(frozen=True)
class QualityReviewerRecord:
    """Separate qualification authority; shared contributor groups mean dependence."""

    reviewer_id: str
    qualification_artifact_sha256: str
    effective_at: datetime
    expires_at: datetime
    contributor_groups: tuple[str, ...]
    prohibited_contributor_groups: tuple[str, ...]
    adjudicator: bool = False

    def __post_init__(self) -> None:
        if (
            not self.reviewer_id.strip()
            or not _hash(self.qualification_artifact_sha256)
            or self.effective_at.tzinfo is None
            or self.expires_at.tzinfo is None
            or self.expires_at <= self.effective_at
            or not self.contributor_groups
            or not _groups(self.contributor_groups)
            or not _groups(self.prohibited_contributor_groups)
            or not isinstance(self.adjudicator, bool)
        ):
            raise ValueError("quality reviewer needs qualification, validity and conflict groups")


@dataclass(frozen=True)
class FakeQualityInspection:
    """Artifact bundle attested by the scorers, bound to one watch and label record.

    Each attribute artifact includes observations, methods, reference/calibration
    anchors and source evidence. Contributor groups include source and batch IDs.
    reviewed_at is the time the quality reviewers scored this inspection.
    """

    watch_entity_id: str
    evidence_record_id: str
    reviewed_at: datetime
    attribute_artifacts: tuple[tuple[str, str], ...]
    contributor_groups: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.watch_entity_id.strip()
            or not self.evidence_record_id.strip()
            or self.reviewed_at.tzinfo is None
            or not isinstance(self.attribute_artifacts, tuple)
            or any(
                not isinstance(pair, tuple)
                or len(pair) != 2
                or pair[0] not in _ATTRIBUTES
                or not _hash(pair[1])
                for pair in self.attribute_artifacts
            )
            or len(dict(self.attribute_artifacts)) != len(self.attribute_artifacts)
            or not self.contributor_groups
            or not _groups(self.contributor_groups)
        ):
            raise ValueError("inspection requires watch, evidence, time and artifact bindings")

    @property
    def sha256(self) -> str:
        return _digest({**asdict(self), "reviewed_at": self.reviewed_at.isoformat()})


def quality_reviewer_digest(reviewer: QualityReviewerRecord) -> str:
    return _digest(
        {
            **asdict(reviewer),
            "effective_at": reviewer.effective_at.isoformat(),
            "expires_at": reviewer.expires_at.isoformat(),
        }
    )


@dataclass(frozen=True)
class QualityReviewDecision:
    reviewer_id: str
    reviewer_registry_sha256: str
    inspection_evidence_sha256: str
    rubric_version: str
    rubric_sha256: str
    reviewed_at: datetime
    score_0_1000: int
    decision_sha256: str


def _decision_payload(decision: QualityReviewDecision) -> dict:
    payload = asdict(decision)
    payload.pop("decision_sha256")
    payload["reviewed_at"] = decision.reviewed_at.isoformat()
    return payload


def submit_quality_review(
    inspection: FakeQualityInspection,
    reviewer: QualityReviewerRecord,
    score_0_1000: int,
    *,
    reviewed_at: datetime,
) -> QualityReviewDecision:
    """Record one draft judgment bound to this exact inspection and rubric."""
    if (
        not isinstance(reviewer, QualityReviewerRecord)
        or not isinstance(score_0_1000, int)
        or isinstance(score_0_1000, bool)
        or not 0 <= score_0_1000 <= 1000
        or reviewed_at.tzinfo is None
        or reviewed_at < inspection.reviewed_at
        or not reviewer.effective_at <= reviewed_at < reviewer.expires_at
    ):
        raise ValueError("qualified active reviewer, review time and integer score required")
    payload = dict(
        reviewer_id=reviewer.reviewer_id,
        reviewer_registry_sha256=quality_reviewer_digest(reviewer),
        inspection_evidence_sha256=inspection.sha256,
        rubric_version=RUBRIC_VERSION,
        rubric_sha256=RUBRIC_SHA256,
        reviewed_at=reviewed_at.isoformat(),
        score_0_1000=score_0_1000,
    )
    return QualityReviewDecision(
        **{**payload, "reviewed_at": reviewed_at}, decision_sha256=_digest(payload)
    )


@dataclass(frozen=True)
class FakeQualityAssessment:
    watch_entity_id: str
    evidence_record_id: str
    inspection_evidence_sha256: str
    rubric_version: str
    rubric_sha256: str
    score_0_1000: int
    reviewer_scores: tuple[tuple[str, int], ...]
    review_decision_sha256s: tuple[str, ...]
    grouping_sha256: str
    adjudicator_score: tuple[str, int] | None
    assessment_sha256: str


def verify_fake_quality(assessment: FakeQualityAssessment) -> None:
    payload = asdict(assessment)
    digest = payload.pop("assessment_sha256")
    if digest != _digest(payload):
        raise ValueError("fake quality assessment hash differs from content")


def assess_fake_quality(
    evidence: EvidenceRecord,
    watch_entity_id: str,
    quality_decisions: Sequence[QualityReviewDecision],
    quality_reviewers: Mapping[str, QualityReviewerRecord],
    rubric_version: str,
    inspection_evidence_sha256: str,
    *,
    candidate: CandidateRecord | None = None,
    grouping: GroupingDecision | None = None,
    inspection: FakeQualityInspection | None = None,
    adjudicator_decision: QualityReviewDecision | None = None,
) -> FakeQualityAssessment | None:
    """Aggregate immutable decisions using the same authoritative grouping as admission.

    candidate and grouping come from the acquisition/entity owners, as they do
    for build_training_record; an inspection cannot declare its own watch identity.
    """
    verify_evidence_record(evidence)
    if (
        candidate is None
        or grouping is None
        or inspection is None
        or evidence.candidate_id != candidate.candidate_id
        or evidence.candidate_sha256 != candidate_digest(candidate)
        or grouping.candidate_id != candidate.candidate_id
        or grouping.quarantined
        or grouping.watch_entity_id != watch_entity_id
        or evidence.accepted_label is not Label.FAKE
        or not evidence.tier.training_eligible
        or rubric_version != RUBRIC_VERSION
        or inspection.sha256 != inspection_evidence_sha256
        or inspection.watch_entity_id != watch_entity_id
        or inspection.evidence_record_id != evidence.evidence_record_id
        or set(dict(inspection.attribute_artifacts)) != _ATTRIBUTES
        or inspection.reviewed_at < max(evidence.review_timestamps)
        or len(quality_decisions) != 2
    ):
        return None
    mandatory_groups = {
        candidate.source_policy_id,
        candidate.acquisition_batch_id,
        grouping.seller_group_id,
    }
    if not mandatory_groups.issubset(inspection.contributor_groups):
        return None
    conflicts = mandatory_groups | set(inspection.contributor_groups)
    votes = tuple(quality_decisions) + (
        (adjudicator_decision,) if adjudicator_decision is not None else ()
    )
    reviewers: list[QualityReviewerRecord] = []
    for decision in votes:
        if not isinstance(decision, QualityReviewDecision):
            return None
        reviewer = quality_reviewers.get(decision.reviewer_id)
        if (
            not isinstance(reviewer, QualityReviewerRecord)
            or reviewer.reviewer_id != decision.reviewer_id
            or decision.reviewer_registry_sha256 != quality_reviewer_digest(reviewer)
            or decision.decision_sha256 != _digest(_decision_payload(decision))
            or decision.inspection_evidence_sha256 != inspection.sha256
            or decision.rubric_version != rubric_version
            or decision.rubric_sha256 != RUBRIC_SHA256
            or decision.reviewed_at.tzinfo is None
            or decision.reviewed_at < inspection.reviewed_at
            or not reviewer.effective_at <= decision.reviewed_at < reviewer.expires_at
            or not isinstance(decision.score_0_1000, int)
            or isinstance(decision.score_0_1000, bool)
            or not 0 <= decision.score_0_1000 <= 1000
            or conflicts & set(reviewer.contributor_groups + reviewer.prohibited_contributor_groups)
        ):
            return None
        if any(
            previous.reviewer_id == reviewer.reviewer_id
            or set(reviewer.contributor_groups)
            & set(previous.contributor_groups + previous.prohibited_contributor_groups)
            or set(previous.contributor_groups) & set(reviewer.prohibited_contributor_groups)
            for previous in reviewers
        ):
            return None
        reviewers.append(reviewer)
    scores = tuple(sorted((d.reviewer_id, d.score_0_1000) for d in quality_decisions))
    needs_adjudicator = abs(scores[0][1] - scores[1][1]) > 100
    if needs_adjudicator != (adjudicator_decision is not None):
        return None
    if adjudicator_decision is not None and (
        not reviewers[-1].adjudicator
        or adjudicator_decision.reviewed_at < max(d.reviewed_at for d in quality_decisions)
    ):
        return None
    adjudicator_score = (
        None
        if adjudicator_decision is None
        else (adjudicator_decision.reviewer_id, adjudicator_decision.score_0_1000)
    )
    score = adjudicator_score[1] if adjudicator_score else (sum(v for _, v in scores) + 1) // 2
    payload = dict(
        watch_entity_id=watch_entity_id,
        evidence_record_id=evidence.evidence_record_id,
        inspection_evidence_sha256=inspection_evidence_sha256,
        rubric_version=rubric_version,
        rubric_sha256=RUBRIC_SHA256,
        score_0_1000=score,
        reviewer_scores=scores,
        review_decision_sha256s=tuple(sorted(d.decision_sha256 for d in votes)),
        grouping_sha256=_digest(asdict(grouping)),
        adjudicator_score=adjudicator_score,
    )
    return FakeQualityAssessment(**payload, assessment_sha256=_digest(payload))
