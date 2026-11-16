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
from typing import Mapping

from acquisition.types import EvidenceRecord
from core.types import Label
from labeling.adjudicate import _digest, verify_evidence_record

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


@dataclass(frozen=True)
class FakeQualityAssessment:
    watch_entity_id: str
    evidence_record_id: str
    inspection_evidence_sha256: str
    rubric_version: str
    rubric_sha256: str
    score_0_1000: int
    reviewer_scores: tuple[tuple[str, int], ...]
    reviewer_registry_sha256s: tuple[tuple[str, str], ...]
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
    reviewer_scores: Mapping[str, int],
    quality_reviewers: Mapping[str, QualityReviewerRecord],
    rubric_version: str,
    inspection_evidence_sha256: str,
    *,
    inspection: FakeQualityInspection | None = None,
    adjudicator_score: tuple[str, int] | None = None,
) -> FakeQualityAssessment | None:
    verify_evidence_record(evidence)
    if (
        evidence.accepted_label is not Label.FAKE
        or not evidence.tier.training_eligible
        or rubric_version != RUBRIC_VERSION
        or inspection is None
        or inspection.sha256 != inspection_evidence_sha256
        or inspection.watch_entity_id != watch_entity_id
        or inspection.evidence_record_id != evidence.evidence_record_id
        or set(dict(inspection.attribute_artifacts)) != _ATTRIBUTES
        or inspection.reviewed_at < max(evidence.review_timestamps)
        or len(reviewer_scores) != 2
    ):
        return None
    scores = tuple(sorted(reviewer_scores.items()))
    adjudicate = (
        abs(scores[0][1] - scores[1][1]) > 100
        if all(isinstance(score, int) and not isinstance(score, bool) for _, score in scores)
        else False
    )
    if adjudicate and adjudicator_score is None:
        return None
    if not adjudicate and adjudicator_score is not None:
        return None
    votes = scores + ((adjudicator_score,) if adjudicator_score is not None else ())
    if len({key for key, _ in votes}) != len(votes):
        return None
    reviewers = []
    for reviewer_id, score in votes:
        reviewer = quality_reviewers.get(reviewer_id)
        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 1000
            or not isinstance(reviewer, QualityReviewerRecord)
            or reviewer.reviewer_id != reviewer_id
            or not reviewer.effective_at <= inspection.reviewed_at < reviewer.expires_at
            or set(inspection.contributor_groups)
            & set(reviewer.contributor_groups + reviewer.prohibited_contributor_groups)
        ):
            return None
        if any(
            set(reviewer.contributor_groups)
            & set(previous.contributor_groups + previous.prohibited_contributor_groups)
            or set(previous.contributor_groups) & set(reviewer.prohibited_contributor_groups)
            for previous in reviewers
        ):
            return None
        reviewers.append(reviewer)
    if adjudicator_score is not None and not reviewers[-1].adjudicator:
        return None
    score = adjudicator_score[1] if adjudicator_score else (sum(v for _, v in scores) + 1) // 2
    bindings = tuple(
        (
            reviewer.reviewer_id,
            _digest(
                {
                    **asdict(reviewer),
                    "effective_at": reviewer.effective_at.isoformat(),
                    "expires_at": reviewer.expires_at.isoformat(),
                }
            ),
        )
        for reviewer in reviewers
    )
    payload = dict(
        watch_entity_id=watch_entity_id,
        evidence_record_id=evidence.evidence_record_id,
        inspection_evidence_sha256=inspection_evidence_sha256,
        rubric_version=rubric_version,
        rubric_sha256=RUBRIC_SHA256,
        score_0_1000=score,
        reviewer_scores=scores,
        reviewer_registry_sha256s=bindings,
        adjudicator_score=adjudicator_score,
    )
    return FakeQualityAssessment(**payload, assessment_sha256=_digest(payload))
