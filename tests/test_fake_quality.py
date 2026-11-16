from dataclasses import replace
from datetime import timedelta

import pytest

from acquisition.types import EvidenceTier
from core.types import Label
from labeling.adjudicate import adjudicate
from labeling.quality import (
    RUBRIC_VERSION,
    FakeQualityInspection,
    QualityReviewerRecord,
    assess_fake_quality,
    verify_fake_quality,
)
from labeling.review import ReviewerRecord, submit_review


@pytest.fixture
def quality_case(candidate):
    now = candidate.collected_at
    label_reviewer = ReviewerRecord(
        "label-reviewer",
        ("replica_transaction", "seller_text", "owner_claim"),
        "a" * 64,
        now - timedelta(days=1),
        now + timedelta(days=1),
        (),
        False,
    )
    review = submit_review(
        candidate,
        label_reviewer,
        proposed_label=Label.FAKE,
        proposed_grade=4,
        proposed_tier=EvidenceTier.B,
        evidence_type="replica_transaction",
        supporting_artifact_hashes=("b" * 64,),
        reviewed_at=now,
    )
    evidence = adjudicate(candidate, [review], {label_reviewer.reviewer_id: label_reviewer})
    inspection = FakeQualityInspection(
        "watch-1",
        evidence.evidence_record_id,
        now,
        (("exterior", "c" * 64), ("inside", "d" * 64), ("mechanical", "e" * 64)),
        (candidate.source_policy_id, candidate.acquisition_batch_id),
    )
    first = QualityReviewerRecord(
        "quality-1",
        "f" * 64,
        now - timedelta(days=1),
        now + timedelta(days=1),
        ("expert-1",),
        (),
        False,
    )
    second = replace(first, reviewer_id="quality-2", contributor_groups=("expert-2",))
    registry = {r.reviewer_id: r for r in (first, second)}
    return evidence, inspection, registry


def assess(case, scores=None, **kwargs):
    evidence, inspection, registry = case
    return assess_fake_quality(
        evidence,
        "watch-1",
        scores or {"quality-1": 900, "quality-2": 901},
        registry,
        RUBRIC_VERSION,
        inspection.sha256,
        inspection=inspection,
        **kwargs,
    )


def test_two_qualified_independent_scores_publish_hash_bound_midpoint(quality_case):
    result = assess(quality_case)
    assert result.score_0_1000 == 901  # halves round up, not banker's rounding
    assert result.reviewer_scores == (("quality-1", 900), ("quality-2", 901))
    assert result.inspection_evidence_sha256 == quality_case[1].sha256
    verify_fake_quality(result)
    with pytest.raises(ValueError):
        verify_fake_quality(replace(result, score_0_1000=1000))
    assert quality_case[0].accepted_grade == 4


@pytest.mark.parametrize(
    "scores",
    [
        {"quality-1": 900},
        {"quality-1": -1, "quality-2": 1001},
        {"quality-1": True, "quality-2": 900},
        {"quality-1": 800, "quality-2": 901},
        {"quality-1": 900, "unknown": 900},
    ],
)
def test_draft_invalid_unqualified_or_disagreeing_scores_are_unassessed(quality_case, scores):
    assert assess(quality_case, scores) is None


@pytest.mark.parametrize("change", ["inside", "mechanical", "watch", "hash", "rubric", "absent"])
def test_missing_or_mismatched_inspection_cannot_publish(quality_case, change):
    evidence, inspection, registry = quality_case
    if change in {"inside", "mechanical"}:
        inspection = replace(
            inspection,
            attribute_artifacts=tuple(
                pair for pair in inspection.attribute_artifacts if pair[0] != change
            ),
        )
    elif change == "watch":
        inspection = replace(inspection, watch_entity_id="other")
    result = assess_fake_quality(
        evidence,
        "watch-1",
        {"quality-1": 900, "quality-2": 900},
        registry,
        "unknown" if change == "rubric" else RUBRIC_VERSION,
        "0" * 64 if change == "hash" else inspection.sha256,
        inspection=None if change == "absent" else inspection,
    )
    assert result is None


@pytest.mark.parametrize(
    "change", ["expired", "not-yet-valid", "conflict", "shared-group", "label-only"]
)
def test_quality_authority_is_separate_current_and_independent(quality_case, change):
    evidence, inspection, registry = quality_case
    first = registry["quality-1"]
    changes = {
        "expired": {"expires_at": inspection.reviewed_at},
        "not-yet-valid": {"effective_at": inspection.reviewed_at + timedelta(seconds=1)},
        "conflict": {"prohibited_contributor_groups": inspection.contributor_groups},
        "shared-group": {"contributor_groups": ("expert-2",)},
    }
    registry = dict(registry)
    registry["quality-1"] = (
        object() if change == "label-only" else replace(first, **changes[change])
    )
    assert assess((evidence, inspection, registry)) is None


def test_disagreement_needs_separate_qualified_adjudicator(quality_case):
    evidence, inspection, registry = quality_case
    third = replace(
        registry["quality-1"],
        reviewer_id="adjudicator",
        contributor_groups=("expert-3",),
        adjudicator=True,
    )
    registry = {**registry, third.reviewer_id: third}
    result = assess(
        (evidence, inspection, registry),
        {"quality-1": 700, "quality-2": 950},
        adjudicator_score=("adjudicator", 825),
    )
    assert result.score_0_1000 == 825
    assert result.adjudicator_score == ("adjudicator", 825)
    assert (
        assess(
            (evidence, inspection, registry),
            {"quality-1": 700, "quality-2": 950},
            adjudicator_score=("quality-1", 825),
        )
        is None
    )


@pytest.mark.parametrize(
    "evidence_type,label,grade",
    [
        ("seller_text", Label.FAKE, 4),
        ("owner_claim", Label.GENUINE, None),
    ],
)
def test_seller_claim_and_unsupported_original_never_publish(
    candidate,
    quality_case,
    evidence_type,
    label,
    grade,
):
    _, inspection, registry = quality_case
    reviewer = ReviewerRecord(
        "label-reviewer",
        (evidence_type,),
        "a" * 64,
        inspection.reviewed_at - timedelta(days=1),
        inspection.reviewed_at + timedelta(days=1),
        (),
        False,
    )
    review = submit_review(
        candidate,
        reviewer,
        proposed_label=label,
        proposed_grade=grade,
        proposed_tier=EvidenceTier.C,
        evidence_type=evidence_type,
        supporting_artifact_hashes=("b" * 64,),
        reviewed_at=inspection.reviewed_at,
    )
    evidence = adjudicate(candidate, [review], {reviewer.reviewer_id: reviewer})
    assert (
        assess(
            (
                evidence,
                replace(inspection, evidence_record_id=evidence.evidence_record_id),
                registry,
            )
        )
        is None
    )


def test_tampered_label_evidence_is_rejected(quality_case):
    evidence, inspection, registry = quality_case
    with pytest.raises(ValueError):
        assess((replace(evidence, accepted_grade=3), inspection, registry))


def test_accepted_genuine_has_no_fake_quality(candidate, quality_case):
    _, inspection, registry = quality_case
    reviewer = ReviewerRecord(
        "label-reviewer",
        ("partner_authenticated_original",),
        "a" * 64,
        inspection.reviewed_at - timedelta(days=1),
        inspection.reviewed_at + timedelta(days=1),
        (),
        False,
    )
    review = submit_review(
        candidate,
        reviewer,
        proposed_label=Label.GENUINE,
        proposed_grade=None,
        proposed_tier=EvidenceTier.B,
        evidence_type="partner_authenticated_original",
        supporting_artifact_hashes=("b" * 64,),
        reviewed_at=inspection.reviewed_at,
    )
    evidence = adjudicate(candidate, [review], {reviewer.reviewer_id: reviewer})
    assert evidence.tier.training_eligible
    assert (
        assess(
            (
                evidence,
                replace(inspection, evidence_record_id=evidence.evidence_record_id),
                registry,
            )
        )
        is None
    )


def test_exact_disagreement_boundary_is_midpoint(quality_case):
    assert assess(quality_case, {"quality-1": 850, "quality-2": 950}).score_0_1000 == 900


def test_unqualified_adjudicator_and_score_out_of_range_stay_unassessed(quality_case):
    evidence, inspection, registry = quality_case
    third = replace(registry["quality-1"], reviewer_id="third", contributor_groups=("expert-3",))
    registry = {**registry, "third": third}
    assert (
        assess(
            (evidence, inspection, registry),
            {"quality-1": 700, "quality-2": 950},
            adjudicator_score=("third", 825),
        )
        is None
    )
    registry["third"] = replace(third, adjudicator=True)
    assert (
        assess(
            (evidence, inspection, registry),
            {"quality-1": 700, "quality-2": 950},
            adjudicator_score=("third", 1001),
        )
        is None
    )


def test_review_registry_and_inspection_are_in_assessment_digest(quality_case):
    evidence, inspection, registry = quality_case
    baseline = assess(quality_case)
    registry = dict(registry)
    registry["quality-1"] = replace(registry["quality-1"], qualification_artifact_sha256="0" * 64)
    assert assess((evidence, inspection, registry)).assessment_sha256 != baseline.assessment_sha256
    inspection = replace(
        inspection,
        attribute_artifacts=(
            ("exterior", "0" * 64),
            ("inside", "d" * 64),
            ("mechanical", "e" * 64),
        ),
    )
    assert (
        assess((evidence, inspection, registry)).inspection_evidence_sha256
        != baseline.inspection_evidence_sha256
    )
