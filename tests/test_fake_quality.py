from dataclasses import dataclass, replace
from datetime import timedelta

import pytest

from acquisition.types import EvidenceTier
from core.types import Label
from dataset.entities import (
    EntityEvidence,
    assign_watch_entities,
    build_content_clusters,
    build_grouping_decision,
)
from labeling.adjudicate import adjudicate
from labeling.quality import (
    RUBRIC_VERSION,
    FakeQualityInspection,
    QualityReviewerRecord,
    assess_fake_quality,
    submit_quality_review,
    verify_fake_quality,
)
from labeling.review import ReviewerRecord, submit_review


@dataclass
class Case:
    candidate: object
    evidence: object
    grouping: object
    inspection: object
    registry: dict
    decisions: tuple


@pytest.fixture
def quality_case(candidate):
    now = candidate.collected_at
    label_reviewer = ReviewerRecord(
        "label-reviewer",
        ("replica_transaction",),
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
    clusters = build_content_clusters([candidate])
    entities = assign_watch_entities(
        [candidate],
        clusters,
        {
            candidate.candidate_id: EntityEvidence(listing_id="listing-1"),
        },
    )
    grouping = build_grouping_decision(candidate, clusters, entities)
    inspection = FakeQualityInspection(
        grouping.watch_entity_id,
        evidence.evidence_record_id,
        now,
        (("exterior", "c" * 64), ("inside", "d" * 64), ("mechanical", "e" * 64)),
        (candidate.source_policy_id, candidate.acquisition_batch_id, grouping.seller_group_id),
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
    decisions = tuple(
        submit_quality_review(inspection, registry[key], score, reviewed_at=now)
        for key, score in (("quality-1", 900), ("quality-2", 901))
    )
    return Case(candidate, evidence, grouping, inspection, registry, decisions)


def assess(case, **kwargs):
    args = dict(candidate=case.candidate, grouping=case.grouping, inspection=case.inspection)
    args.update(kwargs)
    return assess_fake_quality(
        case.evidence,
        case.inspection.watch_entity_id,
        case.decisions,
        case.registry,
        RUBRIC_VERSION,
        case.inspection.sha256,
        **args,
    )


def rescore(case, scores):
    return replace(
        case,
        decisions=tuple(
            submit_quality_review(
                case.inspection,
                case.registry[key],
                score,
                reviewed_at=case.inspection.reviewed_at,
            )
            for key, score in scores
        ),
    )


def test_two_qualified_independent_decisions_publish_hash_bound_midpoint(quality_case):
    result = assess(quality_case)
    assert result.score_0_1000 == 901  # half up, never banker's rounding
    assert result.reviewer_scores == (("quality-1", 900), ("quality-2", 901))
    assert result.inspection_evidence_sha256 == quality_case.inspection.sha256
    assert result.review_decision_sha256s == tuple(
        sorted(d.decision_sha256 for d in quality_case.decisions)
    )
    verify_fake_quality(result)
    with pytest.raises(ValueError):
        verify_fake_quality(replace(result, score_0_1000=1000))
    assert quality_case.evidence.accepted_grade == 4


def test_one_reviewer_and_bare_scores_stay_unassessed(quality_case):
    assert assess(replace(quality_case, decisions=quality_case.decisions[:1])) is None
    assert assess(replace(quality_case, decisions={"quality-1": 900, "quality-2": 901})) is None


@pytest.mark.parametrize("score", [-1, 1001, True, 900.5])
def test_invalid_score_cannot_become_a_decision(quality_case, score):
    with pytest.raises(ValueError):
        submit_quality_review(
            quality_case.inspection,
            quality_case.registry["quality-1"],
            score,
            reviewed_at=quality_case.inspection.reviewed_at,
        )


@pytest.mark.parametrize("attribute", ["inside", "mechanical", "exterior"])
def test_missing_required_inspection_is_unassessed_even_with_new_votes(quality_case, attribute):
    inspection = replace(
        quality_case.inspection,
        attribute_artifacts=tuple(
            pair for pair in quality_case.inspection.attribute_artifacts if pair[0] != attribute
        ),
    )
    case = rescore(
        replace(quality_case, inspection=inspection), (("quality-1", 300), ("quality-2", 320))
    )
    assert assess(case) is None  # no invented full score for obvious exterior-only fakes


@pytest.mark.parametrize(
    "change", ["inspection", "rubric", "review-time", "score", "qualification"]
)
def test_immutable_decisions_cannot_be_replayed_or_tampered(quality_case, change):
    case = quality_case
    first = case.decisions[0]
    if change == "inspection":
        case = replace(
            case,
            inspection=replace(
                case.inspection, reviewed_at=case.inspection.reviewed_at + timedelta(seconds=1)
            ),
        )
    elif change == "qualification":
        registry = dict(case.registry)
        registry["quality-1"] = replace(
            registry["quality-1"], qualification_artifact_sha256="0" * 64
        )
        case = replace(case, registry=registry)
    else:
        fields = {
            "rubric": {"rubric_version": "other"},
            "review-time": {"reviewed_at": first.reviewed_at + timedelta(seconds=1)},
            "score": {"score_0_1000": 1000},
        }
        case = replace(case, decisions=(replace(first, **fields[change]), case.decisions[1]))
    assert assess(case) is None


def test_changed_artifact_cannot_reuse_scores(quality_case):
    inspection = replace(
        quality_case.inspection,
        attribute_artifacts=(
            ("exterior", "0" * 64),
            ("inside", "d" * 64),
            ("mechanical", "e" * 64),
        ),
    )
    assert assess(replace(quality_case, inspection=inspection)) is None


@pytest.mark.parametrize("change", ["watch", "candidate", "candidate-content", "quarantine"])
def test_authoritative_candidate_entity_binding_is_required(quality_case, change):
    case = quality_case
    if change == "watch":
        case = replace(
            case, inspection=replace(case.inspection, watch_entity_id="unconfirmed-watch")
        )
        case = rescore(case, (("quality-1", 900), ("quality-2", 901)))
    elif change == "candidate":
        case = replace(case, grouping=replace(case.grouping, candidate_id="other"))
    elif change == "candidate-content":
        case = replace(case, candidate=replace(case.candidate, raw_sha256="0" * 64))
    else:
        case = replace(case, grouping=replace(case.grouping, quarantined=True))
    assert assess(case) is None


@pytest.mark.parametrize("missing", ["source", "batch", "seller"])
def test_omitted_mandatory_conflict_groups_do_not_publish(quality_case, missing):
    group = {
        "source": quality_case.candidate.source_policy_id,
        "batch": quality_case.candidate.acquisition_batch_id,
        "seller": quality_case.grouping.seller_group_id,
    }[missing]
    inspection = replace(
        quality_case.inspection,
        contributor_groups=tuple(
            value for value in quality_case.inspection.contributor_groups if value != group
        ),
    )
    case = rescore(
        replace(quality_case, inspection=inspection), (("quality-1", 900), ("quality-2", 901))
    )
    assert assess(case) is None


@pytest.mark.parametrize("change", ["expired", "source-conflict", "shared-group", "label-only"])
def test_reviewers_must_remain_qualified_independent_and_conflict_free(quality_case, change):
    registry = dict(quality_case.registry)
    first = registry["quality-1"]
    if change == "label-only":
        registry["quality-1"] = object()
    else:
        changes = {
            "expired": {"expires_at": quality_case.inspection.reviewed_at},
            "source-conflict": {
                "prohibited_contributor_groups": (quality_case.candidate.source_policy_id,)
            },
            "shared-group": {"contributor_groups": ("expert-2",)},
        }
        registry["quality-1"] = replace(first, **changes[change])
    assert assess(replace(quality_case, registry=registry)) is None
    if change in {"source-conflict", "shared-group"}:
        case = rescore(
            replace(quality_case, registry=registry), (("quality-1", 900), ("quality-2", 901))
        )
        assert assess(case) is None  # fresh decisions cannot override conflicts either


def test_large_disagreement_requires_immutable_independent_adjudication(quality_case):
    case = rescore(quality_case, (("quality-1", 700), ("quality-2", 950)))
    assert assess(case) is None
    third = replace(
        case.registry["quality-1"],
        reviewer_id="third",
        contributor_groups=("expert-3",),
        adjudicator=True,
    )
    case = replace(case, registry={**case.registry, "third": third})
    decision = submit_quality_review(
        case.inspection, third, 825, reviewed_at=case.inspection.reviewed_at
    )
    assert assess(case, adjudicator_decision=decision).score_0_1000 == 825
    assert assess(case, adjudicator_decision=case.decisions[0]) is None
    assert assess(case, adjudicator_decision=replace(decision, score_0_1000=826)) is None
    registry = {**case.registry, "third": replace(third, adjudicator=False)}
    case = replace(case, registry=registry)
    unauthorized = submit_quality_review(
        case.inspection, registry["third"], 825, reviewed_at=case.inspection.reviewed_at
    )
    assert assess(case, adjudicator_decision=unauthorized) is None


def test_exact_disagreement_boundary_uses_midpoint(quality_case):
    case = rescore(quality_case, (("quality-1", 850), ("quality-2", 950)))
    assert assess(case).score_0_1000 == 900


@pytest.mark.parametrize(
    "evidence_type,label,grade,tier",
    [
        ("seller_text", Label.FAKE, 4, EvidenceTier.C),
        ("owner_claim", Label.GENUINE, None, EvidenceTier.C),
        ("partner_authenticated_original", Label.GENUINE, None, EvidenceTier.B),
    ],
)
def test_nonfake_or_seller_only_never_gets_quality(quality_case, evidence_type, label, grade, tier):
    case = quality_case
    now = case.inspection.reviewed_at
    reviewer = ReviewerRecord(
        "label",
        (evidence_type,),
        "a" * 64,
        now - timedelta(days=1),
        now + timedelta(days=1),
        (),
        False,
    )
    review = submit_review(
        case.candidate,
        reviewer,
        proposed_label=label,
        proposed_grade=grade,
        proposed_tier=tier,
        evidence_type=evidence_type,
        supporting_artifact_hashes=("b" * 64,),
        reviewed_at=now,
    )
    evidence = adjudicate(case.candidate, [review], {reviewer.reviewer_id: reviewer})
    inspection = replace(case.inspection, evidence_record_id=evidence.evidence_record_id)
    case = rescore(
        replace(case, evidence=evidence, inspection=inspection),
        (("quality-1", 900), ("quality-2", 901)),
    )
    assert assess(case) is None


def test_missing_lineage_or_inspection_is_unassessed(quality_case):
    assert assess(quality_case, candidate=None) is None
    assert assess(quality_case, grouping=None) is None
    assert assess(quality_case, inspection=None) is None


def test_tampered_evidence_rejected(quality_case):
    with pytest.raises(ValueError):
        assess(replace(quality_case, evidence=replace(quality_case.evidence, accepted_grade=3)))
