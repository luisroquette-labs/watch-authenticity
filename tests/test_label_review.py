from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from acquisition.types import Angle, EvidenceTier, ReviewStatus, RightsStatus
from core.types import Label
from dataset.entities import (
    EntityEvidence,
    assign_watch_entities,
    build_content_clusters,
    build_grouping_decision,
)
from labeling.adjudicate import EvidenceTampered, NotTrainingEligible, ReviewerConflict, adjudicate
from labeling.manifest import build_training_record
from labeling.review import (
    ReviewerRecord,
    ReviewStore,
    load_reviewer_registry,
    submit_review,
)


@pytest.fixture
def reviewer(now):
    return ReviewerRecord(
        reviewer_id="reviewer-1",
        allowed_evidence_types=(
            "source_reputation",
            "replica_transaction",
            "physical_examination",
            "partner_authenticated_original",
        ),
        qualification_artifact_sha256="9" * 64,
        effective_at=now - timedelta(days=1),
        expires_at=now + timedelta(days=365),
        prohibited_contributor_groups=(),
        adjudicator=False,
    )


@pytest.fixture
def reviewer_registry(reviewer):
    return {reviewer.reviewer_id: reviewer}


@pytest.fixture
def now(candidate):
    return candidate.collected_at


@pytest.fixture
def grouping(candidate):
    clusters = build_content_clusters([candidate])
    entities = assign_watch_entities(
        [candidate], clusters, {candidate.candidate_id: EntityEvidence(listing_id="listing-1")}
    )
    return build_grouping_decision(candidate, clusters, entities)


@pytest.fixture
def make_review(candidate, reviewer, now):
    def make(
        *,
        label=Label.GENUINE,
        grade=None,
        tier=EvidenceTier.B,
        evidence_type=None,
        reviewer_record=reviewer,
        artifact="a" * 64,
    ):
        resolved_evidence_type = evidence_type or (
            "replica_transaction"
            if label is Label.FAKE
            else "partner_authenticated_original"
        )
        return submit_review(
            candidate,
            reviewer_record,
            proposed_label=label,
            proposed_grade=grade,
            proposed_tier=tier,
            evidence_type=resolved_evidence_type,
            supporting_artifact_hashes=(artifact,),
            reviewed_at=now,
        )

    return make


def test_dealer_reputation_alone_is_tier_c_and_not_trainable(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(
        candidate,
        [make_review(evidence_type="source_reputation", tier=EvidenceTier.A)],
        reviewer_registry,
    )
    assert evidence.tier is EvidenceTier.C
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="casual",
            now=candidate.collected_at,
        )


def test_default_reviewer_registry_fails_closed():
    assert load_reviewer_registry(Path("config/reviewer_registry.json")) == {}


def test_conflicting_reviews_become_inconclusive(candidate, make_review, reviewer, now):
    second = replace(reviewer, reviewer_id="reviewer-2")
    evidence = adjudicate(
        candidate,
        [
            make_review(label=Label.GENUINE),
            make_review(label=Label.FAKE, grade=3, reviewer_record=second),
        ],
        {reviewer.reviewer_id: reviewer, second.reviewer_id: second},
    )
    assert evidence.tier is EvidenceTier.INCONCLUSIVE


def test_disagreement_requires_registered_third_adjudicator(candidate, make_review, reviewer):
    second = replace(reviewer, reviewer_id="reviewer-2")
    third = replace(reviewer, reviewer_id="reviewer-3", adjudicator=True)
    reviews = [
        make_review(label=Label.GENUINE),
        make_review(label=Label.FAKE, grade=3, reviewer_record=second),
        make_review(label=Label.FAKE, grade=3, reviewer_record=third),
    ]
    registry = {item.reviewer_id: item for item in (reviewer, second, third)}

    evidence = adjudicate(candidate, reviews, registry)

    assert evidence.tier is EvidenceTier.B
    assert evidence.accepted_label is Label.FAKE
    assert evidence.accepted_grade == 3


def test_registry_change_invalidates_prior_review(candidate, make_review, reviewer):
    decision = make_review()
    changed = replace(reviewer, expires_at=reviewer.expires_at + timedelta(days=1))

    with pytest.raises(ReviewerConflict, match="registry binding"):
        adjudicate(candidate, [decision], {changed.reviewer_id: changed})


def test_one_qualified_tier_b_review_is_accepted(candidate, make_review, reviewer_registry):
    evidence = adjudicate(
        candidate, [make_review(label=Label.FAKE, grade=4)], reviewer_registry
    )
    assert evidence.tier is EvidenceTier.B
    assert evidence.accepted_label is Label.FAKE
    assert evidence.accepted_grade == 4


@pytest.mark.parametrize(
    ("label", "grade"),
    [(Label.GENUINE, None), (Label.FAKE, 1), (Label.FAKE, 2)],
)
def test_replica_transaction_rejects_incompatible_label_or_grade(
    make_review, label, grade
):
    with pytest.raises(ReviewerConflict):
        make_review(label=label, grade=grade, evidence_type="replica_transaction")


def test_unknown_fake_grade_and_reference_are_inconclusive(
    candidate, make_review, reviewer_registry
):
    unknown_grade = make_review(label=Label.FAKE, grade=None)
    assert adjudicate(candidate, [unknown_grade], reviewer_registry).tier is EvidenceTier.INCONCLUSIVE

    unknown_reference = replace(candidate, declared_reference="unknown-reference")
    matching_review = submit_review(
        unknown_reference,
        next(iter(reviewer_registry.values())),
        proposed_label=Label.FAKE,
        proposed_grade=4,
        proposed_tier=EvidenceTier.B,
        evidence_type="replica_transaction",
        supporting_artifact_hashes=("a" * 64,),
        reviewed_at=candidate.collected_at,
    )
    assert adjudicate(
        unknown_reference, [matching_review], reviewer_registry
    ).tier is EvidenceTier.INCONCLUSIVE


def test_tier_a_requires_two_independent_reviewers(candidate, make_review, reviewer):
    first = make_review(tier=EvidenceTier.A, evidence_type="physical_examination")
    assert adjudicate(candidate, [first], {reviewer.reviewer_id: reviewer}).tier is EvidenceTier.INCONCLUSIVE

    second_reviewer = replace(reviewer, reviewer_id="reviewer-2")
    second = make_review(
        tier=EvidenceTier.A,
        evidence_type="physical_examination",
        reviewer_record=second_reviewer,
    )
    assert adjudicate(
        candidate,
        [first, second],
        {reviewer.reviewer_id: reviewer, second_reviewer.reviewer_id: second_reviewer},
    ).tier is EvidenceTier.A


def test_tier_a_cannot_borrow_a_tier_c_second_vote(candidate, make_review, reviewer):
    second_reviewer = replace(reviewer, reviewer_id="reviewer-2")
    reviews = [
        make_review(tier=EvidenceTier.A, evidence_type="physical_examination"),
        make_review(
            tier=EvidenceTier.C,
            evidence_type="source_reputation",
            reviewer_record=second_reviewer,
        ),
    ]

    evidence = adjudicate(
        candidate,
        reviews,
        {reviewer.reviewer_id: reviewer, second_reviewer.reviewer_id: second_reviewer},
    )

    assert evidence.tier is EvidenceTier.INCONCLUSIVE


def test_gold_candidate_requires_two_reviewers(candidate, make_review, reviewer):
    first = make_review()
    assert adjudicate(
        candidate,
        [first],
        {reviewer.reviewer_id: reviewer},
        gold_candidate_ids=frozenset({candidate.candidate_id}),
    ).tier is EvidenceTier.INCONCLUSIVE

    second_reviewer = replace(reviewer, reviewer_id="reviewer-2")
    second = make_review(reviewer_record=second_reviewer)
    assert adjudicate(
        candidate,
        [first, second],
        {reviewer.reviewer_id: reviewer, second_reviewer.reviewer_id: second_reviewer},
        gold_candidate_ids=frozenset({candidate.candidate_id}),
    ).tier is EvidenceTier.B


def test_self_approval_is_rejected(candidate, reviewer, now):
    conflicted = replace(
        reviewer,
        prohibited_contributor_groups=(candidate.acquisition_batch_id,),
    )
    with pytest.raises(ReviewerConflict):
        submit_review(
            candidate,
            conflicted,
            proposed_label=Label.GENUINE,
            proposed_grade=None,
            proposed_tier=EvidenceTier.B,
            evidence_type="replica_transaction",
            supporting_artifact_hashes=("a" * 64,),
            reviewed_at=now,
        )


def test_tampered_decision_hash_is_rejected(candidate, make_review, reviewer_registry):
    decision = replace(make_review(), decision_sha256="f" * 64)
    with pytest.raises(EvidenceTampered):
        adjudicate(candidate, [decision], reviewer_registry)


def test_forged_adjudicator_authority_is_rejected(
    candidate, make_review, reviewer, reviewer_registry
):
    forged_reviewer = replace(reviewer, adjudicator=True)
    decision = make_review(reviewer_record=forged_reviewer)

    with pytest.raises(ReviewerConflict, match="registry binding"):
        adjudicate(candidate, [decision], reviewer_registry)


def test_same_reviewer_divergent_decisions_are_preserved_and_quarantined(make_review):
    store = ReviewStore()
    store.put(make_review(artifact="a" * 64))
    store.put(make_review(artifact="b" * 64))
    assert len(store.list()) == 2
    assert store.project().status is ReviewStatus.QUARANTINED


def test_same_decision_hash_is_idempotent(make_review):
    store = ReviewStore()
    decision = make_review()
    store.put(decision)
    store.put(decision)
    assert store.list() == (decision,)


def test_cos_store_uses_content_addressed_keys_and_reads_every_page(
    cos, make_review, reviewer
):
    second_reviewer = replace(reviewer, reviewer_id="reviewer-2")
    decisions = (
        make_review(),
        make_review(reviewer_record=second_reviewer, artifact="b" * 64),
    )
    store = ReviewStore(cos, "private")
    for decision in decisions:
        store.put(decision)

    keys = sorted(cos.bodies)
    assert keys == sorted(
        f"datasets/reviews/{decision.candidate_id}/{decision.reviewer_id}/"
        f"{decision.decision_sha256}.json"
        for decision in decisions
    )

    def list_page(*, Bucket, Prefix, ContinuationToken=None):
        del Bucket
        matching = [key for key in keys if key.startswith(Prefix)]
        if ContinuationToken is None:
            return {
                "Contents": [{"Key": matching[0]}],
                "IsTruncated": True,
                "NextContinuationToken": "page-2",
            }
        assert ContinuationToken == "page-2"
        return {"Contents": [{"Key": matching[1]}], "IsTruncated": False}

    cos.list_objects_v2 = list_page
    assert store.list() == tuple(sorted(decisions, key=lambda item: item.decision_sha256))


def test_revoked_rights_and_price_only_grade_cannot_reach_training(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(
        candidate, [make_review(label=Label.FAKE, grade=4)], reviewer_registry
    )
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            grouping,
            replace(rights, status=RightsStatus.REVOKED),
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="casual",
            now=candidate.collected_at,
        )

    price_only = adjudicate(
        candidate,
        [make_review(label=Label.FAKE, grade=4, evidence_type="source_reputation")],
        reviewer_registry,
    )
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            price_only,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="casual",
            now=candidate.collected_at,
        )


def test_build_training_record_uses_only_accepted_evidence(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(
        candidate, [make_review(label=Label.FAKE, grade=3)], reviewer_registry
    )
    record = build_training_record(
        candidate,
        evidence,
        grouping,
        rights,
        object_ref=object_ref,
        angle=Angle.DIAL,
        capture_style="studio",
        now=candidate.collected_at,
    )
    assert record.label is Label.FAKE
    assert record.grade == 3
    assert record.evidence_record_id == evidence.evidence_record_id


def test_build_training_record_rejects_grouping_from_another_candidate(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(candidate, [make_review()], reviewer_registry)

    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            replace(grouping, candidate_id="candidate-other"),
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="studio",
            now=candidate.collected_at,
        )


def test_build_training_record_rejects_tampered_evidence(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(candidate, [make_review()], reviewer_registry)
    tampered = replace(evidence, accepted_label=Label.FAKE, accepted_grade=4)

    with pytest.raises(EvidenceTampered):
        build_training_record(
            candidate,
            tampered,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="studio",
            now=candidate.collected_at,
        )


@pytest.mark.parametrize("offset_days", [-1, 1])
def test_build_training_record_rejects_rights_outside_validity(
    candidate,
    make_review,
    reviewer_registry,
    grouping,
    rights,
    object_ref,
    offset_days,
):
    evidence = adjudicate(candidate, [make_review()], reviewer_registry)
    invalid_now = (
        rights.expires_at
        if offset_days < 0
        else rights.effective_at - timedelta(microseconds=1)
    )

    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            capture_style="studio",
            now=invalid_now,
        )


def test_build_training_record_rejects_unlinked_image(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    evidence = adjudicate(candidate, [make_review()], reviewer_registry)
    unrelated = replace(
        object_ref,
        key="datasets/derivatives/" + "f" * 64,
        sha256="f" * 64,
    )

    with pytest.raises(NotTrainingEligible, match="derivative"):
        build_training_record(
            candidate,
            evidence,
            grouping,
            rights,
            object_ref=unrelated,
            angle=Angle.DIAL,
            capture_style="studio",
            now=candidate.collected_at,
        )
