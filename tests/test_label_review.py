from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from acquisition.types import Angle, EvidenceTier, ReviewStatus, RightsStatus
from core.types import Label
from dataset.angle_review import record_usability_review
from dataset.entities import (
    EntityEvidence,
    assign_watch_entities,
    build_content_clusters,
    build_grouping_decision,
)
from dataset.physical_link import record_physical_link_review
from labeling.adjudicate import (
    EvidenceTampered,
    NotTrainingEligible,
    ReviewerConflict,
    _digest,
    adjudicate,
)
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
    review = make_review(evidence_type="source_reputation", tier=EvidenceTier.A)
    evidence = adjudicate(candidate, [review], reviewer_registry)
    assert evidence.tier is EvidenceTier.C
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="casual",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_copied_official_media_cannot_be_admitted_even_with_review(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    copied = replace(
        candidate,
        raw_sha256="f6552812a866a41d8285ef2b5997aa66b9370414ea26bd5e3cb55faf0f112c98",
    )
    review = make_review(label=Label.FAKE, grade=4)
    evidence = adjudicate(candidate, [review], reviewer_registry)
    with pytest.raises(NotTrainingEligible, match="reuses official brand media"):
        build_training_record(
            copied, evidence, grouping, rights, object_ref=object_ref,
            angle=Angle.DIAL, usability_review=None, capture_style="casual",
            now=candidate.collected_at, reviews=[review], reviewer_registry=reviewer_registry, identity_records=(candidate,),
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
    accepted_review = make_review(label=Label.FAKE, grade=4)
    evidence = adjudicate(candidate, [accepted_review], reviewer_registry)
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            grouping,
            replace(rights, status=RightsStatus.REVOKED),
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="casual",
            now=candidate.collected_at,
            reviews=[accepted_review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )

    price_review = make_review(
        label=Label.FAKE, grade=4, evidence_type="source_reputation"
    )
    price_only = adjudicate(candidate, [price_review], reviewer_registry)
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            price_only,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="casual",
            now=candidate.collected_at,
            reviews=[price_review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_build_training_record_uses_only_accepted_evidence(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review(label=Label.FAKE, grade=3)
    evidence = adjudicate(candidate, [review], reviewer_registry)
    record = build_training_record(
        candidate,
        evidence,
        grouping,
        rights,
        object_ref=object_ref,
        angle=Angle.DIAL,
        usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
        capture_style="studio",
        now=candidate.collected_at,
        reviews=[review],
        reviewer_registry=reviewer_registry, identity_records=(candidate,),
    )
    assert record.label is Label.FAKE
    assert record.grade == 3
    assert record.evidence_record_id == evidence.evidence_record_id


def test_training_admission_recomputes_two_view_identity(
    candidate, make_review, reviewer, reviewer_registry, rights, object_ref
):
    clasp = replace(
        candidate, candidate_id="clasp-photo", source_record_id="same-listing-clasp",
        raw_sha256="b" * 64, derivative_sha256="c" * 64, phash="0123456789abcdef",
    )
    records = (candidate, clasp)
    listing = {item.candidate_id: EntityEvidence(listing_id="listing-1") for item in records}
    artifact = b"continuous source capture showing front, case and clasp of one watch"
    first = replace(reviewer, reviewer_id="identity-1",
                    allowed_evidence_types=(*reviewer.allowed_evidence_types,
                                            "physical_identity_review"))
    second = replace(first, reviewer_id="identity-2", qualification_artifact_sha256="8" * 64)
    registry = {**reviewer_registry, first.reviewer_id: first, second.reviewer_id: second}
    links = [
        record_physical_link_review(candidate, clasp, artifact,
                                    "continuous rotation of one case and bracelet",
                                    expert, candidate.collected_at)
        for expert in (first, second)
    ]
    clusters = build_content_clusters(records)
    assignments = assign_watch_entities(
        records, clusters, listing, physical_link_reviews=links,
        physical_link_artifacts={links[0].artifact_sha256: artifact},
        reviewer_registry=registry,
    )
    grouping = build_grouping_decision(candidate, clusters, assignments)
    authenticity = make_review()
    evidence = adjudicate(candidate, [authenticity], registry)
    kwargs = dict(
        object_ref=object_ref, angle=Angle.DIAL,
        usability_review=record_usability_review(
            candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL,
            True, True, "angle-reviewer", candidate.collected_at,
        ),
        capture_style="casual", now=candidate.collected_at,
        reviews=[authenticity], reviewer_registry=registry,
        identity_records=records, entity_evidence=listing,
        physical_link_reviews=links,
        physical_link_artifacts={links[0].artifact_sha256: artifact},
    )
    admitted = build_training_record(candidate, evidence, grouping, rights, **kwargs)
    assert admitted.watch_entity_id == assignments[clasp.candidate_id]

    with pytest.raises(NotTrainingEligible, match="physical grouping differs"):
        build_training_record(candidate, evidence, grouping, rights,
                              **{**kwargs, "physical_link_reviews": links[:1]})
    with pytest.raises(NotTrainingEligible, match="physical grouping differs"):
        build_training_record(candidate, evidence, replace(grouping, watch_entity_id="forged"),
                              rights, **kwargs)


def test_seller_declared_factory_does_not_become_training_provenance(
    candidate, reviewer, reviewer_registry, grouping, rights, object_ref
):
    declared = replace(candidate, factory="seller-claimed-factory")
    review = submit_review(
        declared, reviewer,
        proposed_label=Label.FAKE, proposed_grade=3,
        proposed_tier=EvidenceTier.B, evidence_type="replica_transaction",
        supporting_artifact_hashes=("a" * 64,), reviewed_at=declared.collected_at,
    )
    evidence = adjudicate(declared, [review], reviewer_registry)
    record = build_training_record(
        declared, evidence, grouping, rights,
        object_ref=object_ref,
        angle=Angle.DIAL,
        usability_review=record_usability_review(
            declared.candidate_id, declared.derivative_sha256, Angle.DIAL,
            True, True, "angle-reviewer", declared.collected_at,
        ),
        capture_style="studio",
        now=declared.collected_at,
        reviews=[review],
        reviewer_registry=reviewer_registry, identity_records=(declared,),
    )
    assert record.factory is None


def test_factory_record_attests_named_factory_for_holdout(
    candidate, reviewer, grouping, rights, object_ref
):
    declared = replace(candidate, factory="verified-factory")
    qualified = replace(reviewer, allowed_evidence_types=("factory_record",))
    with pytest.raises(ReviewerConflict, match="factory identity"):
        submit_review(
            declared, qualified, proposed_label=Label.FAKE, proposed_grade=3,
            proposed_tier=EvidenceTier.B, evidence_type="factory_record",
            supporting_artifact_hashes=("a" * 64,),
            reviewed_at=declared.collected_at, proposed_factory="another-factory",
        )
    review = submit_review(
        declared, qualified, proposed_label=Label.FAKE, proposed_grade=3,
        proposed_tier=EvidenceTier.B, evidence_type="factory_record",
        supporting_artifact_hashes=("a" * 64,),
        reviewed_at=declared.collected_at, proposed_factory="verified-factory",
    )
    evidence = adjudicate(declared, [review], {qualified.reviewer_id: qualified})
    record = build_training_record(
        declared, evidence, grouping, rights, object_ref=object_ref,
        angle=Angle.DIAL,
        usability_review=record_usability_review(
            declared.candidate_id, declared.derivative_sha256, Angle.DIAL,
            True, True, "angle-reviewer", declared.collected_at,
        ),
        capture_style="studio", now=declared.collected_at,
        reviews=[review], reviewer_registry={qualified.reviewer_id: qualified}, identity_records=(declared,),
    )
    assert record.factory == "verified-factory"


def test_unverified_capture_style_cannot_enter_training(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review(label=Label.FAKE, grade=3)
    evidence = adjudicate(candidate, [review], reviewer_registry)
    kwargs = dict(
        object_ref=object_ref,
        angle=Angle.DIAL,
        usability_review=record_usability_review(
            candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL,
            True, True, "angle-reviewer", candidate.collected_at,
        ),
        now=candidate.collected_at,
        reviews=[review],
        reviewer_registry=reviewer_registry, identity_records=(candidate,),
    )

    with pytest.raises(NotTrainingEligible, match="capture style"):
        build_training_record(candidate, evidence, grouping, rights, capture_style="unknown", **kwargs)
    accepted = build_training_record(candidate, evidence, grouping, rights, capture_style="studio", **kwargs)
    with pytest.raises(ValueError, match="capture_style"):
        replace(accepted, capture_style="unknown")


def test_build_training_record_rejects_grouping_from_another_candidate(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)

    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate,
            evidence,
            replace(grouping, candidate_id="candidate-other"),
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_build_training_record_rejects_tampered_evidence(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
    tampered = replace(evidence, accepted_label=Label.FAKE, accepted_grade=4)

    with pytest.raises(EvidenceTampered):
        build_training_record(
            candidate,
            tampered,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_build_training_record_rejects_rehashed_result_not_supported_by_reviews(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
    forged = replace(evidence, accepted_label=Label.FAKE, accepted_grade=4)
    digest = _digest(
        {
            "candidate_id": forged.candidate_id,
            "candidate_sha256": forged.candidate_sha256,
            "accepted_grade": forged.accepted_grade,
            "accepted_label": forged.accepted_label.value,
            "evidence_type": forged.evidence_type,
            "reference_id": forged.reference_id,
            "review_decision_sha256s": list(forged.review_decision_sha256s),
            "review_timestamps": [value.isoformat() for value in forged.review_timestamps],
            "reviewer_ids": list(forged.reviewer_ids),
            "supporting_artifact_hashes": list(forged.supporting_artifact_hashes),
            "tier": forged.tier.value,
        }
    )
    forged = replace(
        forged,
        decision_sha256=digest,
        evidence_record_id=f"evidence-{digest[:24]}",
    )

    with pytest.raises(NotTrainingEligible, match="immutable review decisions"):
        build_training_record(
            candidate,
            forged,
            grouping,
            rights,
            object_ref=object_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_build_training_record_rejects_candidate_changed_after_review(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
    changed_candidate = replace(
        candidate,
        raw_sha256="e" * 64,
        derivative_sha256="f" * 64,
    )
    changed_ref = replace(
        object_ref,
        key="datasets/derivatives/" + "f" * 64,
        sha256="f" * 64,
    )

    with pytest.raises(NotTrainingEligible, match="reviewed candidate"):
        build_training_record(
            changed_candidate,
            evidence,
            grouping,
            rights,
            object_ref=changed_ref,
            angle=Angle.DIAL,
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
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
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
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
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=invalid_now,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


def test_build_training_record_rejects_unlinked_image(
    candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
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
            usability_review=record_usability_review(candidate.candidate_id, candidate.derivative_sha256, Angle.DIAL, True, True, "angle-reviewer", candidate.collected_at),
            capture_style="studio",
            now=candidate.collected_at,
            reviews=[review],
            reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )


@pytest.mark.parametrize(
    ("evidence_type", "label", "grade"),
    [("seller_text", Label.FAKE, 4), ("owner_claim", Label.GENUINE, None)],
)
def test_seller_superclone_and_unsupported_original_remain_candidate_only(
    candidate,
    reviewer,
    evidence_type,
    label,
    grade,
):
    authority = replace(reviewer, allowed_evidence_types=(evidence_type,))
    review = submit_review(
        candidate,
        authority,
        proposed_label=label,
        proposed_grade=grade,
        proposed_tier=EvidenceTier.C,
        evidence_type=evidence_type,
        supporting_artifact_hashes=("a" * 64,),
        reviewed_at=candidate.collected_at,
    )
    evidence = adjudicate(candidate, [review], {authority.reviewer_id: authority})
    assert evidence.tier is EvidenceTier.C
    assert not evidence.tier.training_eligible


@pytest.mark.parametrize("case", ["missing", "hidden", "wrong-angle", "wrong-image"])
def test_training_requires_matching_visible_usable_angle(
    case, candidate, make_review, reviewer_registry, grouping, rights, object_ref
):
    from dataset.angle_review import record_usability_review

    review = make_review()
    evidence = adjudicate(candidate, [review], reviewer_registry)
    usability = None if case == "missing" else record_usability_review(
        candidate.candidate_id,
        "0" * 64 if case == "wrong-image" else candidate.derivative_sha256,
        Angle.CLASP if case == "wrong-angle" else Angle.DIAL,
        case != "hidden", case != "hidden", "angle-reviewer", candidate.collected_at,
    )
    with pytest.raises(NotTrainingEligible, match="usability"):
        build_training_record(
            candidate, evidence, grouping, rights, object_ref=object_ref,
            angle=Angle.DIAL, usability_review=usability, capture_style="studio",
            now=candidate.collected_at, reviews=[review], reviewer_registry=reviewer_registry, identity_records=(candidate,),
        )
