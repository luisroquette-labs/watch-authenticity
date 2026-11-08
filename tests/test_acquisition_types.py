"""Contracts that prevent untraceable images from reaching training."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime
from typing import Any, cast

import pytest

from acquisition.types import (
    Angle,
    BatchReceipt,
    CandidateRecord,
    DataDispositionReport,
    DatasetFreeze,
    EvidenceTier,
    FreezeResult,
    ImpactedArtifactReport,
    ReviewDecision,
    ReviewStatus,
    RightsStatus,
    SourcePolicy,
    validate_contract_bundle,
)
from core.types import Label


def test_only_a_and_b_are_training_eligible():
    assert EvidenceTier.A.training_eligible
    assert EvidenceTier.B.training_eligible
    assert not EvidenceTier.C.training_eligible
    assert not EvidenceTier.INCONCLUSIVE.training_eligible


def test_candidate_is_immutable(candidate: CandidateRecord):
    with pytest.raises(FrozenInstanceError):
        cast(Any, candidate).source_url = "https://changed.invalid"


def test_contract_enums_are_complete():
    assert {angle.value for angle in Angle} == {
        "dial", "cyclops", "rehaut", "crown", "clasp", "caseback",
    }
    assert {status.value for status in RightsStatus} == {
        "active", "expired", "revoked", "blocked",
    }
    assert ReviewStatus.QUARANTINED.value == "quarantined"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", ""),
        ("raw_sha256", "bad"),
        ("collected_at", datetime(2026, 11, 9, 12, 0)),
        ("source_url", "file:///tmp/photo.jpg"),
        ("price_usd", float("nan")),
    ],
)
def test_candidate_rejects_invalid_identity_integrity_and_time(candidate, field, value):
    with pytest.raises(ValueError):
        replace(candidate, **{field: value})


def test_candidate_rejects_grade_on_genuine_label(candidate):
    with pytest.raises(ValueError, match="only fake"):
        replace(candidate, proposed_label=Label.GENUINE, proposed_grade=1)


def test_training_requires_fake_grade_and_tier_a_or_b(valid_contract_bundle):
    _, _, _, training = valid_contract_bundle
    with pytest.raises(ValueError, match="grade is required"):
        replace(training, label=Label.FAKE, grade=None)
    with pytest.raises(ValueError, match="Tier A or B"):
        replace(training, evidence_tier=EvidenceTier.C)


def test_cross_record_lineage_must_match(valid_contract_bundle):
    candidate, rights, evidence, training = valid_contract_bundle
    validate_contract_bundle(candidate, rights, evidence, training)
    with pytest.raises(ValueError, match="candidate lineage"):
        validate_contract_bundle(candidate, rights, replace(evidence, candidate_id="other"), training)


def test_rights_fixture_preserves_expired_status(expired_rights):
    assert expired_rights.status is RightsStatus.EXPIRED
    assert expired_rights.expires_at < expired_rights.disposition_deadline


def test_policy_receipt_freeze_and_reports_are_frozen_real_contracts(
    valid_contract_bundle, object_ref,
):
    _, _, _, training = valid_contract_bundle
    policy = SourcePolicy(
        policy_id="policy-1",
        source="partner-upload",
        allowed_methods=("partner-export",),
        may_download=True,
        may_train=True,
        required_rights_purposes=("download", "train"),
        requires_rightsholder_permission=True,
        policy_evidence_sha256="1" * 64,
    )
    decision = ReviewDecision(
        candidate_id=training.candidate_id,
        reviewer_id="reviewer-1",
        proposed_label=Label.GENUINE,
        proposed_grade=None,
        proposed_tier=EvidenceTier.A,
        evidence_type="controlled-examination",
        supporting_artifact_hashes=("2" * 64,),
        reviewed_at=valid_contract_bundle[2].review_timestamps[0],
        decision_sha256="3" * 64,
    )
    receipt = BatchReceipt("batch-1", "4" * 64, "5" * 64, ("candidate-1",), ())
    freeze = DatasetFreeze(
        "6" * 64,
        "7" * 64,
        "8" * 64,
        (training.asset_id,),
        ("asset-excluded",),
        (("rights expired", 1),),
    )
    impacted = ImpactedArtifactReport(
        ("9" * 64,), "rights revoked", decision.reviewed_at,
    )
    disposition = DataDispositionReport(
        (object_ref,), (), (), "quarantine", decision.reviewed_at, "pending",
    )
    result = FreezeResult(freeze, impacted, (disposition,))
    assert policy.may_train and receipt.accepted_candidate_ids == ("candidate-1",)
    assert result.freeze is freeze and result.impacted_artifact_report is impacted
    with pytest.raises(FrozenInstanceError):
        cast(Any, decision).reviewer_id = "changed"


def test_policy_cannot_train_when_download_is_forbidden():
    with pytest.raises(ValueError, match="allow download"):
        SourcePolicy(
            policy_id="policy-1",
            source="blocked",
            allowed_methods=("official-api",),
            may_download=False,
            may_train=True,
            required_rights_purposes=("train",),
            requires_rightsholder_permission=True,
            policy_evidence_sha256="a" * 64,
        )


def test_unknown_source_can_use_an_empty_deny_all_policy():
    policy = SourcePolicy(
        policy_id="deny-all-v1",
        source="unknown",
        allowed_methods=(),
        may_download=False,
        may_train=False,
        required_rights_purposes=(),
        requires_rightsholder_permission=True,
        policy_evidence_sha256="b" * 64,
    )
    assert not policy.may_download and not policy.may_train
