"""Contract tests for immutable IBM training-plane value objects."""

from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from core.types import Label, ModelClass
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import (
    AuthorizationDecision,
    AuthorizedRun,
    InstanceRef,
    ObjectRef,
    PostPromoEntry,
    ResourceLease,
    RunEvent,
)
from tests.conftest import (
    NOW,
    valid_credit,
    valid_post_promo_plan,
    valid_pricing,
    valid_run_spec,
    valid_usage,
)


def test_run_spec_rejects_mutable_image_tag():
    with pytest.raises(ValueError, match="sha256"):
        replace(valid_run_spec(), image_digest="icr.io/watch/train:latest")


def test_model_classes_include_trained_rejection():
    assert tuple(member.value for member in ModelClass) == (0, 1, 2, 3, 4, 5)
    assert ModelClass.REJECTION.value == 5


def test_rejection_is_a_manifest_training_label():
    assert Label.REJECTION.value == "rejection"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("code_sha", "a" * 39, "Git SHA"),
        ("manifest_sha256", "g" * 64, "SHA-256"),
        ("profile", "gx2-8x64x1v100", "gx3-16x80x1l4"),
        ("region", "", "region"),
        ("zone", " ", "zone"),
        ("bucket", "", "bucket"),
        ("prefix", "", "prefix"),
        ("max_hours", Decimal("0"), "max_hours"),
        ("max_hours", Decimal("24.01"), "max_hours"),
        ("public_gateway_owned_by_run", False, "Public Gateway"),
    ],
)
def test_run_spec_rejects_invalid_contract_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(valid_run_spec(), **{field: value})


def test_run_spec_rejects_invalid_provider_identifier():
    with pytest.raises(ValueError, match="network_id"):
        replace(valid_run_spec(), network_id="not an id")


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        "https://s3.direct.eu-de.cloud-object-storage.appdomain.cloud",
        "https://evil.example/s3.direct.us-south.cloud-object-storage.appdomain.cloud",
    ],
)
def test_run_spec_requires_region_bound_private_or_direct_cos_endpoint(endpoint):
    with pytest.raises(ValueError, match="COS endpoint"):
        replace(valid_run_spec(), cos_endpoint=endpoint)


def test_financial_evidence_rejects_naive_time_and_non_decimal_or_nan():
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(valid_credit(), observed_at=datetime(2026, 11, 5))
    with pytest.raises(TypeError, match="Decimal"):
        replace(valid_credit(), remaining_usd=1.5)
    with pytest.raises(ValueError, match="finite"):
        replace(valid_credit(), remaining_usd=Decimal("NaN"))
    with pytest.raises(ValueError, match="negative"):
        replace(valid_usage(), observed_usage_usd=Decimal("-0.01"))
    with pytest.raises(TypeError, match="UUID"):
        replace(valid_usage(), unreconciled_run_ids=("not-a-uuid",))


def test_pricing_requires_unique_lines_and_usd():
    pricing = valid_pricing()
    with pytest.raises(ValueError, match="duplicate"):
        replace(pricing, lines=(pricing.lines[0], pricing.lines[0]))
    with pytest.raises(ValueError, match="USD"):
        replace(pricing, currency="BRL")


def test_post_promo_plan_requires_unique_services_and_hashes():
    plan = valid_post_promo_plan()
    with pytest.raises(ValueError, match="duplicate"):
        replace(plan, entries=(plan.entries[0], plan.entries[0]))
    with pytest.raises(ValueError, match="SHA-256"):
        replace(plan.entries[0], export_destination_sha256="bad")


def test_run_event_validates_state_sequence_hashes_and_timestamp():
    event = RunEvent(
        event_id=UUID("99999999-9999-4999-8999-999999999999"),
        run_id=valid_run_spec().run_id,
        sequence=0,
        state="planned",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256="9" * 64,
        previous_event_sha256=None,
        payload_sha256="a" * 64,
    )
    assert event.state == "planned"
    with pytest.raises(ValueError, match="state"):
        replace(event, state="unknown")
    with pytest.raises(ValueError, match="previous_event_sha256"):
        replace(event, sequence=1)


def test_resource_instance_and_object_refs_validate_boundaries():
    lease = ResourceLease(
        resource_crn="crn:v1:bluemix:public:is:us-south:a/abc::instance:r006-abc",
        service="vpc-vsi",
        run_id=valid_run_spec().run_id,
        created_at=NOW,
        expires_at=NOW + timedelta(hours=6),
        terminal_policy="delete",
    )
    assert lease.expires_at > lease.created_at
    with pytest.raises(ValueError, match="expires_at"):
        replace(lease, expires_at=NOW - timedelta(seconds=1))

    instance = InstanceRef(
        id="r006-77777777-7777-4777-8777-777777777777",
        name="watch-auth-run",
        profile="gx3-16x80x1l4",
        zone="us-south-1",
        vpc_id="r006-22222222-2222-4222-8222-222222222222",
        subnet_id="0717-33333333-3333-4333-8333-333333333333",
        image_id="r006-88888888-8888-4888-8888-888888888888",
        status="running",
    )
    assert instance.status == "running"

    object_ref = ObjectRef(
        bucket="watch-auth-private",
        key="objects/a",
        sha256="b" * 64,
        length=10,
        etag="etag",
    )
    with pytest.raises(ValueError, match="length"):
        replace(object_ref, length=-1)


def test_authorized_run_only_accepts_allowed_decision_and_freezes_values():
    decision = AuthorizationDecision(
        allowed=True,
        reasons=(),
        estimated_usd=Decimal("12.00"),
        usable_credit_usd=Decimal("17500.00"),
        evidence_hashes=("1" * 64, "2" * 64),
        run_spec_sha256=run_spec_sha256(valid_run_spec()),
    )
    run = AuthorizedRun.from_decision(valid_run_spec(), decision, decision.evidence_hashes)
    assert run.estimated_usd == Decimal("12.00")
    with pytest.raises(FrozenInstanceError):
        run.estimated_usd = Decimal("1")

    denied = replace(decision, allowed=False, reasons=("insufficient credit",))
    with pytest.raises(ValueError, match="denied"):
        AuthorizedRun.from_decision(valid_run_spec(), denied, denied.evidence_hashes)

    other_spec = replace(valid_run_spec(), max_hours=Decimal("7"))
    with pytest.raises(ValueError, match="run spec"):
        AuthorizedRun.from_decision(other_spec, decision, decision.evidence_hashes)

    with pytest.raises(TypeError):
        AuthorizedRun(
            spec=valid_run_spec(),
            estimated_usd=decision.estimated_usd,
            evidence_hashes=decision.evidence_hashes,
        )
    with pytest.raises(TypeError, match="from_decision"):
        AuthorizedRun()


def test_run_spec_requires_account_and_positive_metering_dimensions():
    with pytest.raises(ValueError, match="account_id"):
        replace(valid_run_spec(), account_id="")
    for field in (
        "boot_volume_gb",
        "cos_storage_gb_month",
        "cos_operations",
        "registry_storage_gb_month",
        "code_engine_job_hours",
        "logs_monitoring_gb",
    ):
        with pytest.raises(ValueError, match=field):
            replace(valid_run_spec(), **{field: Decimal("0")})


def test_run_spec_rejects_metering_dimensions_below_conservative_floor():
    for field, value in (
        ("boot_volume_gb", Decimal("99.99")),
        ("cos_storage_gb_month", Decimal("49.99")),
        ("cos_operations", Decimal("999")),
        ("registry_storage_gb_month", Decimal("9.99")),
        ("code_engine_job_hours", Decimal("0.99")),
        ("logs_monitoring_gb", Decimal("0.99")),
    ):
        with pytest.raises(ValueError, match=field):
            replace(valid_run_spec(), **{field: value})


def test_authorization_decision_keeps_allowed_and_reasons_consistent():
    decision = AuthorizationDecision(
        allowed=True,
        reasons=(),
        estimated_usd=Decimal("12.00"),
        usable_credit_usd=Decimal("17500.00"),
        evidence_hashes=("1" * 64,),
        run_spec_sha256=run_spec_sha256(valid_run_spec()),
    )
    with pytest.raises(ValueError, match="allowed"):
        replace(decision, reasons=("unexpected",))
    with pytest.raises(ValueError, match="denied"):
        replace(decision, allowed=False)


def test_all_primary_evidence_objects_are_frozen():
    evidence = valid_pricing()
    with pytest.raises(FrozenInstanceError):
        evidence.currency = "BRL"


def test_post_promo_entry_rejects_naive_deadline():
    with pytest.raises(ValueError, match="timezone-aware"):
        PostPromoEntry("cos-storage", "a" * 64, datetime(2026, 12, 1))


def test_expiry_must_follow_observation():
    with pytest.raises(ValueError, match="expires_at"):
        replace(valid_credit(), expires_at=NOW - timedelta(seconds=1))
