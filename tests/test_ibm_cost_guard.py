"""Fail-closed tests for IBM promotional-credit authorization."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from infra.ibm_cost_guard import REQUIRED_METERED_LINES, authorize_run
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import (
    AuthorizationDecision,
    RunEvent,
    UsageReconciliationReceipt,
    run_event_sha256,
)
from tests.conftest import (
    NOW,
    valid_credit,
    valid_post_promo_plan,
    valid_pricing,
    valid_run_spec,
    valid_usage,
)


def _authorize(**overrides):
    values = {
        "credit": valid_credit(),
        "usage": valid_usage(),
        "pricing": valid_pricing(),
        "post_promo": valid_post_promo_plan(),
        "spec": valid_run_spec(),
        "events": (),
        "now": NOW,
    }
    values.update(overrides)
    return authorize_run(**values)


def test_authorize_run_allows_complete_fresh_evidence_without_provisioning():
    decision = _authorize()
    assert decision.allowed
    assert decision.reasons == ()
    assert decision.estimated_usd == Decimal("14.21")
    assert decision.usable_credit_usd == Decimal("17500.00")
    assert decision.evidence_hashes[:4] == ("1" * 64, "2" * 64, "4" * 64, "8" * 64)
    assert len(decision.evidence_hashes) == 5
    assert REQUIRED_METERED_LINES == frozenset(
        {
            "vpc-vsi",
            "vpc-boot-volume",
            "vpc-public-gateway",
            "cos-storage",
            "cos-operations",
            "container-registry-storage",
            "code-engine",
            "logs-monitoring",
        }
    )


@pytest.mark.parametrize("age", [timedelta(hours=24, seconds=1), timedelta(days=2)])
def test_authorize_run_rejects_stale_credit(age: timedelta):
    decision = _authorize(credit=replace(valid_credit(), observed_at=NOW - age))
    assert not decision.allowed
    assert "credit evidence is stale" in decision.reasons


def test_authorize_run_accepts_24h_boundary_and_rejects_stale_pricing():
    boundary = NOW - timedelta(hours=24)
    assert _authorize(
        credit=replace(valid_credit(), observed_at=boundary),
        usage=replace(valid_usage(), observed_at=boundary),
    ).allowed
    pricing = replace(valid_pricing(), observed_at=NOW - timedelta(hours=24, seconds=1))
    decision = _authorize(pricing=pricing)
    assert not decision.allowed
    assert "pricing evidence is stale" in decision.reasons


def test_authorize_run_rejects_ineligible_service_plan():
    credit = replace(
        valid_credit(),
        eligible_service_plans=tuple(
            pair for pair in valid_credit().eligible_service_plans if pair[0] != "vpc-vsi"
        ),
    )
    decision = _authorize(credit=credit)
    assert not decision.allowed
    assert "pricing includes an ineligible provider SKU" in decision.reasons


def test_authorize_run_requires_48h_expiry_margin():
    exact = replace(valid_credit(), expires_at=NOW + timedelta(hours=54))
    plan = valid_post_promo_plan()
    plan = replace(
        plan,
        promo_expires_at=exact.expires_at,
        entries=tuple(
            replace(entry, delete_or_downgrade_deadline=NOW + timedelta(hours=1))
            for entry in plan.entries
        ),
    )
    assert _authorize(credit=exact, post_promo=plan).allowed
    too_late = replace(exact, expires_at=exact.expires_at - timedelta(seconds=1))
    decision = _authorize(
        credit=too_late,
        post_promo=replace(plan, promo_expires_at=too_late.expires_at),
    )
    assert not decision.allowed
    assert "run would finish less than 48 hours before promo expiry" in decision.reasons


def test_authorize_run_enforces_exact_500_usd_reserve():
    estimated = Decimal("14.21")
    assert _authorize(credit=replace(valid_credit(), remaining_usd=estimated + Decimal("500"))).allowed
    decision = _authorize(
        credit=replace(valid_credit(), remaining_usd=estimated + Decimal("499.99"))
    )
    assert not decision.allowed
    assert "estimate exceeds credit after USD 500.00 safety reserve" in decision.reasons


def test_authorize_run_rejects_missing_required_service_even_when_free():
    pricing = valid_pricing()
    pricing = replace(
        pricing,
        lines=tuple(line for line in pricing.lines if line.service != "code-engine"),
    )
    decision = _authorize(pricing=pricing)
    assert not decision.allowed
    assert "missing required metered service: code-engine" in decision.reasons


def test_authorize_run_rejects_unreconciled_prior_run_even_if_deleted():
    prior_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    event = RunEvent(
        event_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        run_id=prior_id,
        sequence=0,
        state="deleted",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256="a" * 64,
        previous_event_sha256=None,
        payload_sha256="b" * 64,
    )
    usage = replace(valid_usage(), unreconciled_run_ids=(prior_id,))
    decision = _authorize(usage=usage, events=(event,))
    assert not decision.allowed
    assert "prior run usage is unreconciled" in decision.reasons


def test_authorize_run_rejects_nonterminal_prior_event():
    event = RunEvent(
        event_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        run_id=UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        sequence=0,
        state="running",
        observed_at=NOW,
        instance_id="vsi-1",
        lineage_sha256="a" * 64,
        previous_event_sha256=None,
        payload_sha256="b" * 64,
    )
    decision = _authorize(events=(event,))
    assert not decision.allowed
    assert "prior run has no terminal deletion event" in decision.reasons


def test_authorize_run_rejects_non_usd_and_wrong_run_spec_binding():
    with pytest.raises(ValueError, match="USD"):
        replace(valid_pricing(), currency="BRL")
    decision = _authorize(pricing=replace(valid_pricing(), run_spec_sha256="f" * 64))
    assert not decision.allowed
    assert "pricing is bound to a different run spec" in decision.reasons


def test_authorize_run_rejects_duplicate_service_lines():
    pricing = valid_pricing()
    duplicate = replace(pricing.lines[0], plan="standard", sku_id="duplicate-sku")
    pricing = replace(pricing, lines=pricing.lines + (duplicate,))
    decision = _authorize(pricing=pricing)
    assert not decision.allowed
    assert "duplicate metered service: vpc-vsi" in decision.reasons


def test_authorize_run_rejects_unknown_plan_even_when_credit_lists_it():
    pricing = valid_pricing()
    pricing = replace(pricing, lines=(replace(pricing.lines[0], plan="mystery"),) + pricing.lines[1:])
    current = valid_credit().eligible_service_plans[0]
    credit = replace(
        valid_credit(),
        eligible_service_plans=((current[0], "mystery", current[2], current[3]),)
        + valid_credit().eligible_service_plans[1:],
    )
    decision = _authorize(pricing=pricing, credit=credit)
    assert not decision.allowed
    assert "pricing includes an unknown service plan" in decision.reasons


def test_authorize_run_rejects_credit_older_than_usage():
    credit = replace(valid_credit(), observed_at=NOW - timedelta(seconds=1))
    decision = _authorize(credit=credit)
    assert not decision.allowed
    assert "credit evidence predates usage evidence" in decision.reasons


def test_authorize_run_returns_every_financial_refusal_reason():
    pending = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    pricing = valid_pricing()
    pricing = replace(
        pricing,
        observed_at=NOW - timedelta(days=2),
        run_spec_sha256="f" * 64,
        lines=pricing.lines[:-1],
    )
    decision = _authorize(
        credit=replace(
            valid_credit(),
            observed_at=NOW - timedelta(days=2),
            remaining_usd=Decimal("499.99"),
            eligible_service_plans=valid_credit().eligible_service_plans[1:],
        ),
        usage=replace(valid_usage(), unreconciled_run_ids=(pending,)),
        pricing=pricing,
    )
    assert not decision.allowed
    assert {
        "credit evidence is stale",
        "pricing evidence is stale",
        "pricing includes an ineligible provider SKU",
        "prior run usage is unreconciled",
        "missing required metered service: logs-monitoring",
        "pricing is bound to a different run spec",
        "credit does not cover the USD 500.00 safety reserve",
        "estimate exceeds credit after USD 500.00 safety reserve",
    }.issubset(decision.reasons)


def test_authorize_run_rejects_future_evidence_without_raising():
    decision = _authorize(
        credit=replace(valid_credit(), observed_at=NOW + timedelta(seconds=1)),
        pricing=replace(valid_pricing(), observed_at=NOW + timedelta(seconds=1)),
    )
    assert not decision.allowed
    assert "credit evidence timestamp is in the future" in decision.reasons
    assert "pricing evidence timestamp is in the future" in decision.reasons


def test_authorize_run_requires_timezone_aware_now():
    with pytest.raises(ValueError, match="timezone-aware"):
        _authorize(now=datetime(2026, 11, 5))


def test_authorize_run_binds_credit_usage_account_and_current_billing_month():
    mismatch = _authorize(usage=replace(valid_usage(), account_id="account-2"))
    assert not mismatch.allowed
    assert "financial evidence accounts differ from run account" in mismatch.reasons

    old_period = _authorize(
        usage=replace(valid_usage(), billing_period_start=NOW.replace(month=10, day=1))
    )
    assert not old_period.allowed
    assert "usage billing period is not the current month" in old_period.reasons


def test_authorize_run_binds_every_financial_artifact_to_run_account():
    for name, override in (
        ("credit", replace(valid_credit(), account_id="account-2")),
        ("usage", replace(valid_usage(), account_id="account-2")),
        ("pricing", replace(valid_pricing(), account_id="account-2")),
        ("post_promo", replace(valid_post_promo_plan(), account_id="account-2")),
    ):
        decision = _authorize(**{name: override})
        assert not decision.allowed
        assert "financial evidence accounts differ from run account" in decision.reasons


def test_authorize_run_compares_billing_period_in_utc():
    instant_utc = datetime(2026, 12, 1, 0, 30, tzinfo=timezone.utc)
    instant_local = instant_utc.astimezone(timezone(timedelta(hours=-3)))
    spec = valid_run_spec()
    decision = authorize_run(
        valid_credit(instant_utc),
        valid_usage(instant_utc),
        valid_pricing(instant_utc, spec),
        valid_post_promo_plan(instant_utc),
        spec,
        (),
        now=instant_local,
    )
    assert decision.allowed


def test_authorize_run_revalidates_post_promo_evidence_and_expiry_binding():
    stale = _authorize(
        post_promo=replace(valid_post_promo_plan(), observed_at=NOW - timedelta(days=2))
    )
    assert not stale.allowed
    assert "post-promo evidence is stale" in stale.reasons

    wrong_expiry = _authorize(
        post_promo=replace(
            valid_post_promo_plan(), promo_expires_at=NOW + timedelta(days=91)
        )
    )
    assert not wrong_expiry.allowed
    assert "post-promo plan is bound to a different promo expiry" in wrong_expiry.reasons


def test_authorize_run_rejects_zero_quantities_wrong_units_and_short_runtime_lines():
    pricing = valid_pricing()
    zero = replace(pricing, lines=tuple(replace(line, estimated_quantity=Decimal("0")) for line in pricing.lines))
    decision = _authorize(pricing=zero)
    assert not decision.allowed
    assert any("estimated quantity" in reason for reason in decision.reasons)

    wrong_unit = replace(
        pricing,
        lines=(replace(pricing.lines[0], unit="request"),) + pricing.lines[1:],
    )
    assert "pricing unit is invalid for vpc-vsi" in _authorize(pricing=wrong_unit).reasons

    short_vsi = replace(
        pricing,
        lines=(replace(pricing.lines[0], estimated_quantity=Decimal("5.99")),) + pricing.lines[1:],
    )
    assert "vpc-vsi estimate does not cover max_hours" in _authorize(pricing=short_vsi).reasons


def test_authorize_run_rejects_every_underestimated_metered_dimension():
    pricing = valid_pricing()
    minimums = {
        "vpc-boot-volume": "vpc-boot-volume",
        "cos-storage": "cos-storage",
        "cos-operations": "cos-operations",
        "container-registry-storage": "container-registry-storage",
        "code-engine": "code-engine",
        "logs-monitoring": "logs-monitoring",
    }
    for service, message in minimums.items():
        lines = tuple(
            replace(line, estimated_quantity=Decimal("0.000000001"))
            if line.service == service
            else line
            for line in pricing.lines
        )
        decision = _authorize(pricing=replace(pricing, lines=lines))
        assert not decision.allowed
        assert any(message in reason for reason in decision.reasons)


def test_authorize_run_blocks_any_prior_event_fork_even_if_one_branch_deleted():
    prior_run = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    deleted = RunEvent(
        event_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        run_id=prior_run,
        sequence=0,
        state="deleted",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256="a" * 64,
        previous_event_sha256=None,
        payload_sha256="b" * 64,
    )
    running = replace(
        deleted,
        event_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        state="running",
        instance_id="vsi-1",
        payload_sha256="c" * 64,
    )

    decision = _authorize(events=(deleted, running))

    assert not decision.allowed
    assert "prior run event chain is forked" in decision.reasons


def _reconciled_prior_chain() -> tuple[tuple[RunEvent, ...], UsageReconciliationReceipt]:
    prior_run = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    planned = RunEvent(
        event_id=UUID("bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"),
        run_id=prior_run,
        sequence=0,
        state="planned",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256="a" * 64,
        previous_event_sha256=None,
        payload_sha256="b" * 64,
    )
    deleting = replace(
        planned,
        event_id=UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc"),
        sequence=1,
        state="deleting",
        previous_event_sha256=run_event_sha256(planned),
        payload_sha256="c" * 64,
    )
    deleted = replace(
        deleting,
        event_id=UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd"),
        sequence=2,
        state="deleted",
        previous_event_sha256=run_event_sha256(deleting),
        payload_sha256="d" * 64,
    )
    receipt = UsageReconciliationReceipt(
        run_id=prior_run,
        usage_evidence_sha256=valid_usage().evidence_sha256,
        lineage_sha256=planned.lineage_sha256,
        receipt_sha256="e" * 64,
    )
    return (planned, deleting, deleted), receipt


def test_authorize_run_rejects_reused_run_id():
    event = replace(_reconciled_prior_chain()[0][0], run_id=valid_run_spec().run_id)
    decision = _authorize(events=(event,))
    assert not decision.allowed
    assert "run_id already has persisted events" in decision.reasons


@pytest.mark.parametrize("mutation", ["gap", "bad-link", "foreign-lineage", "illegal-state"])
def test_authorize_run_rejects_invalid_prior_chain(mutation: str):
    chain, receipt = _reconciled_prior_chain()
    events = list(chain)
    if mutation == "gap":
        events.pop(1)
    elif mutation == "bad-link":
        events[1] = replace(events[1], previous_event_sha256="f" * 64)
    elif mutation == "foreign-lineage":
        events[1] = replace(events[1], lineage_sha256="f" * 64)
    else:
        events[1] = replace(events[1], state="running")
    usage = replace(valid_usage(), reconciled_run_receipts=(receipt,))
    decision = _authorize(events=tuple(events), usage=usage)
    assert not decision.allowed
    assert "prior run event chain is invalid" in decision.reasons


def test_authorize_run_requires_usage_receipt_bound_to_raw_report():
    chain, receipt = _reconciled_prior_chain()
    assert "prior run usage reconciliation receipt is missing" in _authorize(events=chain).reasons
    wrong = replace(receipt, usage_evidence_sha256="f" * 64)
    decision = _authorize(
        events=chain,
        usage=replace(valid_usage(), reconciled_run_receipts=(wrong,)),
    )
    assert "prior run usage reconciliation receipt is missing" in decision.reasons
    allowed = _authorize(
        events=chain,
        usage=replace(valid_usage(), reconciled_run_receipts=(receipt,)),
    )
    assert allowed.allowed


def test_authorized_run_rejects_forged_allowed_decision():
    with pytest.raises(TypeError, match="authorize_run"):
        AuthorizationDecision(
            allowed=True,
            reasons=(),
            estimated_usd=Decimal("0"),
            usable_credit_usd=Decimal("0"),
            evidence_hashes=("1" * 64,),
            run_spec_sha256=run_spec_sha256(valid_run_spec()),
            authorization_sha256="2" * 64,
        )


def test_authorize_run_requires_provider_sku_and_region_eligibility():
    pricing = valid_pricing()
    wrong_sku = replace(
        pricing,
        lines=(replace(pricing.lines[0], sku_id="wrong-sku"),) + pricing.lines[1:],
    )
    assert "pricing includes an ineligible provider SKU" in _authorize(pricing=wrong_sku).reasons
    wrong_region = replace(
        pricing,
        lines=(replace(pricing.lines[0], region="eu-de"),) + pricing.lines[1:],
    )
    assert "pricing SKU region differs from run region" in _authorize(pricing=wrong_region).reasons
