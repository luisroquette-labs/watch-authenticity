"""Fail-closed tests for IBM promotional-credit authorization."""

from dataclasses import replace
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from infra.ibm_cost_guard import REQUIRED_METERED_LINES, authorize_run
from infra.ibm_types import RunEvent
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
    credit = replace(valid_credit(), eligible_service_plans=("standard",))
    decision = _authorize(credit=credit)
    assert not decision.allowed
    assert "pricing includes an ineligible service plan" in decision.reasons


def test_authorize_run_requires_48h_expiry_margin():
    exact = replace(valid_credit(), expires_at=NOW + timedelta(hours=54))
    plan = valid_post_promo_plan()
    plan = replace(
        plan,
        entries=tuple(
            replace(entry, delete_or_downgrade_deadline=NOW + timedelta(hours=1))
            for entry in plan.entries
        ),
    )
    assert _authorize(credit=exact, post_promo=plan).allowed
    too_late = replace(exact, expires_at=exact.expires_at - timedelta(seconds=1))
    decision = _authorize(credit=too_late, post_promo=plan)
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
    duplicate = replace(pricing.lines[0], plan="standard")
    pricing = replace(pricing, lines=pricing.lines + (duplicate,))
    decision = _authorize(pricing=pricing)
    assert not decision.allowed
    assert "duplicate metered service: vpc-vsi" in decision.reasons


def test_authorize_run_rejects_unknown_plan_even_when_credit_lists_it():
    pricing = valid_pricing()
    pricing = replace(pricing, lines=(replace(pricing.lines[0], plan="mystery"),) + pricing.lines[1:])
    credit = replace(valid_credit(), eligible_service_plans=("mystery", "standard"))
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
            eligible_service_plans=("standard",),
        ),
        usage=replace(valid_usage(), unreconciled_run_ids=(pending,)),
        pricing=pricing,
    )
    assert not decision.allowed
    assert {
        "credit evidence is stale",
        "pricing evidence is stale",
        "pricing includes an ineligible service plan",
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
