"""Provider usage lag, durable terminal binding and controller fence regressions."""

import json
from dataclasses import replace
from datetime import timedelta
from unittest.mock import Mock
from uuid import uuid4

import pytest

from infra.ibm_bootstrap import bootstrap_sha256_for
from infra.ibm_cos import IntegrityError, append_event, load_event_inventory, load_immutable_json
from infra.ibm_evidence import fetch_usage_evidence, run_spec_sha256
from infra.ibm_train import (
    _publish_authorization_evidence,
    authorize_training,
    consume_authorization,
    publish_run_spec,
)
from infra.ibm_types import RunEvent, run_event_sha256
from infra.ibm_watchdog import _persist_cleanup_context, run_watchdog_once
from tests.conftest import NOW, valid_credit, valid_post_promo_plan, valid_pricing
from tests.test_ibm_cos import FakeCOS
from tests.test_ibm_train import _decision, _proof, _spec
from tests.test_ibm_watchdog import _context


def _resources(spec):
    context = _context()
    return [
        {
            "account_id": spec.account_id,
            "region": spec.region,
            "resource_instance_id": context[name],
            "resource_id": f"is-{name}",
            "plan_id": "standard",
            "pricing_plan_id": f"ibm-{name}-sku",
            "currency_code": "USD", "month": "2026-11", "pending": False,
            "deleted_at": NOW.isoformat(),
            "usage": [{"metric": "hours", "quantity": 2, "cost": 1.25}],
        }
        for name in ("instance_id", "volume_id", "public_gateway_id", "public_gateway_floating_ip_id")
    ]


def _prepared(tmp_path):
    spec = _spec()
    decision = _decision(spec)
    store = tmp_path / "auth"
    run = consume_authorization(
        spec, decision, decision.evidence_hashes,
        watchdog_proof=_proof(spec), now=NOW, store=store,
    )
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    cos._watch_auth_spec = spec
    publish_run_spec(cos, spec)
    _publish_authorization_evidence(cos, run)
    _persist_cleanup_context(cos, spec, _context())
    events = []
    for sequence, state in enumerate(("planned", "authorized", "creating", "running", "deleting", "deleted")):
        event = RunEvent(
            event_id=uuid4(), run_id=spec.run_id, sequence=sequence, state=state,
            observed_at=NOW - timedelta(minutes=5 - sequence),
            instance_id=_context()["instance_id"] if sequence >= 2 else None,
            lineage_sha256=run_spec_sha256(spec),
            previous_event_sha256=run_event_sha256(events[-1]) if events else None,
            payload_sha256="a" * 64,
        )
        append_event(cos, spec.bucket, spec.prefix, event)
        events.append(event)
    return spec, cos, store


def _watch(spec, cos, resources, now):
    usage = Mock()
    usage.get_resource_usage_account.return_value.get_result.return_value = {
        "resources": resources, "count": len(resources),
    }
    vpc = Mock()
    run_watchdog_once(
        vpc, Mock(), cos, usage, account_id=spec.account_id,
        resource_group_id=spec.resource_group_id, region=spec.region,
        bucket=spec.bucket, runs_prefix="runs", now=now,
    )
    assert not vpc.method_calls


def _next_run(spec, cos, store, now, resources=None):
    run_id = uuid4()
    candidate = replace(spec, run_id=run_id, prefix=f"runs/{run_id}")
    candidate = replace(candidate, bootstrap_sha256=bootstrap_sha256_for(candidate))
    resources = _resources(spec) if resources is None else resources
    usage_client = Mock()
    def response(_account, period, **_kwargs):
        selected = [resource for resource in resources if resource["month"] == period]
        result = Mock()
        result.get_result.return_value = {"resources": selected, "count": len(selected)}
        return result
    usage_client.get_resource_usage_account.side_effect = response
    return authorize_training(
        cos, usage_client, candidate, valid_credit(now), valid_pricing(now, candidate),
        valid_post_promo_plan(now),
        watchdog_proof=_proof(candidate, observed_at=now, canary_deleted_at=now - timedelta(minutes=1), expires_at=now + timedelta(hours=24)),
        now=now, store=store,
    )


def test_delayed_usage_stays_fenced_then_stable_provider_usage_releases_in_controller(tmp_path):
    spec, cos, store = _prepared(tmp_path)
    first = NOW + timedelta(minutes=30)
    _watch(spec, cos, [], first)
    with pytest.raises(IntegrityError, match="reconciliation is missing"):
        _next_run(spec, cos, store, first)
    assert len(tuple(store.glob("account-*.pending.json"))) == 1

    resources = _resources(spec)
    first = NOW + timedelta(hours=1)
    _watch(spec, cos, resources, first)
    _watch(spec, cos, resources, first + timedelta(minutes=15))
    with pytest.raises(IntegrityError, match="reconciliation is missing"):
        _next_run(spec, cos, store, first)

    later = first + timedelta(hours=24)
    _watch(spec, cos, resources, later)
    receipt = load_immutable_json(cos, spec, f"{spec.prefix}/evidence/usage-reconciliation.json")
    assert receipt["reconciled_run_ids"] == [str(spec.run_id)]
    assert len(receipt["observation_keys"]) == 2
    new_run = _next_run(spec, cos, store, later)
    assert new_run.spec.run_id != spec.run_id
    assert len(tuple(store.glob("account-*.reconciled-*.json"))) == 1
    assert len(tuple(store.glob("account-*.pending.json"))) == 1


@pytest.mark.parametrize("defect", ["missing-resource", "missing-sku", "not-deleted", "zero-usage", "changing-cost"])
def test_incomplete_or_changing_usage_never_reconciles(tmp_path, defect):
    spec, cos, store = _prepared(tmp_path)
    resources = _resources(spec)
    if defect == "missing-resource":
        resources.pop()
    elif defect == "missing-sku":
        resources[0].pop("pricing_plan_id")
    elif defect == "not-deleted":
        resources[0].pop("deleted_at")
    elif defect == "zero-usage":
        resources[0]["usage"][0]["quantity"] = 0
    first = NOW + timedelta(hours=1)
    _watch(spec, cos, resources, first)
    if defect == "changing-cost":
        resources[0]["usage"][0]["cost"] = 4
    _watch(spec, cos, resources, first + timedelta(hours=24))
    with pytest.raises(IntegrityError, match="reconciliation is missing"):
        _next_run(spec, cos, store, first)


def test_usage_adapter_cannot_accept_asserted_reconciled_ids():
    spec = _spec()
    client = Mock()
    with pytest.raises(TypeError, match="reconciled_run_ids"):
        fetch_usage_evidence(client, spec.account_id, "2026-11", (), now=NOW,
                             reconciled_run_ids=(spec.run_id,))
    client.get_resource_usage_account.assert_not_called()


def test_authorization_rechecks_provider_usage_after_stable_receipt(tmp_path):
    spec, cos, store = _prepared(tmp_path)
    first, later = NOW + timedelta(hours=1), NOW + timedelta(hours=25)
    _watch(spec, cos, _resources(spec), first)
    _watch(spec, cos, _resources(spec), later)
    resources = _resources(spec)
    resources[0]["usage"][0]["cost"] = 9
    with pytest.raises(IntegrityError, match="changed after reconciliation"):
        _next_run(spec, cos, store, later, resources)
    assert tuple(store.glob("account-*.pending.json"))


def test_billing_outage_cannot_prevent_expired_owned_resource_cleanup(monkeypatch, tmp_path):
    spec, cos, store = _prepared(tmp_path)
    for key, (body, _) in list(cos.objects.items()):
        if "/events/" in key and json.loads(body)["state"] in {"deleting", "deleted"}:
            del cos.objects[key]
    usage_client = Mock()
    usage_client.get_resource_usage_account.side_effect = RuntimeError("billing unavailable")
    tagger = Mock()
    tagger.list_tags.return_value = {"items": [
        {"name": "project:watch-authenticity"}, {"name": f"run:{spec.run_id}"},
    ]}
    deleted = Mock(side_effect=lambda *_a, **kw: kw["cleanup_context_sink"](_context()))
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)
    with pytest.raises(RuntimeError, match="watchdog failed"):
        run_watchdog_once(
            Mock(), tagger, cos, usage_client, account_id=spec.account_id,
            resource_group_id=spec.resource_group_id, region=spec.region,
            bucket=spec.bucket, runs_prefix="runs", now=NOW + timedelta(hours=7),
        )
    deleted.assert_called_once()
    assert load_event_inventory(cos, spec).events[-1].state == "deleted"
    assert tuple(store.glob("account-*.pending.json"))
    assert load_immutable_json(cos, spec, f"{spec.prefix}/evidence/usage-reconciliation.json") is None


def test_intermediate_usage_change_restarts_the_stability_window(tmp_path):
    spec, cos, store = _prepared(tmp_path)
    first = NOW + timedelta(hours=1)
    resources = _resources(spec)
    _watch(spec, cos, resources, first)
    resources[0]["usage"][0]["cost"] = 4
    _watch(spec, cos, resources, first + timedelta(hours=12))
    resources[0]["usage"][0]["cost"] = 1.25
    _watch(spec, cos, resources, first + timedelta(hours=24))
    with pytest.raises(IntegrityError, match="reconciliation is missing"):
        _next_run(spec, cos, store, first + timedelta(hours=24))


def test_previous_billing_month_receipt_can_release_next_month_authorization(tmp_path):
    spec, cos, store = _prepared(tmp_path)
    _watch(spec, cos, _resources(spec), NOW + timedelta(hours=1))
    _watch(spec, cos, _resources(spec), NOW + timedelta(hours=25))
    next_month = NOW.replace(month=12, day=1)
    assert _next_run(spec, cos, store, next_month).spec.run_id != spec.run_id


def test_documented_vpc_usage_without_classic_pending_field_reconciles(tmp_path):
    from ibm_platform_services.usage_reports_v4 import InstanceUsage

    spec, cos, store = _prepared(tmp_path)
    resources = _resources(spec)
    for resource in resources:
        del resource["pending"]
        resource.update(pricing_country="USA", billable=True)
        resource["usage"][0].update(rated_cost=1.25, discounts=[])
    # ibm-platform-services 0.77.1: InstanceUsage.pending is optional and classic-only.
    resources = [InstanceUsage.from_dict(resource).to_dict() for resource in resources]
    assert all("pending" not in resource for resource in resources)
    _watch(spec, cos, resources, NOW + timedelta(hours=1))
    later = NOW + timedelta(hours=25)
    _watch(spec, cos, resources, later)
    assert _next_run(spec, cos, store, later, resources).spec.run_id != spec.run_id


@pytest.mark.parametrize("pending", [True, 0, "false"])
def test_sdk_boundary_rejects_pending_or_invalid_flags(pending):
    spec = _spec()
    resources = _resources(spec)
    resources[0]["pending"] = pending
    client = Mock()
    client.get_resource_usage_account.return_value.get_result.return_value = {
        "resources": resources, "count": len(resources),
    }
    with pytest.raises(ValueError, match="pending"):
        fetch_usage_evidence(client, spec.account_id, "2026-11", (spec.run_id,), now=NOW)


@pytest.mark.parametrize("field", ["terminal_event_sha256", "consumption_sha256", "evidence_sha256"])
def test_controller_rejects_receipt_rebound_to_another_event_or_authorization(tmp_path, field):
    from infra.ibm_cos import immutable_json_key
    from infra.ibm_watchdog import _canonical

    spec, cos, store = _prepared(tmp_path)
    first, later = NOW + timedelta(hours=1), NOW + timedelta(hours=25)
    _watch(spec, cos, _resources(spec), first)
    _watch(spec, cos, _resources(spec), later)
    receipt = load_immutable_json(cos, spec, f"{spec.prefix}/evidence/usage-reconciliation.json")
    receipt[field] = "f" * 64
    key = next(key for key in cos.objects if "/usage-reconciliation-" in key)
    del cos.objects[key]
    body = _canonical(receipt)
    cos.objects[immutable_json_key(f"{spec.prefix}/evidence/usage-reconciliation.json", body)] = (body, {})
    with pytest.raises(IntegrityError):
        _next_run(spec, cos, store, later)
    assert tuple(store.glob("account-*.pending.json"))
