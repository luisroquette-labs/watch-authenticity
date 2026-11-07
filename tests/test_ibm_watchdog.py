"""Fail-closed tests for the independent IBM cleanup watchdog."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import ResourceLease, RunEvent, run_event_sha256
from infra.ibm_watchdog import reconcile_expired_resources
from tests.conftest import NOW, valid_run_spec, vpc_instance


def _events(*, instance_id: str | None = None) -> tuple[RunEvent, ...]:
    spec = valid_run_spec()
    states = ("planned", "authorized", "creating", "running")
    events: list[RunEvent] = []
    for sequence, state in enumerate(states):
        event = RunEvent(
            event_id=UUID(f"00000000-0000-4000-8000-{sequence + 1:012d}"),
            run_id=spec.run_id,
            sequence=sequence,
            state=state,
            observed_at=NOW - timedelta(hours=2) + timedelta(minutes=sequence),
            instance_id=instance_id if sequence >= 2 else None,
            lineage_sha256=run_spec_sha256(spec),
            previous_event_sha256=run_event_sha256(events[-1]) if events else None,
            payload_sha256=f"{sequence + 1:x}" * 64,
        )
        events.append(event)
    return tuple(events)


def _lease(*, active: bool = False) -> ResourceLease:
    spec = valid_run_spec()
    instance_id = "r006-77777777-7777-4777-8777-777777777777"
    return ResourceLease(
        resource_crn=(
            "crn:v1:bluemix:public:is:us-south-1:a/account-1::instance:" f"{instance_id}"
        ),
        service="vpc-vsi",
        run_id=spec.run_id,
        created_at=NOW - timedelta(hours=1),
        expires_at=NOW + timedelta(minutes=15) if active else NOW - timedelta(minutes=1),
        terminal_policy="delete",
    )


def _clients(monkeypatch: pytest.MonkeyPatch, initial: tuple[RunEvent, ...]):
    spec = valid_run_spec()
    stored = list(initial)
    cos = SimpleNamespace(
        _watch_auth_spec=spec,
        _watch_auth_bucket=spec.bucket,
        _watch_auth_prefix=spec.prefix,
        _watch_auth_configuration_verified=True,
    )
    tagger = Mock()
    tagger.list_tags.return_value = {
        "items": [
            {"name": "project:watch-authenticity"},
            {"name": f"run:{spec.run_id}"},
        ]
    }
    vpc = Mock()
    vpc._watch_auth_tagger = tagger

    monkeypatch.setattr(
        "infra.ibm_watchdog.load_event_inventory",
        lambda *_args, **_kwargs: SimpleNamespace(events=tuple(stored)),
    )

    def append(_cos, _bucket, _prefix, event):
        stored.append(event)

    monkeypatch.setattr("infra.ibm_watchdog.append_event", append)
    return vpc, cos, stored


def test_expired_exactly_owned_vsi_is_deleted_and_observed(monkeypatch):
    instance_id = "r006-77777777-7777-4777-8777-777777777777"
    initial = _events(instance_id=instance_id)
    vpc, cos, _stored = _clients(monkeypatch, initial)
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = reconcile_expired_resources(vpc, cos, (_lease(),), initial, now=NOW)

    assert tuple(event.state for event in result[-2:]) == ("deleting", "deleted")
    deleted.assert_called_once()
    assert deleted.call_args.args[:3] == (vpc, valid_run_spec(), instance_id)
    assert not any(call for call in vpc.method_calls if call[0].startswith("create"))


def test_active_valid_vsi_is_not_deleted(monkeypatch):
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, _stored = _clients(monkeypatch, initial)
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    assert (
        reconcile_expired_resources(vpc, cos, (_lease(active=True),), initial, now=NOW) == initial
    )
    deleted.assert_not_called()


def test_untagged_vsi_is_quarantined_without_delete(monkeypatch):
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, _stored = _clients(monkeypatch, initial)
    vpc._watch_auth_tagger.list_tags.return_value = {"items": []}
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = reconcile_expired_resources(vpc, cos, (_lease(),), initial, now=NOW)

    assert result[-1].state == "quarantined"
    deleted.assert_not_called()


def test_malformed_ttl_is_quarantined_without_delete(monkeypatch):
    spec = valid_run_spec()
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, _stored = _clients(monkeypatch, initial)
    malformed = SimpleNamespace(run_id=spec.run_id, expires_at="not-a-time")
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = reconcile_expired_resources(vpc, cos, (malformed,), initial, now=NOW)

    assert result[-1].state == "quarantined"
    deleted.assert_not_called()


def test_lost_create_with_duplicate_named_matches_is_quarantined(monkeypatch):
    initial = _events()
    vpc, cos, _stored = _clients(monkeypatch, initial)
    vpc.list_instances.return_value = {
        "instances": [vpc_instance(), vpc_instance(id="r006-duplicate")],
        "total_count": 2,
    }
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = reconcile_expired_resources(vpc, cos, (_lease(),), initial, now=NOW)

    assert result[-1].state == "quarantined"
    deleted.assert_not_called()


def test_delete_timeout_is_quarantined_and_propagated(monkeypatch):
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, stored = _clients(monkeypatch, initial)
    monkeypatch.setattr(
        "infra.ibm_watchdog.delete_and_wait",
        Mock(side_effect=TimeoutError("provider timeout")),
    )

    with pytest.raises(TimeoutError, match="provider timeout"):
        reconcile_expired_resources(vpc, cos, (_lease(),), initial, now=NOW)

    assert tuple(event.state for event in stored[-2:]) == ("deleting", "quarantined")


def test_deploy_script_is_digest_pinned_scoped_and_keyless():
    script = (
        valid_run_spec.__code__.co_filename.rsplit("/tests/", 1)[0]
        + "/scripts/deploy_ibm_watchdog.sh"
    )
    body = __import__("pathlib").Path(script).read_text(encoding="utf-8")
    assert "@sha256:" in body
    assert "ibmcloud ce job create" in body and "ibmcloud ce job update" in body
    assert "ibmcloud ce sub cron create" in body and "ibmcloud ce sub cron update" in body
    assert "*/15 * * * *" in body
    assert "service-account" in body
    assert "API_KEY" not in body and "INGESTION_KEY" not in body
    assert "project create" not in body and "resource group-create" not in body
