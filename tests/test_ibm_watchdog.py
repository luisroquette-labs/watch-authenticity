"""Fail-closed tests for the independent IBM cleanup watchdog."""

from __future__ import annotations

import io
import json
import logging
import sys
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest

from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import ResourceLease, RunEvent, run_event_sha256
from infra.ibm_watchdog import reconcile_expired_resources
from tests.conftest import NOW, valid_run_spec, valid_usage, vpc_instance


class _MissingObject(Exception):
    response = {"ResponseMetadata": {"HTTPStatusCode": 404}}


class _Cos:
    def __init__(self, spec):
        self._watch_auth_spec = spec
        self._watch_auth_bucket = spec.bucket
        self._watch_auth_prefix = spec.prefix
        self._watch_auth_configuration_verified = True
        self.objects: dict[str, bytes] = {}

    def put_object(self, **kwargs):
        key = kwargs["Key"]
        if key in self.objects:
            error = RuntimeError("exists")
            error.response = {"ResponseMetadata": {"HTTPStatusCode": 412}}
            raise error
        self.objects[key] = kwargs["Body"]

    def get_object(self, **kwargs):
        try:
            body = self.objects[kwargs["Key"]]
        except KeyError as exc:
            raise _MissingObject from exc
        return {"Body": io.BytesIO(body)}

    def list_objects_v2(self, *, Bucket, Prefix):
        return {"Contents": [{"Key": key} for key in self.objects if key.startswith(Prefix)]}


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
    cos = _Cos(spec)
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


def _context() -> dict[str, object]:
    spec = valid_run_spec()
    return {
        "run_id": str(spec.run_id),
        "account_id": spec.account_id,
        "instance_id": "r006-77777777-7777-4777-8777-777777777777",
        "volume_id": "r006-volume-1",
        "subnet_id": spec.subnet_id,
        "public_gateway_id": spec.public_gateway_id,
        "public_gateway_floating_ip_id": spec.public_gateway_floating_ip_id,
    }


def _reconcile(vpc, cos, leases, events, *, now=NOW, usage_reconciler=None):
    terminal_usage = replace(
        valid_usage(),
        reconciled_run_ids=(valid_run_spec().run_id,),
    )
    return reconcile_expired_resources(
        vpc,
        cos,
        leases,
        events,
        now=now,
        estimated_usd=Decimal("14.33"),
        usage=valid_usage(),
        usage_reconciler=usage_reconciler or (lambda: terminal_usage),
    )


def test_expired_exactly_owned_vsi_is_deleted_and_observed(monkeypatch):
    instance_id = "r006-77777777-7777-4777-8777-777777777777"
    initial = _events(instance_id=instance_id)
    vpc, cos, _stored = _clients(monkeypatch, initial)
    deleted = Mock(side_effect=lambda *_a, **kw: kw["cleanup_context_sink"](_context()))
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = _reconcile(vpc, cos, (_lease(),), initial)

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
        _reconcile(vpc, cos, (_lease(active=True),), initial) == initial
    )
    deleted.assert_not_called()


def test_untagged_vsi_is_quarantined_without_delete(monkeypatch):
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, _stored = _clients(monkeypatch, initial)
    vpc._watch_auth_tagger.list_tags.return_value = {"items": []}
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = _reconcile(vpc, cos, (_lease(),), initial)

    assert result[-1].state == "quarantined"
    deleted.assert_not_called()


def test_malformed_ttl_is_quarantined_without_delete(monkeypatch):
    spec = valid_run_spec()
    initial = _events(instance_id="r006-77777777-7777-4777-8777-777777777777")
    vpc, cos, _stored = _clients(monkeypatch, initial)
    malformed = SimpleNamespace(run_id=spec.run_id, expires_at="not-a-time")
    deleted = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", deleted)

    result = _reconcile(vpc, cos, (malformed,), initial)

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

    result = _reconcile(vpc, cos, (_lease(),), initial)

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
        _reconcile(vpc, cos, (_lease(),), initial)

    assert tuple(event.state for event in stored[-2:]) == ("deleting", "quarantined")


def test_fresh_job_restores_cleanup_and_persists_five_404_receipts(
    monkeypatch, tmp_path
):
    import infra.ibm_vpc as ibm_vpc

    instance_id = "r006-77777777-7777-4777-8777-777777777777"
    initial = _events(instance_id=instance_id)
    vpc, cos, stored = _clients(monkeypatch, initial)
    first_recovery = tmp_path / "first"
    second_recovery = tmp_path / "second"
    monkeypatch.setattr(ibm_vpc, "_RECOVERY_DIR", first_recovery)

    def first_delete(*_args, **kwargs):
        kwargs["cleanup_context_sink"](_context())

    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", first_delete)
    with pytest.raises(RuntimeError, match="usage unavailable"):
        _reconcile(
            vpc,
            cos,
            (_lease(),),
            initial,
            usage_reconciler=lambda: (_ for _ in ()).throw(
                RuntimeError("usage unavailable")
            ),
        )

    monkeypatch.setattr(ibm_vpc, "_RECOVERY_DIR", second_recovery)
    vpc._watch_auth_tagger.list_tags.side_effect = _MissingObject()

    def second_delete(_vpc, spec, observed_id, **kwargs):
        assert observed_id == instance_id
        assert ibm_vpc._load_cleanup(spec, observed_id)[0] == "r006-volume-1"
        kwargs["cleanup_context_sink"](_context())

    monkeypatch.setattr("infra.ibm_watchdog.delete_and_wait", second_delete)
    result = _reconcile(vpc, cos, (_lease(),), tuple(stored))

    assert result[-1].state == "deleted"
    receipt_keys = [key for key in cos.objects if "/receipts/deletion/" in key]
    assert len(receipt_keys) == 5
    payload = next(
        json.loads(body)
        for key, body in cos.objects.items()
        if "/event-payloads/" in key
    )
    assert set(payload["deletion_receipt_sha256"]) == {
        "vpc-vsi",
        "boot-volume",
        "subnet",
        "public-gateway",
        "floating-ip",
    }
    assert result[-1].payload_sha256 in next(
        key for key in cos.objects if "/event-payloads/" in key
    )


def test_main_uses_trusted_profile_and_info_logging(monkeypatch):
    created: dict[str, object] = {}

    class Auth:
        def __init__(self, *, iam_profile_id):
            created["profile"] = iam_profile_id
            self.token_manager = object()

    class Vpc:
        def __init__(self, *, authenticator):
            self.authenticator = authenticator

        def set_service_url(self, url):
            created["vpc_url"] = url

    class Service:
        def __init__(self, *, authenticator):
            self.authenticator = authenticator

    monkeypatch.setitem(
        sys.modules,
        "ibm_cloud_sdk_core.authenticators",
        SimpleNamespace(ContainerAuthenticator=Auth),
    )
    monkeypatch.setitem(sys.modules, "ibm_vpc", SimpleNamespace(VpcV1=Vpc))
    monkeypatch.setitem(
        sys.modules,
        "ibm_platform_services",
        SimpleNamespace(GlobalTaggingV1=Service, UsageReportsV4=Service),
    )
    monkeypatch.setitem(
        sys.modules,
        "ibm_boto3",
        SimpleNamespace(client=lambda *_a, **_kw: object()),
    )
    for name, value in {
        "WATCH_AUTH_ACCOUNT_ID": "account-1",
        "WATCH_AUTH_RESOURCE_GROUP_ID": "rg-1",
        "WATCH_AUTH_REGION": "us-south",
        "WATCH_AUTH_COS_BUCKET": "bucket",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
    }.items():
        monkeypatch.setenv(name, value)
    basic_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    run = Mock()
    monkeypatch.setattr("infra.ibm_watchdog.run_watchdog_once", run)

    from infra.ibm_watchdog import main

    main()

    assert created == {
        "profile": "profile-1",
        "vpc_url": "https://us-south.iaas.cloud.ibm.com/v1",
    }
    basic_config.assert_called_once_with(level=logging.INFO)
    run.assert_called_once()


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
    assert "WATCH_AUTH_TRUSTED_PROFILE_ID" in body
    assert "credentialless platform routing proof" in body
    assert "API_KEY" not in body and "INGESTION_KEY" not in body
    assert "project create" not in body and "resource group-create" not in body
