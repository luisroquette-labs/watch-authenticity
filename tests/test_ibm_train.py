"""Focused regressions for the authorized IBM training orchestrator."""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

import infra.ibm_train as subject
import training.worker as worker
from infra.ibm_bootstrap import bootstrap_sha256_for
from infra.ibm_cos import IntegrityError
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_train import (
    IBMClients,
    consume_authorization,
    launch_training,
    parse_run_spec,
    reconcile_pending_authorization,
)
from infra.ibm_types import (
    AuthorizationDecision,
    InstanceRef,
    RunEvent,
    WatchdogCanaryProof,
    run_spec_transport_document,
    run_spec_transport_sha256,
)
from tests.conftest import NOW, valid_run_spec


def _spec():
    candidate = valid_run_spec()
    return replace(candidate, bootstrap_sha256=bootstrap_sha256_for(candidate))


def _decision(spec=None, **overrides):
    spec = spec or _spec()
    values = {
        "allowed": True,
        "reasons": (),
        "estimated_usd": Decimal("14.33"),
        "usable_credit_usd": Decimal("17500"),
        "evidence_hashes": ("1" * 64, "2" * 64),
        "run_spec_sha256": run_spec_sha256(spec),
        "authorization_sha256": "3" * 64,
        "issued_at": NOW,
        "expires_at": NOW + timedelta(minutes=5),
        "nonce": uuid4(),
    }
    values.update(overrides)
    return AuthorizationDecision(**values)


def _proof(spec=None, **overrides):
    spec = spec or _spec()
    values = {
        "account_id": spec.account_id,
        "region": spec.region,
        "resource_group_id": spec.resource_group_id,
        "watchdog_job_crn": (
            f"crn:v1:bluemix:public:codeengine:{spec.region}:"
            f"a/{spec.account_id}:watch-auth::job:watchdog"
        ),
        "canary_run_id": UUID("99999999-9999-4999-8999-999999999999"),
        "canary_instance_sha256": "7" * 64,
        "canary_deleted_at": NOW - timedelta(minutes=1),
        "observed_at": NOW,
        "expires_at": NOW + timedelta(hours=24),
        "evidence_sha256": "8" * 64,
        "provider_verified": True,
    }
    values.update(overrides)
    return WatchdogCanaryProof(**values)


def _authorized(tmp_path: Path):
    spec = _spec()
    decision = _decision(spec)
    return consume_authorization(
        spec,
        decision,
        decision.evidence_hashes,
        watchdog_proof=_proof(spec),
        now=NOW,
        store=tmp_path / "auth",
    )


def test_concurrent_same_run_consumes_only_one_account_launch(tmp_path):
    spec = _spec()
    first = _decision(spec)
    second = _decision(spec)
    store = tmp_path / "auth"

    def consume(decision):
        try:
            consume_authorization(
                spec,
                decision,
                decision.evidence_hashes,
                watchdog_proof=_proof(spec),
                now=NOW,
                store=store,
            )
            return "allowed"
        except PermissionError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(consume, (first, second)))

    assert sorted(outcomes) == ["allowed", "blocked"]
    assert len(tuple(store.glob("account-*.pending.json"))) == 1


def test_crash_after_consume_blocks_replay_and_different_run(tmp_path):
    spec = _spec()
    first = _decision(spec)
    store = tmp_path / "auth"
    consume_authorization(
        spec,
        first,
        first.evidence_hashes,
        watchdog_proof=_proof(spec),
        now=NOW,
        store=store,
    )

    with pytest.raises(PermissionError, match="pending provider usage"):
        consume_authorization(
            spec,
            first,
            first.evidence_hashes,
            watchdog_proof=_proof(spec),
            now=NOW,
            store=store,
        )

    different_candidate = replace(
        spec,
        run_id=UUID("22222222-2222-4222-8222-222222222222"),
        prefix="runs/22222222-2222-4222-8222-222222222222",
        bootstrap_sha256="0" * 64,
    )
    different = replace(
        different_candidate,
        bootstrap_sha256=bootstrap_sha256_for(different_candidate),
    )
    second = _decision(different)
    with pytest.raises(PermissionError, match="pending provider usage"):
        consume_authorization(
            different,
            second,
            second.evidence_hashes,
            watchdog_proof=_proof(different),
            now=NOW,
            store=store,
        )

    assert len(tuple(store.glob("account-*.pending.json"))) == 1


def test_terminal_cos_usage_receipt_releases_account_fence(tmp_path):
    spec = _spec()
    decision = _decision(spec)
    store = tmp_path / "auth"
    consume_authorization(
        spec,
        decision,
        decision.evidence_hashes,
        watchdog_proof=_proof(spec),
        now=NOW,
        store=store,
    )
    receipt = subject._canonical(
        {
            "account_id": spec.account_id,
            "evidence_sha256": "9" * 64,
            "observed_at": NOW.isoformat(),
            "observed_usage_usd": "2.50",
            "provider_verified": True,
            "resource_count": 1,
            "run_id": str(spec.run_id),
            "reconciled_run_ids": [str(spec.run_id)],
            "unreconciled_run_ids": [],
        }
    )
    key = f"{spec.prefix}/evidence/usage-reconciliation-{'9' * 64}.json"
    cos = SimpleNamespace(
        list_objects_v2=lambda **_kwargs: {
            "Contents": [{"Key": key}],
            "IsTruncated": False,
        },
        get_object=lambda **_kwargs: {"Body": io.BytesIO(receipt)},
    )

    reconcile_pending_authorization(cos, spec, store=store)

    assert not tuple(store.glob("account-*.pending.json"))
    marker = next(store.glob("account-*.reconciled-*.json"))
    assert json.loads(marker.read_bytes())["provider_verified"] is True


def test_authorization_estimate_is_published_without_local_paths(monkeypatch, tmp_path):
    run = _authorized(tmp_path)
    captured: list[tuple[str, bytes]] = []
    monkeypatch.setattr(
        subject,
        "_put_immutable",
        lambda _client, _spec, key, body: captured.append((key, body)),
    )

    subject._publish_authorization_evidence(object(), run)

    key, body = captured[0]
    assert key == f"{run.spec.prefix}/evidence/authorization-consumed.json"
    assert json.loads(body)["estimated_usd"] == "14.33"
    assert b"pending_record_path" not in body


def test_default_account_fence_is_shared_across_working_directories(monkeypatch, tmp_path):
    spec = _spec()
    first = _decision(spec)
    second = _decision(spec)
    shared = tmp_path / "shared-auth"
    first_cwd = tmp_path / "one"
    second_cwd = tmp_path / "two"
    first_cwd.mkdir()
    second_cwd.mkdir()
    monkeypatch.setattr(subject, "_AUTHORIZATION_DIR", shared)

    monkeypatch.chdir(first_cwd)
    consume_authorization(
        spec,
        first,
        first.evidence_hashes,
        watchdog_proof=_proof(spec),
        now=NOW,
    )
    monkeypatch.chdir(second_cwd)
    with pytest.raises(PermissionError, match="pending provider usage"):
        consume_authorization(
            spec,
            second,
            second.evidence_hashes,
            watchdog_proof=_proof(spec),
            now=NOW,
        )


def test_provider_deadline_repeats_when_sdk_swallows_first_timeout():
    with pytest.raises(TimeoutError, match="deadline expired"):
        with subject._hard_deadline(time.monotonic() + 0.02):
            try:
                time.sleep(0.04)
            except OSError:
                pass
            time.sleep(0.1)


def test_denied_or_expired_decision_never_consumes_nonce(tmp_path):
    spec = _spec()
    denied = _decision(spec, allowed=False, reasons=("credit unavailable",))
    with pytest.raises(PermissionError, match="denied"):
        consume_authorization(
            spec,
            denied,
            denied.evidence_hashes,
            watchdog_proof=_proof(spec),
            now=NOW,
            store=tmp_path / "auth",
        )
    assert not (tmp_path / "auth").exists()


def test_missing_watchdog_canary_proof_blocks_before_create(monkeypatch, tmp_path):
    spec = _spec()
    decision = _decision(spec)
    create_calls: list[str] = []
    monkeypatch.setattr(
        subject,
        "ensure_single_instance",
        lambda *args, **kwargs: create_calls.append("create"),
    )

    with pytest.raises(PermissionError, match="watchdog deletion canary"):
        consume_authorization(
            spec,
            decision,
            decision.evidence_hashes,
            watchdog_proof=None,
            now=NOW,
            store=tmp_path / "auth",
        )

    assert create_calls == []
    assert not (tmp_path / "auth").exists()


def test_watchdog_proof_from_different_environment_blocks_before_create(tmp_path):
    spec = _spec()
    decision = _decision(spec)
    foreign = _proof(
        spec,
        region="eu-de",
        watchdog_job_crn=(
            f"crn:v1:bluemix:public:codeengine:eu-de:"
            f"a/{spec.account_id}:watch-auth::job:watchdog"
        ),
    )

    with pytest.raises(PermissionError, match="region differs"):
        consume_authorization(
            spec,
            decision,
            decision.evidence_hashes,
            watchdog_proof=foreign,
            now=NOW,
            store=tmp_path / "auth",
        )

    assert not (tmp_path / "auth").exists()


def test_consumed_authorization_is_revalidated_at_launch(monkeypatch, tmp_path):
    create_calls: list[str] = []
    monkeypatch.setattr(
        subject,
        "ensure_single_instance",
        lambda *args, **kwargs: create_calls.append("create"),
    )

    with pytest.raises(PermissionError, match="authorization is expired"):
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW + timedelta(days=40),
        )

    assert create_calls == []


def test_tampered_run_spec_transport_is_rejected():
    document = run_spec_transport_document(_spec())
    document["image_digest"] = document["image_digest"].replace("b", "c")
    parsed = parse_run_spec(document)

    assert run_spec_sha256(parsed) != run_spec_sha256(_spec())
    document["unexpected"] = True
    with pytest.raises(IntegrityError, match="schema"):
        parse_run_spec(document)


def test_worker_rejects_run_spec_that_differs_from_bootstrap():
    spec = _spec()
    document = run_spec_transport_document(spec)
    document["image_digest"] = document["image_digest"].replace("b", "c")
    body = subject._canonical(document)
    cos = SimpleNamespace(
        get_object=lambda **kwargs: {"Body": SimpleNamespace(read=lambda: body)}
    )
    env = {
        "RUN_ID": str(spec.run_id),
        "WATCH_AUTH_REGION": spec.region,
        "WATCH_AUTH_COS_ENDPOINT": spec.cos_endpoint,
        "WATCH_AUTH_COS_BUCKET": spec.bucket,
        "WATCH_AUTH_COS_PREFIX": spec.prefix,
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": spec.cos_service_instance_id,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": spec.trusted_profile_id,
        "WATCH_AUTH_CODE_SHA": spec.code_sha,
        "WATCH_AUTH_IMAGE_DIGEST": spec.image_digest,
        "WATCH_AUTH_MANIFEST_SHA256": spec.manifest_sha256,
        "WATCH_AUTH_SPLITS_SHA256": spec.splits_sha256,
        "WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256": run_spec_transport_sha256(spec),
    }

    with pytest.raises(IntegrityError, match="bootstrap identity"):
        worker._load_spec(cos, env)


def _instance() -> InstanceRef:
    spec = _spec()
    return InstanceRef(
        id="vsi-1",
        name=f"watch-auth-{spec.run_id}",
        profile=spec.profile,
        zone=spec.zone,
        vpc_id=spec.network_id,
        subnet_id=spec.subnet_id,
        image_id=spec.image_id,
        status="running",
    )


def _patch_lifecycle(monkeypatch, states, deleted, receipt_error=None):
    monkeypatch.setattr(subject, "verify_cos_configuration", lambda *_: None)
    monkeypatch.setattr(subject, "publish_run_spec", lambda *_: None)
    monkeypatch.setattr(subject, "_publish_authorization_evidence", lambda *_: None)

    def append(_cos, spec, state, *, now, instance_id, details=None, deadline):
        del details, deadline
        states.append(state)
        return RunEvent(
            event_id=uuid4(),
            run_id=spec.run_id,
            sequence=len(states) - 1,
            state=state,
            observed_at=now,
            instance_id=instance_id,
            lineage_sha256=run_spec_sha256(spec),
            previous_event_sha256=None if len(states) == 1 else "f" * 64,
            payload_sha256="e" * 64,
        )

    monkeypatch.setattr(subject, "_append_state", append)
    monkeypatch.setattr(subject, "ensure_single_instance", lambda *args, **kwargs: _instance())
    monkeypatch.setattr(subject, "wait_running", lambda *args, **kwargs: _instance())

    def receipt(*args, **kwargs):
        if receipt_error is not None:
            raise receipt_error
        return {"artifacts": []}

    monkeypatch.setattr(subject, "wait_for_training_receipt", receipt)

    def delete(*args, **kwargs):
        deleted.append(args[2])

    monkeypatch.setattr(subject, "delete_and_wait", delete)


def test_success_writes_terminal_result_then_deletes(monkeypatch, tmp_path):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted)

    event = launch_training(
        IBMClients(vpc=object(), tagger=object(), cos=object()),
        _authorized(tmp_path),
        confirm=True,
        now=NOW,
    )

    assert event.state == "deleted"
    assert states == [
        "planned",
        "authorized",
        "creating",
        "running",
        "artifact_sync",
        "succeeded",
        "deleting",
        "deleted",
    ]
    assert deleted == ["vsi-1"]


@pytest.mark.parametrize("failure", [RuntimeError("training failed"), KeyboardInterrupt()])
def test_training_failure_or_ctrl_c_still_deletes(monkeypatch, tmp_path, failure):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted, failure)

    with pytest.raises(type(failure)):
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW,
        )

    assert "failed" in states
    assert states[-2:] == ["deleting", "deleted"]
    assert deleted == ["vsi-1"]


def test_create_response_failure_recovers_accepted_id_and_deletes(monkeypatch, tmp_path):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted)
    monkeypatch.setattr(
        subject,
        "ensure_single_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("lost response")),
    )
    monkeypatch.setattr(subject, "recover_created_instance_id", lambda *args: "vsi-1")

    with pytest.raises(RuntimeError, match="lost response"):
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW,
        )

    assert states == ["planned", "authorized", "creating", "failed", "deleting", "deleted"]
    assert deleted == ["vsi-1"]


def test_cleanup_window_recovers_unpersisted_accepted_create(monkeypatch, tmp_path):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted)
    monkeypatch.setattr(
        subject,
        "ensure_single_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("create timeout")),
    )
    monkeypatch.setattr(
        subject,
        "recover_created_instance_id",
        lambda *args: (_ for _ in ()).throw(RuntimeError("no persisted ID")),
    )
    provider_recovery: list[str] = []

    def recover(*args, **kwargs):
        provider_recovery.append("vsi-1")
        return "vsi-1"

    monkeypatch.setattr(subject, "recover_accepted_instance", recover)

    with pytest.raises(TimeoutError, match="create timeout"):
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW,
        )

    assert provider_recovery == ["vsi-1"]
    assert states[-3:] == ["failed", "deleting", "deleted"]
    assert deleted == ["vsi-1"]


def test_partial_provider_recovery_rereads_durable_id_then_deletes(monkeypatch, tmp_path):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted)
    monkeypatch.setattr(
        subject,
        "ensure_single_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("create timeout")),
    )
    durable_reads = iter((RuntimeError("no persisted ID"), "vsi-1"))

    def read_durable(*args):
        result = next(durable_reads)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(subject, "recover_created_instance_id", read_durable)
    monkeypatch.setattr(
        subject,
        "recover_accepted_instance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("accepted create cleanup context is unavailable")
        ),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW,
        )

    assert {str(error) for error in raised.value.exceptions} == {
        "create timeout",
        "accepted create cleanup context is unavailable",
    }
    assert states[-3:] == ["failed", "deleting", "deleted"]
    assert deleted == ["vsi-1"]


def test_recovery_record_failure_cannot_skip_teardown(monkeypatch, tmp_path):
    states: list[str] = []
    deleted: list[str] = []
    _patch_lifecycle(monkeypatch, states, deleted)
    original_append = subject._append_state

    def fail_deleting(*args, **kwargs):
        if args[2] == "deleting":
            raise OSError("COS unavailable")
        return original_append(*args, **kwargs)

    monkeypatch.setattr(subject, "_append_state", fail_deleting)
    monkeypatch.setattr(
        subject,
        "_record_recovery",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=True,
            now=NOW,
        )

    assert {str(error) for error in raised.value.exceptions} == {
        "COS unavailable",
        "disk full",
    }
    assert deleted == ["vsi-1"]


def test_confirm_false_has_no_effect(monkeypatch, tmp_path):
    monkeypatch.setattr(
        subject,
        "verify_cos_configuration",
        lambda *_: pytest.fail("COS must not be touched"),
    )
    with pytest.raises(PermissionError, match="confirmation"):
        launch_training(
            IBMClients(vpc=object(), tagger=object(), cos=object()),
            _authorized(tmp_path),
            confirm=False,
            now=NOW,
        )


def test_tampered_receipt_artifact_is_rejected():
    spec = _spec()
    body = b"tampered"
    digest = hashlib.sha256(b"expected").hexdigest()
    names = ("checkpoint.pt", "class-map.json", "metrics.json", "run-result.json")
    artifacts = [
        {
            "key": f"{spec.prefix}/artifacts/{name}/{digest}",
            "length": len(body),
            "name": name,
            "sha256": digest,
        }
        for name in names
    ]
    client = SimpleNamespace(
        get_object=lambda **kwargs: {"Body": SimpleNamespace(read=lambda: body)}
    )

    with pytest.raises(IntegrityError, match="bytes differ"):
        subject._verify_receipt_artifacts(client, spec, artifacts)


def test_run_result_with_wrong_lineage_is_rejected():
    spec = _spec()
    bodies = {
        "checkpoint.pt": b"checkpoint",
        "class-map.json": b'{"genuine":0}',
        "metrics.json": b'{"epochs_completed":1}',
    }
    digests = {name: hashlib.sha256(body).hexdigest() for name, body in bodies.items()}
    result = subject._canonical(
        {
            "artifacts": digests,
            "best_epoch": 1,
            "best_validation_nll": 0.5,
            "epochs_completed": 1,
            "lineage": {
                "code_sha": "0" * 40,
                "image_digest": spec.image_digest,
                "manifest_sha256": spec.manifest_sha256,
                "splits_sha256": spec.splits_sha256,
                "class_map_sha256": digests["class-map.json"],
            },
        }
    )
    bodies["run-result.json"] = result
    digests["run-result.json"] = hashlib.sha256(result).hexdigest()
    artifacts = [
        {
            "key": f"{spec.prefix}/artifacts/{name}/{digests[name]}",
            "length": len(body),
            "name": name,
            "sha256": digests[name],
        }
        for name, body in bodies.items()
    ]
    by_key = {item["key"]: bodies[item["name"]] for item in artifacts}
    client = SimpleNamespace(
        get_object=lambda **kwargs: {
            "Body": SimpleNamespace(read=lambda: by_key[kwargs["Key"]])
        }
    )

    with pytest.raises(IntegrityError, match="lineage differs"):
        subject._verify_receipt_artifacts(client, spec, artifacts)


def test_worker_persists_failure_receipt_when_training_fails(monkeypatch):
    spec = _spec()
    receipts: list[bytes] = []
    monkeypatch.setattr(worker, "_environment", lambda: {})
    monkeypatch.setattr(worker, "_load_spec", lambda *args: spec)
    monkeypatch.setattr(worker, "verify_cos_configuration", lambda *args: None)
    monkeypatch.setattr(worker, "_require_gpu", lambda: None)
    monkeypatch.setattr(worker, "load_training_data", lambda *args: object())
    monkeypatch.setattr(
        worker,
        "run_training",
        lambda *args: (_ for _ in ()).throw(RuntimeError("training failed")),
    )
    monkeypatch.setattr(
        worker,
        "_put_immutable",
        lambda _cos, _spec, _key, body: receipts.append(body),
    )

    with pytest.raises(RuntimeError, match="training failed"):
        worker.execute_worker(cos=object())

    assert len(receipts) == 1
    assert b'"status":"failed"' in receipts[0]
    assert b'"error_type":"RuntimeError"' in receipts[0]
    assert b"training failed" not in receipts[0]
