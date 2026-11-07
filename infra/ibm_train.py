"""Authorize and orchestrate one IBM GPU training run fail closed."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from infra.ibm_bootstrap import bootstrap_sha256_for, render_bootstrap
from infra.ibm_cos import (
    IntegrityError,
    append_event,
    load_event_inventory,
    load_immutable_json,
    persist_deletion_receipts,
    put_immutable_json,
    validate_event_chain,
    verify_cos_configuration,
)
from infra.ibm_cost_guard import authorize_run
from infra.ibm_evidence import (
    fetch_usage_evidence,
    run_spec_sha256,
    stable_usage_observations,
    usage_observation,
)
from infra.ibm_types import (
    AuthorizationDecision,
    AuthorizedRun,
    CreditEvidence,
    PostPromoPlan,
    PricingEvidence,
    RunEvent,
    RunSpec,
    UsageEvidence,
    WatchdogCanaryProof,
    run_event_sha256,
    run_spec_transport_document,
)
from infra.ibm_vpc import (
    build_instance_prototype,
    delete_and_wait,
    ensure_single_instance,
    recover_accepted_instance,
    recover_created_instance_id,
    wait_running,
)

_STATE_DIR = Path.home() / ".local/state/watch-authenticity/ibm-training"
_AUTHORIZATION_DIR = _STATE_DIR / "authorizations"
_RECOVERY_DIR = _STATE_DIR / "recovery"
_CREATE_TIMEOUT_SECONDS = 15 * 60
_DELETE_TIMEOUT_SECONDS = 30 * 60
_RECOVERY_TIMEOUT_SECONDS = 5 * 60
_COS_EVIDENCE_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class IBMClients:
    vpc: Any
    tagger: Any
    cos: Any


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@contextmanager
def _hard_deadline(deadline: float):
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("provider deadlines require the main thread")
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        raise RuntimeError("platform cannot enforce provider deadlines")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("provider deadline expired")
    if signal.getitimer(signal.ITIMER_REAL)[0] > 0:
        raise RuntimeError("an existing process alarm blocks provider calls")
    previous = signal.getsignal(signal.SIGALRM)

    def expire(_signum: int, _frame: object) -> None:
        raise TimeoutError("provider deadline expired")

    signal.signal(signal.SIGALRM, expire)
    interval = min(0.05, max(0.001, remaining))
    signal.setitimer(signal.ITIMER_REAL, remaining, interval)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def parse_run_spec(document: object) -> RunSpec:
    expected = {field.name for field in fields(RunSpec)} - {"bootstrap_sha256"}
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != 1
        or set(document) != expected | {"schema_version"}
    ):
        raise IntegrityError("RunSpec document schema is invalid")
    values = dict(document)
    del values["schema_version"]
    try:
        values["run_id"] = UUID(values["run_id"])
        for field in (
            "max_hours",
            "boot_volume_gb",
            "cos_storage_gb_month",
            "cos_operations",
            "registry_storage_gb_month",
            "code_engine_job_hours",
            "logs_monitoring_gb",
        ):
            values[field] = Decimal(values[field])
        candidate = RunSpec(bootstrap_sha256="0" * 64, **values)
        return RunSpec(bootstrap_sha256=bootstrap_sha256_for(candidate), **values)
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise IntegrityError("RunSpec document is invalid") from exc


def _atomic_record(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _cos_run_scope(cos: Any, spec: RunSpec):
    bindings = {name: getattr(cos, name, None) for name in (
        "_watch_auth_spec", "_watch_auth_bucket", "_watch_auth_prefix",
        "_watch_auth_configuration_verified",
    )}
    try:
        cos._watch_auth_spec = spec
        cos._watch_auth_bucket = spec.bucket
        cos._watch_auth_prefix = spec.prefix
        verify_cos_configuration(cos, spec)
        yield
    finally:
        for name, value in bindings.items():
            setattr(cos, name, value)


def authorize_training(
    cos: Any,
    usage_client: Any,
    spec: RunSpec,
    credit: CreditEvidence,
    pricing: PricingEvidence,
    post_promo: PostPromoPlan,
    *,
    watchdog_proof: WatchdogCanaryProof,
    now: datetime,
    store: Path | None = None,
) -> AuthorizedRun:
    """Read prior terminal receipts before granting the next account launch capability."""

    from infra.ibm_watchdog import _load_run_spec, _run_spec_keys

    usage = fetch_usage_evidence(
        usage_client, spec.account_id, now.astimezone(timezone.utc).strftime("%Y-%m"), (), now=now,
    )
    prior_ids: list[UUID] = []
    events: list[RunEvent] = []
    receipts: list[str] = []
    for key in _run_spec_keys(cos, spec.bucket, "runs"):
        prior = _load_run_spec(cos, spec.bucket, key)
        if (
            prior.account_id != spec.account_id or prior.bucket != spec.bucket
            or prior.region != spec.region or prior.resource_group_id != spec.resource_group_id
            or prior.run_id == spec.run_id or prior.run_id in prior_ids
        ):
            raise IntegrityError("prior run inventory is ambiguous or outside controller scope")
        with _cos_run_scope(cos, prior):
            authorization = load_immutable_json(cos, prior, f"{prior.prefix}/evidence/authorization-consumed.json")
            if authorization is None:
                raise IntegrityError("prior launch authorization is missing")
            receipt = _validated_usage_receipt(cos, prior, authorization)
            context = load_immutable_json(cos, prior, f"{prior.prefix}/watchdog/cleanup-context.json")
            if context is None:
                raise IntegrityError("provider resource inventory disappeared")
            latest = load_immutable_json(cos, prior, receipt["observation_keys"][1])
            if latest is None:
                raise IntegrityError("provider usage observation disappeared")
            period = latest["resources"][0]["month"]
            prior_usage = usage if period == usage.billing_period_start.strftime("%Y-%m") else fetch_usage_evidence(
                usage_client, spec.account_id, period, (prior.run_id,), now=now,
            )
            current = usage_observation(prior, prior_usage, context)
            if current is None or current["resources"] != latest["resources"]:
                raise IntegrityError("prior provider usage changed after reconciliation")
            events.extend(load_event_inventory(cos, prior).events)
            receipts.append(hashlib.sha256(_canonical(receipt)).hexdigest())
            prior_ids.append(prior.run_id)
    usage = replace(
        usage, reconciled_run_ids=tuple(prior_ids),
        evidence_sha256=hashlib.sha256(_canonical({
            "provider_usage": usage.evidence_sha256, "receipts": sorted(receipts),
        })).hexdigest(),
    )
    decision = authorize_run(credit, usage, pricing, post_promo, spec, tuple(events), now=now)
    return consume_authorization(
        spec, decision, decision.evidence_hashes, watchdog_proof=watchdog_proof,
        now=now, store=store, cos=cos,
    )


def consume_authorization(
    spec: RunSpec,
    decision: AuthorizationDecision,
    evidence_hashes: tuple[str, ...],
    *,
    watchdog_proof: WatchdogCanaryProof | None,
    now: datetime,
    store: Path | None = None,
    cos: Any | None = None,
) -> AuthorizedRun:
    """Atomically turn one fresh decision nonce into one launch capability."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if watchdog_proof is None:
        raise PermissionError("provider-verified watchdog deletion canary proof is required")
    _validate_watchdog_proof(spec, watchdog_proof, now=now)
    if not decision.allowed:
        raise PermissionError("authorization decision denied the run")
    if not decision.issued_at <= now <= decision.expires_at:
        raise PermissionError("authorization decision is not currently valid")
    if decision.run_spec_sha256 != run_spec_sha256(spec):
        raise IntegrityError("authorization RunSpec hash mismatch")
    if evidence_hashes != decision.evidence_hashes:
        raise IntegrityError("authorization evidence hashes mismatch")

    payload = _consumption_payload(
        spec=spec,
        authorization_sha256=decision.authorization_sha256,
        evidence_hashes=evidence_hashes,
        issued_at=decision.issued_at,
        expires_at=decision.expires_at,
        nonce=decision.nonce,
        watchdog_proof=watchdog_proof,
    )
    consumption_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    body = _canonical({**payload, "consumption_sha256": consumption_sha256})
    store = (store or _AUTHORIZATION_DIR).resolve()
    store.mkdir(parents=True, exist_ok=True)
    account_key = hashlib.sha256(spec.account_id.encode()).hexdigest()
    lock_path = store / f"account-{account_key}.lock"
    pending_path = store / f"account-{account_key}.pending.json"
    path = store / f"nonce-{decision.nonce}.json"
    if pending_path.exists() and cos is not None:
        pending = json.loads(pending_path.read_bytes())
        previous_spec = parse_run_spec(pending.get("run_spec"))
        if previous_spec.account_id != spec.account_id or previous_spec.bucket != spec.bucket:
            raise IntegrityError("pending authorization escapes controller account or bucket")
        with _cos_run_scope(cos, previous_spec):
            reconcile_pending_authorization(cos, previous_spec, store=store)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if pending_path.exists():
            raise PermissionError("account has a run pending provider usage reconciliation")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise PermissionError("authorization nonce was already consumed") from exc
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            pending_descriptor = os.open(
                pending_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(pending_descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
            directory = os.open(store, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            path.unlink(missing_ok=True)
            pending_path.unlink(missing_ok=True)
            raise
    return AuthorizedRun._from_consumed_decision(
        spec,
        decision,
        watchdog_proof,
        consumption_sha256,
        str(path),
        str(pending_path),
    )


def _validated_usage_receipt(cos: Any, spec: RunSpec, consumed: dict[str, Any]) -> dict[str, Any]:
    receipt = load_immutable_json(cos, spec, f"{spec.prefix}/evidence/usage-reconciliation.json")
    if receipt is None:
        raise IntegrityError("terminal usage reconciliation is missing")
    if (
        receipt.get("account_id") != spec.account_id
        or receipt.get("run_id") != str(spec.run_id)
        or receipt.get("provider_verified") is not True
        or receipt.get("unreconciled_run_ids") != []
        or receipt.get("reconciled_run_ids") != [str(spec.run_id)]
        or receipt.get("run_spec_sha256") != run_spec_sha256(spec)
        or receipt.get("consumption_sha256") != consumed.get("consumption_sha256")
        or receipt.get("authorization_sha256") != consumed.get("authorization_sha256")
    ):
        raise IntegrityError("usage reconciliation is not terminal for this run")
    events = load_event_inventory(cos, spec).events
    validate_event_chain(events, spec)
    if (
        not events or events[-1].state != "deleted"
        or receipt.get("terminal_event_sha256") != run_event_sha256(events[-1])
    ):
        raise IntegrityError("usage reconciliation differs from terminal event")
    keys = receipt.get("observation_keys")
    if (
        not isinstance(keys, list) or len(keys) != 2
        or any(not isinstance(key, str) or not key.startswith(
            f"{spec.prefix}/evidence/usage-observations/"
        ) for key in keys)
    ):
        raise IntegrityError("usage reconciliation observation binding is invalid")
    first, later = (load_immutable_json(cos, spec, key) for key in keys)
    if first is None or later is None or not stable_usage_observations(first, later, spec):
        raise IntegrityError("provider usage has not stabilized")
    if receipt.get("evidence_sha256") != later.get("evidence_sha256"):
        raise IntegrityError("usage receipt differs from provider observation")
    context = load_immutable_json(cos, spec, f"{spec.prefix}/watchdog/cleanup-context.json")
    if context is None:
        raise IntegrityError("provider resource inventory is missing")
    for observation in (first, later):
        observed_at = datetime.fromisoformat(observation["observed_at"])
        if observed_at <= events[-1].observed_at:
            raise IntegrityError("usage observation predates terminal deletion")
        resources = observation["resources"]
        usage = UsageEvidence(
            account_id=spec.account_id, observed_at=observed_at,
            billing_period_start=datetime.strptime(resources[0]["month"], "%Y-%m").replace(tzinfo=timezone.utc),
            observed_usage_usd=Decimal("0"), resource_count=len(resources),
            unreconciled_run_ids=(spec.run_id,), reconciled_run_ids=(),
            evidence_sha256=observation["evidence_sha256"], provider_verified=True,
            resource_records=tuple(_canonical(resource).decode() for resource in resources),
        )
        if usage_observation(spec, usage, context) != observation:
            raise IntegrityError("usage observation lacks run resource and SKU binding")
    return receipt


def reconcile_pending_authorization(
    cos: Any,
    spec: RunSpec,
    *,
    store: Path | None = None,
) -> None:
    """Release the account fence only from a durable provider-usage receipt."""

    store = (store or _AUTHORIZATION_DIR).resolve()
    account_key = hashlib.sha256(spec.account_id.encode()).hexdigest()
    lock_path = store / f"account-{account_key}.lock"
    pending_path = store / f"account-{account_key}.pending.json"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        pending = pending_path.read_bytes()
        try:
            consumed = json.loads(pending)
        except json.JSONDecodeError as exc:
            raise IntegrityError("pending authorization is not JSON") from exc
        if (
            not isinstance(consumed, dict)
            or consumed.get("run_id") != str(spec.run_id)
            or consumed.get("run_spec_sha256") != run_spec_sha256(spec)
        ):
            raise IntegrityError("pending authorization differs from RunSpec")
        receipt = _validated_usage_receipt(cos, spec, consumed)
        receipt_body = _canonical(receipt)
        marker = store / f"account-{account_key}.reconciled-{spec.run_id}.json"
        _atomic_record(marker, receipt_body)
        pending_path.unlink()
        directory = os.open(store, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _watchdog_proof_document(proof: WatchdogCanaryProof) -> dict[str, object]:
    return {
        "account_id": proof.account_id,
        "canary_deleted_at": proof.canary_deleted_at.isoformat(),
        "canary_instance_sha256": proof.canary_instance_sha256,
        "canary_run_id": str(proof.canary_run_id),
        "canary_nonce": str(proof.canary_nonce),
        "logs_crn": proof.logs_crn,
        "monitoring_crn": proof.monitoring_crn,
        "canary_log_sha256": proof.canary_log_sha256,
        "canary_metric_sha256": proof.canary_metric_sha256,
        "canary_404_sha256": proof.canary_404_sha256,
        "evidence_sha256": proof.evidence_sha256,
        "expires_at": proof.expires_at.isoformat(),
        "observed_at": proof.observed_at.isoformat(),
        "provider_verified": proof.provider_verified,
        "region": proof.region,
        "resource_group_id": proof.resource_group_id,
        "watchdog_job_crn": proof.watchdog_job_crn,
    }


def _validate_watchdog_proof(
    spec: RunSpec, proof: WatchdogCanaryProof, *, now: datetime
) -> None:
    if proof.account_id != spec.account_id:
        raise PermissionError("watchdog proof account differs from RunSpec")
    if proof.region != spec.region:
        raise PermissionError("watchdog proof region differs from RunSpec")
    if proof.resource_group_id != spec.resource_group_id:
        raise PermissionError("watchdog proof resource group differs from RunSpec")
    if proof.canary_run_id == spec.run_id:
        raise PermissionError("watchdog deletion canary must be independent from training run")
    if f":{spec.region}:a/{spec.account_id}:" not in proof.watchdog_job_crn:
        raise PermissionError("watchdog job CRN differs from RunSpec environment")
    if not proof.observed_at <= now <= proof.expires_at:
        raise PermissionError("watchdog deletion canary proof is not currently valid")


def _consumption_payload(
    *,
    spec: RunSpec,
    authorization_sha256: str,
    evidence_hashes: tuple[str, ...],
    issued_at: datetime,
    expires_at: datetime,
    nonce: UUID,
    watchdog_proof: WatchdogCanaryProof,
) -> dict[str, object]:
    return {
        "authorization_sha256": authorization_sha256,
        "evidence_hashes": list(evidence_hashes),
        "expires_at": expires_at.isoformat(),
        "issued_at": issued_at.isoformat(),
        "nonce": str(nonce),
        "run_id": str(spec.run_id),
        "run_spec_sha256": run_spec_sha256(spec),
        "run_spec": run_spec_transport_document(spec),
        "watchdog_proof": _watchdog_proof_document(watchdog_proof),
    }


def _revalidate_consumed_authorization(run: AuthorizedRun, *, now: datetime) -> None:
    spec = run.spec
    if not run.issued_at <= now <= run.expires_at:
        raise PermissionError("consumed launch authorization is expired")
    if run.run_spec_sha256 != run_spec_sha256(spec):
        raise IntegrityError("consumed authorization RunSpec binding changed")
    _validate_watchdog_proof(spec, run.watchdog_proof, now=now)
    payload = _consumption_payload(
        spec=spec,
        authorization_sha256=run.authorization_sha256,
        evidence_hashes=run.evidence_hashes,
        issued_at=run.issued_at,
        expires_at=run.expires_at,
        nonce=run.nonce,
        watchdog_proof=run.watchdog_proof,
    )
    consumption_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
    if consumption_sha256 != run.consumption_sha256:
        raise IntegrityError("consumed authorization binding changed")
    expected = _canonical({**payload, "consumption_sha256": consumption_sha256})
    for record_path in (run.authorization_record_path, run.pending_record_path):
        try:
            observed = Path(record_path).read_bytes()
        except OSError as exc:
            raise IntegrityError("consumed authorization record is unavailable") from exc
        if observed != expected:
            raise IntegrityError("consumed authorization record differs from launch")


def _response_bytes(response: object) -> bytes:
    if not isinstance(response, dict) or "Body" not in response:
        raise IntegrityError("COS object response is invalid")
    body = response["Body"]
    raw = body.read() if hasattr(body, "read") else body
    if not isinstance(raw, bytes):
        raise IntegrityError("COS object body is not bytes")
    return raw


def _put_immutable(client: Any, spec: RunSpec, key: str, body: bytes) -> None:
    put_immutable_json(client, spec, key, body)


def publish_run_spec(client: Any, spec: RunSpec) -> None:
    body = _canonical(run_spec_transport_document(spec))
    _put_immutable(client, spec, f"{spec.prefix}/run-spec.json", body)


def _publish_authorization_evidence(client: Any, run: AuthorizedRun) -> None:
    spec = run.spec
    body = _canonical(
        {
            "account_id": spec.account_id,
            "authorization_sha256": run.authorization_sha256,
            "consumption_sha256": run.consumption_sha256,
            "estimated_usd": str(run.estimated_usd),
            "run_id": str(spec.run_id),
            "run_spec_sha256": run.run_spec_sha256,
        }
    )
    _put_immutable(
        client,
        spec,
        f"{spec.prefix}/evidence/authorization-consumed.json",
        body,
    )


def _event_payload(state: str, details: object | None = None) -> str:
    return hashlib.sha256(_canonical({"details": details, "state": state})).hexdigest()


def _append_state(
    client: Any,
    spec: RunSpec,
    state: str,
    *,
    now: datetime,
    instance_id: str | None,
    details: object | None = None,
    deadline: float,
) -> RunEvent:
    with _hard_deadline(deadline):
        snapshot = load_event_inventory(client, spec)
        events = snapshot.events
        if events:
            validate_event_chain(events, spec)
        event = RunEvent(
            event_id=uuid4(),
            run_id=spec.run_id,
            sequence=len(events),
            state=state,
            observed_at=now,
            instance_id=instance_id,
            lineage_sha256=run_spec_sha256(spec),
            previous_event_sha256=run_event_sha256(events[-1]) if events else None,
            payload_sha256=_event_payload(state, details),
        )
        append_event(client, spec.bucket, spec.prefix, event)
        updated = load_event_inventory(client, spec).events
        validate_event_chain(updated, spec)
        if updated[-1] != event:
            raise IntegrityError("persisted lifecycle event is not authoritative")
        return event


def _receipt_key(spec: RunSpec) -> str:
    return f"{spec.prefix}/receipts/training.json"


def wait_for_training_receipt(client: Any, spec: RunSpec, *, deadline: float) -> dict[str, object]:
    with _hard_deadline(deadline):
        while time.monotonic() < deadline:
            receipt = load_immutable_json(client, spec, _receipt_key(spec))
            if receipt is None:
                time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
                continue
            if not isinstance(receipt, dict) or receipt.get("run_id") != str(spec.run_id):
                raise IntegrityError("training receipt run mismatch")
            if receipt.get("run_spec_sha256") != run_spec_sha256(spec):
                raise IntegrityError("training receipt RunSpec hash mismatch")
            status = receipt.get("status")
            if status == "failed":
                if set(receipt) != {"error_type", "run_id", "run_spec_sha256", "status"}:
                    raise IntegrityError("failed training receipt schema is invalid")
                raise RuntimeError(
                    f"training worker failed: {receipt.get('error_type', 'unknown')}"
                )
            if (
                status != "succeeded"
                or set(receipt) != {"artifacts", "run_id", "run_spec_sha256", "status"}
                or not isinstance(receipt.get("artifacts"), list)
            ):
                raise IntegrityError("training receipt schema is invalid")
            _verify_receipt_artifacts(client, spec, receipt["artifacts"])
            return receipt
    raise TimeoutError("training receipt did not arrive before deadline")


def _verify_receipt_artifacts(client: Any, spec: RunSpec, artifacts: object) -> None:
    if not isinstance(artifacts, list) or len(artifacts) != 4:
        raise IntegrityError("training receipt must contain exactly four artifacts")
    expected_names = {"checkpoint.pt", "class-map.json", "metrics.json", "run-result.json"}
    names: set[str] = set()
    bodies: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"key", "length", "name", "sha256"}:
            raise IntegrityError("training artifact receipt schema is invalid")
        name, key, digest, length = (
            item["name"], item["key"], item["sha256"], item["length"]
        )
        if (
            not isinstance(name, str)
            or not isinstance(key, str)
            or not key.startswith(f"{spec.prefix}/artifacts/{name}/")
            or not isinstance(digest, str)
            or len(digest) != 64
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
        ):
            raise IntegrityError("training artifact receipt values are invalid")
        body = _response_bytes(client.get_object(Bucket=spec.bucket, Key=key))
        if len(body) != length or hashlib.sha256(body).hexdigest() != digest:
            raise IntegrityError("training artifact bytes differ from receipt")
        names.add(name)
        bodies[name] = body
        digests[name] = digest
    if names != expected_names:
        raise IntegrityError("training receipt artifact names are invalid")
    try:
        result = json.loads(bodies["run-result.json"])
    except json.JSONDecodeError as exc:
        raise IntegrityError("run-result artifact is not JSON") from exc
    if _canonical(result) != bodies["run-result.json"] or not isinstance(result, dict):
        raise IntegrityError("run-result artifact is not canonical JSON")
    if set(result) != {
        "artifacts",
        "best_epoch",
        "best_validation_nll",
        "epochs_completed",
        "lineage",
    }:
        raise IntegrityError("run-result artifact schema is invalid")
    expected_artifacts = {
        name: digests[name] for name in ("checkpoint.pt", "class-map.json", "metrics.json")
    }
    lineage = result.get("lineage")
    if result.get("artifacts") != expected_artifacts or not isinstance(lineage, dict):
        raise IntegrityError("run-result artifact references differ from receipt")
    expected_lineage = {
        "code_sha": spec.code_sha,
        "image_digest": spec.image_digest,
        "manifest_sha256": spec.manifest_sha256,
        "splits_sha256": spec.splits_sha256,
        "class_map_sha256": digests["class-map.json"],
    }
    if any(lineage.get(name) != value for name, value in expected_lineage.items()):
        raise IntegrityError("run-result lineage differs from RunSpec")


def _record_recovery(spec: RunSpec, stage: str, error: BaseException) -> None:
    body = _canonical(
        {
            "error_type": type(error).__name__,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "run_id": str(spec.run_id),
            "stage": stage,
        }
    )
    _atomic_record(_RECOVERY_DIR / f"{spec.run_id}.json", body)


def _try_record_recovery(
    spec: RunSpec, stage: str, error: BaseException
) -> BaseException | None:
    try:
        _record_recovery(spec, stage, error)
    except BaseException as recovery_error:
        return recovery_error
    return None


def _evidence_deadline(limit: float) -> float:
    return min(time.monotonic() + _COS_EVIDENCE_TIMEOUT_SECONDS, limit)


def launch_training(
    clients: IBMClients,
    run: AuthorizedRun,
    *,
    confirm: bool,
    now: datetime,
) -> RunEvent:
    """Run one authorized lifecycle and always tear down a created/adopted VSI."""

    if confirm is not True:
        raise PermissionError("explicit launch confirmation is required")
    spec = run.spec
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    _revalidate_consumed_authorization(run, now=now)
    started_at = time.monotonic()
    total_deadline = started_at + float(spec.max_hours * Decimal(3600))
    operation_deadline = total_deadline - _DELETE_TIMEOUT_SECONDS
    if operation_deadline <= started_at + 60:
        raise ValueError("max_hours must reserve 30 minutes for teardown")

    with _hard_deadline(_evidence_deadline(operation_deadline)):
        verify_cos_configuration(clients.cos, spec)
    with _hard_deadline(_evidence_deadline(operation_deadline)):
        publish_run_spec(clients.cos, spec)
    with _hard_deadline(_evidence_deadline(operation_deadline)):
        _publish_authorization_evidence(clients.cos, run)
    _append_state(
        clients.cos,
        spec,
        "planned",
        now=now,
        instance_id=None,
        deadline=_evidence_deadline(operation_deadline),
    )
    _append_state(
        clients.cos,
        spec,
        "authorized",
        now=now,
        instance_id=None,
        deadline=_evidence_deadline(operation_deadline),
    )
    bootstrap = render_bootstrap(spec)
    prototype = build_instance_prototype(spec, bootstrap)
    instance_id: str | None = None
    original_error: BaseException | None = None
    terminal: RunEvent | None = None
    create_started = False
    evidence_errors: list[BaseException] = []
    try:
        _append_state(
            clients.cos,
            spec,
            "creating",
            now=now,
            instance_id=None,
            deadline=_evidence_deadline(operation_deadline),
        )
        create_started = True
        instance = ensure_single_instance(
            clients.vpc,
            spec,
            prototype,
            tagger=clients.tagger,
            deadline=min(time.monotonic() + _CREATE_TIMEOUT_SECONDS, operation_deadline),
        )
        instance_id = instance.id
        running = wait_running(
            clients.vpc,
            instance_id,
            deadline=min(time.monotonic() + _CREATE_TIMEOUT_SECONDS, operation_deadline),
        )
        _append_state(
            clients.cos,
            spec,
            "running",
            now=now,
            instance_id=running.id,
            deadline=_evidence_deadline(operation_deadline),
        )
        receipt = wait_for_training_receipt(
            clients.cos,
            spec,
            deadline=operation_deadline,
        )
        _append_state(
            clients.cos,
            spec,
            "artifact_sync",
            now=now,
            instance_id=instance_id,
            details=receipt["artifacts"],
            deadline=_evidence_deadline(operation_deadline),
        )
        terminal = _append_state(
            clients.cos,
            spec,
            "succeeded",
            now=now,
            instance_id=instance_id,
            deadline=_evidence_deadline(operation_deadline),
        )
    except BaseException as exc:
        original_error = exc
        if create_started:
            try:
                if instance_id is None:
                    try:
                        instance_id = recover_created_instance_id(spec, prototype)
                    except Exception:
                        try:
                            instance_id = recover_accepted_instance(
                                clients.vpc,
                                spec,
                                prototype,
                                deadline=min(
                                    time.monotonic() + _RECOVERY_TIMEOUT_SECONDS,
                                    total_deadline,
                                ),
                            )
                        except BaseException as provider_recovery_error:
                            evidence_errors.append(provider_recovery_error)
                            try:
                                instance_id = recover_created_instance_id(
                                    spec, prototype
                                )
                            except BaseException as durable_recovery_error:
                                evidence_errors.append(durable_recovery_error)
                _append_state(
                    clients.cos,
                    spec,
                    "failed",
                    now=now,
                    instance_id=instance_id,
                    details=type(exc).__name__,
                    deadline=_evidence_deadline(total_deadline),
                )
            except BaseException as evidence_error:
                evidence_errors.append(evidence_error)
                recovery_error = _try_record_recovery(
                    spec, "failure-event", evidence_error
                )
                if recovery_error is not None:
                    evidence_errors.append(recovery_error)
    finally:
        cleanup_error: BaseException | None = None
        if create_started and instance_id is not None:
            try:
                _append_state(
                    clients.cos,
                    spec,
                    "deleting",
                    now=now,
                    instance_id=instance_id,
                    deadline=_evidence_deadline(total_deadline),
                )
            except BaseException as evidence_error:
                evidence_errors.append(evidence_error)
                recovery_error = _try_record_recovery(
                    spec, "deleting-event", evidence_error
                )
                if recovery_error is not None:
                    evidence_errors.append(recovery_error)
            try:
                delete_and_wait(
                    clients.vpc,
                    spec,
                    instance_id,
                    deadline=min(time.monotonic() + _DELETE_TIMEOUT_SECONDS, total_deadline),
                    tagger=clients.tagger,
                    cleanup_context_sink=lambda context: _put_immutable(
                        clients.cos, spec, f"{spec.prefix}/watchdog/cleanup-context.json",
                        _canonical(context),
                    ),
                )
                with _hard_deadline(_evidence_deadline(total_deadline)):
                    context = load_immutable_json(
                        clients.cos, spec, f"{spec.prefix}/watchdog/cleanup-context.json"
                    )
                    if context is None:
                        raise IntegrityError("durable cleanup context is missing after teardown")
                    deletion_receipts = persist_deletion_receipts(clients.cos, spec, context)
                terminal = _append_state(
                    clients.cos,
                    spec,
                    "deleted",
                    now=now,
                    instance_id=instance_id,
                    details={"deletion_receipt_sha256": deletion_receipts},
                    deadline=_evidence_deadline(total_deadline),
                )
            except BaseException as exc:
                cleanup_error = exc
                recovery_error = _try_record_recovery(spec, "teardown", exc)
                if recovery_error is not None:
                    evidence_errors.append(recovery_error)
        errors = [
            error
            for error in (original_error, cleanup_error, *evidence_errors)
            if error is not None
        ]
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("IBM training lifecycle failures", errors)
    if terminal is None or terminal.state != "deleted":
        raise RuntimeError("run ended without an observed deleted event")
    return terminal
