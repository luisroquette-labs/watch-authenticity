"""Independent, reconcile-only cleanup for expired IBM training resources."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from uuid import uuid4

from infra.ibm_cos import (
    IntegrityError,
    append_event,
    load_event_inventory,
    validate_event_chain,
    verify_cos_configuration,
)
from infra.ibm_evidence import fetch_usage_evidence, run_spec_sha256
from infra.ibm_types import ResourceLease, RunEvent, RunSpec, UsageEvidence, run_event_sha256
from infra.ibm_vpc import (
    AmbiguousProvisioning,
    _field,
    _list_named,
    delete_and_wait,
    restore_cleanup_context,
)

_LOG = logging.getLogger(__name__)
_DELETE_TIMEOUT_SECONDS = 30 * 60
_PROJECT_TAG = "project:watch-authenticity"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _result(response: object) -> dict[str, Any]:
    if isinstance(response, dict):
        result = response
    else:
        getter = getattr(response, "get_result", None)
        result = getter() if callable(getter) else None
    if not isinstance(result, dict):
        raise AmbiguousProvisioning("IBM tag response is not a mapping")
    return result


def _bound_spec(cos: Any) -> RunSpec:
    spec = getattr(cos, "_watch_auth_spec", None)
    if not isinstance(spec, RunSpec):
        raise IntegrityError("COS client is not bound to an immutable RunSpec")
    if getattr(cos, "_watch_auth_configuration_verified", None) is not True:
        raise IntegrityError("COS configuration is not verified")
    if getattr(cos, "_watch_auth_bucket", None) != spec.bucket or getattr(
        cos, "_watch_auth_prefix", None
    ) != spec.prefix.strip("/"):
        raise IntegrityError("COS client binding differs from RunSpec")
    return spec


def _ordered(events: tuple[RunEvent, ...], spec: RunSpec) -> tuple[RunEvent, ...]:
    validate_event_chain(events, spec)
    return tuple(sorted(events, key=lambda event: event.sequence))


def _authoritative_events(
    cos: Any, spec: RunSpec, expected: tuple[RunEvent, ...]
) -> tuple[RunEvent, ...]:
    observed = load_event_inventory(cos, spec).events
    observed = _ordered(observed, spec)
    if observed != expected:
        raise IntegrityError("provided events differ from the authoritative COS chain")
    return observed


def _payload_hash(state: str, *, resource: str, reason: str) -> str:
    return hashlib.sha256(
        _canonical({"reason": reason, "resource": resource, "state": state})
    ).hexdigest()


def _cos_status_code(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    metadata = response.get("ResponseMetadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("HTTPStatusCode"), int):
        return metadata["HTTPStatusCode"]
    error = response.get("Error")
    code = error.get("Code") if isinstance(error, dict) else None
    if not isinstance(code, (str, int)) or isinstance(code, bool):
        return None
    try:
        return int(code)
    except (TypeError, ValueError):
        return 404 if code in {"NoSuchKey", "NotFound"} else None


def _put_immutable_document(cos: Any, spec: RunSpec, key: str, document: object) -> str:
    body = _canonical(document)
    digest = hashlib.sha256(body).hexdigest()
    try:
        cos.put_object(
            Bucket=spec.bucket,
            Key=key,
            Body=body,
            Metadata={"sha256": digest, "length": str(len(body))},
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as exc:
        if _cos_status_code(exc) != 412:
            raise
    observed = _body_bytes(cos.get_object(Bucket=spec.bucket, Key=key))
    if observed != body:
        raise IntegrityError(f"immutable COS object differs: {key}")
    return digest


def _load_optional_document(cos: Any, spec: RunSpec, key: str) -> dict[str, object] | None:
    try:
        body = _body_bytes(cos.get_object(Bucket=spec.bucket, Key=key))
    except Exception as exc:
        if _cos_status_code(exc) == 404:
            return None
        raise
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"COS object is not JSON: {key}") from exc
    if not isinstance(document, dict) or body != _canonical(document):
        raise IntegrityError(f"COS object is not canonical JSON: {key}")
    return document


def _cleanup_context_key(spec: RunSpec) -> str:
    return f"{spec.prefix}/watchdog/cleanup-context.json"


def _persist_cleanup_context(cos: Any, spec: RunSpec, context: dict[str, object]) -> None:
    _put_immutable_document(cos, spec, _cleanup_context_key(spec), context)


def _restore_cleanup_context(cos: Any, spec: RunSpec, instance_id: str) -> None:
    context = _load_optional_document(cos, spec, _cleanup_context_key(spec))
    if context is not None:
        restore_cleanup_context(spec, instance_id, context)


def _require_identity_or_durable_cleanup(
    tagger: Any,
    lease: ResourceLease,
    cos: Any,
    spec: RunSpec,
    instance_id: str,
) -> None:
    try:
        _require_identity_tags(tagger, lease)
    except Exception:
        context = _load_optional_document(cos, spec, _cleanup_context_key(spec))
        if context is None:
            raise
        restore_cleanup_context(spec, instance_id, context)


def _persist_deletion_receipts(
    cos: Any, spec: RunSpec, context: dict[str, object]
) -> dict[str, str]:
    resources = {
        "vpc-vsi": context["instance_id"],
        "boot-volume": context["volume_id"],
        "subnet": context["subnet_id"],
        "public-gateway": context["public_gateway_id"],
        "floating-ip": context["public_gateway_floating_ip_id"],
    }
    hashes: dict[str, str] = {}
    for resource_type, resource_id in resources.items():
        receipt = {
            "provider_state": "404",
            "resource_id": resource_id,
            "resource_type": resource_type,
            "run_id": str(spec.run_id),
        }
        digest = hashlib.sha256(_canonical(receipt)).hexdigest()
        key = f"{spec.prefix}/receipts/deletion/{resource_type}-{digest}.json"
        hashes[resource_type] = _put_immutable_document(cos, spec, key, receipt)
    return hashes


def _persist_usage_reconciliation(
    cos: Any, spec: RunSpec, usage: UsageEvidence
) -> str:
    if (
        usage.account_id != spec.account_id
        or usage.unreconciled_run_ids
        or usage.reconciled_run_ids != (spec.run_id,)
    ):
        raise IntegrityError("provider usage is not terminally reconciled for this account")
    document = {
        "account_id": usage.account_id,
        "evidence_sha256": usage.evidence_sha256,
        "observed_at": usage.observed_at.isoformat(),
        "observed_usage_usd": str(usage.observed_usage_usd),
        "provider_verified": usage.provider_verified,
        "resource_count": usage.resource_count,
        "run_id": str(spec.run_id),
        "reconciled_run_ids": [str(value) for value in usage.reconciled_run_ids],
        "unreconciled_run_ids": [],
    }
    return _put_immutable_document(
        cos,
        spec,
        f"{spec.prefix}/evidence/usage-reconciliation-{usage.evidence_sha256}.json",
        document,
    )


def _append_state(
    cos: Any,
    spec: RunSpec,
    events: tuple[RunEvent, ...],
    state: str,
    *,
    now: datetime,
    instance_id: str | None,
    resource: str,
    reason: str,
    details: dict[str, object] | None = None,
) -> tuple[RunEvent, ...]:
    current = _authoritative_events(cos, spec, events)
    payload_sha256 = _payload_hash(state, resource=resource, reason=reason)
    if details is not None:
        payload = {"reason": reason, "resource": resource, "state": state, **details}
        payload_sha256 = hashlib.sha256(_canonical(payload)).hexdigest()
        _put_immutable_document(
            cos,
            spec,
            f"{spec.prefix}/event-payloads/{payload_sha256}.json",
            payload,
        )
    event = RunEvent(
        event_id=uuid4(),
        run_id=spec.run_id,
        sequence=len(current),
        state=state,
        observed_at=now,
        instance_id=instance_id,
        lineage_sha256=run_spec_sha256(spec),
        previous_event_sha256=run_event_sha256(current[-1]),
        payload_sha256=payload_sha256,
    )
    append_event(cos, spec.bucket, spec.prefix, event)
    updated = _ordered(load_event_inventory(cos, spec).events, spec)
    if updated != current + (event,):
        raise IntegrityError("watchdog event append was not authoritative")
    return updated


def _instance_id_from_crn(resource_crn: str) -> str:
    parts = resource_crn.split(":")
    if len(parts) != 10 or parts[4] != "is" or parts[8] != "instance" or not parts[9]:
        raise ValueError("lease resource_crn must identify one VPC instance")
    return parts[9]


def _validate_lease(lease: object, spec: RunSpec, now: datetime) -> ResourceLease:
    if not isinstance(lease, ResourceLease):
        raise ValueError("lease must be a validated ResourceLease")
    if lease.run_id != spec.run_id:
        raise ValueError("lease run differs from RunSpec")
    if lease.service != "vpc-vsi" or lease.terminal_policy != "delete":
        raise ValueError("lease is not an auto-delete VPC VSI lease")
    if lease.created_at > now:
        raise ValueError("lease creation time is in the future")
    _instance_id_from_crn(lease.resource_crn)
    return lease


def _recorded_instance_id(events: tuple[RunEvent, ...]) -> str | None:
    ids = {event.instance_id for event in events if event.instance_id is not None}
    if len(ids) > 1:
        raise AmbiguousProvisioning("event chain records multiple instance IDs")
    return next(iter(ids), None)


def _resolve_instance_id(
    vpc: Any, spec: RunSpec, lease: ResourceLease, events: tuple[RunEvent, ...]
) -> str:
    leased_id = _instance_id_from_crn(lease.resource_crn)
    recorded_id = _recorded_instance_id(events)
    if recorded_id is not None:
        if recorded_id != leased_id:
            raise AmbiguousProvisioning("event and lease instance IDs differ")
        return recorded_id
    matches = _list_named(vpc, spec)
    if len(matches) != 1:
        raise AmbiguousProvisioning("lost-create reconciliation is not unique")
    instance_id = _field(matches[0], "id")
    if instance_id != leased_id:
        raise AmbiguousProvisioning("named instance differs from the lease")
    return instance_id


def _require_identity_tags(tagger: Any, lease: ResourceLease) -> None:
    result = _result(
        tagger.list_tags(
            attached_to=lease.resource_crn,
            tag_type="user",
        )
    )
    items = result.get("items")
    if not isinstance(items, list):
        raise AmbiguousProvisioning("instance tag response is malformed")
    names = {
        item.get("name")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    required = {_PROJECT_TAG, f"run:{lease.run_id}"}
    conflicting = {
        name
        for name in names
        if isinstance(name, str)
        and (name.startswith("project:") or name.startswith("run:"))
        and name not in required
    }
    if not required <= names or conflicting:
        raise AmbiguousProvisioning("instance does not have the exact watchdog identity tags")


def _log_state(
    lease: ResourceLease,
    state: str,
    *,
    now: datetime,
    instance_id: str | None,
    estimated_usd: Decimal,
    observed_usage_usd: Decimal,
) -> None:
    record = {
        "age_seconds": max(0, int((now - lease.created_at).total_seconds())),
        "estimated_usd": str(estimated_usd),
        "observed_usage_usd": str(observed_usage_usd),
        "run_id": str(lease.run_id),
        "state": state,
        "vsi_id_hash": hashlib.sha256(instance_id.encode()).hexdigest() if instance_id else None,
    }
    _LOG.info("watchdog_event %s", json.dumps(record, sort_keys=True, separators=(",", ":")))


def reconcile_expired_resources(
    vpc: Any,
    cos: Any,
    leases: tuple[ResourceLease, ...],
    events: tuple[RunEvent, ...],
    *,
    now: datetime,
    estimated_usd: Decimal,
    usage: UsageEvidence,
    usage_reconciler: Callable[[], UsageEvidence],
) -> tuple[RunEvent, ...]:
    """Delete one expired, exactly-owned run; never create or broaden ownership."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    spec = _bound_spec(cos)
    if usage.account_id != spec.account_id or not usage.provider_verified:
        raise IntegrityError("watchdog usage evidence is not authoritative")
    ordered = _ordered(events, spec)
    _authoritative_events(cos, spec, ordered)
    if ordered[-1].state == "deleted":
        return ordered

    candidates = tuple(lease for lease in leases if getattr(lease, "run_id", None) == spec.run_id)
    instance_id: str | None = None
    try:
        if len(candidates) != 1:
            raise AmbiguousProvisioning("run must have exactly one VSI lease")
        lease = _validate_lease(candidates[0], spec, now)
        if now < lease.expires_at:
            _log_state(
                lease,
                "active",
                now=now,
                instance_id=None,
                estimated_usd=estimated_usd,
                observed_usage_usd=usage.observed_usage_usd,
            )
            return ordered
        instance_id = _resolve_instance_id(vpc, spec, lease, ordered)
        tagger = getattr(vpc, "_watch_auth_tagger", None)
        if tagger is None:
            raise AmbiguousProvisioning("VPC client has no bound tagging client")
        _require_identity_or_durable_cleanup(
            tagger, lease, cos, spec, instance_id
        )
    except (AmbiguousProvisioning, ValueError) as exc:
        quarantine_lease = (
            candidates[0]
            if len(candidates) == 1 and isinstance(candidates[0], ResourceLease)
            else None
        )
        if ordered[-1].state != "quarantined":
            ordered = _append_state(
                cos,
                spec,
                ordered,
                "quarantined",
                now=now,
                instance_id=instance_id,
                resource="vpc-vsi",
                reason=type(exc).__name__,
            )
        if quarantine_lease is not None:
            _log_state(
                quarantine_lease,
                "quarantined",
                now=now,
                instance_id=instance_id,
                estimated_usd=estimated_usd,
                observed_usage_usd=usage.observed_usage_usd,
            )
        return ordered

    assert isinstance(lease, ResourceLease)
    if ordered[-1].state != "deleting":
        ordered = _append_state(
            cos,
            spec,
            ordered,
            "deleting",
            now=now,
            instance_id=instance_id,
            resource="vpc-vsi",
            reason="lease_expired",
        )
    _log_state(
        lease,
        "deleting",
        now=now,
        instance_id=instance_id,
        estimated_usd=estimated_usd,
        observed_usage_usd=usage.observed_usage_usd,
    )
    try:
        _restore_cleanup_context(cos, spec, instance_id)
        delete_and_wait(
            vpc,
            spec,
            instance_id,
            deadline=time.monotonic() + _DELETE_TIMEOUT_SECONDS,
            tagger=getattr(vpc, "_watch_auth_tagger"),
            cleanup_context_sink=lambda context: _persist_cleanup_context(
                cos, spec, context
            ),
        )
    except Exception as exc:
        ordered = _append_state(
            cos,
            spec,
            ordered,
            "quarantined",
            now=now,
            instance_id=instance_id,
            resource="vpc-vsi",
            reason=type(exc).__name__,
        )
        _log_state(
            lease,
            "quarantined",
            now=now,
            instance_id=instance_id,
            estimated_usd=estimated_usd,
            observed_usage_usd=usage.observed_usage_usd,
        )
        raise
    context = _load_optional_document(cos, spec, _cleanup_context_key(spec))
    if context is None:
        raise IntegrityError("durable cleanup context is missing after teardown")
    deletion_receipts = _persist_deletion_receipts(cos, spec, context)
    reconciled_usage = usage_reconciler()
    usage_receipt_sha256 = _persist_usage_reconciliation(cos, spec, reconciled_usage)
    ordered = _append_state(
        cos,
        spec,
        ordered,
        "deleted",
        now=now,
        instance_id=instance_id,
        resource="all-run-resources",
        reason="provider_404_confirmed",
        details={
            "deletion_receipt_sha256": deletion_receipts,
            "estimated_usd": str(estimated_usd),
            "observed_usage_usd": str(reconciled_usage.observed_usage_usd),
            "usage_receipt_sha256": usage_receipt_sha256,
        },
    )
    _log_state(
        lease,
        "deleted",
        now=now,
        instance_id=instance_id,
        estimated_usd=estimated_usd,
        observed_usage_usd=reconciled_usage.observed_usage_usd,
    )
    return ordered


def _body_bytes(response: object) -> bytes:
    if not isinstance(response, dict):
        raise IntegrityError("COS object response is invalid")
    body = response.get("Body")
    reader = getattr(body, "read", None)
    raw = reader() if callable(reader) else body
    if not isinstance(raw, bytes):
        raise IntegrityError("COS object body is unavailable")
    return raw


def _parse_run_spec(document: object) -> RunSpec:
    from infra.ibm_train import parse_run_spec

    return parse_run_spec(document)


def _run_spec_keys(cos: Any, bucket: str, prefix: str) -> tuple[str, ...]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix.strip("/") + "/"}
        if token is not None:
            kwargs["ContinuationToken"] = token
        page = cos.list_objects_v2(**kwargs)
        if not isinstance(page, dict) or not isinstance(page.get("Contents", []), list):
            raise IntegrityError("COS run-spec listing is malformed")
        for item in page.get("Contents", []):
            key = item.get("Key") if isinstance(item, dict) else None
            if isinstance(key, str) and key.endswith("/run-spec.json"):
                keys.append(key)
        if not page.get("IsTruncated"):
            return tuple(sorted(set(keys)))
        next_token = page.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise IntegrityError("COS run-spec listing pagination is incomplete")
        token = next_token


def _load_run_spec(cos: Any, bucket: str, key: str) -> RunSpec:
    body = _body_bytes(cos.get_object(Bucket=bucket, Key=key))
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise IntegrityError("run-spec object is not JSON") from exc
    if body != _canonical(document):
        raise IntegrityError("run-spec object is not canonical JSON")
    spec = _parse_run_spec(document)
    if key != f"{spec.prefix}/run-spec.json":
        raise IntegrityError("run-spec key differs from its RunSpec")
    return spec


def _load_authorization_estimate(cos: Any, spec: RunSpec) -> Decimal:
    key = f"{spec.prefix}/evidence/authorization-consumed.json"
    document = _load_optional_document(cos, spec, key)
    if document is None:
        raise IntegrityError("durable launch authorization is missing")
    expected = {
        "account_id",
        "authorization_sha256",
        "consumption_sha256",
        "estimated_usd",
        "run_id",
        "run_spec_sha256",
    }
    if set(document) != expected:
        raise IntegrityError("durable launch authorization schema differs")
    if (
        document["account_id"] != spec.account_id
        or document["run_id"] != str(spec.run_id)
        or document["run_spec_sha256"] != run_spec_sha256(spec)
    ):
        raise IntegrityError("durable launch authorization differs from RunSpec")
    for name in ("authorization_sha256", "consumption_sha256"):
        value = document[name]
        if not isinstance(value, str) or len(value) != 64:
            raise IntegrityError(f"durable launch {name} is invalid")
    try:
        estimated = Decimal(str(document["estimated_usd"]))
    except InvalidOperation as exc:
        raise IntegrityError("durable launch estimate is invalid") from exc
    if not estimated.is_finite() or estimated < 0:
        raise IntegrityError("durable launch estimate is invalid")
    return estimated


def _lease_for_run(
    vpc: Any, spec: RunSpec, events: tuple[RunEvent, ...]
) -> tuple[ResourceLease, ...]:
    instance_id = _recorded_instance_id(events)
    if instance_id is None:
        matches = _list_named(vpc, spec)
        if len(matches) != 1:
            return ()
        instance_id = _field(matches[0], "id")
        resource_crn = _field(matches[0], "crn")
    else:
        resource_crn = (
            f"crn:v1:bluemix:public:is:{spec.zone}:a/{spec.account_id}::instance:{instance_id}"
        )
    if not resource_crn:
        return ()
    created_at = min(event.observed_at for event in events)
    return (
        ResourceLease(
            resource_crn=resource_crn,
            service="vpc-vsi",
            run_id=spec.run_id,
            created_at=created_at,
            expires_at=created_at + timedelta(hours=float(spec.max_hours)),
            terminal_policy="delete",
        ),
    )


def run_watchdog_once(
    vpc: Any,
    tagger: Any,
    cos: Any,
    usage_client: Any,
    *,
    account_id: str,
    resource_group_id: str,
    region: str,
    bucket: str,
    runs_prefix: str,
    now: datetime,
) -> None:
    """Reconcile every immutable run spec in the dedicated bucket once."""

    setattr(vpc, "_watch_auth_tagger", tagger)
    failures: list[tuple[str, str]] = []
    for key in _run_spec_keys(cos, bucket, runs_prefix):
        run_id = "unknown"
        try:
            spec = _load_run_spec(cos, bucket, key)
            run_id = str(spec.run_id)
            if (
                spec.account_id != account_id
                or spec.resource_group_id != resource_group_id
                or spec.region != region
                or spec.bucket != bucket
            ):
                raise IntegrityError("RunSpec escapes the watchdog's IBM scope")
            setattr(cos, "_watch_auth_bucket", spec.bucket)
            setattr(cos, "_watch_auth_prefix", spec.prefix.strip("/"))
            setattr(cos, "_watch_auth_spec", spec)
            setattr(cos, "_watch_auth_configuration_verified", False)
            verify_cos_configuration(cos, spec)
            events = load_event_inventory(cos, spec).events
            if not events or events[-1].state == "deleted":
                continue
            estimated_usd = _load_authorization_estimate(cos, spec)
            billing_period = now.astimezone(timezone.utc).strftime("%Y-%m")
            usage = fetch_usage_evidence(
                usage_client,
                spec.account_id,
                billing_period,
                (spec.run_id,),
                now=now,
            )
            leases = _lease_for_run(vpc, spec, events)
            reconcile_expired_resources(
                vpc,
                cos,
                leases,
                events,
                now=now,
                estimated_usd=estimated_usd,
                usage=usage,
                usage_reconciler=lambda: fetch_usage_evidence(
                    usage_client,
                    spec.account_id,
                    billing_period,
                    (),
                    now=now,
                reconciled_run_ids=(spec.run_id,),
                ),
            )
        except Exception as exc:
            failures.append((run_id, type(exc).__name__))
            _LOG.error(
                "watchdog_run_failed %s",
                json.dumps(
                    {"error_type": type(exc).__name__, "run_id": run_id},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
    if failures:
        raise RuntimeError(f"watchdog failed for {len(failures)} run(s)")


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> None:
    """Code Engine entry point using workload identity; no persistent API key."""

    import ibm_boto3
    from ibm_cloud_sdk_core.authenticators import ContainerAuthenticator
    from ibm_platform_services import GlobalTaggingV1, UsageReportsV4
    from ibm_vpc import VpcV1

    account_id = _required_env("WATCH_AUTH_ACCOUNT_ID")
    resource_group_id = _required_env("WATCH_AUTH_RESOURCE_GROUP_ID")
    region = _required_env("WATCH_AUTH_REGION")
    bucket = _required_env("WATCH_AUTH_COS_BUCKET")
    endpoint = _required_env("WATCH_AUTH_COS_ENDPOINT")
    service_instance_id = _required_env("WATCH_AUTH_COS_SERVICE_INSTANCE_ID")
    trusted_profile_id = _required_env("WATCH_AUTH_TRUSTED_PROFILE_ID")
    runs_prefix = os.environ.get("WATCH_AUTH_RUNS_PREFIX", "runs").strip("/")
    if not runs_prefix:
        raise RuntimeError("WATCH_AUTH_RUNS_PREFIX must be non-empty")

    logging.basicConfig(level=logging.INFO)
    auth = ContainerAuthenticator(iam_profile_id=trusted_profile_id)
    vpc = VpcV1(authenticator=auth)
    vpc.set_service_url(f"https://{region}.iaas.cloud.ibm.com/v1")
    tagger = GlobalTaggingV1(authenticator=auth)
    usage = UsageReportsV4(authenticator=auth)
    cos = ibm_boto3.client(
        "s3",
        endpoint_url=endpoint,
        ibm_service_instance_id=service_instance_id,
        token_manager=auth.token_manager,
    )
    run_watchdog_once(
        vpc,
        tagger,
        cos,
        usage,
        account_id=account_id,
        resource_group_id=resource_group_id,
        region=region,
        bucket=bucket,
        runs_prefix=runs_prefix,
        now=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    main()
