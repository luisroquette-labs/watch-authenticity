"""Load runtime-only IBM billing evidence without inferring financial values."""

from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from infra.ibm_types import (
    CreditEvidence,
    PostPromoEntry,
    PostPromoPlan,
    PriceLine,
    PricingEvidence,
    RunSpec,
    UsageEvidence,
)

if TYPE_CHECKING:
    from ibm_platform_services import UsageReportsV4

_MAX_AGE = timedelta(hours=24)
_PERSISTENT_SERVICES = frozenset(
    {"cos-storage", "container-registry-storage", "logs-monitoring"}
)
_SERVICE_PLANS = {
    "vpc-vsi": frozenset({"gx3-16x80x1l4"}),
    "vpc-boot-volume": frozenset({"standard"}),
    "vpc-public-gateway": frozenset({"standard"}),
    "cos-storage": frozenset({"standard"}),
    "cos-operations": frozenset({"standard"}),
    "container-registry-storage": frozenset({"standard"}),
    "code-engine": frozenset({"standard"}),
    "logs-monitoring": frozenset({"standard"}),
}


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _parse_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from exc
    _require_aware(parsed, name)
    return parsed


def _fresh(observed_at: datetime, now: datetime) -> None:
    _require_aware(now, "now")
    if observed_at > now:
        raise ValueError("financial evidence timestamp is in the future")
    if now - observed_at > _MAX_AGE:
        raise ValueError("financial evidence is stale")


def _decimal_string(value: object, name: str) -> Decimal:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal string") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name} must be a finite non-negative decimal string")
    return parsed


def _require_exact_keys(record: dict[str, Any], expected: set[str]) -> None:
    actual = set(record)
    if actual != expected:
        raise ValueError(
            f"record schema fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _is_runtime_path(path: Path) -> bool:
    parts = path.resolve().parts
    return any(parts[index : index + 3] == (".runtime", "ibm", "evidence") for index in range(len(parts) - 2))


def _load_record(record_path: Path) -> tuple[dict[str, Any], bytes]:
    if not _is_runtime_path(record_path):
        raise ValueError("evidence record must be under .runtime/ibm/evidence")
    try:
        raw = record_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("evidence record is missing") from exc
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("evidence record must be valid JSON") from exc
    if not isinstance(record, dict):
        raise ValueError("evidence record must be a JSON object")
    if record.get("schema_version") != 1:
        raise ValueError("unsupported evidence schema_version")
    return record, raw


def _verified_artifact(
    record_path: Path,
    artifact_path: Path,
    expected_sha256: object,
) -> str:
    if record_path.parent.resolve() != artifact_path.parent.resolve():
        raise ValueError("authoritative artifact must be beside its evidence record")
    try:
        content = artifact_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("authoritative artifact is missing") from exc
    actual = hashlib.sha256(content).hexdigest()
    if not isinstance(expected_sha256, str) or expected_sha256 != actual:
        raise ValueError("authoritative artifact hash mismatch")
    return actual


def _bound_evidence_hash(record_bytes: bytes, artifact_sha256: str) -> str:
    return hashlib.sha256(record_bytes + b"\0" + artifact_sha256.encode("ascii")).hexdigest()


def run_spec_sha256(spec: RunSpec) -> str:
    """Return a stable digest of every immutable RunSpec field."""

    values: dict[str, object] = {}
    for field in fields(spec):
        value = getattr(spec, field.name)
        if isinstance(value, (Decimal, UUID)):
            value = str(value)
        values[field.name] = value
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_credit_evidence(
    record_path: Path,
    console_capture_path: Path,
    *,
    now: datetime,
) -> CreditEvidence:
    record, raw = _load_record(record_path)
    _require_exact_keys(
        record,
        {
            "schema_version",
            "observed_at",
            "account_id",
            "currency",
            "remaining_usd",
            "expires_at",
            "eligible_service_plans",
            "console_capture_sha256",
        },
    )
    artifact_hash = _verified_artifact(
        record_path, console_capture_path, record["console_capture_sha256"]
    )
    observed_at = _parse_datetime(record["observed_at"], "observed_at")
    _fresh(observed_at, now)
    if record["currency"] != "USD":
        raise ValueError("credit currency must be USD")
    account_id = record["account_id"]
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("account_id must be non-empty")
    raw_plans = record["eligible_service_plans"]
    if not isinstance(raw_plans, list) or not raw_plans:
        raise ValueError("eligible_service_plans must be a non-empty list")
    plans: list[tuple[str, str, str, str]] = []
    for item in raw_plans:
        if not isinstance(item, dict):
            raise ValueError("eligible_service_plans items must be objects")
        _require_exact_keys(item, {"service", "plan", "sku_id", "region"})
        service = item["service"]
        plan = item["plan"]
        sku_id = item["sku_id"]
        region = item["region"]
        if (
            not isinstance(service, str)
            or service not in _SERVICE_PLANS
            or not isinstance(plan, str)
            or plan not in _SERVICE_PLANS[service]
        ):
            raise ValueError(f"unknown service plan: {service!r}/{plan!r}")
        if not isinstance(sku_id, str) or not sku_id.strip():
            raise ValueError("eligible SKU ID must be non-empty")
        if not isinstance(region, str) or not region.strip():
            raise ValueError("eligible SKU region must be non-empty")
        plans.append((service, plan, sku_id, region))
    return CreditEvidence(
        account_id=account_id,
        observed_at=observed_at,
        remaining_usd=_decimal_string(record["remaining_usd"], "remaining_usd"),
        expires_at=_parse_datetime(record["expires_at"], "expires_at"),
        eligible_service_plans=tuple(plans),
        evidence_sha256=_bound_evidence_hash(raw, artifact_hash),
        provider_verified=False,
    )


def load_pricing_evidence(
    record_path: Path,
    quote_capture_path: Path,
    spec: RunSpec,
    *,
    now: datetime,
) -> PricingEvidence:
    record, raw = _load_record(record_path)
    _require_exact_keys(
        record,
        {
            "schema_version",
            "observed_at",
            "account_id",
            "currency",
            "run_spec_sha256",
            "lines",
            "quote_capture_sha256",
        },
    )
    artifact_hash = _verified_artifact(
        record_path, quote_capture_path, record["quote_capture_sha256"]
    )
    observed_at = _parse_datetime(record["observed_at"], "observed_at")
    _fresh(observed_at, now)
    if record["currency"] != "USD":
        raise ValueError("pricing currency must be USD")
    account_id = record["account_id"]
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("pricing account_id must be non-empty")
    if account_id != spec.account_id:
        raise ValueError("pricing evidence account differs from run account")
    expected_spec_hash = run_spec_sha256(spec)
    if record["run_spec_sha256"] != expected_spec_hash:
        raise ValueError("pricing evidence is bound to a different run spec")
    raw_lines = record["lines"]
    if not isinstance(raw_lines, list) or not raw_lines:
        raise ValueError("pricing lines must be a non-empty list")
    lines = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            raise ValueError("each pricing line must be an object")
        _require_exact_keys(
            raw_line,
            {
                "service",
                "plan",
                "sku_id",
                "region",
                "unit",
                "estimated_quantity",
                "unit_price_usd",
            },
        )
        service = raw_line["service"]
        plan = raw_line["plan"]
        if not isinstance(service, str) or service not in _SERVICE_PLANS:
            raise ValueError(f"unknown metered service: {service!r}")
        if not isinstance(plan, str) or plan not in _SERVICE_PLANS[service]:
            raise ValueError(f"unknown service plan for {service}: {plan!r}")
        if not isinstance(raw_line["unit"], str):
            raise TypeError("pricing unit must be a string")
        lines.append(
            PriceLine(
                service=service,
                plan=plan,
                sku_id=raw_line["sku_id"],
                region=raw_line["region"],
                unit=raw_line["unit"],
                estimated_quantity=_decimal_string(
                    raw_line["estimated_quantity"], "estimated_quantity"
                ),
                unit_price_usd=_decimal_string(raw_line["unit_price_usd"], "unit_price_usd"),
            )
        )
    return PricingEvidence(
        account_id=account_id,
        observed_at=observed_at,
        currency="USD",
        run_spec_sha256=expected_spec_hash,
        lines=tuple(lines),
        evidence_sha256=_bound_evidence_hash(raw, artifact_hash),
        provider_verified=False,
    )


def load_post_promo_plan(
    record_path: Path,
    authorization_artifact: Path,
    *,
    now: datetime,
) -> PostPromoPlan:
    record, raw = _load_record(record_path)
    _require_exact_keys(
        record,
        {
            "schema_version",
            "observed_at",
            "account_id",
            "promo_expires_at",
            "entries",
            "authorization_capture_sha256",
        },
    )
    authorization_hash = _verified_artifact(
        record_path, authorization_artifact, record["authorization_capture_sha256"]
    )
    observed_at = _parse_datetime(record["observed_at"], "observed_at")
    _fresh(observed_at, now)
    account_id = record["account_id"]
    if not isinstance(account_id, str) or not account_id.strip():
        raise ValueError("post-promo account_id must be non-empty")
    promo_expires_at = _parse_datetime(record["promo_expires_at"], "promo_expires_at")
    latest_deadline = promo_expires_at - timedelta(hours=48)
    raw_entries = record["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("post-promo entries must be a list")
    entries = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("each post-promo entry must be an object")
        _require_exact_keys(
            raw_entry,
            {
                "service",
                "export_destination_sha256",
                "delete_or_downgrade_deadline",
            },
        )
        deadline = _parse_datetime(
            raw_entry["delete_or_downgrade_deadline"], "delete_or_downgrade_deadline"
        )
        if deadline > latest_deadline:
            raise ValueError("post-promo deadline must be at least 48 hours before expiry")
        entries.append(
            PostPromoEntry(
                service=raw_entry["service"],
                export_destination_sha256=raw_entry["export_destination_sha256"],
                delete_or_downgrade_deadline=deadline,
            )
        )
    if {entry.service for entry in entries} != _PERSISTENT_SERVICES:
        raise ValueError("post-promo plan must cover exactly the persistent metered services")
    return PostPromoPlan(
        account_id=account_id,
        observed_at=observed_at,
        promo_expires_at=promo_expires_at,
        entries=tuple(entries),
        owner_authorization_sha256=authorization_hash,
        evidence_sha256=_bound_evidence_hash(raw, authorization_hash),
    )


def _usage_decimal(value: object, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"{name} must be numeric")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def fetch_usage_evidence(
    client: UsageReportsV4,
    account_id: str,
    billing_period: str,
    pending_run_ids: tuple[UUID, ...],
    *,
    now: datetime,
) -> UsageEvidence:
    """Fetch and normalize every page of an IBM Usage Reports response."""

    _require_aware(now, "now")
    try:
        billing_period_start = datetime.strptime(billing_period, "%Y-%m").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError("billing_period must use YYYY-MM") from exc
    if billing_period_start.strftime("%Y-%m") != billing_period:
        raise ValueError("billing_period must use YYYY-MM")
    if billing_period != now.astimezone(timezone.utc).strftime("%Y-%m"):
        raise ValueError("billing_period must be the current month")

    pages: list[dict[str, Any]] = []
    resource_ids: set[str] = set()
    total = Decimal("0")
    start: str | None = None
    seen_offsets: set[str] = set()
    while True:
        kwargs: dict[str, object] = {"limit": 200}
        if start is not None:
            kwargs["start"] = start
        response = client.get_resource_usage_account(account_id, billing_period, **kwargs)
        result = response.get_result()
        if not isinstance(result, dict):
            raise ValueError("usage response must be an object")
        pages.append(result)
        resources = result.get("resources")
        count = result.get("count")
        if not isinstance(resources, list) or not isinstance(count, int) or count != len(resources):
            raise ValueError("incomplete billing period response")
        for resource in resources:
            if not isinstance(resource, dict):
                raise ValueError("usage resource must be an object")
            if resource.get("account_id") != account_id:
                raise ValueError("usage response account mismatch")
            if resource.get("currency_code") != "USD":
                raise ValueError("usage response currency must be USD")
            if resource.get("month") != billing_period:
                raise ValueError("usage response billing period mismatch")
            pending = resource.get("pending")
            if pending is True:
                raise ValueError("incomplete billing period response has pending usage")
            if pending not in {None, False}:
                raise ValueError("usage pending flag is invalid")
            resource_id = resource.get("resource_instance_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("usage resource_instance_id is missing")
            if resource_id in resource_ids:
                raise ValueError("duplicate usage resource")
            resource_ids.add(resource_id)
            metrics = resource.get("usage")
            if not isinstance(metrics, list):
                raise ValueError("usage metrics are missing")
            for metric in metrics:
                if not isinstance(metric, dict) or "cost" not in metric:
                    raise ValueError("usage metric cost is missing")
                total += _usage_decimal(metric["cost"], "usage metric cost")

        next_page = result.get("next")
        if next_page is None:
            break
        if not isinstance(next_page, dict):
            raise ValueError("usage pagination gap")
        offset = next_page.get("offset")
        if not isinstance(offset, str):
            raise ValueError("usage pagination gap")
        start = offset
        if not start or start in seen_offsets:
            raise ValueError("usage pagination gap")
        seen_offsets.add(start)

    raw_hash = hashlib.sha256(
        json.dumps(
            {"account_id": account_id, "billing_period": billing_period, "pages": pages},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return UsageEvidence(
        account_id=account_id,
        observed_at=now,
        billing_period_start=billing_period_start,
        observed_usage_usd=total,
        resource_count=len(resource_ids),
        unreconciled_run_ids=pending_run_ids,
        evidence_sha256=raw_hash,
        provider_verified=True,
    )
