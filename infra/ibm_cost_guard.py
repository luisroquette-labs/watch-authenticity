"""Pure fail-closed authorization for one promotional-credit IBM run."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import (
    AuthorizationDecision,
    CreditEvidence,
    PostPromoPlan,
    PricingEvidence,
    RunEvent,
    RunSpec,
    UsageEvidence,
)

REQUIRED_METERED_LINES = frozenset(
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
PERSISTENT_METERED_SERVICES = frozenset(
    {"cos-storage", "container-registry-storage", "logs-monitoring"}
)
SAFETY_RESERVE_USD = Decimal("500.00")
_MAX_EVIDENCE_AGE = timedelta(hours=24)
_EXPECTED_UNITS = {
    "vpc-vsi": "instance-hour",
    "vpc-boot-volume": "gb-hour",
    "vpc-public-gateway": "gateway-hour",
    "cos-storage": "gb-month",
    "cos-operations": "request",
    "container-registry-storage": "gb-month",
    "code-engine": "job-hour",
    "logs-monitoring": "gb",
}


def _timestamp_reasons(name: str, observed_at: datetime, now: datetime) -> list[str]:
    if observed_at > now:
        return [f"{name} evidence timestamp is in the future"]
    if now - observed_at > _MAX_EVIDENCE_AGE:
        return [f"{name} evidence is stale"]
    return []


def _latest_prior_events(events: tuple[RunEvent, ...], spec: RunSpec) -> dict[UUID, RunEvent]:
    latest: dict[UUID, RunEvent] = {}
    for event in events:
        if event.run_id == spec.run_id:
            continue
        current = latest.get(event.run_id)
        if current is None or event.sequence > current.sequence:
            latest[event.run_id] = event
    return latest


def _prior_event_chain_is_forked(events: tuple[RunEvent, ...], spec: RunSpec) -> bool:
    branches: dict[tuple[UUID, int], set[RunEvent]] = {}
    for event in events:
        if event.run_id != spec.run_id:
            branches.setdefault((event.run_id, event.sequence), set()).add(event)
    return any(len(values) > 1 for values in branches.values())


def authorize_run(
    credit: CreditEvidence,
    usage: UsageEvidence,
    pricing: PricingEvidence,
    post_promo: PostPromoPlan,
    spec: RunSpec,
    events: tuple[RunEvent, ...],
    *,
    now: datetime,
) -> AuthorizationDecision:
    """Return every refusal reason; this function never contacts IBM or provisions."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    reasons = []
    reasons.extend(_timestamp_reasons("credit", credit.observed_at, now))
    reasons.extend(_timestamp_reasons("pricing", pricing.observed_at, now))
    reasons.extend(_timestamp_reasons("usage", usage.observed_at, now))
    reasons.extend(_timestamp_reasons("post-promo", post_promo.observed_at, now))
    if credit.observed_at < usage.observed_at:
        reasons.append("credit evidence predates usage evidence")
    evidence_accounts = {
        credit.account_id,
        usage.account_id,
        pricing.account_id,
        post_promo.account_id,
    }
    if evidence_accounts != {spec.account_id}:
        reasons.append("financial evidence accounts differ from run account")
    current_month_start = now.astimezone(timezone.utc).replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    if usage.billing_period_start.astimezone(timezone.utc) != current_month_start:
        reasons.append("usage billing period is not the current month")

    services = [line.service for line in pricing.lines]
    counts = Counter(services)
    for service in sorted(service for service, count in counts.items() if count > 1):
        reasons.append(f"duplicate metered service: {service}")
    for service in sorted(REQUIRED_METERED_LINES - set(services)):
        reasons.append(f"missing required metered service: {service}")
    for service in sorted(set(services) - REQUIRED_METERED_LINES):
        reasons.append(f"unexpected metered service: {service}")

    if any(
        line.plan != (spec.profile if line.service == "vpc-vsi" else "standard")
        for line in pricing.lines
        if line.service in REQUIRED_METERED_LINES
    ):
        reasons.append("pricing includes an unknown service plan")
    eligible_pairs = frozenset(credit.eligible_service_plans)
    if any((line.service, line.plan) not in eligible_pairs for line in pricing.lines):
        reasons.append("pricing includes an ineligible service plan")
    if pricing.run_spec_sha256 != run_spec_sha256(spec):
        reasons.append("pricing is bound to a different run spec")
    for line in pricing.lines:
        expected_unit = _EXPECTED_UNITS.get(line.service)
        if expected_unit is not None and line.unit != expected_unit:
            reasons.append(f"pricing unit is invalid for {line.service}")
        if line.estimated_quantity <= 0:
            reasons.append(f"estimated quantity must be positive for {line.service}")
    quantities = {line.service: line.estimated_quantity for line in pricing.lines}
    minimum_quantities = {
        "vpc-vsi": spec.max_hours,
        "vpc-boot-volume": spec.boot_volume_gb * spec.max_hours,
        "vpc-public-gateway": spec.max_hours,
        "cos-storage": spec.cos_storage_gb_month,
        "cos-operations": spec.cos_operations,
        "container-registry-storage": spec.registry_storage_gb_month,
        "code-engine": spec.code_engine_job_hours,
        "logs-monitoring": spec.logs_monitoring_gb,
    }
    for service, minimum in minimum_quantities.items():
        if quantities.get(service, Decimal("0")) < minimum:
            if service in {"vpc-vsi", "vpc-public-gateway"}:
                reasons.append(f"{service} estimate does not cover max_hours")
            else:
                reasons.append(f"{service} estimate does not cover RunSpec minimum")

    if usage.unreconciled_run_ids:
        reasons.append("prior run usage is unreconciled")
    if _prior_event_chain_is_forked(events, spec):
        reasons.append("prior run event chain is forked")
    if any(event.state != "deleted" for event in _latest_prior_events(events, spec).values()):
        reasons.append("prior run has no terminal deletion event")

    persistent_entries = {entry.service: entry for entry in post_promo.entries}
    if set(persistent_entries) != PERSISTENT_METERED_SERVICES:
        reasons.append("post-promo plan does not cover exactly the persistent metered services")
    deadline = credit.expires_at - timedelta(hours=48)
    if post_promo.promo_expires_at != credit.expires_at:
        reasons.append("post-promo plan is bound to a different promo expiry")
    if any(entry.delete_or_downgrade_deadline > deadline for entry in post_promo.entries):
        reasons.append("post-promo deadline is less than 48 hours before promo expiry")
    if any(entry.delete_or_downgrade_deadline < now for entry in post_promo.entries):
        reasons.append("post-promo plan deadline is already overdue")

    run_seconds = int(spec.max_hours * Decimal("3600"))
    if now + timedelta(seconds=run_seconds, hours=48) > credit.expires_at:
        reasons.append("run would finish less than 48 hours before promo expiry")

    estimated = sum(
        (line.estimated_quantity * line.unit_price_usd for line in pricing.lines),
        start=Decimal("0"),
    )
    raw_usable = credit.remaining_usd - SAFETY_RESERVE_USD
    usable = max(raw_usable, Decimal("0"))
    if raw_usable < 0:
        reasons.append("credit does not cover the USD 500.00 safety reserve")
    if estimated > raw_usable:
        reasons.append("estimate exceeds credit after USD 500.00 safety reserve")

    evidence_hashes_with_duplicates = (
        credit.evidence_sha256,
        usage.evidence_sha256,
        pricing.evidence_sha256,
        post_promo.owner_authorization_sha256,
        post_promo.evidence_sha256,
    )
    if len(set(evidence_hashes_with_duplicates)) != len(evidence_hashes_with_duplicates):
        reasons.append("evidence hashes are not distinct")
    evidence_hashes = tuple(dict.fromkeys(evidence_hashes_with_duplicates))
    return AuthorizationDecision(
        allowed=not reasons,
        reasons=tuple(reasons),
        estimated_usd=estimated,
        usable_credit_usd=usable,
        evidence_hashes=evidence_hashes,
        run_spec_sha256=run_spec_sha256(spec),
    )
