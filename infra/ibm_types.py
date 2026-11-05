"""Immutable domain contracts for the IBM Cloud training plane.

This module has no IBM SDK dependency. Provider adapters consume these
validated values at the infrastructure boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from urllib.parse import urlsplit
from uuid import UUID

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_OCI_DIGEST = re.compile(r"^private\.[a-z0-9-]+\.icr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$")
_IBM_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_IBM_CRN = re.compile(r"^crn:v1:[A-Za-z0-9._:/-]+$")

RUN_STATES = frozenset(
    {
        "planned",
        "authorized",
        "creating",
        "running",
        "artifact_sync",
        "succeeded",
        "failed",
        "deleting",
        "deleted",
        "quarantined",
    }
)


def _aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _hash(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")


def _decimal(value: Decimal, field: str, *, allow_zero: bool = True) -> None:
    if not isinstance(value, Decimal):
        raise TypeError(f"{field} must be Decimal")
    if not value.is_finite():
        raise ValueError(f"{field} must be finite")
    if value < 0:
        raise ValueError(f"{field} cannot be negative")
    if not allow_zero and value == 0:
        raise ValueError(f"{field} must be positive")


def _identifier(value: str, field: str) -> None:
    _nonempty(value, field)
    if _IBM_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} is not a valid IBM identifier")


def _crn(value: str, field: str) -> None:
    if not isinstance(value, str) or _IBM_CRN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a valid IBM CRN")


def _uuid_string(value: str, field: str) -> None:
    try:
        UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid UUID") from exc


@dataclass(frozen=True)
class CreditEvidence:
    account_id: str
    observed_at: datetime
    remaining_usd: Decimal
    expires_at: datetime
    eligible_service_plans: tuple[tuple[str, str, str, str], ...]
    evidence_sha256: str
    provider_verified: bool

    def __post_init__(self) -> None:
        _nonempty(self.account_id, "account_id")
        _aware(self.observed_at, "observed_at")
        _aware(self.expires_at, "expires_at")
        _decimal(self.remaining_usd, "remaining_usd")
        if self.expires_at <= self.observed_at:
            raise ValueError("expires_at must be after observed_at")
        if not self.eligible_service_plans:
            raise ValueError("eligible_service_plans must be non-empty")
        for item in self.eligible_service_plans:
            if not isinstance(item, tuple) or len(item) != 4:
                raise TypeError(
                    "eligible_service_plans items must be "
                    "(service, plan, sku_id, region) tuples"
                )
            service, plan, sku_id, region = item
            _nonempty(service, "eligible_service_plans service")
            _nonempty(plan, "eligible_service_plans plan")
            _nonempty(sku_id, "eligible_service_plans sku_id")
            _nonempty(region, "eligible_service_plans region")
        if len(set(self.eligible_service_plans)) != len(self.eligible_service_plans):
            raise ValueError("eligible_service_plans contains duplicate values")
        _hash(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.provider_verified, bool):
            raise TypeError("provider_verified must be bool")


@dataclass(frozen=True)
class UsageEvidence:
    account_id: str
    observed_at: datetime
    billing_period_start: datetime
    observed_usage_usd: Decimal
    resource_count: int
    unreconciled_run_ids: tuple[UUID, ...]
    evidence_sha256: str
    provider_verified: bool

    def __post_init__(self) -> None:
        _nonempty(self.account_id, "account_id")
        _aware(self.observed_at, "observed_at")
        _aware(self.billing_period_start, "billing_period_start")
        if self.billing_period_start > self.observed_at:
            raise ValueError("billing_period_start cannot follow observed_at")
        _decimal(self.observed_usage_usd, "observed_usage_usd")
        if (
            not isinstance(self.resource_count, int)
            or isinstance(self.resource_count, bool)
            or self.resource_count < 0
        ):
            raise ValueError("resource_count must be a non-negative integer")
        if any(not isinstance(run_id, UUID) for run_id in self.unreconciled_run_ids):
            raise TypeError("unreconciled_run_ids must contain UUID values")
        if len(set(self.unreconciled_run_ids)) != len(self.unreconciled_run_ids):
            raise ValueError("unreconciled_run_ids contains duplicate values")
        _hash(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.provider_verified, bool):
            raise TypeError("provider_verified must be bool")


@dataclass(frozen=True)
class PriceLine:
    service: str
    plan: str
    sku_id: str
    region: str
    unit: str
    estimated_quantity: Decimal
    unit_price_usd: Decimal

    def __post_init__(self) -> None:
        _nonempty(self.service, "service")
        _nonempty(self.plan, "plan")
        _nonempty(self.sku_id, "sku_id")
        _nonempty(self.region, "region")
        _nonempty(self.unit, "unit")
        _decimal(self.estimated_quantity, "estimated_quantity")
        _decimal(self.unit_price_usd, "unit_price_usd")


@dataclass(frozen=True)
class PricingEvidence:
    account_id: str
    observed_at: datetime
    currency: str
    run_spec_sha256: str
    lines: tuple[PriceLine, ...]
    evidence_sha256: str
    provider_verified: bool

    def __post_init__(self) -> None:
        _nonempty(self.account_id, "account_id")
        _aware(self.observed_at, "observed_at")
        if self.currency != "USD":
            raise ValueError("currency must be USD")
        _hash(self.run_spec_sha256, "run_spec_sha256")
        if not self.lines:
            raise ValueError("lines must be non-empty")
        keys = [(line.service, line.sku_id, line.region, line.unit) for line in self.lines]
        if len(set(keys)) != len(keys):
            raise ValueError("lines contains duplicate service/plan/unit entries")
        _hash(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.provider_verified, bool):
            raise TypeError("provider_verified must be bool")


@dataclass(frozen=True)
class PostPromoEntry:
    service: str
    export_destination_sha256: str
    delete_or_downgrade_deadline: datetime

    def __post_init__(self) -> None:
        _nonempty(self.service, "service")
        _hash(self.export_destination_sha256, "export_destination_sha256")
        _aware(self.delete_or_downgrade_deadline, "delete_or_downgrade_deadline")


@dataclass(frozen=True)
class PostPromoPlan:
    account_id: str
    observed_at: datetime
    promo_expires_at: datetime
    entries: tuple[PostPromoEntry, ...]
    owner_authorization_sha256: str
    evidence_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.account_id, "account_id")
        _aware(self.observed_at, "observed_at")
        _aware(self.promo_expires_at, "promo_expires_at")
        if self.promo_expires_at <= self.observed_at:
            raise ValueError("promo_expires_at must be after observed_at")
        if not self.entries:
            raise ValueError("entries must be non-empty")
        services = [entry.service for entry in self.entries]
        if len(set(services)) != len(services):
            raise ValueError("entries contains duplicate services")
        _hash(self.owner_authorization_sha256, "owner_authorization_sha256")
        _hash(self.evidence_sha256, "evidence_sha256")


@dataclass(frozen=True)
class RunSpec:
    run_id: UUID
    account_id: str
    code_sha: str
    image_digest: str
    manifest_sha256: str
    splits_sha256: str
    profile: str
    region: str
    zone: str
    cos_endpoint: str
    max_hours: Decimal
    boot_volume_gb: Decimal
    cos_storage_gb_month: Decimal
    cos_operations: Decimal
    registry_storage_gb_month: Decimal
    code_engine_job_hours: Decimal
    logs_monitoring_gb: Decimal
    bucket: str
    prefix: str
    cos_service_instance_id: str
    network_id: str
    subnet_id: str
    public_gateway_id: str
    security_group_id: str
    trusted_profile_id: str
    public_gateway_owned_by_run: bool
    cos_private_access_only: bool
    cos_versioning_enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be UUID")
        _nonempty(self.account_id, "account_id")
        if not isinstance(self.code_sha, str) or _GIT_SHA.fullmatch(self.code_sha) is None:
            raise ValueError("code_sha must be a lowercase 40-character Git SHA")
        if not isinstance(self.image_digest, str) or _OCI_DIGEST.fullmatch(self.image_digest) is None:
            raise ValueError("image_digest must be a private IBM OCI reference pinned with @sha256")
        _hash(self.manifest_sha256, "manifest_sha256")
        _hash(self.splits_sha256, "splits_sha256")
        if self.profile != "gx3-16x80x1l4":
            raise ValueError("profile must be exactly gx3-16x80x1l4")
        for field, value in (
            ("region", self.region),
            ("zone", self.zone),
            ("bucket", self.bucket),
            ("prefix", self.prefix),
        ):
            _nonempty(value, field)
        endpoint = urlsplit(self.cos_endpoint)
        allowed_hosts = {
            f"s3.direct.{self.region}.cloud-object-storage.appdomain.cloud",
            f"s3.private.{self.region}.cloud-object-storage.appdomain.cloud",
        }
        if (
            endpoint.scheme != "https"
            or endpoint.hostname not in allowed_hosts
            or endpoint.port is not None
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path not in {"", "/"}
            or endpoint.query
            or endpoint.fragment
        ):
            raise ValueError("COS endpoint must be an IBM private/direct endpoint for the RunSpec region")
        _decimal(self.max_hours, "max_hours", allow_zero=False)
        if self.max_hours > Decimal("24"):
            raise ValueError("max_hours must be at most 24")
        for dimension_field, dimension_value, minimum in (
            ("boot_volume_gb", self.boot_volume_gb, Decimal("100")),
            ("cos_storage_gb_month", self.cos_storage_gb_month, Decimal("50")),
            ("cos_operations", self.cos_operations, Decimal("1000")),
            ("registry_storage_gb_month", self.registry_storage_gb_month, Decimal("10")),
            ("code_engine_job_hours", self.code_engine_job_hours, Decimal("1")),
            ("logs_monitoring_gb", self.logs_monitoring_gb, Decimal("1")),
        ):
            _decimal(dimension_value, dimension_field, allow_zero=False)
            if dimension_value < minimum:
                raise ValueError(f"{dimension_field} must be at least {minimum}")
        _crn(self.cos_service_instance_id, "cos_service_instance_id")
        if f":a/{self.account_id}:" not in self.cos_service_instance_id:
            raise ValueError("COS service instance CRN must belong to run account")
        if self.prefix.strip("/") != f"runs/{self.run_id}":
            raise ValueError("prefix must be the immutable run prefix")
        if self.cos_private_access_only is not True:
            raise ValueError("COS bucket must block public access")
        if self.cos_versioning_enabled is not True:
            raise ValueError("COS bucket versioning must be enabled")
        for field, value in (
            ("network_id", self.network_id),
            ("subnet_id", self.subnet_id),
            ("public_gateway_id", self.public_gateway_id),
            ("security_group_id", self.security_group_id),
        ):
            _identifier(value, field)
        _uuid_string(self.trusted_profile_id, "trusted_profile_id")
        if self.public_gateway_owned_by_run is not True:
            raise ValueError("Public Gateway must be dedicated to and owned by the run")


@dataclass(frozen=True)
class AuthorizationDecision:
    """Auditable gate result, never a launch capability.

    Local Python code is trusted. Provider responses and persisted evidence are
    not. Task 8 must re-authorize and atomically consume ``nonce`` at launch.
    """

    allowed: bool
    reasons: tuple[str, ...]
    estimated_usd: Decimal
    usable_credit_usd: Decimal
    evidence_hashes: tuple[str, ...]
    run_spec_sha256: str
    authorization_sha256: str
    issued_at: datetime
    expires_at: datetime
    nonce: UUID

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be bool")
        for reason in self.reasons:
            _nonempty(reason, "reason")
        if self.allowed and self.reasons:
            raise ValueError("an allowed decision cannot contain refusal reasons")
        if not self.allowed and not self.reasons:
            raise ValueError("a denied decision must contain at least one reason")
        _decimal(self.estimated_usd, "estimated_usd")
        _decimal(self.usable_credit_usd, "usable_credit_usd")
        if not self.evidence_hashes:
            raise ValueError("evidence_hashes must be non-empty")
        for value in self.evidence_hashes:
            _hash(value, "evidence_hash")
        if len(set(self.evidence_hashes)) != len(self.evidence_hashes):
            raise ValueError("evidence_hashes contains duplicate values")
        _hash(self.run_spec_sha256, "run_spec_sha256")
        _hash(self.authorization_sha256, "authorization_sha256")
        _aware(self.issued_at, "issued_at")
        _aware(self.expires_at, "expires_at")
        if self.expires_at != self.issued_at + timedelta(minutes=5):
            raise ValueError("authorization decision must expire after exactly five minutes")
        if not isinstance(self.nonce, UUID):
            raise TypeError("nonce must be UUID")


@dataclass(frozen=True, init=False)
class AuthorizedRun:
    spec: RunSpec
    estimated_usd: Decimal
    evidence_hashes: tuple[str, ...]

    def __new__(cls) -> AuthorizedRun:
        raise TypeError("AuthorizedRun must be created with from_decision")

    def _validate(self) -> None:
        _decimal(self.estimated_usd, "estimated_usd")
        if not self.evidence_hashes:
            raise ValueError("evidence_hashes must be non-empty")
        for value in self.evidence_hashes:
            _hash(value, "evidence_hash")

    @classmethod
    def from_decision(
        cls,
        spec: RunSpec,
        decision: AuthorizationDecision,
        evidence_hashes: tuple[str, ...],
    ) -> AuthorizedRun:
        del spec, decision, evidence_hashes
        raise RuntimeError(
            "launch authorization requires Task 8 atomic nonce consumption"
        )


@dataclass(frozen=True)
class RunEvent:
    event_id: UUID
    run_id: UUID
    sequence: int
    state: str
    observed_at: datetime
    instance_id: str | None
    lineage_sha256: str
    previous_event_sha256: str | None
    payload_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, UUID) or not isinstance(self.run_id, UUID):
            raise TypeError("event_id and run_id must be UUID values")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if self.state not in RUN_STATES:
            raise ValueError(f"state must be one of {sorted(RUN_STATES)}")
        _aware(self.observed_at, "observed_at")
        if self.instance_id is not None:
            _identifier(self.instance_id, "instance_id")
        _hash(self.lineage_sha256, "lineage_sha256")
        if self.sequence == 0:
            if self.previous_event_sha256 is not None:
                raise ValueError("previous_event_sha256 must be None for sequence zero")
        elif self.previous_event_sha256 is None:
            raise ValueError("previous_event_sha256 is required after sequence zero")
        else:
            _hash(self.previous_event_sha256, "previous_event_sha256")
        _hash(self.payload_sha256, "payload_sha256")


def run_event_sha256(event: RunEvent) -> str:
    """Hash the complete immutable event for previous-event linkage."""

    payload = {
        "event_id": str(event.event_id),
        "run_id": str(event.run_id),
        "sequence": event.sequence,
        "state": event.state,
        "observed_at": event.observed_at.isoformat(),
        "instance_id": event.instance_id,
        "lineage_sha256": event.lineage_sha256,
        "previous_event_sha256": event.previous_event_sha256,
        "payload_sha256": event.payload_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


_RUN_TRANSITIONS = {
    "planned": frozenset({"authorized", "deleting", "quarantined"}),
    "authorized": frozenset({"creating", "deleting", "quarantined"}),
    "creating": frozenset({"running", "failed", "deleting", "quarantined"}),
    "running": frozenset({"artifact_sync", "failed", "deleting", "quarantined"}),
    "artifact_sync": frozenset({"succeeded", "failed", "deleting", "quarantined"}),
    "succeeded": frozenset({"deleting", "quarantined"}),
    "failed": frozenset({"deleting", "quarantined"}),
    "deleting": frozenset({"deleted", "quarantined"}),
    "quarantined": frozenset({"deleting"}),
    "deleted": frozenset(),
}


def validate_run_event_chain(events: tuple[RunEvent, ...], run_id: UUID) -> None:
    """Validate one complete, linear run lifecycle chain."""

    if not events or any(event.run_id != run_id for event in events):
        raise ValueError("event run_id does not match chain")
    by_sequence: dict[int, RunEvent] = {}
    for event in events:
        previous = by_sequence.get(event.sequence)
        if previous is not None and previous != event:
            raise ValueError(f"event chain fork at sequence {event.sequence}")
        by_sequence[event.sequence] = event
    ordered = tuple(by_sequence[index] for index in sorted(by_sequence))
    if tuple(event.sequence for event in ordered) != tuple(range(len(ordered))):
        raise ValueError("event chain sequence is not contiguous from zero")
    if ordered[0].state != "planned":
        raise ValueError("event chain must start in planned state")
    lineage = ordered[0].lineage_sha256
    for previous, current in zip(ordered, ordered[1:]):
        if current.lineage_sha256 != lineage:
            raise ValueError("event chain lineage changed")
        if current.previous_event_sha256 != run_event_sha256(previous):
            raise ValueError("event chain previous hash mismatch")
        if current.state not in _RUN_TRANSITIONS[previous.state]:
            raise ValueError("event chain state transition is invalid")


@dataclass(frozen=True)
class InstanceRef:
    id: str
    name: str
    profile: str
    zone: str
    vpc_id: str
    subnet_id: str
    image_id: str
    status: str

    def __post_init__(self) -> None:
        for field, value in (
            ("id", self.id),
            ("vpc_id", self.vpc_id),
            ("subnet_id", self.subnet_id),
            ("image_id", self.image_id),
        ):
            _identifier(value, field)
        for field, value in (
            ("name", self.name),
            ("zone", self.zone),
            ("status", self.status),
        ):
            _nonempty(value, field)
        if self.profile != "gx3-16x80x1l4":
            raise ValueError("profile must be exactly gx3-16x80x1l4")


@dataclass(frozen=True)
class ObjectRef:
    bucket: str
    key: str
    sha256: str
    length: int
    etag: str

    def __post_init__(self) -> None:
        _nonempty(self.bucket, "bucket")
        _nonempty(self.key, "key")
        _hash(self.sha256, "sha256")
        if not isinstance(self.length, int) or isinstance(self.length, bool) or self.length < 0:
            raise ValueError("length must be a non-negative integer")
        _nonempty(self.etag, "etag")


@dataclass(frozen=True)
class ResourceLease:
    resource_crn: str
    service: str
    run_id: UUID
    created_at: datetime
    expires_at: datetime
    terminal_policy: str

    def __post_init__(self) -> None:
        _crn(self.resource_crn, "resource_crn")
        _nonempty(self.service, "service")
        if not isinstance(self.run_id, UUID):
            raise TypeError("run_id must be UUID")
        _aware(self.created_at, "created_at")
        _aware(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        _nonempty(self.terminal_policy, "terminal_policy")
