"""Default-deny source policy and rights authorization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from acquisition.types import (
    RightsRecord,
    RightsStatus,
    SourceAccessRecord,
    SourceAccessStatus,
    SourcePolicy,
)

_DENY_ALL_HASH = "0" * 64


@dataclass(frozen=True)
class AcquisitionDecision:
    may_download: bool
    may_train: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DiscoveryDecision:
    allowed: bool
    reasons: tuple[str, ...]


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON string array")
    return tuple(value)


def _evidence_hash(values: Mapping[str, Any], field: str) -> str:
    payload = {key: value for key, value in values.items() if key != field}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source_policies(path: Path) -> dict[str, SourcePolicy]:
    """Load and validate the versioned source registry."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load source policies: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("source policy registry must be a JSON array")

    policies: dict[str, SourcePolicy] = {}
    policy_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each source policy must be a JSON object")
        values = dict(entry)
        if values.get("policy_evidence_sha256") != _evidence_hash(
            values, "policy_evidence_sha256"
        ):
            raise ValueError("policy_evidence_sha256 does not match policy content")
        values["allowed_methods"] = _string_tuple(values.get("allowed_methods"), "allowed_methods")
        values["required_rights_purposes"] = _string_tuple(
            values.get("required_rights_purposes"), "required_rights_purposes"
        )
        try:
            policy = SourcePolicy(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid source policy: {exc}") from exc
        source = policy.source.strip().lower()
        if source in policies:
            raise ValueError(f"duplicate source policy: {source}")
        if policy.policy_id in policy_ids:
            raise ValueError(f"duplicate policy_id: {policy.policy_id}")
        policies[source] = policy
        policy_ids.add(policy.policy_id)
    return policies


def load_source_access_registry(path: Path) -> dict[str, SourceAccessRecord]:
    """Load the current, expiring evidence for source discovery access."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load source access registry: {exc}") from exc
    if not isinstance(raw, list):
        raise ValueError("source access registry must be a JSON array")

    records: dict[str, SourceAccessRecord] = {}
    record_ids: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("each source access record must be a JSON object")
        values = dict(entry)
        if values.get("evidence_sha256") != _evidence_hash(values, "evidence_sha256"):
            raise ValueError("source access evidence_sha256 does not match record content")
        values["status"] = _access_status(values.get("status"))
        values["verified_at"] = _timestamp(values.get("verified_at"), "verified_at")
        values["expires_at"] = _timestamp(values.get("expires_at"), "expires_at")
        values["allowed_image_hosts"] = _string_tuple(
            values.get("allowed_image_hosts"), "allowed_image_hosts"
        )
        try:
            record = SourceAccessRecord(**values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid source access record: {exc}") from exc
        source = record.source.strip().lower()
        if source in records:
            raise ValueError(f"duplicate source access record: {source}")
        if record.access_record_id in record_ids:
            raise ValueError(f"duplicate access_record_id: {record.access_record_id}")
        records[source] = record
        record_ids.add(record.access_record_id)
    return records


def _access_status(value: object) -> SourceAccessStatus:
    try:
        return SourceAccessStatus(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("source access status is invalid") from exc


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 string") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp


def get_source_policy(
    policies: Mapping[str, SourcePolicy], source: str
) -> SourcePolicy:
    """Return an immutable deny-all policy when the source is absent."""
    normalized = source.strip().lower()
    policy = policies.get(normalized)
    if policy is not None:
        return policy
    return SourcePolicy(
        policy_id="deny-all-v1",
        source=normalized or "unknown",
        allowed_methods=(),
        may_download=False,
        may_train=False,
        required_rights_purposes=(),
        requires_rightsholder_permission=True,
        policy_evidence_sha256=_DENY_ALL_HASH,
    )


def _denied(reason: str) -> AcquisitionDecision:
    return AcquisitionDecision(False, False, (reason,))


def authorize_discovery(
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
) -> DiscoveryDecision:
    """Authorize metadata discovery separately from image rights."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if access is None:
        return DiscoveryDecision(False, ("source access evidence missing",))
    if access.status is not SourceAccessStatus.ACTIVE:
        return DiscoveryDecision(False, (f"source access {access.status.value}",))
    if now < access.verified_at:
        return DiscoveryDecision(False, ("source access not effective",))
    if now >= access.expires_at:
        return DiscoveryDecision(False, ("source access expired",))
    if access.source.strip().lower() != policy.source.strip().lower():
        return DiscoveryDecision(False, ("source access policy mismatch",))
    if access.acquisition_method not in policy.allowed_methods:
        return DiscoveryDecision(False, ("source access method not allowed",))
    return DiscoveryDecision(True, ())


def authorize_candidate(
    policy: SourcePolicy,
    rights: RightsRecord | None,
    *,
    now: datetime,
) -> AcquisitionDecision:
    """Authorize bytes only when policy, method, purpose and time all agree."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if not policy.may_download:
        return _denied("source policy denies download")
    if rights is None:
        return _denied("rights missing")
    if rights.status is RightsStatus.EXPIRED or now >= rights.expires_at:
        return _denied("rights expired")
    if rights.status is RightsStatus.REVOKED:
        return _denied("rights revoked")
    if rights.status is RightsStatus.BLOCKED:
        return _denied("rights blocked")
    if rights.status is not RightsStatus.ACTIVE:
        return _denied("rights inactive")
    if now < rights.effective_at:
        return _denied("rights not effective")
    if rights.source.strip().lower() != policy.source.strip().lower():
        return _denied("rights source mismatch")
    if rights.acquisition_method not in policy.allowed_methods:
        return _denied("acquisition method not allowed")
    if not set(policy.required_rights_purposes).issubset(rights.allowed_purposes):
        return _denied("rights purpose mismatch")
    if not policy.may_train:
        return AcquisitionDecision(True, False, ("source policy denies training",))
    return AcquisitionDecision(True, True, ())
