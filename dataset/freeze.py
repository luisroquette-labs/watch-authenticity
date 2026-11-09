"""Content-addressed dataset freeze and fail-closed rights exclusion."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from acquisition.source_policy import authorize_candidate
from acquisition.types import (
    DataDispositionReport,
    DatasetFreeze,
    FreezeResult,
    ImpactedArtifactReport,
    RightsRecord,
    SourcePolicy,
    TrainingRecord,
)
from dataset.readiness import evaluate_readiness
from dataset.splits import DatasetSplits
from infra.ibm_types import ObjectRef


class FrozenDatasetRefused(PermissionError):
    pass


@dataclass(frozen=True)
class CandidateObjectInventory:
    raw: tuple[ObjectRef, ...]
    derivatives: tuple[ObjectRef, ...]
    evidence: tuple[ObjectRef, ...]

    def __post_init__(self) -> None:
        if not self.raw or not self.derivatives or not self.evidence:
            raise ValueError("candidate inventory requires raw, derivative and evidence objects")
        if not all(
            isinstance(item, ObjectRef)
            for item in self.raw + self.derivatives + self.evidence
        ):
            raise ValueError("candidate inventory must contain ObjectRef values")


def require_launchable_freeze(freeze: DatasetFreeze) -> None:
    if freeze.blocked:
        raise FrozenDatasetRefused("dataset freeze is blocked")


def require_frozen_asset_read(freeze: DatasetFreeze, asset_id: str) -> None:
    require_launchable_freeze(freeze)
    if asset_id not in freeze.eligible_asset_ids:
        raise FrozenDatasetRefused("asset is not eligible in this freeze")


def freeze_dataset(
    records: Sequence[TrainingRecord],
    splits: DatasetSplits,
    policies: Mapping[str, SourcePolicy],
    rights: Mapping[str, RightsRecord],
    *,
    previous_freeze: DatasetFreeze | None,
    published_artifact_hashes: Collection[str],
    now: datetime,
    object_inventory: Mapping[str, CandidateObjectInventory],
    artifact_lineage: Mapping[str, Collection[str]],
) -> FreezeResult:
    """Freeze eligible assets while reporting every rights-driven removal."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    split_by_asset = splits.split_by_asset
    asset_ids = [record.asset_id for record in records]
    if len(asset_ids) != len(set(asset_ids)) or set(split_by_asset) != set(asset_ids):
        raise ValueError("split assignments must match records")

    exclusions: dict[str, str] = {}
    dispositions: list[DataDispositionReport] = []
    disposed_candidates: set[str] = set()
    for record in records:
        rights_record = rights.get(record.rights_record_id)
        policy = _policy_for_rights(policies, rights_record)
        reasons = (
            ("rights record is missing",)
            if rights_record is None
            else authorize_candidate(policy, rights_record, now=now).reasons
        )
        if reasons:
            reason = "; ".join(reasons)
            exclusions[record.asset_id] = reason
            if (
                rights_record is not None
                and rights_record.required_disposition != "retain"
                and record.candidate_id not in disposed_candidates
            ):
                inventory = object_inventory.get(record.candidate_id)
                if inventory is None:
                    raise ValueError("revoked candidate object inventory is missing")
                dispositions.append(
                    DataDispositionReport(
                        raw_object_refs=inventory.raw,
                        derivative_object_refs=inventory.derivatives,
                        evidence_object_refs=inventory.evidence,
                        required_action=rights_record.required_disposition,
                        deadline=rights_record.disposition_deadline,
                        status="blocked" if rights_record.required_disposition == "delete" else "pending",
                    )
                )
                disposed_candidates.add(record.candidate_id)

    eligible = tuple(sorted(record.asset_id for record in records if record.asset_id not in exclusions))
    excluded = tuple(sorted(exclusions))
    eligible_records = [record for record in records if record.asset_id in eligible]
    references = {record.reference_id for record in records}
    if len(references) != 1:
        raise ValueError("one freeze must contain exactly one watch reference")
    eligible_splits = DatasetSplits(
        assignments=tuple(item for item in splits.assignments if item[0] in eligible),
        temporal_cutoff=splits.temporal_cutoff,
        seed=splits.seed,
    )
    readiness = evaluate_readiness(
        eligible_records,
        next(iter(references)),
        splits=eligible_splits,
    )
    manifest_sha256 = _digest([_record_json(record) for record in eligible_records])
    split_sha256 = _digest(
        {
            "assignments": [item for item in splits.assignments if item[0] in eligible],
            "seed": splits.seed,
            "temporal_cutoff": (
                None if splits.temporal_cutoff is None else splits.temporal_cutoff.isoformat()
            ),
        }
    )
    policy_sha256 = _digest([_policy_json(policy) for policy in sorted(policies.values(), key=lambda item: item.policy_id)])
    rights_sha256 = _digest([_rights_json(item) for item in sorted(rights.values(), key=lambda value: value.rights_record_id)])
    readiness_sha256 = _digest(asdict(readiness))
    reason_counts = tuple(sorted(Counter(exclusions.values()).items()))
    freeze = DatasetFreeze(
        manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        policy_set_sha256=policy_sha256,
        eligible_asset_ids=eligible,
        excluded_asset_ids=excluded,
        exclusion_reason_counts=reason_counts,
        readiness_sha256=readiness_sha256,
        rights_sha256=rights_sha256,
        exclusions=tuple(sorted(exclusions.items())),
        blocked=not readiness.ready,
    )
    impacted = _impacted_report(
        previous_freeze,
        published_artifact_hashes,
        artifact_lineage,
        exclusions,
        now,
    )
    return FreezeResult(freeze, impacted, tuple(dispositions))


def _policy_for_rights(
    policies: Mapping[str, SourcePolicy], rights: RightsRecord | None
) -> SourcePolicy:
    if rights is not None:
        matches = [policy for policy in policies.values() if policy.source == rights.source]
        if len(matches) == 1:
            return matches[0]
    return SourcePolicy(
        policy_id="deny-missing-policy",
        source="missing",
        allowed_methods=(),
        may_download=False,
        may_train=False,
        required_rights_purposes=(),
        requires_rightsholder_permission=True,
        policy_evidence_sha256="0" * 64,
    )


def _impacted_report(
    previous: DatasetFreeze | None,
    published: Collection[str],
    lineage: Mapping[str, Collection[str]],
    exclusions: Mapping[str, str],
    now: datetime,
) -> ImpactedArtifactReport | None:
    if previous is None or not exclusions:
        return None
    impacted = tuple(
        sorted(
            artifact
            for artifact in published
            if previous.manifest_sha256 in set(lineage.get(artifact, ()))
        )
    )
    if not impacted:
        return None
    return ImpactedArtifactReport(
        impacted_artifact_hashes=impacted,
        reason="rights changed for assets in the previous freeze",
        generated_at=now,
    )


def _record_json(record: TrainingRecord) -> dict[str, object]:
    raw = asdict(record)
    raw["angle"] = None if record.angle is None else record.angle.value
    raw["collected_at"] = record.collected_at.isoformat()
    raw["label"] = record.label.value
    raw["evidence_tier"] = record.evidence_tier.value
    return raw


def _policy_json(policy: SourcePolicy) -> dict[str, object]:
    return asdict(policy)


def _rights_json(rights: RightsRecord) -> dict[str, object]:
    raw = asdict(rights)
    raw["effective_at"] = rights.effective_at.isoformat()
    raw["expires_at"] = rights.expires_at.isoformat()
    raw["disposition_deadline"] = rights.disposition_deadline.isoformat()
    raw["status"] = rights.status.value
    return raw


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
