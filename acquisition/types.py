"""Immutable acquisition, rights, evidence and dataset-freeze contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from urllib.parse import urlsplit

from core.types import Grade, Label
from infra.ibm_types import ObjectRef

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PHASH = re.compile(r"[0-9a-f]+")
_DISPOSITIONS = {"retain", "quarantine", "delete"}
_DISPOSITION_STATES = {"pending", "completed", "blocked"}
_HOSTNAME = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def _nonempty(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a non-empty single-line string")


def _sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")


def _aware(value: datetime, field: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _unique_nonempty(values: tuple[str, ...], field: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise ValueError(f"{field} must be a non-empty tuple")
    for value in values:
        _nonempty(value, field)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")


def _hashes(values: tuple[str, ...], field: str, *, allow_empty: bool = False) -> None:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise ValueError(f"{field} must be a non-empty tuple")
    for value in values:
        _sha256(value, field)
    if len(set(values)) != len(values):
        raise ValueError(f"{field} must not contain duplicates")


def _grade(value: int | None, field: str, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required")
        return
    if isinstance(value, bool) or not isinstance(value, int) or value not in set(Grade):
        raise ValueError(f"{field} must be a fake grade from 1 to 4")


def _label_grade(label: Label, grade: int | None, *, fake_grade_required: bool) -> None:
    if not isinstance(label, Label):
        raise ValueError("label must be Label")
    if label is Label.FAKE:
        _grade(grade, "grade", required=fake_grade_required)
    elif grade is not None:
        raise ValueError("only fake labels may carry a grade")


class EvidenceTier(Enum):
    A = "a"
    B = "b"
    C = "c"
    INCONCLUSIVE = "inconclusive"

    @property
    def training_eligible(self) -> bool:
        return self in {EvidenceTier.A, EvidenceTier.B}


class RightsStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    BLOCKED = "blocked"


class SourceAccessStatus(Enum):
    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    BLOCKED = "blocked"


class ReviewStatus(Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class Angle(Enum):
    DIAL = "dial"
    CYCLOPS = "cyclops"
    REHAUT = "rehaut"
    CROWN = "crown"
    CLASP = "clasp"
    CASEBACK = "caseback"


@dataclass(frozen=True)
class SourcePolicy:
    policy_id: str
    source: str
    allowed_methods: tuple[str, ...]
    may_download: bool
    may_train: bool
    required_rights_purposes: tuple[str, ...]
    requires_rightsholder_permission: bool
    policy_evidence_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.policy_id, "policy_id")
        _nonempty(self.source, "source")
        _unique_nonempty(self.allowed_methods, "allowed_methods", allow_empty=True)
        _unique_nonempty(
            self.required_rights_purposes,
            "required_rights_purposes",
            allow_empty=True,
        )
        _sha256(self.policy_evidence_sha256, "policy_evidence_sha256")
        for field, value in (
            ("may_download", self.may_download),
            ("may_train", self.may_train),
            ("requires_rightsholder_permission", self.requires_rightsholder_permission),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be bool")
        if self.may_train and not self.may_download:
            raise ValueError("training policy must also allow download")
        if (self.may_download or self.may_train) and (
            not self.allowed_methods or not self.required_rights_purposes
        ):
            raise ValueError("enabled policy requires methods and rights purposes")


@dataclass(frozen=True)
class SourceAccessRecord:
    access_record_id: str
    source: str
    acquisition_method: str
    status: SourceAccessStatus
    evidence_url: str | None
    evidence_sha256: str
    verified_at: datetime
    expires_at: datetime
    allowed_image_hosts: tuple[str, ...]
    requests_per_minute: int | None
    reason: str

    def __post_init__(self) -> None:
        for field in ("access_record_id", "source", "acquisition_method", "reason"):
            _nonempty(getattr(self, field), field)
        if not isinstance(self.status, SourceAccessStatus):
            raise ValueError("status must be SourceAccessStatus")
        _sha256(self.evidence_sha256, "evidence_sha256")
        _aware(self.verified_at, "verified_at")
        _aware(self.expires_at, "expires_at")
        if self.expires_at <= self.verified_at:
            raise ValueError("source access expiry must follow verification")
        _unique_nonempty(self.allowed_image_hosts, "allowed_image_hosts", allow_empty=True)
        for host in self.allowed_image_hosts:
            if (
                host != host.lower()
                or _HOSTNAME.fullmatch(host) is None
                or urlsplit(f"//{host}").hostname != host
                or ":" in host
            ):
                raise ValueError("allowed_image_hosts must contain exact lowercase hostnames")
        if self.requests_per_minute is not None and (
            isinstance(self.requests_per_minute, bool)
            or not isinstance(self.requests_per_minute, int)
            or self.requests_per_minute < 1
        ):
            raise ValueError("requests_per_minute must be a positive integer")
        if self.evidence_url is not None:
            parsed = urlsplit(self.evidence_url)
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError("evidence_url must be an absolute HTTPS URL")
        if self.status is SourceAccessStatus.ACTIVE:
            network_method = self.acquisition_method in {"official-api", "licensed-feed"}
            if self.evidence_url is None or (
                network_method
                and (not self.allowed_image_hosts or self.requests_per_minute is None)
            ):
                raise ValueError("active source access requires real evidence and network limits")


@dataclass(frozen=True)
class RightsRecord:
    rights_record_id: str
    source: str
    acquisition_method: str
    allowed_purposes: tuple[str, ...]
    effective_at: datetime
    expires_at: datetime
    required_disposition: str
    disposition_deadline: datetime
    rightsholder: str
    evidence_sha256: str
    status: RightsStatus

    def __post_init__(self) -> None:
        for field, value in (
            ("rights_record_id", self.rights_record_id),
            ("source", self.source),
            ("acquisition_method", self.acquisition_method),
            ("rightsholder", self.rightsholder),
        ):
            _nonempty(value, field)
        _unique_nonempty(self.allowed_purposes, "allowed_purposes")
        _aware(self.effective_at, "effective_at")
        _aware(self.expires_at, "expires_at")
        _aware(self.disposition_deadline, "disposition_deadline")
        if self.expires_at <= self.effective_at:
            raise ValueError("expires_at must be after effective_at")
        if self.disposition_deadline < self.expires_at:
            raise ValueError("disposition_deadline must not precede expires_at")
        if self.required_disposition not in _DISPOSITIONS:
            raise ValueError("required_disposition must be retain, quarantine or delete")
        _sha256(self.evidence_sha256, "evidence_sha256")
        if not isinstance(self.status, RightsStatus):
            raise ValueError("status must be RightsStatus")


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    acquisition_batch_id: str
    source_policy_id: str
    source_record_id: str
    source_url: str
    collected_at: datetime
    declared_reference: str
    proposed_label: Label | None
    proposed_grade: int | None
    factory: str | None
    price_usd: Decimal | None
    raw_sha256: str
    derivative_sha256: str
    phash: str
    rights_record_id: str

    def __post_init__(self) -> None:
        for field, value in (
            ("candidate_id", self.candidate_id),
            ("acquisition_batch_id", self.acquisition_batch_id),
            ("source_policy_id", self.source_policy_id),
            ("source_record_id", self.source_record_id),
            ("declared_reference", self.declared_reference),
            ("rights_record_id", self.rights_record_id),
        ):
            _nonempty(value, field)
        parsed_url = urlsplit(self.source_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("source_url must be an absolute HTTP(S) URL")
        _aware(self.collected_at, "collected_at")
        if self.proposed_label is not None:
            _label_grade(self.proposed_label, self.proposed_grade, fake_grade_required=False)
        elif self.proposed_grade is not None:
            raise ValueError("proposed_grade requires proposed_label")
        if self.factory is not None:
            _nonempty(self.factory, "factory")
        if self.price_usd is not None and (
            not isinstance(self.price_usd, Decimal)
            or not self.price_usd.is_finite()
            or self.price_usd < 0
        ):
            raise ValueError("price_usd must be a finite non-negative Decimal")
        _sha256(self.raw_sha256, "raw_sha256")
        _sha256(self.derivative_sha256, "derivative_sha256")
        if not isinstance(self.phash, str) or _PHASH.fullmatch(self.phash) is None:
            raise ValueError("phash must be lowercase hexadecimal")


@dataclass(frozen=True)
class ReviewDecision:
    candidate_id: str
    reviewer_id: str
    proposed_label: Label | None
    proposed_grade: int | None
    proposed_tier: EvidenceTier
    evidence_type: str
    supporting_artifact_hashes: tuple[str, ...]
    reviewed_at: datetime
    decision_sha256: str
    candidate_sha256: str
    proposed_reference_id: str
    reviewer_registry_sha256: str
    adjudicator: bool

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        _nonempty(self.reviewer_id, "reviewer_id")
        if self.proposed_label is not None:
            _label_grade(self.proposed_label, self.proposed_grade, fake_grade_required=False)
        elif self.proposed_grade is not None:
            raise ValueError("proposed_grade requires proposed_label")
        if not isinstance(self.proposed_tier, EvidenceTier):
            raise ValueError("proposed_tier must be EvidenceTier")
        _nonempty(self.evidence_type, "evidence_type")
        _hashes(self.supporting_artifact_hashes, "supporting_artifact_hashes")
        _aware(self.reviewed_at, "reviewed_at")
        _sha256(self.decision_sha256, "decision_sha256")
        _sha256(self.candidate_sha256, "candidate_sha256")
        _nonempty(self.proposed_reference_id, "proposed_reference_id")
        _sha256(self.reviewer_registry_sha256, "reviewer_registry_sha256")
        if not isinstance(self.adjudicator, bool):
            raise ValueError("adjudicator must be bool")


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_record_id: str
    candidate_id: str
    candidate_sha256: str
    accepted_label: Label | None
    accepted_grade: int | None
    reference_id: str
    tier: EvidenceTier
    evidence_type: str
    reviewer_ids: tuple[str, ...]
    review_timestamps: tuple[datetime, ...]
    supporting_artifact_hashes: tuple[str, ...]
    review_decision_sha256s: tuple[str, ...]
    decision_sha256: str

    def __post_init__(self) -> None:
        _nonempty(self.evidence_record_id, "evidence_record_id")
        _nonempty(self.candidate_id, "candidate_id")
        _sha256(self.candidate_sha256, "candidate_sha256")
        _nonempty(self.reference_id, "reference_id")
        if not isinstance(self.tier, EvidenceTier):
            raise ValueError("tier must be EvidenceTier")
        if self.accepted_label is None:
            if self.accepted_grade is not None or self.tier.training_eligible:
                raise ValueError("eligible evidence requires an accepted label")
        else:
            _label_grade(
                self.accepted_label,
                self.accepted_grade,
                fake_grade_required=self.tier.training_eligible,
            )
            if self.accepted_label is Label.EXCLUDED and self.tier.training_eligible:
                raise ValueError("excluded evidence cannot be training-eligible")
        _nonempty(self.evidence_type, "evidence_type")
        _unique_nonempty(self.reviewer_ids, "reviewer_ids")
        if len(self.review_timestamps) != len(self.reviewer_ids):
            raise ValueError("review_timestamps must align with reviewer_ids")
        if len(self.review_decision_sha256s) != len(self.reviewer_ids):
            raise ValueError("review decision hashes must align with reviewer_ids")
        for timestamp in self.review_timestamps:
            _aware(timestamp, "review_timestamps")
        _hashes(self.supporting_artifact_hashes, "supporting_artifact_hashes")
        _hashes(self.review_decision_sha256s, "review_decision_sha256s")
        _sha256(self.decision_sha256, "decision_sha256")


@dataclass(frozen=True)
class TrainingRecord:
    asset_id: str
    candidate_id: str
    object_ref: ObjectRef
    reference_id: str
    watch_entity_id: str
    content_cluster_id: str
    seller_group_id: str
    acquisition_batch_id: str
    angle: Angle | None
    collected_at: datetime
    original_sha256: str
    is_augmented: bool
    is_gold: bool
    rejection_subtype: str | None
    rejection_reference_id: str | None
    source_domain: str
    capture_style: str
    label: Label
    grade: int | None
    factory: str | None
    evidence_tier: EvidenceTier
    evidence_record_id: str
    rights_record_id: str
    image_sha256: str
    phash: str

    def __post_init__(self) -> None:
        for field, value in (
            ("asset_id", self.asset_id),
            ("candidate_id", self.candidate_id),
            ("reference_id", self.reference_id),
            ("watch_entity_id", self.watch_entity_id),
            ("content_cluster_id", self.content_cluster_id),
            ("seller_group_id", self.seller_group_id),
            ("acquisition_batch_id", self.acquisition_batch_id),
            ("source_domain", self.source_domain),
            ("capture_style", self.capture_style),
            ("evidence_record_id", self.evidence_record_id),
            ("rights_record_id", self.rights_record_id),
        ):
            _nonempty(value, field)
        if not isinstance(self.object_ref, ObjectRef):
            raise ValueError("object_ref must be ObjectRef")
        _aware(self.collected_at, "collected_at")
        _sha256(self.original_sha256, "original_sha256")
        if not isinstance(self.is_augmented, bool) or not isinstance(self.is_gold, bool):
            raise ValueError("augmentation and gold flags must be bool")
        if self.label not in {Label.GENUINE, Label.FAKE, Label.REJECTION}:
            raise ValueError("training label must be genuine, fake or rejection")
        if self.label is Label.REJECTION:
            if self.angle is not None:
                raise ValueError("rejection records cannot claim a guided angle")
            if self.rejection_subtype not in {"no-watch", "non-submariner"}:
                raise ValueError("rejection_subtype is required for rejection records")
            if self.rejection_subtype == "non-submariner":
                if self.rejection_reference_id is None:
                    raise ValueError("non-submariner records require a reference")
                _nonempty(self.rejection_reference_id, "rejection_reference_id")
            elif self.rejection_reference_id is not None:
                raise ValueError("no-watch records cannot claim a reference")
        elif (
            not isinstance(self.angle, Angle)
            or self.rejection_subtype is not None
            or self.rejection_reference_id is not None
        ):
            raise ValueError("authenticity records require an angle and no rejection metadata")
        _label_grade(self.label, self.grade, fake_grade_required=True)
        if self.factory is not None:
            _nonempty(self.factory, "factory")
        if not isinstance(self.evidence_tier, EvidenceTier) or not self.evidence_tier.training_eligible:
            raise ValueError("training requires Tier A or B evidence")
        _sha256(self.image_sha256, "image_sha256")
        if self.image_sha256 != self.object_ref.sha256:
            raise ValueError("image_sha256 must match object_ref")
        if not isinstance(self.phash, str) or _PHASH.fullmatch(self.phash) is None:
            raise ValueError("phash must be lowercase hexadecimal")


def validate_contract_bundle(
    candidate: CandidateRecord,
    rights: RightsRecord,
    evidence: EvidenceRecord,
    training: TrainingRecord,
) -> None:
    """Reject lineage mismatches before a reviewed candidate becomes training data."""

    if candidate.candidate_id != evidence.candidate_id or candidate.candidate_id != training.candidate_id:
        raise ValueError("candidate lineage differs")
    if candidate.acquisition_batch_id != training.acquisition_batch_id:
        raise ValueError("acquisition batch lineage differs")
    if candidate.rights_record_id != rights.rights_record_id or rights.rights_record_id != training.rights_record_id:
        raise ValueError("rights lineage differs")
    if evidence.evidence_record_id != training.evidence_record_id:
        raise ValueError("evidence lineage differs")
    if (
        evidence.accepted_label is not training.label
        or evidence.accepted_grade != training.grade
        or evidence.reference_id != training.reference_id
        or evidence.tier is not training.evidence_tier
    ):
        raise ValueError("accepted evidence differs from training record")
    candidate_domain = urlsplit(candidate.source_url).hostname
    if candidate_domain is None or candidate_domain.rstrip(".").lower() != training.source_domain:
        raise ValueError("source domain differs from candidate URL")


@dataclass(frozen=True)
class BatchReceipt:
    batch_id: str
    batch_sha256: str
    policy_sha256: str
    rights_sha256: str
    accepted_candidate_ids: tuple[str, ...]
    refused_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.batch_id, "batch_id")
        _sha256(self.batch_sha256, "batch_sha256")
        _sha256(self.policy_sha256, "policy_sha256")
        _sha256(self.rights_sha256, "rights_sha256")
        _unique_nonempty(self.accepted_candidate_ids, "accepted_candidate_ids", allow_empty=True)
        _unique_nonempty(self.refused_candidate_ids, "refused_candidate_ids", allow_empty=True)
        if set(self.accepted_candidate_ids) & set(self.refused_candidate_ids):
            raise ValueError("accepted and refused candidates must be disjoint")


@dataclass(frozen=True)
class DatasetFreeze:
    manifest_sha256: str
    split_sha256: str
    policy_set_sha256: str
    eligible_asset_ids: tuple[str, ...]
    excluded_asset_ids: tuple[str, ...]
    exclusion_reason_counts: tuple[tuple[str, int], ...]
    readiness_sha256: str | None = None
    rights_sha256: str | None = None
    exclusions: tuple[tuple[str, str], ...] = ()
    blocked: bool = False
    asset_rights: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _sha256(self.manifest_sha256, "manifest_sha256")
        _sha256(self.split_sha256, "split_sha256")
        _sha256(self.policy_set_sha256, "policy_set_sha256")
        _unique_nonempty(self.eligible_asset_ids, "eligible_asset_ids", allow_empty=True)
        _unique_nonempty(self.excluded_asset_ids, "excluded_asset_ids", allow_empty=True)
        if set(self.eligible_asset_ids) & set(self.excluded_asset_ids):
            raise ValueError("eligible and excluded assets must be disjoint")
        reasons = [reason for reason, _ in self.exclusion_reason_counts]
        if len(reasons) != len(set(reasons)):
            raise ValueError("exclusion reasons must be unique")
        for reason, count in self.exclusion_reason_counts:
            _nonempty(reason, "exclusion reason")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("exclusion reason count must be a positive integer")
        for value, field in (
            (self.readiness_sha256, "readiness_sha256"),
            (self.rights_sha256, "rights_sha256"),
        ):
            if value is not None:
                _sha256(value, field)
        if len({asset_id for asset_id, _ in self.exclusions}) != len(self.exclusions):
            raise ValueError("freeze exclusions must contain unique assets")
        if self.exclusions and {asset_id for asset_id, _ in self.exclusions} != set(
            self.excluded_asset_ids
        ):
            raise ValueError("freeze exclusion details must match excluded assets")
        for asset_id, reason in self.exclusions:
            _nonempty(asset_id, "excluded asset")
            _nonempty(reason, "exclusion reason")
        if not isinstance(self.blocked, bool):
            raise ValueError("blocked must be bool")
        if len({asset_id for asset_id, _ in self.asset_rights}) != len(self.asset_rights):
            raise ValueError("freeze asset rights must contain unique assets")
        if self.asset_rights and {asset_id for asset_id, _ in self.asset_rights} != set(
            self.eligible_asset_ids
        ):
            raise ValueError("freeze asset rights must match eligible assets")
        for asset_id, rights_id in self.asset_rights:
            _nonempty(asset_id, "rights asset")
            _nonempty(rights_id, "rights record")


@dataclass(frozen=True)
class ImpactedArtifactReport:
    impacted_artifact_hashes: tuple[str, ...]
    reason: str
    generated_at: datetime

    def __post_init__(self) -> None:
        _hashes(self.impacted_artifact_hashes, "impacted_artifact_hashes")
        _nonempty(self.reason, "reason")
        _aware(self.generated_at, "generated_at")


@dataclass(frozen=True)
class DataDispositionReport:
    raw_object_refs: tuple[ObjectRef, ...]
    derivative_object_refs: tuple[ObjectRef, ...]
    evidence_object_refs: tuple[ObjectRef, ...]
    required_action: str
    deadline: datetime
    status: str

    def __post_init__(self) -> None:
        all_refs = self.raw_object_refs + self.derivative_object_refs + self.evidence_object_refs
        if not all_refs or not all(isinstance(item, ObjectRef) for item in all_refs):
            raise ValueError("data disposition must list ObjectRef values")
        if self.required_action not in _DISPOSITIONS - {"retain"}:
            raise ValueError("required_action must be quarantine or delete")
        _aware(self.deadline, "deadline")
        if self.status not in _DISPOSITION_STATES:
            raise ValueError("status must be pending, completed or blocked")


@dataclass(frozen=True)
class FreezeResult:
    freeze: DatasetFreeze
    impacted_artifact_report: ImpactedArtifactReport | None
    data_disposition_reports: tuple[DataDispositionReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.freeze, DatasetFreeze):
            raise ValueError("freeze must be DatasetFreeze")
        if self.impacted_artifact_report is not None and not isinstance(
            self.impacted_artifact_report, ImpactedArtifactReport
        ):
            raise ValueError("impacted_artifact_report must be ImpactedArtifactReport")
        if not isinstance(self.data_disposition_reports, tuple) or not all(
            isinstance(item, DataDispositionReport) for item in self.data_disposition_reports
        ):
            raise ValueError("data_disposition_reports must contain DataDispositionReport values")
