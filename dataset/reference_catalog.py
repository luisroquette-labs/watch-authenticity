"""Immutable, per-reference admission catalog for non-universal heads."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from acquisition.types import Angle
from adapters.backbone import BACKBONE_CHECKPOINT, BACKBONE_REVISION
from dataset.readiness import ReadinessReport

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)+$")
_REFERENCE = re.compile(r"^[a-z0-9]+$")
_CATALOG_FIELDS = {"generation", "baseline", "references"}
_BASELINE_FIELDS = {"reference_id", "backbone_checkpoint", "backbone_revision", "generation"}
_ENTRY_FIELDS = {
    "reference_id",
    "brand",
    "family",
    "reference",
    "aliases",
    "identity_evidence_sha256",
    "required_angles",
    "gold_freeze_sha256",
    "holdout_freeze_sha256",
    "status",
    "head",
}
_HEAD_FIELDS = {"reference_id", "generation", "checkpoint_sha256", "calibration_sha256"}


class ReferenceCatalogError(ValueError):
    pass


class ReferenceNotAdmitted(LookupError):
    pass


@dataclass(frozen=True)
class HeadConfig:
    reference_id: str
    generation: int
    checkpoint_sha256: str
    calibration_sha256: str


@dataclass(frozen=True)
class ReferenceEntry:
    reference_id: str
    brand: str
    family: str
    reference: str
    aliases: tuple[str, ...]
    identity_evidence_sha256: str | None
    required_angles: tuple[Angle, ...]
    gold_freeze_sha256: str | None
    holdout_freeze_sha256: str | None
    status: str
    head: HeadConfig | None


@dataclass(frozen=True)
class ReferenceAdmission:
    reference_id: str
    allowed: bool
    reasons: tuple[str, ...]
    catalog_sha256: str


@dataclass(frozen=True)
class ReferenceCatalog:
    generation: int
    baseline_reference_id: str
    backbone_checkpoint: str
    backbone_revision: str
    entries: MappingProxyType[str, ReferenceEntry]
    aliases: MappingProxyType[str, str]
    sha256: str

    def entry_for(self, reference_id: str) -> ReferenceEntry:
        canonical = self.aliases.get(reference_id, reference_id)
        try:
            return self.entries[canonical]
        except KeyError as exc:
            raise ReferenceNotAdmitted(f"reference is absent from catalog: {reference_id}") from exc

    def head_for(self, reference_id: str) -> HeadConfig:
        entry = self.entry_for(reference_id)
        if entry.status != "admitted" or entry.head is None:
            raise ReferenceNotAdmitted(f"reference has no admitted head: {reference_id}")
        return entry.head


def catalog_document_sha256(document: object) -> str:
    return hashlib.sha256(_canonical(document)).hexdigest()


def load_reference_catalog(
    path: str | Path,
    *,
    expected_sha256: str,
    previous: ReferenceCatalog | None = None,
) -> ReferenceCatalog:
    try:
        document = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceCatalogError("reference catalog is unreadable") from exc
    return parse_reference_catalog(document, expected_sha256=expected_sha256, previous=previous)


def parse_reference_catalog(
    document: object,
    *,
    expected_sha256: str,
    previous: ReferenceCatalog | None = None,
) -> ReferenceCatalog:
    if _SHA256.fullmatch(expected_sha256) is None or catalog_document_sha256(document) != expected_sha256:
        raise ReferenceCatalogError("reference catalog hash mismatch")
    if not isinstance(document, dict) or set(document) != _CATALOG_FIELDS:
        raise ReferenceCatalogError("reference catalog schema is invalid")
    generation = _positive_int(document["generation"], "generation")
    baseline = document["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != _BASELINE_FIELDS:
        raise ReferenceCatalogError("baseline schema is invalid")
    if (
        baseline["generation"] != generation
        or baseline["backbone_checkpoint"] != BACKBONE_CHECKPOINT
        or baseline["backbone_revision"] != BACKBONE_REVISION
    ):
        raise ReferenceCatalogError("baseline generation or backbone is incompatible")
    baseline_reference_id = _slug(baseline["reference_id"], "baseline reference")
    raw_entries = document["references"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ReferenceCatalogError("catalog requires reference entries")
    entries: dict[str, ReferenceEntry] = {}
    aliases: dict[str, str] = {}
    for raw in raw_entries:
        entry = _parse_entry(raw, generation)
        if entry.reference_id in entries:
            raise ReferenceCatalogError("reference IDs must be unique")
        entries[entry.reference_id] = entry
        for alias in (entry.reference_id, *entry.aliases):
            if alias in aliases:
                raise ReferenceCatalogError("reference aliases collide")
            aliases[alias] = entry.reference_id
    if baseline_reference_id not in entries:
        raise ReferenceCatalogError("baseline reference is absent")
    _validate_catalog_invariants(entries, baseline_reference_id)
    if previous is not None:
        for reference_id, prior in previous.entries.items():
            if prior.status == "admitted" and entries.get(reference_id) != prior:
                raise ReferenceCatalogError("an admitted reference entry is immutable")
    return ReferenceCatalog(
        generation,
        baseline_reference_id,
        str(baseline["backbone_checkpoint"]),
        str(baseline["backbone_revision"]),
        MappingProxyType(entries),
        MappingProxyType(aliases),
        expected_sha256,
    )


def admit_reference(
    reference_id: str,
    report: ReadinessReport,
    catalog: ReferenceCatalog,
) -> ReferenceAdmission:
    reasons: list[str] = []
    try:
        entry = catalog.entry_for(reference_id)
    except ReferenceNotAdmitted:
        return ReferenceAdmission(reference_id, False, ("reference is absent from catalog",), catalog.sha256)
    if entry.reference_id != reference_id or report.reference_id != entry.reference_id:
        reasons.append("readiness report belongs to another reference")
    baseline = catalog.entries[catalog.baseline_reference_id]
    if (
        entry.reference_id != catalog.baseline_reference_id
        and (baseline.status != "admitted" or baseline.head is None)
    ):
        reasons.append("trained baseline head is not admitted")
    if not report.ready or report.reasons:
        reasons.append("readiness report failed")
    if entry.status != "admitted" or entry.head is None:
        reasons.append("reference head is not admitted")
    if entry.gold_freeze_sha256 is None or entry.holdout_freeze_sha256 is None:
        reasons.append("reference lacks its own gold or holdout freeze")
    return ReferenceAdmission(entry.reference_id, not reasons, tuple(reasons), catalog.sha256)


def _parse_entry(raw: object, generation: int) -> ReferenceEntry:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
        raise ReferenceCatalogError("reference entry schema is invalid")
    brand = _token(raw["brand"], "brand")
    family = _token(raw["family"], "family")
    reference = _token(raw["reference"], "reference")
    reference_id = _slug(raw["reference_id"], "reference_id")
    if reference_id != f"{brand}-{family}-{reference}":
        raise ReferenceCatalogError("reference identity is not normalized")
    aliases_raw = raw["aliases"]
    if not isinstance(aliases_raw, list):
        raise ReferenceCatalogError("aliases must be a list")
    aliases = tuple(_slug(alias, "alias") for alias in aliases_raw)
    if len(set(aliases)) != len(aliases) or reference_id in aliases:
        raise ReferenceCatalogError("aliases must be unique and non-canonical")
    if any(
        not alias.startswith(f"{brand}-{family}-")
        or not alias.endswith(f"-{reference}")
        for alias in aliases
    ):
        raise ReferenceCatalogError("alias must preserve brand, family and reference")
    identity_hash = _optional_hash(raw["identity_evidence_sha256"], "identity evidence")
    angles_raw = raw["required_angles"]
    if not isinstance(angles_raw, list) or not angles_raw:
        raise ReferenceCatalogError("required angles are missing")
    try:
        angles = tuple(Angle(value) for value in angles_raw)
    except (TypeError, ValueError) as exc:
        raise ReferenceCatalogError("required angle is invalid") from exc
    if len(set(angles)) != len(angles):
        raise ReferenceCatalogError("required angles must be unique")
    status = raw["status"]
    if status not in {"pending", "admitted"}:
        raise ReferenceCatalogError("reference status is invalid")
    gold = _optional_hash(raw["gold_freeze_sha256"], "gold freeze")
    holdout = _optional_hash(raw["holdout_freeze_sha256"], "holdout freeze")
    head = _parse_head(raw["head"], reference_id, generation)
    if status == "admitted" and (
        head is None or gold is None or holdout is None or identity_hash is None
    ):
        raise ReferenceCatalogError("admitted reference lacks identity, head, gold or holdout")
    if status == "pending" and head is not None:
        raise ReferenceCatalogError("pending reference cannot carry a head")
    return ReferenceEntry(
        reference_id,
        brand,
        family,
        reference,
        aliases,
        identity_hash,
        angles,
        gold,
        holdout,
        status,
        head,
    )


def _validate_catalog_invariants(
    entries: dict[str, ReferenceEntry], baseline_reference_id: str
) -> None:
    baseline = entries[baseline_reference_id]
    if any(
        entry.reference_id != baseline_reference_id and entry.status == "admitted"
        for entry in entries.values()
    ) and (baseline.status != "admitted" or baseline.head is None):
        raise ReferenceCatalogError("later reference requires an admitted baseline head")

    owners: dict[str, tuple[str, str]] = {}
    for entry in entries.values():
        artifacts = {
            "identity evidence": entry.identity_evidence_sha256,
            "gold freeze": entry.gold_freeze_sha256,
            "holdout freeze": entry.holdout_freeze_sha256,
            "head checkpoint": entry.head.checkpoint_sha256 if entry.head else None,
            "calibration": entry.head.calibration_sha256 if entry.head else None,
        }
        for kind, digest in artifacts.items():
            if digest is None:
                continue
            prior = owners.get(digest)
            if prior is not None:
                raise ReferenceCatalogError(
                    f"{kind} reuses {prior[1]} from reference {prior[0]}"
                )
            owners[digest] = (entry.reference_id, kind)


def _parse_head(raw: object, reference_id: str, generation: int) -> HeadConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict) or set(raw) != _HEAD_FIELDS:
        raise ReferenceCatalogError("head schema is invalid")
    if raw["reference_id"] != reference_id or raw["generation"] != generation:
        raise ReferenceCatalogError("head belongs to another reference or generation")
    return HeadConfig(
        reference_id,
        generation,
        _hash(raw["checkpoint_sha256"], "checkpoint"),
        _hash(raw["calibration_sha256"], "calibration"),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReferenceCatalogError(f"{field} must be a positive integer")
    return value


def _slug(value: object, field: str) -> str:
    if not isinstance(value, str) or _SLUG.fullmatch(value) is None:
        raise ReferenceCatalogError(f"{field} must be a normalized slug")
    return value


def _token(value: object, field: str) -> str:
    if not isinstance(value, str) or _REFERENCE.fullmatch(value) is None:
        raise ReferenceCatalogError(f"{field} must be normalized")
    return value


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReferenceCatalogError(f"{field} must be SHA-256")
    return value


def _optional_hash(value: object, field: str) -> str | None:
    return None if value is None else _hash(value, field)
