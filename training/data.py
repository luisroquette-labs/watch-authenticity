"""Fail-closed loader for one content-addressed COS training freeze."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.rules import model_class_for_label, require_all_model_classes
from core.types import Label, ModelClass
from dataset.crop import ANGLE_CROP_FRACTIONS
from dataset.splits import (
    CROSS_SOURCE,
    FACTORY_HOLDOUT,
    GOLD,
    TEST,
    TRAIN,
    VAL,
    ensure_rejection_groups_are_disjoint,
)
from infra.ibm_cos import IntegrityError, download_verified
from infra.ibm_types import RunSpec

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")
_SPLITS = frozenset({TRAIN, VAL, TEST, CROSS_SOURCE, FACTORY_HOLDOUT, GOLD})
_RIGHTS = frozenset({"owner", "licensed"})
_REJECTION_SUBTYPES = frozenset({"no-watch", "non-submariner"})
_PLACEHOLDER_DOMAINS = frozenset({"example.com", "example.net", "example.org"})
_MAX_IMAGE_BYTES = 25 * 1024 * 1024
_RECORD_FIELDS = frozenset(
    {
        "asset_id",
        "object_key",
        "image_sha256",
        "label",
        "grade",
        "angle",
        "source_domain",
        "subject_id",
        "provenance_id",
        "rights",
        "rejection_subtype",
    }
)


class TrainingDataRefused(RuntimeError):
    """The frozen dataset cannot safely or completely train the declared heads."""


@dataclass(frozen=True)
class TrainingRecord:
    asset_id: str
    object_key: str
    image_sha256: str
    label: Label
    grade: int | None
    angle: str | None
    source_domain: str
    subject_id: str
    provenance_id: str
    rights: str
    rejection_subtype: str | None
    model_class: ModelClass


@dataclass(frozen=True)
class TrainingBundle:
    cos: Any
    bucket: str
    records: tuple[TrainingRecord, ...]
    splits: Mapping[str, tuple[str, ...]]
    split_by_asset: Mapping[str, str]
    class_counts: Mapping[ModelClass, int]
    head_class_counts: Mapping[str, Mapping[ModelClass, int]]
    manifest_sha256: str
    splits_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingDataRefused(f"{field} must be a non-empty string")
    return value


def _source_domain(value: object) -> str:
    domain = _nonempty(value, "source_domain")
    if (
        domain != domain.lower()
        or _DOMAIN.fullmatch(domain) is None
        or "." not in domain
        or any(domain == root or domain.endswith(f".{root}") for root in _PLACEHOLDER_DOMAINS)
        or domain.rsplit(".", 1)[-1] in {"invalid", "test", "localhost", "local"}
    ):
        raise TrainingDataRefused("source_domain must be a real lowercase DNS domain")
    return domain


def _parse_record(raw: object, spec: RunSpec) -> TrainingRecord:
    if not isinstance(raw, dict) or set(raw) != _RECORD_FIELDS:
        raise TrainingDataRefused("manifest record schema is invalid")
    asset_id = _nonempty(raw["asset_id"], "asset_id")
    digest = _nonempty(raw["image_sha256"], "image_sha256")
    if _SHA256.fullmatch(digest) is None:
        raise TrainingDataRefused("image_sha256 must be lowercase SHA-256")
    object_key = _nonempty(raw["object_key"], "object_key")
    expected_key = f"{spec.prefix}/inputs/images/{digest}"
    if object_key != expected_key:
        raise TrainingDataRefused("image object key is not content-addressed inside the run prefix")
    try:
        label = Label(_nonempty(raw["label"], "label"))
    except ValueError as exc:
        raise TrainingDataRefused("label is not trainable") from exc
    grade = raw["grade"]
    if grade is not None and (not isinstance(grade, int) or isinstance(grade, bool)):
        raise TrainingDataRefused("grade must be an integer or null")
    try:
        model_class = model_class_for_label(label, grade)
    except ValueError as exc:
        raise TrainingDataRefused(str(exc)) from exc
    angle = raw["angle"]
    if label is Label.REJECTION:
        if angle is not None:
            raise TrainingDataRefused("rejection records must not claim one guided angle")
        if raw["rejection_subtype"] not in _REJECTION_SUBTYPES:
            raise TrainingDataRefused("rejection_subtype is invalid")
    else:
        if angle not in ANGLE_CROP_FRACTIONS:
            raise TrainingDataRefused("authenticity record angle is invalid")
        if raw["rejection_subtype"] is not None:
            raise TrainingDataRefused("authenticity records cannot have a rejection subtype")
    rights = _nonempty(raw["rights"], "rights")
    if rights not in _RIGHTS:
        raise TrainingDataRefused("training records require owner or licensed rights")
    return TrainingRecord(
        asset_id=asset_id,
        object_key=object_key,
        image_sha256=digest,
        label=label,
        grade=grade,
        angle=angle if isinstance(angle, str) else None,
        source_domain=_source_domain(raw["source_domain"]),
        subject_id=_nonempty(raw["subject_id"], "subject_id"),
        provenance_id=_nonempty(raw["provenance_id"], "provenance_id"),
        rights=rights,
        rejection_subtype=(
            raw["rejection_subtype"] if isinstance(raw["rejection_subtype"], str) else None
        ),
        model_class=model_class,
    )


def _load_json(cos: Any, spec: RunSpec, kind: str, digest: str) -> object:
    key = f"{spec.prefix}/inputs/{kind}/{digest}.json"
    with tempfile.TemporaryDirectory(prefix="watch-auth-input-") as directory:
        path = Path(directory) / f"{kind}.json"
        try:
            download_verified(cos, spec.bucket, key, path, digest)
        except (FileNotFoundError, IntegrityError) as exc:
            raise TrainingDataRefused(f"{kind} object is missing or corrupt") from exc
        body = path.read_bytes()
    try:
        parsed = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingDataRefused(f"{kind} is not valid JSON") from exc
    if body != _canonical(parsed):
        raise TrainingDataRefused(f"{kind} must use canonical JSON bytes")
    return parsed


def _parse_splits(raw: object, asset_ids: set[str]) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "splits"}:
        raise TrainingDataRefused("split document schema is invalid")
    if raw["schema_version"] != 1 or not isinstance(raw["splits"], dict):
        raise TrainingDataRefused("split document version is unsupported")
    split_values = raw["splits"]
    if set(split_values) != _SPLITS:
        raise TrainingDataRefused("split document must contain the six frozen splits")
    parsed: dict[str, tuple[str, ...]] = {}
    split_by_asset: dict[str, str] = {}
    for split, members in split_values.items():
        if not isinstance(members, list) or any(not isinstance(item, str) for item in members):
            raise TrainingDataRefused("split members must be asset ID lists")
        if len(set(members)) != len(members):
            raise TrainingDataRefused("a split contains a duplicate asset")
        parsed[split] = tuple(members)
        for asset_id in members:
            if asset_id in split_by_asset:
                raise TrainingDataRefused("an asset crosses dataset splits")
            split_by_asset[asset_id] = split
    if set(split_by_asset) != asset_ids:
        raise TrainingDataRefused("split assignments do not match the manifest assets")
    if not parsed[TRAIN] or not parsed[VAL]:
        raise TrainingDataRefused("train and validation splits must be non-empty")
    return parsed, split_by_asset


def _verify_image_metadata(cos: Any, spec: RunSpec, record: TrainingRecord) -> None:
    try:
        response = cos.head_object(Bucket=spec.bucket, Key=record.object_key)
    except Exception as exc:
        error = getattr(exc, "response", {}).get("Error", {})
        if str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}:
            raise TrainingDataRefused(f"image object is missing: {record.asset_id}") from exc
        raise
    if not isinstance(response, dict) or not isinstance(response.get("Metadata"), dict):
        raise TrainingDataRefused("image object metadata is missing")
    metadata = response["Metadata"]
    try:
        metadata_length = int(metadata.get("length", ""))
        content_length = int(response["ContentLength"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingDataRefused("image object length metadata is invalid") from exc
    if (
        metadata.get("sha256") != record.image_sha256
        or metadata_length != content_length
        or not 0 < content_length <= _MAX_IMAGE_BYTES
    ):
        raise TrainingDataRefused("image object hash or length metadata does not match")


def load_training_data(cos: Any, spec: RunSpec) -> TrainingBundle:
    """Load and validate one immutable manifest/split pair without provisioning."""
    manifest_raw = _load_json(cos, spec, "manifest", spec.manifest_sha256)
    splits_raw = _load_json(cos, spec, "splits", spec.splits_sha256)
    if not isinstance(manifest_raw, dict) or set(manifest_raw) != {"schema_version", "records"}:
        raise TrainingDataRefused("manifest document schema is invalid")
    if manifest_raw["schema_version"] != 1 or not isinstance(manifest_raw["records"], list):
        raise TrainingDataRefused("manifest document version is unsupported")
    records = tuple(_parse_record(raw, spec) for raw in manifest_raw["records"])
    asset_ids = [record.asset_id for record in records]
    if not records or len(set(asset_ids)) != len(asset_ids):
        raise TrainingDataRefused("manifest must contain unique assets")
    splits, split_by_asset = _parse_splits(splits_raw, set(asset_ids))
    splits_by_digest: dict[str, set[str]] = {}
    for record in records:
        splits_by_digest.setdefault(record.image_sha256, set()).add(
            split_by_asset[record.asset_id]
        )
    if any(len(assigned) > 1 for assigned in splits_by_digest.values()):
        raise TrainingDataRefused("image content crosses dataset splits")
    try:
        ensure_rejection_groups_are_disjoint(records, split_by_asset)
    except ValueError as exc:
        raise TrainingDataRefused(str(exc)) from exc

    train_records = [record for record in records if split_by_asset[record.asset_id] == TRAIN]
    class_counts = Counter(record.model_class for record in train_records)
    try:
        require_all_model_classes(class_counts)
    except ValueError as exc:
        raise TrainingDataRefused(str(exc)) from exc
    def count_heads(selected: list[TrainingRecord]) -> dict[str, Counter[ModelClass]]:
        counts_by_angle: dict[str, Counter[ModelClass]] = {
            angle: Counter() for angle in ANGLE_CROP_FRACTIONS
        }
        for record in selected:
            angles = ANGLE_CROP_FRACTIONS if record.label is Label.REJECTION else (record.angle,)
            for angle in angles:
                assert isinstance(angle, str)
                counts_by_angle[angle][record.model_class] += 1
        return counts_by_angle

    head_counts = count_heads(train_records)
    validation_records = [
        record for record in records if split_by_asset[record.asset_id] == VAL
    ]
    try:
        for counts in (*head_counts.values(), *count_heads(validation_records).values()):
            require_all_model_classes(counts)
    except ValueError as exc:
        raise TrainingDataRefused(
            "every angle head requires every model class in train and validation"
        ) from exc
    for label in (Label.GENUINE, Label.FAKE):
        domains = {record.source_domain for record in train_records if record.label is label}
        if len(domains) < 2:
            raise TrainingDataRefused(f"{label.value} training requires at least two source domains")
    for record in records:
        _verify_image_metadata(cos, spec, record)

    return TrainingBundle(
        cos=cos,
        bucket=spec.bucket,
        records=records,
        splits=MappingProxyType(splits),
        split_by_asset=MappingProxyType(split_by_asset),
        class_counts=MappingProxyType(dict(class_counts)),
        head_class_counts=MappingProxyType(
            {
                angle: MappingProxyType(dict(counts))
                for angle, counts in head_counts.items()
            }
        ),
        manifest_sha256=spec.manifest_sha256,
        splits_sha256=spec.splits_sha256,
    )
