"""Quantitative admission gates for one evidence-backed watch reference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from acquisition.types import Angle, EvidenceTier, TrainingRecord
from core.types import Label
from dataset.splits import TRAIN, DatasetSplits

MIN_CELL_IMAGES = 500
MIN_CELL_ENTITIES = 100
MIN_REJECTION_IMAGES = 500
MIN_REJECTION_SUBJECTS = 250
MIN_SOURCES_PER_LABEL = 3
MIN_GOLD_IMAGES = 200


@dataclass(frozen=True)
class ReadinessReport:
    reference_id: str
    ready: bool
    reasons: tuple[str, ...]
    cell_counts: tuple[tuple[str, int, int], ...]
    required_angles: tuple[Angle, ...]


def evaluate_readiness(
    records: Sequence[TrainingRecord],
    reference_id: str,
    *,
    splits: DatasetSplits,
    required_angles: Sequence[Angle],
    reference_aliases: Mapping[str, str] | None = None,
) -> ReadinessReport:
    """Return every readiness deficit without inflating crops or augmentations."""
    angles = tuple(required_angles)
    if not angles or any(not isinstance(angle, Angle) for angle in angles) or len(set(angles)) != len(angles):
        raise ValueError("required_angles must be nonempty, unique Angle values")
    scoped = [record for record in records if record.reference_id == reference_id]
    components = _independent_entity_components(
        [record for record in scoped if not record.is_augmented]
    )
    reasons: list[str] = []
    counts: list[tuple[str, int, int]] = []
    outputs = (("genuine", Label.GENUINE, None),) + tuple(
        (f"fake-{grade}", Label.FAKE, grade) for grade in range(1, 5)
    )
    for output, label, grade in outputs:
        for angle in angles:
            cell = [
                record
                for record in scoped
                if record.label is label
                and record.grade == grade
                and record.angle is angle
                and not record.is_augmented
            ]
            images = len({record.content_cluster_id for record in cell})
            entities = len({components[record.asset_id] for record in cell})
            cell_name = f"{output}/{angle.value}"
            counts.append((cell_name, images, entities))
            if images < MIN_CELL_IMAGES:
                reasons.append(f"{cell_name}: requires 500 distinct original images")
            if entities < MIN_CELL_ENTITIES:
                reasons.append(f"{cell_name}: requires 100 distinct watch entities")

    rejections = [
        record
        for record in scoped
        if record.label is Label.REJECTION and not record.is_augmented
    ]
    rejection_images = len({record.content_cluster_id for record in rejections})
    rejection_subjects = len({components[record.asset_id] for record in rejections})
    if rejection_images < MIN_REJECTION_IMAGES:
        reasons.append("rejection: requires 500 distinct original images")
    if rejection_subjects < MIN_REJECTION_SUBJECTS:
        reasons.append("rejection: requires 250 distinct subjects")
    other_watch_subtype = "non-submariner" if reference_id == "rolex-submariner-126610ln" else "non-target-reference"
    subtypes = {record.rejection_subtype for record in rejections}
    if not {"no-watch", other_watch_subtype}.issubset(subtypes):
        reasons.append(f"rejection: requires no-watch and {other_watch_subtype} examples")
    aliases = reference_aliases or {}
    canonical_reference_id = aliases.get(reference_id, reference_id)
    if any(
        record.label is Label.REJECTION
        and record.rejection_subtype in {"non-submariner", "non-target-reference"}
        and record.rejection_reference_id is not None
        and aliases.get(record.rejection_reference_id, record.rejection_reference_id) == canonical_reference_id
        for record in scoped
    ):
        reasons.append("rejection: other-watch example names the target reference or an alias")
    references = {
        aliases.get(record.rejection_reference_id, record.rejection_reference_id)
        for record in rejections
        if record.rejection_subtype == other_watch_subtype
        and record.rejection_reference_id is not None
    }
    if len(references - {None, reference_id}) < 5:
        reasons.append(f"rejection: requires five {other_watch_subtype} references")

    for name, label in (("genuine", Label.GENUINE), ("fake", Label.FAKE)):
        label_records = _unique_content_records(
            record
            for record in scoped
            if record.label is label and not record.is_augmented
        )
        if len({record.source_domain for record in label_records}) < MIN_SOURCES_PER_LABEL:
            reasons.append(f"{name}: requires three independent source domains")
        styles = {record.capture_style for record in label_records}
        if not {"studio", "casual"}.issubset(styles):
            reasons.append(f"{name}: requires studio and casual examples")

    gold = [
        record
        for record in scoped
        if record.is_gold
        and record.evidence_tier is EvidenceTier.A
        and not record.is_augmented
    ]
    gold_outputs = {_output_name(record) for record in gold}
    if len({record.content_cluster_id for record in gold}) < MIN_GOLD_IMAGES:
        reasons.append("gold: requires 200 distinct Tier-A images")
    if gold_outputs != {"genuine", "fake-1", "fake-2", "fake-3", "fake-4", "rejection"}:
        reasons.append("gold: must span all six outputs")

    train_ids = set(splits.asset_ids(TRAIN))
    train = [record for record in scoped if record.asset_id in train_ids]
    if not train:
        reasons.append("train split is empty")
    else:
        _append_dominance_reasons(train, components, reasons)
    return ReadinessReport(
        reference_id=reference_id,
        ready=not reasons,
        reasons=tuple(sorted(reasons)),
        cell_counts=tuple(counts),
        required_angles=angles,
    )


def _append_dominance_reasons(
    records: Sequence[TrainingRecord], components: dict[str, str], reasons: list[str]
) -> None:
    records = [record for record in records if not record.is_augmented]
    by_label: dict[Label, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    seller_entities: dict[str, set[str]] = defaultdict(set)
    all_entities = {components[record.asset_id] for record in records}
    for record in records:
        if record.label in {Label.GENUINE, Label.FAKE}:
            by_label[record.label][record.source_domain].add(components[record.asset_id])
        seller_entities[record.seller_group_id].add(components[record.asset_id])
    for label, domains in by_label.items():
        label_entities = set().union(*domains.values()) if domains else set()
        for domain, entities in domains.items():
            if len(entities) * 100 > len(label_entities) * 40:
                reasons.append(f"{label.value}: source {domain} exceeds 40% of train entities")
    for seller, entities in seller_entities.items():
        if len(entities) * 100 > len(all_entities) * 10:
            reasons.append(f"seller {seller} exceeds 10% of train entities")


def _output_name(record: TrainingRecord) -> str:
    if record.label is Label.FAKE:
        return f"fake-{record.grade}"
    return record.label.value


def _unique_content_records(records: Iterable[TrainingRecord]) -> list[TrainingRecord]:
    unique: dict[str, TrainingRecord] = {}
    for record in sorted(records, key=lambda item: item.asset_id):
        unique.setdefault(record.content_cluster_id, record)
    return list(unique.values())


def _independent_entity_components(records: Sequence[TrainingRecord]) -> dict[str, str]:
    """Use one conservative watch/content graph across every view and output."""
    parent = {record.watch_entity_id: record.watch_entity_id for record in records}

    def root(entity: str) -> str:
        while parent[entity] != entity:
            entity = parent[entity]
        return entity

    first_by_content: dict[str, str] = {}
    for record in records:
        entity = record.watch_entity_id
        first = first_by_content.setdefault(record.content_cluster_id, entity)
        parent[root(entity)] = root(first)
    return {record.asset_id: root(record.watch_entity_id) for record in records}
