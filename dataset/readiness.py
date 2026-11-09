"""Quantitative admission gates for one evidence-backed watch reference."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
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


def evaluate_readiness(
    records: Sequence[TrainingRecord],
    reference_id: str,
    *,
    splits: DatasetSplits,
) -> ReadinessReport:
    """Return every readiness deficit without inflating crops or augmentations."""
    scoped = [record for record in records if record.reference_id == reference_id]
    reasons: list[str] = []
    counts: list[tuple[str, int, int]] = []
    outputs = (("genuine", Label.GENUINE, None),) + tuple(
        (f"fake-{grade}", Label.FAKE, grade) for grade in range(1, 5)
    )
    for output, label, grade in outputs:
        for angle in Angle:
            cell = [
                record
                for record in scoped
                if record.label is label
                and record.grade == grade
                and record.angle is angle
                and not record.is_augmented
            ]
            images = len({record.content_cluster_id for record in cell})
            entities = len({record.watch_entity_id for record in cell})
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
    rejection_subjects = len({record.watch_entity_id for record in rejections})
    if rejection_images < MIN_REJECTION_IMAGES:
        reasons.append("rejection: requires 500 distinct original images")
    if rejection_subjects < MIN_REJECTION_SUBJECTS:
        reasons.append("rejection: requires 250 distinct subjects")
    subtypes = {record.rejection_subtype for record in rejections}
    if not {"no-watch", "non-submariner"}.issubset(subtypes):
        reasons.append("rejection: requires no-watch and non-submariner examples")
    references = {
        record.rejection_reference_id
        for record in rejections
        if record.rejection_subtype == "non-submariner"
    }
    if len(references - {None}) < 5:
        reasons.append("rejection: requires five non-submariner references")

    for name, label in (("genuine", Label.GENUINE), ("fake", Label.FAKE)):
        label_records = [record for record in scoped if record.label is label]
        if len({record.source_domain for record in label_records}) < MIN_SOURCES_PER_LABEL:
            reasons.append(f"{name}: requires three independent source domains")
        styles = {record.capture_style for record in label_records}
        if not {"studio", "casual"}.issubset(styles):
            reasons.append(f"{name}: requires studio and casual examples")

    gold = [record for record in scoped if record.is_gold and record.evidence_tier is EvidenceTier.A]
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
        _append_dominance_reasons(train, reasons)
    return ReadinessReport(
        reference_id=reference_id,
        ready=not reasons,
        reasons=tuple(reasons),
        cell_counts=tuple(counts),
    )


def _append_dominance_reasons(records: Sequence[TrainingRecord], reasons: list[str]) -> None:
    by_label: dict[Label, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    seller_entities: dict[str, set[str]] = defaultdict(set)
    all_entities = {record.watch_entity_id for record in records}
    for record in records:
        if record.label in {Label.GENUINE, Label.FAKE}:
            by_label[record.label][record.source_domain].add(record.watch_entity_id)
        seller_entities[record.seller_group_id].add(record.watch_entity_id)
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
