"""Six-way dataset split builder over manifest rows (CK-7, CK-9, CK-12, item 13).

Builds train / val / test / cross-source / factory-holdout / gold splits
from a flat list of `core.types.ManifestRecord` rows. The three
evaluation-only splits are pulled out of the training-eligible pool BEFORE
train/val/test is cut, so a gold-set, factory-holdout, or cross-source row
leaking into train is structurally impossible rather than merely checked
for afterward — this is the no-leak guarantee the Success Criteria and the
task's heaviest-weighted rubric (Generalization & Regression Evaluation
Rigor, 0.25) both depend on.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import ceil
from typing import Protocol

from acquisition.types import TrainingRecord as AcceptedTrainingRecord
from core.types import Label, ManifestRecord
from dataset.entities import find_near_duplicate_pairs

DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15
DEFAULT_SEED = 42

TRAIN = "train"
VAL = "val"
TEST = "test"
CROSS_SOURCE = "cross_source"
FACTORY_HOLDOUT = "factory_holdout"
GOLD = "gold"
TEMPORAL_HOLDOUT = "temporal_holdout"


class RejectionGroupRecord(Protocol):
    @property
    def asset_id(self) -> str: ...

    @property
    def label(self) -> Label: ...

    @property
    def provenance_id(self) -> str: ...

    @property
    def subject_id(self) -> str: ...


def ensure_rejection_groups_are_disjoint(
    records: Sequence[RejectionGroupRecord], split_by_asset: Mapping[str, str]
) -> None:
    """Keep every approved negative provenance/subject family in one split."""
    group_splits: dict[tuple[str, str], str] = {}
    for record in records:
        if record.label is not Label.REJECTION:
            continue
        split = split_by_asset[record.asset_id]
        group = (record.provenance_id, record.subject_id)
        previous = group_splits.setdefault(group, split)
        if previous != split:
            raise ValueError("rejection group crosses dataset splits")


def build_manifest_splits(
    records: Sequence[ManifestRecord],
    *,
    gold_paths: Collection[str] = (),
    holdout_sources: Collection[str] = (),
    cross_source_sources: Collection[str] = (),
    val_fraction: float = DEFAULT_VAL_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[ManifestRecord]]:
    """Partition manifest rows into the six disjoint splits.

    `gold_paths` (the frozen ~200-photo gold set) and `holdout_sources`
    (factories/sources reserved for the never-seen-in-training evaluation
    split) are the caller's membership decisions — neither is derivable
    from a manifest row alone. The cross-source domains are also explicit. Both pools
    are removed from the pool before train/val/test is cut.
    """
    gold_path_set = set(gold_paths)
    holdout_source_set = set(holdout_sources)
    cross_source_set = set(cross_source_sources)
    if holdout_source_set & cross_source_set:
        raise ValueError("factory and cross-source holdouts must use distinct sources")

    gold = [record for record in records if record.path in gold_path_set]
    remaining = [record for record in records if record.path not in gold_path_set]

    factory_holdout = [record for record in remaining if record.source in holdout_source_set]
    remaining = [record for record in remaining if record.source not in holdout_source_set]

    cross_source = [record for record in remaining if record.source in cross_source_set]
    cross_source_paths = {record.path for record in cross_source}
    remaining = [record for record in remaining if record.path not in cross_source_paths]

    train, val, test = _split_train_val_test(
        remaining, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    return {
        TRAIN: train,
        VAL: val,
        TEST: test,
        CROSS_SOURCE: cross_source,
        FACTORY_HOLDOUT: factory_holdout,
        GOLD: gold,
    }


def _split_train_val_test(
    records: Sequence[ManifestRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[ManifestRecord], list[ManifestRecord], list[ManifestRecord]]:
    """Deterministically shuffle the remaining pool, then cut it three ways."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_size = round(total * val_fraction)
    test_size = round(total * test_fraction)

    val = shuffled[:val_size]
    test = shuffled[val_size : val_size + test_size]
    train = shuffled[val_size + test_size :]
    return train, val, test


@dataclass(frozen=True)
class DatasetSplits:
    assignments: tuple[tuple[str, str], ...]
    temporal_cutoff: datetime | None
    seed: int
    held_out_factories: tuple[str, ...] = ()
    held_out_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        asset_ids = [asset_id for asset_id, _ in self.assignments]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError("split assignments must contain unique assets")
        allowed = {TRAIN, VAL, TEST, CROSS_SOURCE, FACTORY_HOLDOUT, GOLD, TEMPORAL_HOLDOUT}
        if any(split not in allowed for _, split in self.assignments):
            raise ValueError("split assignment is unsupported")
        if self.temporal_cutoff is not None and (
            self.temporal_cutoff.tzinfo is None or self.temporal_cutoff.utcoffset() is None
        ):
            raise ValueError("temporal cutoff must be timezone-aware")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if tuple(sorted(set(self.held_out_factories))) != self.held_out_factories:
            raise ValueError("held-out factories must be sorted and unique")
        if tuple(sorted(set(self.held_out_sources))) != self.held_out_sources:
            raise ValueError("held-out sources must be sorted and unique")

    @property
    def split_by_asset(self) -> Mapping[str, str]:
        return dict(self.assignments)

    def asset_ids(self, split: str) -> tuple[str, ...]:
        return tuple(asset_id for asset_id, assigned in self.assignments if assigned == split)


class _ComponentUnion:
    def __init__(self, asset_ids: Sequence[str]) -> None:
        self.parent = {asset_id: asset_id for asset_id in asset_ids}

    def find(self, asset_id: str) -> str:
        parent = self.parent[asset_id]
        if parent != asset_id:
            self.parent[asset_id] = self.find(parent)
        return self.parent[asset_id]

    def union(self, left: str, right: str) -> None:
        roots = sorted((self.find(left), self.find(right)))
        if roots[0] != roots[1]:
            self.parent[roots[1]] = roots[0]


def build_splits(
    records: Sequence[AcceptedTrainingRecord],
    *,
    seed: int,
    held_out_factories: Collection[str],
    held_out_sources: Collection[str] = (),
) -> DatasetSplits:
    """Assign every transitive leakage family to one deterministic split."""
    asset_ids = [record.asset_id for record in records]
    if len(asset_ids) != len(set(asset_ids)):
        raise ValueError("training asset IDs must be unique")
    union = _ComponentUnion(asset_ids)
    seen: dict[tuple[str, str], str] = {}
    for record in sorted(records, key=lambda item: item.asset_id):
        for field, value in (
            ("watch", record.watch_entity_id),
            ("family", record.split_family_id),
            ("content", record.content_cluster_id),
            ("seller", record.seller_group_id),
            ("batch", record.acquisition_batch_id),
        ):
            prior = seen.setdefault((field, value), record.asset_id)
            union.union(prior, record.asset_id)
    by_phash: dict[str, str] = {}
    labels_by_phash: dict[str, tuple[Label, int | None]] = {}
    for record in records:
        label = (record.label, record.grade)
        previous_label = labels_by_phash.setdefault(record.phash, label)
        if previous_label != label:
            raise ValueError("conflicting labels for similar images")
        prior = by_phash.setdefault(record.phash, record.asset_id)
        union.union(prior, record.asset_id)
    unique_phashes = sorted(by_phash)
    for left, right in find_near_duplicate_pairs(unique_phashes):
        if labels_by_phash[unique_phashes[left]] != labels_by_phash[unique_phashes[right]]:
            raise ValueError("conflicting labels for similar images")
        union.union(by_phash[unique_phashes[left]], by_phash[unique_phashes[right]])

    components: dict[str, list[AcceptedTrainingRecord]] = {}
    for record in records:
        components.setdefault(union.find(record.asset_id), []).append(record)
    ordered_components = sorted(
        (tuple(sorted(items, key=lambda item: item.asset_id)) for items in components.values()),
        key=lambda items: items[0].asset_id,
    )
    first_seen = sorted(min(item.collected_at for item in component) for component in ordered_components)
    cutoff = first_seen[ceil(0.8 * len(first_seen)) - 1] if first_seen else None

    assigned_components: dict[str, str] = {}
    remainder: list[tuple[AcceptedTrainingRecord, ...]] = []
    held_out = set(held_out_factories)
    held_out_domains = set(held_out_sources)
    for component in ordered_components:
        component_id = component[0].asset_id
        if any(item.is_gold for item in component):
            assigned_components[component_id] = GOLD
        elif any(item.factory in held_out for item in component):
            assigned_components[component_id] = FACTORY_HOLDOUT
        elif cutoff is not None and min(item.collected_at for item in component) >= cutoff:
            assigned_components[component_id] = TEMPORAL_HOLDOUT
        elif any(item.source_domain in held_out_domains for item in component):
            assigned_components[component_id] = CROSS_SOURCE
        else:
            remainder.append(component)

    strata: dict[tuple[str, ...], list[tuple[AcceptedTrainingRecord, ...]]] = {}
    for component in remainder:
        key = tuple(sorted({_class_key(item) for item in component}))
        strata.setdefault(key, []).append(component)
    rng = random.Random(seed)
    for key in sorted(strata):
        stratum = strata[key]
        rng.shuffle(stratum)
        validation_count = round(len(stratum) * DEFAULT_VAL_FRACTION)
        test_count = round(len(stratum) * DEFAULT_TEST_FRACTION)
        for index, component in enumerate(stratum):
            split = (
                VAL
                if index < validation_count
                else TEST
                if index < validation_count + test_count
                else TRAIN
            )
            assigned_components[component[0].asset_id] = split

    assignments = tuple(
        sorted(
            (
                item.asset_id,
                assigned_components[component[0].asset_id],
            )
            for component in ordered_components
            for item in component
        )
    )
    result = DatasetSplits(
        assignments=assignments,
        temporal_cutoff=cutoff,
        seed=seed,
        held_out_factories=tuple(sorted(held_out)),
        held_out_sources=tuple(sorted(held_out_domains)),
    )
    assert_no_connected_component_crosses_splits(records, result)
    return result


def assert_no_connected_component_crosses_splits(
    records: Sequence[AcceptedTrainingRecord], splits: DatasetSplits
) -> None:
    split_by_asset = splits.split_by_asset
    observed: dict[tuple[str, str], str] = {}
    for record in records:
        split = split_by_asset[record.asset_id]
        for field, value in (
            ("watch", record.watch_entity_id),
            ("family", record.split_family_id),
            ("content", record.content_cluster_id),
            ("seller", record.seller_group_id),
            ("batch", record.acquisition_batch_id),
        ):
            previous = observed.setdefault((field, value), split)
            if previous != split:
                raise ValueError(f"{field} group crosses dataset splits")
    by_phash: dict[str, str] = {}
    for record in records:
        split = split_by_asset[record.asset_id]
        previous = by_phash.setdefault(record.phash, split)
        if previous != split:
            raise ValueError("pHash group crosses dataset splits")
    unique_phashes = sorted(by_phash)
    for left, right in find_near_duplicate_pairs(unique_phashes):
        if by_phash[unique_phashes[left]] != by_phash[unique_phashes[right]]:
            raise ValueError("near-pHash group crosses dataset splits")


def _class_key(record: AcceptedTrainingRecord) -> str:
    return f"{record.label.value}:{record.grade if record.grade is not None else '-'}"
