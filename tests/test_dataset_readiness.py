from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from acquisition.types import Angle, DatasetFreeze, EvidenceTier, RightsStatus, TrainingRecord
from core.types import Label
from dataset import readiness
from dataset.freeze import (
    CandidateObjectInventory,
    FrozenDatasetRefused,
    freeze_dataset,
    require_frozen_asset_read,
)
from dataset.readiness import evaluate_readiness
from dataset.splits import TRAIN, DatasetSplits, build_splits
from infra.ibm_types import ObjectRef
from tests.conftest import NOW


def _record(
    index: int,
    *,
    label: Label = Label.GENUINE,
    grade: int | None = None,
    angle: Angle | None = Angle.DIAL,
    source: str | None = None,
    style: str | None = None,
    gold: bool = False,
    subtype: str | None = None,
    rejection_reference: str | None = None,
) -> TrainingRecord:
    digest = f"{index:064x}"
    return TrainingRecord(
        asset_id=f"asset-{index}",
        candidate_id=f"candidate-{index}",
        object_ref=ObjectRef("bucket", f"datasets/derivatives/{digest}", digest, 1, "etag"),
        reference_id="rolex-submariner-126610ln",
        watch_entity_id=f"watch-{index}",
        content_cluster_id=f"content-{index}",
        seller_group_id=f"seller-{index}",
        acquisition_batch_id=f"batch-{index}",
        angle=angle,
        collected_at=NOW + timedelta(seconds=index),
        original_sha256=digest,
        is_augmented=False,
        is_gold=gold,
        rejection_subtype=subtype,
        rejection_reference_id=rejection_reference,
        source_domain=source or f"source-{index % 3}.example.co.za",
        capture_style=style or ("studio" if index % 2 else "casual"),
        label=label,
        grade=grade,
        factory="vsf" if label is Label.FAKE else None,
        evidence_tier=EvidenceTier.A if gold else EvidenceTier.B,
        evidence_record_id=f"evidence-{index}",
        rights_record_id=f"rights-{index}",
        image_sha256=digest,
        phash=f"{index:016x}",
    )


def _small_complete_records() -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    index = 1
    for label, grade in ((Label.GENUINE, None), *((Label.FAKE, grade) for grade in range(1, 5))):
        for angle in Angle:
            records.append(_record(index, label=label, grade=grade, angle=angle, gold=angle is Angle.DIAL))
            index += 1
    records.append(
        _record(index, label=Label.REJECTION, angle=None, gold=True, subtype="no-watch")
    )
    index += 1
    for reference in range(5):
        records.append(
            _record(
                index,
                label=Label.REJECTION,
                angle=None,
                subtype="non-submariner",
                rejection_reference=f"other-{reference}",
            )
        )
        index += 1
    return records


def test_small_complete_dataset_passes_when_thresholds_are_scaled(monkeypatch):
    records = _small_complete_records()
    monkeypatch.setattr(readiness, "MIN_CELL_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_CELL_ENTITIES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_SUBJECTS", 1)
    monkeypatch.setattr(readiness, "MIN_GOLD_IMAGES", 1)
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, "rolex-submariner-126610ln", splits=splits)

    assert report.ready
    assert report.reasons == ()


def test_augmented_or_repeated_original_does_not_increase_cell_count(monkeypatch):
    original = _record(1)
    repeated = replace(original, asset_id="crop-2", watch_entity_id="watch-2")
    augmented = replace(original, asset_id="augmented-3", original_sha256="3" * 64, is_augmented=True)
    records = [original, repeated, augmented]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, original.reference_id, splits=splits)

    dial = next(cell for cell in report.cell_counts if cell[0] == "genuine/dial")
    assert dial == ("genuine/dial", 1, 2)


def test_source_share_above_forty_percent_is_reported(monkeypatch):
    records = [
        _record(index, source="dominant.example.co.za" if index < 4 else f"other-{index}.co.za")
        for index in range(1, 6)
    ]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits)

    assert any("exceeds 40%" in reason for reason in report.reasons)


def test_exactly_forty_percent_source_share_is_allowed():
    records = [
        _record(
            index,
            source="boundary.example.co.za" if index <= 2 else f"other-{index}.co.za",
        )
        for index in range(1, 6)
    ]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits)

    assert not any("boundary.example.co.za exceeds 40%" in reason for reason in report.reasons)


def test_grouped_splits_keep_transitive_relations_together():
    first = _record(1)
    second = replace(_record(2), watch_entity_id=first.watch_entity_id)
    third = replace(_record(3), content_cluster_id=second.content_cluster_id)
    fourth = _record(4, gold=True)
    records = [first, second, third, fourth, _record(5)]

    splits = build_splits(records, seed=42, held_out_factories=set())

    assigned = splits.split_by_asset
    assert assigned[first.asset_id] == assigned[second.asset_id] == assigned[third.asset_id]
    assert assigned[fourth.asset_id] == "gold"
    assert splits.temporal_cutoff is not None


def test_grouped_splits_are_deterministic_for_input_order():
    records = [_record(index) for index in range(1, 21)]

    forward = build_splits(records, seed=42, held_out_factories=set())
    reverse = build_splits(list(reversed(records)), seed=42, held_out_factories=set())

    assert forward == reverse


def test_readiness_thresholds_match_the_mvp_contract():
    assert (
        readiness.MIN_CELL_IMAGES,
        readiness.MIN_CELL_ENTITIES,
        readiness.MIN_REJECTION_IMAGES,
        readiness.MIN_REJECTION_SUBJECTS,
        readiness.MIN_GOLD_IMAGES,
    ) == (500, 100, 500, 250, 200)


def test_freeze_excludes_expired_rights_and_reports_impacted_artifact(
    rights, policy_set
):
    record = replace(_record(1), rights_record_id=rights.rights_record_id)
    splits = build_splits([record], seed=42, held_out_factories=set())
    raw = ObjectRef("bucket", "datasets/raw/" + "a" * 64, "a" * 64, 1, "raw")
    evidence = ObjectRef(
        "bucket", "datasets/reviews/" + "c" * 64, "c" * 64, 1, "evidence"
    )
    inventory = CandidateObjectInventory((raw,), (record.object_ref,), (evidence,))
    previous = DatasetFreeze("d" * 64, "e" * 64, "f" * 64, (record.asset_id,), (), ())
    artifact = "9" * 64
    expired = replace(
        rights,
        effective_at=NOW - timedelta(days=1),
        expires_at=NOW,
        status=RightsStatus.ACTIVE,
    )

    result = freeze_dataset(
        [record],
        splits,
        policy_set,
        {expired.rights_record_id: expired},
        previous_freeze=previous,
        published_artifact_hashes={artifact},
        now=NOW,
        object_inventory={record.candidate_id: inventory},
        artifact_lineage={artifact: {previous.manifest_sha256}},
    )

    assert result.freeze.blocked
    assert result.freeze.eligible_asset_ids == ()
    assert result.freeze.excluded_asset_ids == (record.asset_id,)
    assert result.impacted_artifact_report is not None
    assert result.impacted_artifact_report.impacted_artifact_hashes == (artifact,)
    assert result.data_disposition_reports[0].status == "pending"
    with pytest.raises(FrozenDatasetRefused):
        require_frozen_asset_read(
            result.freeze,
            record.asset_id,
            {expired.rights_record_id: expired},
            policy_set,
            now=NOW,
        )


def test_prior_freeze_rechecks_current_rights(policy_set, rights):
    record = replace(_record(1), rights_record_id=rights.rights_record_id)
    freeze = DatasetFreeze(
        "a" * 64,
        "b" * 64,
        "c" * 64,
        (record.asset_id,),
        (),
        (),
        asset_rights=((record.asset_id, rights.rights_record_id),),
    )
    require_frozen_asset_read(
        freeze,
        record.asset_id,
        {rights.rights_record_id: rights},
        policy_set,
        now=NOW,
    )

    revoked = replace(rights, status=RightsStatus.REVOKED)
    with pytest.raises(FrozenDatasetRefused, match="current rights"):
        require_frozen_asset_read(
            freeze,
            record.asset_id,
            {rights.rights_record_id: revoked},
            policy_set,
            now=NOW,
        )


def test_freeze_hashes_do_not_depend_on_record_order(policy_set, rights):
    records = [
        replace(_record(index), rights_record_id=rights.rights_record_id)
        for index in range(1, 6)
    ]
    splits = build_splits(records, seed=42, held_out_factories=set())

    first = freeze_dataset(
        records,
        splits,
        policy_set,
        {rights.rights_record_id: rights},
        previous_freeze=None,
        published_artifact_hashes=(),
        now=NOW,
        object_inventory={},
        artifact_lineage={},
    )
    second = freeze_dataset(
        list(reversed(records)),
        splits,
        policy_set,
        {rights.rights_record_id: rights},
        previous_freeze=None,
        published_artifact_hashes=(),
        now=NOW,
        object_inventory={},
        artifact_lineage={},
    )

    assert first.freeze == second.freeze


def test_freeze_rejects_noncanonical_split_assignments(policy_set, rights):
    records = [
        replace(_record(index), rights_record_id=rights.rights_record_id)
        for index in range(1, 6)
    ]
    canonical = build_splits(records, seed=42, held_out_factories=set())
    first_asset, first_split = canonical.assignments[0]
    forged = replace(
        canonical,
        assignments=((first_asset, TRAIN if first_split != TRAIN else "val"),)
        + canonical.assignments[1:],
    )

    with pytest.raises(ValueError, match="canonical grouped splits"):
        freeze_dataset(
            records,
            forged,
            policy_set,
            {rights.rights_record_id: rights},
            previous_freeze=None,
            published_artifact_hashes=(),
            now=NOW,
            object_inventory={},
            artifact_lineage={},
        )


def test_augmented_gold_does_not_satisfy_gold_gate(monkeypatch):
    records = [replace(record, is_augmented=record.is_gold) for record in _small_complete_records()]
    monkeypatch.setattr(readiness, "MIN_CELL_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_CELL_ENTITIES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_SUBJECTS", 1)
    monkeypatch.setattr(readiness, "MIN_GOLD_IMAGES", 1)
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits)

    assert any(reason.startswith("gold:") for reason in report.reasons)
