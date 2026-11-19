from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import timedelta

import pytest

from acquisition.types import Angle, DatasetFreeze, EvidenceTier, RightsStatus, TrainingRecord
from core.types import Label
from dataset import readiness
from dataset.angle_review import record_usability_review
from dataset.freeze import (
    CandidateObjectInventory,
    FrozenDatasetRefused,
    freeze_dataset,
    require_frozen_asset_read,
)
from dataset.readiness import evaluate_readiness
from dataset.splits import (
    CROSS_SOURCE,
    TRAIN,
    DatasetSplits,
    assert_no_connected_component_crosses_splits,
    build_splits,
)
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
        phash=hashlib.sha256(str(index).encode()).hexdigest()[:16],
        usability_review=None if label is Label.REJECTION else record_usability_review(f"candidate-{index}", digest, Angle(angle), True, True, "angle-reviewer", NOW),
    )


def _small_complete_records() -> list[TrainingRecord]:
    records: list[TrainingRecord] = []
    index = 1
    for label, grade in ((Label.GENUINE, None), *((Label.FAKE, grade) for grade in range(1, 5))):
        for angle in Angle:
            records.append(_record(index, label=label, grade=grade, angle=angle,
                                   source=f"source-{index % 4}.example.co.za",
                                   gold=angle is Angle.DIAL))
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
    splits = DatasetSplits(tuple((record.asset_id, CROSS_SOURCE if record.source_domain == "source-0.example.co.za" else TRAIN) for record in records), NOW, 42,
                           held_out_sources=("source-0.example.co.za",))

    report = evaluate_readiness(records, "rolex-submariner-126610ln", splits=splits, required_angles=tuple(Angle))

    assert report.ready
    assert report.reasons == ()


def test_patek_readiness_uses_its_views_and_non_target_rejections(monkeypatch):
    reference = "patek-nautilus-58111g001"
    required_angles = (Angle.DIAL, Angle.CROWN, Angle.CLASP, Angle.CASEBACK)
    records = [
        replace(
            record,
            reference_id=reference,
            rejection_subtype=(
                "non-target-reference" if record.rejection_subtype == "non-submariner"
                else record.rejection_subtype
            ),
        )
        for record in _small_complete_records()
        if record.angle is None or record.angle in required_angles
    ]
    monkeypatch.setattr(readiness, "MIN_CELL_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_CELL_ENTITIES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_SUBJECTS", 1)
    monkeypatch.setattr(readiness, "MIN_GOLD_IMAGES", 1)
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, reference, splits=splits, required_angles=required_angles)

    assert not any("rejection:" in reason for reason in report.reasons)
    assert not any("cyclops" in reason or "rehaut" in reason for reason in report.reasons)
    assert report.required_angles == required_angles
    assert len(report.cell_counts) == 20
    assert not any("cyclops" in cell or "rehaut" in cell for cell, _, _ in report.cell_counts)


def test_training_style_shortcut_and_missing_unseen_source_block_readiness():
    records = [
        _record(1, label=Label.GENUINE, style="studio"),
        _record(2, label=Label.FAKE, grade=1, style="casual"),
        _record(3, label=Label.GENUINE, style="casual"),
        _record(4, label=Label.FAKE, grade=1, style="studio"),
    ]
    splits = DatasetSplits(tuple((record.asset_id, CROSS_SOURCE if record.asset_id in {"asset-3", "asset-4"} else TRAIN)
                                  for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits,
                                required_angles=(Angle.DIAL,))

    assert "train/genuine: requires studio and casual examples" in report.reasons
    assert "train/fake: requires studio and casual examples" in report.reasons
    assert "cross-source: source also appears in train" in report.reasons


def test_explicit_source_holdout_leaves_both_styles_per_class_in_train():
    records = [
        _record(index, label=label, grade=1 if label is Label.FAKE else None,
                style=style, source=source)
        for index, label, style, source in (
            (1, Label.GENUINE, "casual", "unseen-original.example"),
            (2, Label.FAKE, "studio", "unseen-fake.example"),
            (3, Label.GENUINE, "studio", "studio-original.example"),
            (4, Label.GENUINE, "casual", "casual-original.example"),
            (5, Label.FAKE, "studio", "studio-fake.example"),
            (6, Label.FAKE, "casual", "casual-fake.example"),
        )
    ] + [_record(index, label=Label.REJECTION, angle=None, subtype="no-watch") for index in (7, 8)]
    splits = build_splits(records, seed=42, held_out_factories=(),
                          held_out_sources={"unseen-original.example", "unseen-fake.example"})
    train = {record.asset_id for record in records if splits.split_by_asset[record.asset_id] == TRAIN}
    cross = {record.asset_id for record in records if splits.split_by_asset[record.asset_id] == CROSS_SOURCE}

    assert cross == {"asset-1", "asset-2"}
    assert {record.capture_style for record in records if record.asset_id in train and record.label is Label.GENUINE} == {"studio", "casual"}
    assert {record.capture_style for record in records if record.asset_id in train and record.label is Label.FAKE} == {"studio", "casual"}


def test_other_watch_rejection_cannot_name_its_target_reference():
    with pytest.raises(ValueError, match="cannot name the target"):
        _record(
            99,
            label=Label.REJECTION,
            angle=None,
            subtype="non-target-reference",
            rejection_reference="rolex-submariner-126610ln",
        )


@pytest.mark.parametrize("target, self_reference", [
    ("rolex-datejust-126334", "rolex-datejust-41-126334"),
    ("rolex-datejust-41-126334", "rolex-datejust-126334"),
])
def test_rejection_aliases_do_not_count_as_other_references(catalog_fixture, target, self_reference):
    aliases = catalog_fixture.catalog.aliases
    records = [
        replace(_record(index, label=Label.REJECTION, angle=None,
                        subtype="non-target-reference", rejection_reference=reference),
                reference_id=target)
        for index, reference in enumerate(
            (self_reference, "other-1", "other-2", "other-3", "other-4", "other-5"), 1
        )
    ]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, target, splits=splits,
                                required_angles=(Angle.DIAL,), reference_aliases=aliases)

    assert "rejection: other-watch example names the target reference or an alias" in report.reasons
    assert "rejection: requires five non-target-reference references" not in report.reasons


def test_augmented_or_repeated_original_does_not_increase_cell_count(monkeypatch):
    original = _record(1)
    repeated = replace(original, asset_id="crop-2", watch_entity_id="watch-2")
    augmented = replace(original, asset_id="augmented-3", original_sha256="3" * 64, is_augmented=True)
    records = [original, repeated, augmented]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, original.reference_id, splits=splits, required_angles=tuple(Angle))

    dial = next(cell for cell in report.cell_counts if cell[0] == "genuine/dial")
    assert dial == ("genuine/dial", 1, 1)


def test_ambiguous_content_links_do_not_inflate_independent_watch_count():
    first = _record(1)
    second = replace(_record(2), watch_entity_id=first.watch_entity_id)
    third = replace(_record(3), content_cluster_id=second.content_cluster_id)
    records = [first, second, third]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, first.reference_id, splits=splits, required_angles=tuple(Angle))

    assert next(cell for cell in report.cell_counts if cell[0] == "genuine/dial") == (
        "genuine/dial", 2, 1
    )


def test_cross_angle_content_link_reduces_dial_entity_count():
    first = _record(1)
    bridge = replace(_record(2, angle=Angle.CROWN), watch_entity_id=first.watch_entity_id)
    other_view = replace(_record(3, angle=Angle.CROWN), content_cluster_id=bridge.content_cluster_id)
    second_dial = replace(_record(4), watch_entity_id=other_view.watch_entity_id)
    records = [first, bridge, other_view, second_dial]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, first.reference_id, splits=splits, required_angles=tuple(Angle))

    assert next(cell for cell in report.cell_counts if cell[0] == "genuine/dial") == (
        "genuine/dial", 2, 1
    )


def test_rejection_subject_count_uses_content_components(monkeypatch):
    monkeypatch.setattr(readiness, "MIN_REJECTION_SUBJECTS", 2)
    first = _record(1, label=Label.REJECTION, angle=None, subtype="no-watch")
    repeated = replace(
        _record(2, label=Label.REJECTION, angle=None, subtype="no-watch"),
        content_cluster_id=first.content_cluster_id,
    )
    splits = DatasetSplits(((first.asset_id, TRAIN), (repeated.asset_id, TRAIN)), NOW, 42)

    report = evaluate_readiness([first, repeated], first.reference_id, splits=splits, required_angles=tuple(Angle))

    assert "rejection: requires 250 distinct subjects" in report.reasons


def test_ambiguous_aliases_cannot_hide_source_or_seller_dominance():
    records = [
        _record(1, source="dominant.example.co.za"),
        _record(2, source="dominant.example.co.za"),
        _record(3, source="other.example.co.za"),
    ]
    records.extend(
        replace(_record(index, source="other.example.co.za"), content_cluster_id=records[2].content_cluster_id)
        for index in range(4, 15)
    )
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits, required_angles=tuple(Angle))

    assert any("dominant.example.co.za exceeds 40%" in reason for reason in report.reasons)
    assert any("seller seller-1 exceeds 10%" in reason for reason in report.reasons)


def test_source_share_above_forty_percent_is_reported(monkeypatch):
    records = [
        _record(index, source="dominant.example.co.za" if index < 4 else f"other-{index}.co.za")
        for index in range(1, 6)
    ]
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits, required_angles=tuple(Angle))

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

    report = evaluate_readiness(records, records[0].reference_id, splits=splits, required_angles=tuple(Angle))

    assert not any("boundary.example.co.za exceeds 40%" in reason for reason in report.reasons)


def test_dominant_source_and_seller_are_seen_across_multiple_photos_per_entity():
    records: list[TrainingRecord] = []
    for entity in range(1, 11):
        dominant = replace(
            _record(entity * 2),
            asset_id=f"z-dominant-{entity}",
            watch_entity_id=f"watch-{entity}",
            source_domain="dominant.co.za",
            seller_group_id="dominant-seller",
        )
        alternate = replace(
            _record(entity * 2 + 1),
            asset_id=f"a-alternate-{entity}",
            watch_entity_id=f"watch-{entity}",
            source_domain=f"other-{entity}.co.za",
            seller_group_id=f"other-seller-{entity}",
        )
        records.extend((alternate, dominant))
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits, required_angles=tuple(Angle))

    assert any("source dominant.co.za exceeds 40%" in reason for reason in report.reasons)
    assert any("seller dominant-seller exceeds 10%" in reason for reason in report.reasons)


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


def test_near_phash_cannot_cross_train_and_gold_splits():
    first = replace(_record(1), phash="bed0d503522f2f54")
    second = replace(_record(2, gold=True), phash="bed0d503522f2fd0")

    splits = build_splits([first, second], seed=42, held_out_factories=set())

    assert splits.split_by_asset[first.asset_id] == splits.split_by_asset[second.asset_id] == "gold"
    with pytest.raises(ValueError, match="near-pHash group crosses dataset splits"):
        assert_no_connected_component_crosses_splits(
            [first, second], DatasetSplits(((first.asset_id, TRAIN), (second.asset_id, "gold")), NOW, 42)
        )


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
    rights, policy_set, catalog_fixture
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
        reference_catalog=catalog_fixture.catalog,
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


def test_freeze_hashes_do_not_depend_on_record_order(policy_set, rights, catalog_fixture):
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
        reference_catalog=catalog_fixture.catalog,
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
        reference_catalog=catalog_fixture.catalog,
    )

    assert first.freeze == second.freeze


def test_freeze_commits_source_holdout_to_split_hash(policy_set, rights, catalog_fixture):
    records = [replace(_record(index, source="heldout.example" if index == 1 else f"source-{index}.example"),
                       rights_record_id=rights.rights_record_id) for index in range(1, 7)]

    def frozen(held_out_sources):
        splits = build_splits(records, seed=42, held_out_factories=(),
                              held_out_sources=held_out_sources)
        return freeze_dataset(records, splits, policy_set, {rights.rights_record_id: rights},
                              previous_freeze=None, published_artifact_hashes=(), now=NOW,
                              object_inventory={}, artifact_lineage={},
                              reference_catalog=catalog_fixture.catalog).freeze

    assert frozen({"heldout.example"}).split_sha256 != frozen(()).split_sha256


def test_freeze_rejects_noncanonical_split_assignments(policy_set, rights, catalog_fixture):
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
            reference_catalog=catalog_fixture.catalog,
        )


def test_augmented_gold_does_not_satisfy_gold_gate(monkeypatch):
    records = [replace(record, is_augmented=record.is_gold) for record in _small_complete_records()]
    monkeypatch.setattr(readiness, "MIN_CELL_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_CELL_ENTITIES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_IMAGES", 1)
    monkeypatch.setattr(readiness, "MIN_REJECTION_SUBJECTS", 1)
    monkeypatch.setattr(readiness, "MIN_GOLD_IMAGES", 1)
    splits = DatasetSplits(tuple((record.asset_id, TRAIN) for record in records), NOW, 42)

    report = evaluate_readiness(records, records[0].reference_id, splits=splits, required_angles=tuple(Angle))

    assert any(reason.startswith("gold:") for reason in report.reasons)


def test_training_contract_refuses_unreviewed_angle_and_preserves_raw_audit():
    from dataset.angle_review import usability_review_from_json
    from dataset.freeze import _record_json

    record = _record(1, angle=Angle.CLASP)
    with pytest.raises(ValueError, match="usability review"):
        replace(record, usability_review=None)
    raw = _record_json(record)
    assert raw["original_sha256"] == record.original_sha256
    restored = usability_review_from_json(raw["usability_review"])
    assert restored == record.usability_review
    assert restored.angle is Angle.CLASP
