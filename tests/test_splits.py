"""Tests for the six-way dataset split builder in `dataset/splits.py`.

Covers this step's Success Criteria: all six splits are produced from
manifest rows, and gold + factory-holdout + cross-source rows never leak
into train — the no-leak guarantee the Generalization & Regression
Evaluation Rigor rubric (weight 0.25, this task's heaviest) depends on.
"""

from datetime import date

import pytest

from core.types import Label, ManifestRecord
from dataset.splits import (
    CROSS_SOURCE,
    FACTORY_HOLDOUT,
    GOLD,
    TEST,
    TRAIN,
    VAL,
    build_splits,
)


def _record(path: str, source: str, label: Label, grade: int | None) -> ManifestRecord:
    return ManifestRecord(
        path=path,
        source=source,
        url=f"https://example.test/{path}",
        label=label,
        grade=grade,
        phash=f"phash-{path}",
        scraped_at=date(2026, 1, 1),
    )


@pytest.fixture
def manifest_rows() -> list[ManifestRecord]:
    """A manifest with every source category this step's splits must separate.

    `chrono24` (genuine, studio-style dealer photos) and `reptime` (fake,
    forum-style) are the typical pairing and stay eligible for train/val/
    test. `watchuseek` (genuine, casual forum style) and `chinese_replica`
    (fake, studio/catalog style) are the atypical pairing the cross-source
    split exists to isolate. `unseen_forum` stands in for a factory/source
    the training manifest reserves exclusively for factory-holdout.
    """
    genuine_studio = [
        _record(f"genuine_chrono24_{i}.jpg", "chrono24", Label.GENUINE, None) for i in range(20)
    ]
    genuine_casual = [
        _record(f"genuine_watchuseek_{i}.jpg", "watchuseek", Label.GENUINE, None)
        for i in range(10)
    ]
    fake_forum = [_record(f"fake_reptime_{i}.jpg", "reptime", Label.FAKE, 3) for i in range(20)]
    fake_studio = [
        _record(f"fake_studio_{i}.jpg", "chinese_replica", Label.FAKE, 2) for i in range(10)
    ]
    fake_unseen_factory = [
        _record(f"fake_unseen_{i}.jpg", "unseen_forum", Label.FAKE, 3) for i in range(8)
    ]
    gold = [_record(f"gold_{i}.jpg", "chrono24", Label.GENUINE, None) for i in range(5)]
    return (
        genuine_studio + genuine_casual + fake_forum + fake_studio + fake_unseen_factory + gold
    )


def _gold_paths(records: list[ManifestRecord]) -> set[str]:
    return {record.path for record in records if record.path.startswith("gold_")}


def test_build_splits_returns_all_six_split_keys(manifest_rows: list[ManifestRecord]):
    splits = build_splits(
        manifest_rows,
        gold_paths=_gold_paths(manifest_rows),
        holdout_sources={"unseen_forum"},
    )

    assert set(splits) == {TRAIN, VAL, TEST, CROSS_SOURCE, FACTORY_HOLDOUT, GOLD}


def test_gold_set_never_appears_in_train(manifest_rows: list[ManifestRecord]):
    gold_paths = _gold_paths(manifest_rows)
    splits = build_splits(manifest_rows, gold_paths=gold_paths, holdout_sources={"unseen_forum"})

    train_paths = {record.path for record in splits[TRAIN]}
    assert train_paths.isdisjoint(gold_paths)
    assert {record.path for record in splits[GOLD]} == gold_paths


def test_factory_holdout_never_appears_in_train(manifest_rows: list[ManifestRecord]):
    holdout_sources = {"unseen_forum"}
    splits = build_splits(
        manifest_rows, gold_paths=_gold_paths(manifest_rows), holdout_sources=holdout_sources
    )

    train_sources = {record.source for record in splits[TRAIN]}
    assert train_sources.isdisjoint(holdout_sources)
    assert all(record.source in holdout_sources for record in splits[FACTORY_HOLDOUT])
    assert len(splits[FACTORY_HOLDOUT]) == 8


def test_cross_source_split_pairs_genuine_casual_with_fake_studio(
    manifest_rows: list[ManifestRecord],
):
    splits = build_splits(
        manifest_rows, gold_paths=_gold_paths(manifest_rows), holdout_sources={"unseen_forum"}
    )

    cross_source = splits[CROSS_SOURCE]
    assert any(record.label is Label.GENUINE for record in cross_source)
    assert any(record.label is Label.FAKE for record in cross_source)
    assert all(
        (record.label is Label.GENUINE and record.source == "watchuseek")
        or (record.label is Label.FAKE and record.source == "chinese_replica")
        for record in cross_source
    )
    assert len(cross_source) == 20


def test_cross_source_split_never_appears_in_train(manifest_rows: list[ManifestRecord]):
    splits = build_splits(
        manifest_rows, gold_paths=_gold_paths(manifest_rows), holdout_sources={"unseen_forum"}
    )

    cross_source_paths = {record.path for record in splits[CROSS_SOURCE]}
    train_paths = {record.path for record in splits[TRAIN]}
    assert train_paths.isdisjoint(cross_source_paths)


def test_train_val_test_partition_the_remaining_pool_without_overlap(
    manifest_rows: list[ManifestRecord],
):
    splits = build_splits(
        manifest_rows, gold_paths=_gold_paths(manifest_rows), holdout_sources={"unseen_forum"}
    )

    reserved_paths = (
        {record.path for record in splits[GOLD]}
        | {record.path for record in splits[FACTORY_HOLDOUT]}
        | {record.path for record in splits[CROSS_SOURCE]}
    )
    train_paths = {record.path for record in splits[TRAIN]}
    val_paths = {record.path for record in splits[VAL]}
    test_paths = {record.path for record in splits[TEST]}
    all_paths = {record.path for record in manifest_rows}

    assert train_paths | val_paths | test_paths | reserved_paths == all_paths
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
