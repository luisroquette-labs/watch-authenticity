from __future__ import annotations

import pytest

from dataset.splits import CROSS_SOURCE, FACTORY_HOLDOUT, GOLD, TEMPORAL_HOLDOUT, TEST
from evaluation.evaluate import SourceProbeReport
from evaluation.release_gate import (
    EvaluationReport,
    SliceMetric,
    build_slice_census,
    release_gate,
)
from tests.conftest import valid_run_spec
from tests.test_training_run import _png, _record, _seed, _valid_payloads
from training.data import load_training_data

MODEL_SHA = "b" * 64


@pytest.fixture
def release_bundle():
    base = valid_run_spec()
    manifest, splits, images = _valid_payloads(base)
    records = manifest["records"]
    split_values = splits["splits"]
    assert isinstance(records, list) and isinstance(split_values, dict)
    evaluation_splits = (TEST, CROSS_SOURCE, FACTORY_HOLDOUT, TEMPORAL_HOLDOUT, GOLD)
    for split_index, split in enumerate(evaluation_splits):
        for item_index in range(2):
            index = 1_000 + split_index * 2 + item_index
            image = _png(index)
            is_factory = split == FACTORY_HOLDOUT
            record = _record(
                base,
                f"evaluation-{split}-{item_index}",
                image,
                label="fake" if is_factory else "genuine",
                grade=4 if is_factory else None,
                angle="dial",
                source_domain=f"evaluation-{split_index}.example.co.za",
                subject_id=f"evaluation-watch-{split_index}",
                provenance_id=f"evaluation-proof-{index}",
                factory="vsf" if is_factory else None,
            )
            records.append(record)
            split_values[split].append(record["asset_id"])
            images[str(record["object_key"])] = image
    cos, spec = _seed(manifest, splits, images)
    return load_training_data(cos, spec)


@pytest.fixture
def empty_release_bundle():
    manifest, splits, images = _valid_payloads(valid_run_spec())
    cos, spec = _seed(manifest, splits, images)
    return load_training_data(cos, spec)


def _probe(freeze_sha256: str, *, blocked: bool = False) -> SourceProbeReport:
    if blocked:
        return SourceProbeReport(
            freeze_sha256,
            MODEL_SHA,
            0.90,
            0.85,
            0.95,
            0.40,
            0.50,
            ("source.example",),
        )
    return SourceProbeReport(
        freeze_sha256, MODEL_SHA, 0.80, 0.70, 0.85, 0.50, 0.65, ("source.example",)
    )


def _required_slices(bundle) -> tuple[SliceMetric, ...]:
    return tuple(
        SliceMetric(name, 0.80, max(1, count), 0.70, 0.90)
        for name, count in build_slice_census(bundle).counts
    )


def test_release_blocks_source_shortcut(release_bundle):
    decision = release_gate(_probe(release_bundle.freeze_sha256, blocked=True), None, None)

    assert not decision.allowed
    assert "source-domain shortcut" in decision.reasons


def test_release_reports_named_weak_slice_instead_of_hiding_it(release_bundle):
    report = EvaluationReport(
        freeze_sha256=release_bundle.freeze_sha256,
        model_sha256=MODEL_SHA,
        expected_calibration_error=0.03,
        gold_baseline_accuracy=0.78,
        slices=(
            *_required_slices(release_bundle),
            SliceMetric("grade-4/caseback", 0.49, 100, 0.40, 0.58),
        ),
    )
    decision = release_gate(
        _probe(release_bundle.freeze_sha256), report, release_bundle
    )

    assert not decision.allowed
    assert decision.reasons == ("weak required slice: grade-4/caseback",)


def test_release_requires_complete_evaluation(release_bundle):
    decision = release_gate(_probe(release_bundle.freeze_sha256), None, None)

    assert not decision.allowed
    assert decision.reasons == ("complete freeze-derived per-slice evaluation is missing",)


def test_release_derives_cross_source_and_gold_failures_from_slices(release_bundle):
    slices = [
        SliceMetric(metric.name, 0.95, 100, 0.90, 0.98)
        if metric.name == "standard"
        else SliceMetric(metric.name, 0.51, 100, 0.42, 0.60)
        if metric.name in {"cross-source", "gold"}
        else metric
        for metric in _required_slices(release_bundle)
    ]
    report = EvaluationReport(
        release_bundle.freeze_sha256, MODEL_SHA, 0.03, 0.99, tuple(slices)
    )

    decision = release_gate(
        _probe(release_bundle.freeze_sha256), report, release_bundle
    )

    assert "cross-source degradation exceeds release limit" in decision.reasons
    assert "gold-set regression" in decision.reasons


def test_release_cannot_replace_freeze_census_with_one_good_metric(release_bundle):
    report = EvaluationReport(
        release_bundle.freeze_sha256,
        MODEL_SHA,
        0.03,
        0.80,
        (SliceMetric("anything", 0.99, 100, 0.95, 1.0),),
    )

    decision = release_gate(
        _probe(release_bundle.freeze_sha256), report, release_bundle
    )

    assert not decision.allowed
    assert any("required evaluation slice is missing" in reason for reason in decision.reasons)


def test_release_rejects_claimed_metrics_for_empty_frozen_splits(empty_release_bundle):
    report = EvaluationReport(
        empty_release_bundle.freeze_sha256,
        MODEL_SHA,
        0.03,
        0.80,
        tuple(
            SliceMetric(name, 0.99, 100, 0.95, 1.0)
            for name in ("standard", "cross-source", "unseen-factory", "temporal", "gold")
        ),
    )

    decision = release_gate(
        _probe(empty_release_bundle.freeze_sha256), report, empty_release_bundle
    )

    assert not decision.allowed
    assert any("required evaluation split is empty" in reason for reason in decision.reasons)
    assert any("slice count differs from freeze" in reason for reason in decision.reasons)
