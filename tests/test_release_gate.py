from __future__ import annotations

import pytest

from evaluation.evaluate import SourceProbeReport
from evaluation.release_gate import (
    EvaluationReport,
    SliceMetric,
    release_gate,
)
from tests.conftest import valid_run_spec
from tests.test_training_run import _seed, _valid_payloads
from training.data import load_training_data

MODEL_SHA = "b" * 64


@pytest.fixture
def release_bundle():
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


def _required_slices() -> tuple[SliceMetric, ...]:
    return tuple(
        SliceMetric(name, 0.80, 100, 0.70, 0.90)
        for name in ("standard", "cross-source", "unseen-factory", "temporal", "gold")
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
            *_required_slices(),
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
        for metric in _required_slices()
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
