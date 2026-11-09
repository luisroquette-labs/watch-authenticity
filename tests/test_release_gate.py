from __future__ import annotations

from evaluation.evaluate import SourceProbeReport
from evaluation.release_gate import (
    EvaluationReport,
    SliceCensus,
    SliceMetric,
    release_gate,
)

FREEZE_SHA = "a" * 64
MODEL_SHA = "b" * 64


def _probe(*, blocked: bool = False) -> SourceProbeReport:
    if blocked:
        return SourceProbeReport(
            FREEZE_SHA, MODEL_SHA, 0.90, 0.85, 0.95, 0.40, 0.50, ("source.example",)
        )
    return SourceProbeReport(
        FREEZE_SHA, MODEL_SHA, 0.80, 0.70, 0.85, 0.50, 0.65, ("source.example",)
    )


def _required_slices() -> tuple[SliceMetric, ...]:
    return tuple(
        SliceMetric(name, 0.80, 100, 0.70, 0.90)
        for name in ("standard", "cross-source", "unseen-factory", "temporal", "gold")
    )


def test_release_blocks_source_shortcut():
    decision = release_gate(_probe(blocked=True), None, None)

    assert not decision.allowed
    assert "source-domain shortcut" in decision.reasons


def test_release_reports_named_weak_slice_instead_of_hiding_it():
    report = EvaluationReport(
        freeze_sha256=FREEZE_SHA,
        model_sha256=MODEL_SHA,
        expected_calibration_error=0.03,
        gold_baseline_accuracy=0.78,
        slices=(
            *_required_slices(),
            SliceMetric("grade-4/caseback", 0.49, 100, 0.40, 0.58),
        ),
    )
    census = SliceCensus(
        FREEZE_SHA,
        tuple(sorted({metric.name for metric in report.slices})),
    )

    decision = release_gate(_probe(), report, census)

    assert not decision.allowed
    assert decision.reasons == ("weak required slice: grade-4/caseback",)


def test_release_requires_complete_evaluation():
    decision = release_gate(_probe(), None, None)

    assert not decision.allowed
    assert decision.reasons == ("complete freeze-derived per-slice evaluation is missing",)


def test_release_derives_cross_source_and_gold_failures_from_slices():
    slices = [
        SliceMetric(metric.name, 0.95, 100, 0.90, 0.98)
        if metric.name == "standard"
        else SliceMetric(metric.name, 0.51, 100, 0.42, 0.60)
        if metric.name in {"cross-source", "gold"}
        else metric
        for metric in _required_slices()
    ]
    report = EvaluationReport(FREEZE_SHA, MODEL_SHA, 0.03, 0.99, tuple(slices))
    census = SliceCensus(FREEZE_SHA, tuple(sorted(metric.name for metric in slices)))

    decision = release_gate(_probe(), report, census)

    assert "cross-source degradation exceeds release limit" in decision.reasons
    assert "gold-set regression" in decision.reasons
