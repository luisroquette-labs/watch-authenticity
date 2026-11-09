from __future__ import annotations

from evaluation.evaluate import SourceProbeReport
from evaluation.release_gate import EvaluationReport, SliceMetric, release_gate


def _probe(*, blocked: bool = False) -> SourceProbeReport:
    if blocked:
        return SourceProbeReport(0.90, 0.85, 0.95, 0.40, 0.50, ("source.example",))
    return SourceProbeReport(0.80, 0.70, 0.85, 0.50, 0.65, ("source.example",))


def _required_slices() -> tuple[SliceMetric, ...]:
    return tuple(
        SliceMetric(name, 0.80, 100)
        for name in ("standard", "cross-source", "unseen-factory", "temporal", "gold")
    )


def test_release_blocks_source_shortcut():
    decision = release_gate(_probe(blocked=True), None)

    assert not decision.allowed
    assert "source-domain shortcut" in decision.reasons


def test_release_reports_named_weak_slice_instead_of_hiding_it():
    report = EvaluationReport(
        macro_balanced_accuracy=0.90,
        expected_calibration_error=0.03,
        cross_source_delta=0.04,
        gold_accuracy=0.89,
        gold_baseline_accuracy=0.88,
        slices=(*_required_slices(), SliceMetric("grade-4/caseback", 0.49, 100)),
    )

    decision = release_gate(_probe(), report)

    assert not decision.allowed
    assert decision.reasons == ("weak required slice: grade-4/caseback",)


def test_release_requires_complete_evaluation():
    decision = release_gate(_probe(), None)

    assert not decision.allowed
    assert decision.reasons == ("complete per-slice evaluation is missing",)
