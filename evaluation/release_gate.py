"""Release decision that keeps weak slices and shortcut evidence visible."""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.rules import goldset_regression_gate, publish_gate
from evaluation.evaluate import SourceProbeReport

_REQUIRED_EVALUATION_SLICES = frozenset(
    {"standard", "cross-source", "unseen-factory", "temporal", "gold"}
)


@dataclass(frozen=True)
class SliceMetric:
    name: str
    balanced_accuracy: float
    count: int
    ci_low: float | None = None
    ci_high: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not 0 <= self.balanced_accuracy <= 1 or self.count < 1:
            raise ValueError("slice metric is invalid")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("slice confidence interval must be complete")
        low, high = self.ci_low, self.ci_high
        if low is not None and high is not None and not 0 <= low <= high <= 1:
            raise ValueError("slice confidence interval is invalid")


@dataclass(frozen=True)
class EvaluationReport:
    macro_balanced_accuracy: float
    expected_calibration_error: float
    cross_source_delta: float
    gold_accuracy: float
    gold_baseline_accuracy: float
    slices: tuple[SliceMetric, ...]

    def __post_init__(self) -> None:
        values = (
            self.macro_balanced_accuracy,
            self.expected_calibration_error,
            self.cross_source_delta,
            self.gold_accuracy,
            self.gold_baseline_accuracy,
        )
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            raise ValueError("evaluation metrics must be finite probabilities")
        if not self.slices or len({item.name for item in self.slices}) != len(self.slices):
            raise ValueError("evaluation requires unique named slices")


@dataclass(frozen=True)
class ReleaseDecision:
    allowed: bool
    reasons: tuple[str, ...]


def release_gate(
    source_probe: SourceProbeReport,
    report: EvaluationReport | None,
    *,
    minimum_slice_accuracy: float = 0.50,
    maximum_calibration_error: float = 0.05,
    maximum_cross_source_delta: float = 0.10,
) -> ReleaseDecision:
    reasons: list[str] = []
    if source_probe.blocked:
        reasons.append("source-domain shortcut")
    if report is None:
        reasons.append("complete per-slice evaluation is missing")
        return ReleaseDecision(False, tuple(reasons))
    if report.expected_calibration_error > maximum_calibration_error:
        reasons.append("calibration error exceeds release limit")
    if not publish_gate(
        report.macro_balanced_accuracy,
        report.macro_balanced_accuracy - report.cross_source_delta,
        maximum_cross_source_delta,
    ):
        reasons.append("cross-source degradation exceeds release limit")
    if not goldset_regression_gate(report.gold_accuracy, report.gold_baseline_accuracy):
        reasons.append("gold-set regression")
    missing = sorted(_REQUIRED_EVALUATION_SLICES - {metric.name for metric in report.slices})
    reasons.extend(f"required evaluation slice is missing: {name}" for name in missing)
    reasons.extend(
        f"weak required slice: {metric.name}"
        for metric in report.slices
        if metric.balanced_accuracy < minimum_slice_accuracy
    )
    return ReleaseDecision(not reasons, tuple(reasons))
