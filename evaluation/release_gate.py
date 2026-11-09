"""Freeze-bound release decision with complete, uncertainty-aware slices."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.rules import goldset_regression_gate, publish_gate
from core.types import Label
from dataset.crop import ANGLE_CROP_FRACTIONS
from dataset.splits import FACTORY_HOLDOUT, TRAIN, VAL
from evaluation.evaluate import SourceProbeReport
from training.data import TrainingBundle

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AGGREGATE_SLICES = frozenset(
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
class SliceCensus:
    freeze_sha256: str
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            _SHA256.fullmatch(self.freeze_sha256) is None
            or not self.names
            or tuple(sorted(set(self.names))) != self.names
        ):
            raise ValueError("slice census is invalid")


@dataclass(frozen=True)
class EvaluationReport:
    freeze_sha256: str
    model_sha256: str
    expected_calibration_error: float
    gold_baseline_accuracy: float
    slices: tuple[SliceMetric, ...]

    def __post_init__(self) -> None:
        if (
            _SHA256.fullmatch(self.freeze_sha256) is None
            or _SHA256.fullmatch(self.model_sha256) is None
            or not 0 <= self.expected_calibration_error <= 1
            or not 0 <= self.gold_baseline_accuracy <= 1
            or not self.slices
            or len({item.name for item in self.slices}) != len(self.slices)
        ):
            raise ValueError("evaluation report is invalid")


@dataclass(frozen=True)
class ReleaseDecision:
    allowed: bool
    reasons: tuple[str, ...]


def build_slice_census(bundle: TrainingBundle) -> SliceCensus:
    """Derive every required evaluation slice from the frozen dataset."""
    names = set(_AGGREGATE_SLICES)
    for record in bundle.records:
        split = bundle.split_by_asset[record.asset_id]
        if split in {TRAIN, VAL}:
            continue
        angles = ANGLE_CROP_FRACTIONS if record.label is Label.REJECTION else (record.angle,)
        for angle in angles:
            assert isinstance(angle, str)
            names.add(f"class-angle:{record.model_class.name.lower()}:{angle}")
        names.add(f"source:{record.source_domain}")
        names.add(f"entity:{record.subject_id}")
        if split == FACTORY_HOLDOUT and record.label is Label.FAKE:
            if record.factory is None:
                raise ValueError("factory holdout record lacks a named factory")
            names.add(f"factory-grade:{record.factory}:{record.grade}")
    return SliceCensus(bundle.freeze_sha256, tuple(sorted(names)))


def release_gate(
    source_probe: SourceProbeReport,
    report: EvaluationReport | None,
    census: SliceCensus | None,
    *,
    minimum_slice_accuracy: float = 0.50,
    minimum_slice_count: int = 2,
    maximum_calibration_error: float = 0.05,
    maximum_cross_source_delta: float = 0.10,
) -> ReleaseDecision:
    reasons: list[str] = []
    if source_probe.blocked:
        reasons.append("source-domain shortcut")
    if report is None or census is None:
        reasons.append("complete freeze-derived per-slice evaluation is missing")
        return ReleaseDecision(False, tuple(reasons))
    if not (
        source_probe.freeze_sha256 == report.freeze_sha256 == census.freeze_sha256
        and source_probe.model_sha256 == report.model_sha256
    ):
        reasons.append("evaluation identity differs from source probe or frozen dataset")
    if report.expected_calibration_error > maximum_calibration_error:
        reasons.append("calibration error exceeds release limit")

    metrics = {metric.name: metric for metric in report.slices}
    missing = sorted(set(census.names) - set(metrics))
    reasons.extend(f"required evaluation slice is missing: {name}" for name in missing)
    for name in census.names:
        metric = metrics.get(name)
        if metric is None:
            continue
        if metric.count < minimum_slice_count or metric.ci_low is None or metric.ci_high is None:
            reasons.append(f"required slice lacks uncertainty evidence: {name}")
        if metric.balanced_accuracy < minimum_slice_accuracy:
            reasons.append(f"weak required slice: {name}")

    standard = metrics.get("standard")
    cross_source = metrics.get("cross-source")
    if standard is not None and cross_source is not None and not publish_gate(
        standard.balanced_accuracy,
        cross_source.balanced_accuracy,
        maximum_cross_source_delta,
    ):
        reasons.append("cross-source degradation exceeds release limit")
    gold = metrics.get("gold")
    if gold is not None and not goldset_regression_gate(
        gold.balanced_accuracy, report.gold_baseline_accuracy
    ):
        reasons.append("gold-set regression")
    return ReleaseDecision(not reasons, tuple(reasons))
