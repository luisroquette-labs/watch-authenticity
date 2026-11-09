"""Deterministic shortcut diagnostics and collection saturation decisions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class SourceEmbedding:
    vector: tuple[float, ...]
    source_domain: str

    def __post_init__(self) -> None:
        if not self.vector or not all(math.isfinite(value) for value in self.vector):
            raise ValueError("embedding must contain finite values")
        if not self.source_domain.strip():
            raise ValueError("source domain is required")


@dataclass(frozen=True)
class SourceProbeReport:
    real_accuracy: float
    real_ci_low: float
    real_ci_high: float
    shuffled_accuracy: float
    shuffled_ci_high: float
    driving_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.real_accuracy,
            self.real_ci_low,
            self.real_ci_high,
            self.shuffled_accuracy,
            self.shuffled_ci_high,
        )
        if (
            not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
            or not self.real_ci_low <= self.real_accuracy <= self.real_ci_high
            or self.shuffled_accuracy > self.shuffled_ci_high
            or not self.driving_sources
        ):
            raise ValueError("source probe report is invalid")

    @property
    def blocked(self) -> bool:
        return self.real_ci_low > self.shuffled_ci_high + 0.10


@dataclass(frozen=True)
class CollectionWave:
    new_entity_gain: float
    macro_gain_pp: float
    worst_slice_gain_pp: float
    weak_slice: str | None = None
    readiness_deficit: str | None = None

    def __post_init__(self) -> None:
        values = (self.new_entity_gain, self.macro_gain_pp, self.worst_slice_gain_pp)
        if not all(math.isfinite(value) for value in values) or not 0 <= self.new_entity_gain <= 1:
            raise ValueError("collection wave metrics are invalid")


@dataclass(frozen=True)
class CollectionDecision:
    stop_same_distribution: bool
    continue_collection: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class LearningCurvePoint:
    fraction: float
    entity_count: int
    macro_balanced_accuracy: float
    worst_required_slice_accuracy: float
    expected_calibration_error: float
    cross_source_delta: float

    def __post_init__(self) -> None:
        probabilities = (
            self.macro_balanced_accuracy,
            self.worst_required_slice_accuracy,
            self.expected_calibration_error,
            self.cross_source_delta,
        )
        if (
            self.fraction not in {0.25, 0.5, 0.75, 1.0}
            or self.entity_count < 1
            or not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities)
        ):
            raise ValueError("learning-curve point is invalid")


@dataclass(frozen=True)
class LearningCurveReport:
    points: tuple[LearningCurvePoint, ...]


def learning_curve_report(
    points: list[LearningCurvePoint] | tuple[LearningCurvePoint, ...],
) -> LearningCurveReport:
    ordered = tuple(sorted(points, key=lambda point: point.fraction))
    if tuple(point.fraction for point in ordered) != (0.25, 0.5, 0.75, 1.0):
        raise ValueError("learning curve requires the 25/50/75/100% entity waves")
    if any(left.entity_count > right.entity_count for left, right in zip(ordered, ordered[1:])):
        raise ValueError("learning-curve entity counts must be monotonic")
    return LearningCurveReport(ordered)


def evaluate_source_probe(
    embeddings: list[SourceEmbedding] | tuple[SourceEmbedding, ...],
    *,
    permutations: int = 1_000,
    seed: int = 42,
) -> SourceProbeReport:
    """Compare a leave-one-out source probe with shuffled-source controls."""
    rows = tuple(embeddings)
    if permutations < 20:
        raise ValueError("at least 20 permutations are required")
    if len(rows) < 4 or len({row.vector for row in rows}) < 2:
        raise ValueError("source probe requires varied embeddings")
    dimensions = {len(row.vector) for row in rows}
    labels = [row.source_domain for row in rows]
    sources = sorted(set(labels))
    if len(dimensions) != 1 or len(sources) < 2 or any(labels.count(source) < 2 for source in sources):
        raise ValueError("source probe requires two samples for each source in one vector space")

    vectors = [row.vector for row in rows]
    real_correct = _leave_one_out_correct(vectors, labels)
    rng = random.Random(seed)
    bootstrap = [
        mean(real_correct[rng.randrange(len(real_correct))] for _ in real_correct)
        for _ in range(permutations)
    ]
    shuffled_scores: list[float] = []
    for _ in range(permutations):
        shuffled = list(labels)
        rng.shuffle(shuffled)
        shuffled_scores.append(mean(_leave_one_out_correct(vectors, shuffled)))
    real_accuracy = mean(real_correct)
    real_ci_low = _quantile(bootstrap, 0.025)
    real_ci_high = _quantile(bootstrap, 0.975)
    shuffled_accuracy = mean(shuffled_scores)
    shuffled_ci_high = _quantile(shuffled_scores, 0.975)
    driving_sources = tuple(
        source
        for source in sources
        if mean(
            correct for correct, label in zip(real_correct, labels, strict=True) if label == source
        )
        >= real_accuracy
    )
    return SourceProbeReport(
        real_accuracy,
        real_ci_low,
        real_ci_high,
        shuffled_accuracy,
        shuffled_ci_high,
        driving_sources,
    )


def collection_decision(waves: list[CollectionWave] | tuple[CollectionWave, ...]) -> CollectionDecision:
    """Stop two low-yield, sufficiently novel waves unless a named deficit remains."""
    recent = tuple(waves)[-2:]
    named_need = next(
        (
            wave.weak_slice or wave.readiness_deficit
            for wave in reversed(recent)
            if wave.weak_slice or wave.readiness_deficit
        ),
        None,
    )
    saturated = len(recent) == 2 and all(
        wave.new_entity_gain >= 0.20
        and wave.macro_gain_pp < 1.0
        and wave.worst_slice_gain_pp < 2.0
        for wave in recent
    )
    stop = saturated and named_need is None
    if named_need is not None:
        reasons = (f"continue for named deficit: {named_need}",)
    elif stop:
        reasons = ("two consecutive diverse waves produced immaterial gains",)
    else:
        reasons = ("saturation threshold has not been met",)
    return CollectionDecision(stop, not stop, reasons)


def _leave_one_out_correct(
    vectors: list[tuple[float, ...]], labels: list[str]
) -> list[bool]:
    outcomes: list[bool] = []
    for index, vector in enumerate(vectors):
        centroids: dict[str, tuple[float, ...]] = {}
        for source in sorted(set(labels)):
            members = [
                candidate
                for member_index, candidate in enumerate(vectors)
                if member_index != index and labels[member_index] == source
            ]
            if members:
                centroids[source] = tuple(mean(values) for values in zip(*members, strict=True))
        predicted = min(
            centroids,
            key=lambda source: (
                sum((left - right) ** 2 for left, right in zip(vector, centroids[source], strict=True)),
                source,
            ),
        )
        outcomes.append(predicted == labels[index])
    return outcomes


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
