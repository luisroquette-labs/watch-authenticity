"""Deterministic shortcut diagnostics and collection saturation decisions."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from statistics import mean
from typing import TYPE_CHECKING

from core.types import Label
from labeling.quality import (
    RUBRIC_SHA256,
    RUBRIC_VERSION,
    FakeQualityAssessment,
    verify_fake_quality,
)

if TYPE_CHECKING:
    from training.data import TrainingBundle

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MIN_GROUPS_PER_SOURCE = 5
_ENTITY_CONCLUSIONS = frozenset({"fake_likely", "no_visible_inconsistency", "inconclusive", "rejection"})
HIGH_QUALITY_FAKE_MIN_SCORE = 800


@dataclass(frozen=True)
class EntityOutcome:
    watch_entity_id: str
    conclusion: str

    def __post_init__(self) -> None:
        if not self.watch_entity_id.strip() or self.conclusion not in _ENTITY_CONCLUSIONS:
            raise ValueError("entity outcome requires one valid watch-level conclusion")


@dataclass(frozen=True)
class EntitySafetyReport:
    freeze_sha256: str
    model_sha256: str
    outcomes: tuple[EntityOutcome, ...]
    quality_assessments: tuple[FakeQualityAssessment, ...]

    def __post_init__(self) -> None:
        if (
            _SHA256.fullmatch(self.freeze_sha256) is None
            or _SHA256.fullmatch(self.model_sha256) is None
            or len({item.watch_entity_id for item in self.outcomes}) != len(self.outcomes)
            or len({item.watch_entity_id for item in self.quality_assessments}) != len(self.quality_assessments)
        ):
            raise ValueError("entity safety report has invalid identity or duplicate watches")


@dataclass(frozen=True)
class EntityCohort:
    name: str
    answered: int
    total: int
    false_reassurance: int
    false_reassurance_upper95: float | None

    @property
    def abstention_rate(self) -> float:
        return 1 - self.answered / self.total if self.total else 1.0


def false_reassurance_upper95(errors: int, total: int) -> float:
    """Exact one-sided binomial upper bound by inversion of the CDF."""
    if not 0 <= errors <= total or total < 1:
        raise ValueError("false-reassurance bound requires a nonempty valid count")
    if errors == total:
        return 1.0
    if errors == 0:
        return 1 - 0.05 ** (1 / total)
    low, high = 0.0, 1.0
    for _ in range(55):
        middle = (low + high) / 2
        cdf = sum(
            math.exp(
                math.lgamma(total + 1) - math.lgamma(index + 1)
                - math.lgamma(total - index + 1)
                + index * math.log(middle)
                + (total - index) * math.log1p(-middle)
            )
            for index in range(errors + 1)
        )
        if cdf > 0.05:
            low = middle
        else:
            high = middle
    return high


def evaluate_entity_safety(bundle: TrainingBundle, report: EntitySafetyReport) -> tuple[EntityCohort, ...]:
    """Reconcile one result per frozen holdout watch with confirmed quality evidence."""
    from dataset.splits import TRAIN, VAL

    if report.freeze_sha256 != bundle.freeze_sha256:
        raise ValueError("entity outcomes belong to another frozen dataset")
    holdout: dict[str, list] = {}
    for record in bundle.records:
        if bundle.split_by_asset[record.asset_id] not in {TRAIN, VAL}:
            holdout.setdefault(record.subject_id, []).append(record)
    if set(holdout) != {item.watch_entity_id for item in report.outcomes}:
        raise ValueError("entity outcomes must cover every frozen holdout watch exactly once")
    outcomes = {item.watch_entity_id: item for item in report.outcomes}
    if any(len({(record.label, record.grade) for record in records}) != 1 for records in holdout.values()):
        raise ValueError("one holdout watch cannot have conflicting labels or grades")
    fake_entities = {
        entity: records for entity, records in holdout.items()
        if records[0].label is Label.FAKE
    }
    assessed = {item.watch_entity_id: item for item in report.quality_assessments}
    if set(assessed) != set(fake_entities):
        raise ValueError("every frozen fake holdout watch requires a separate quality assessment")
    for entity, assessment in assessed.items():
        verify_fake_quality(assessment)
        if (
            assessment.rubric_version != RUBRIC_VERSION
            or assessment.rubric_sha256 != RUBRIC_SHA256
            or not isinstance(assessment.score_0_1000, int)
            or isinstance(assessment.score_0_1000, bool)
            or not 0 <= assessment.score_0_1000 <= 1000
            or len(assessment.reviewer_scores) != 2
            or len({reviewer for reviewer, _ in assessment.reviewer_scores}) != 2
            or any(not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 1000 for _, score in assessment.reviewer_scores)
            or len(assessment.review_decision_sha256s) != 2
            or any(_SHA256.fullmatch(digest) is None for digest in assessment.review_decision_sha256s)
            or assessment.evidence_record_id not in {
                record.provenance_id for record in fake_entities[entity]
            }
        ):
            raise ValueError("fake quality is not bound to this frozen watch and evidence")

    def cohort(name: str, entities: set[str], *, fake: bool = False) -> EntityCohort:
        values = [outcomes[entity].conclusion for entity in entities]
        answered = sum(value not in {"inconclusive", "rejection"} for value in values)
        errors = sum(value == "no_visible_inconsistency" for value in values) if fake else 0
        return EntityCohort(
            name, answered, len(values), errors,
            false_reassurance_upper95(errors, len(values)) if fake and values else None,
        )

    genuine_entities = {entity for entity, records in holdout.items() if records[0].label is Label.GENUINE}
    high_quality = {
        entity for entity, assessment in assessed.items()
        if assessment.score_0_1000 >= HIGH_QUALITY_FAKE_MIN_SCORE
    }
    return (
        cohort("genuine", genuine_entities),
        cohort("fake", set(fake_entities), fake=True),
        *(cohort(f"fake-{grade}", {
            entity for entity, records in fake_entities.items() if records[0].grade == grade
        }, fake=True) for grade in range(1, 5)),
        cohort("high-quality-fake", high_quality, fake=True),
    )


@dataclass(frozen=True)
class SourceEmbedding:
    vector: tuple[float, ...]
    source_domain: str
    leakage_group_id: str

    def __post_init__(self) -> None:
        if not self.vector or not all(math.isfinite(value) for value in self.vector):
            raise ValueError("embedding must contain finite values")
        if not self.source_domain.strip():
            raise ValueError("source domain is required")
        if not self.leakage_group_id.strip():
            raise ValueError("leakage group is required")


@dataclass(frozen=True)
class SourceProbeReport:
    freeze_sha256: str
    model_sha256: str
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
            _SHA256.fullmatch(self.freeze_sha256) is None
            or _SHA256.fullmatch(self.model_sha256) is None
            or not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
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
    freeze_sha256: str
    model_sha256: str
    points: tuple[LearningCurvePoint, ...]


def learning_curve_report(
    points: list[LearningCurvePoint] | tuple[LearningCurvePoint, ...],
    *,
    freeze_sha256: str,
    model_sha256: str,
) -> LearningCurveReport:
    ordered = tuple(sorted(points, key=lambda point: point.fraction))
    if tuple(point.fraction for point in ordered) != (0.25, 0.5, 0.75, 1.0):
        raise ValueError("learning curve requires the 25/50/75/100% entity waves")
    if any(left.entity_count > right.entity_count for left, right in zip(ordered, ordered[1:])):
        raise ValueError("learning-curve entity counts must be monotonic")
    if _SHA256.fullmatch(freeze_sha256) is None or _SHA256.fullmatch(model_sha256) is None:
        raise ValueError("learning curve requires freeze and model SHA-256 identities")
    return LearningCurveReport(freeze_sha256, model_sha256, ordered)


def evaluate_source_probe(
    embeddings: list[SourceEmbedding] | tuple[SourceEmbedding, ...],
    *,
    permutations: int = 1_000,
    seed: int = 42,
    freeze_sha256: str,
    model_sha256: str,
) -> SourceProbeReport:
    """Compare a leave-one-out source probe with shuffled-source controls."""
    rows = tuple(embeddings)
    if permutations < 20:
        raise ValueError("at least 20 permutations are required")
    if (
        _SHA256.fullmatch(freeze_sha256) is None
        or _SHA256.fullmatch(model_sha256) is None
        or len(rows) < 4
        or len({row.vector for row in rows}) < 2
    ):
        raise ValueError("source probe requires varied embeddings")
    dimensions = {len(row.vector) for row in rows}
    grouped: dict[str, list[SourceEmbedding]] = {}
    for row in rows:
        grouped.setdefault(row.leakage_group_id, []).append(row)
    if any(len({row.source_domain for row in members}) != 1 for members in grouped.values()):
        raise ValueError("one leakage group cannot claim multiple sources")
    vectors = [
        tuple(mean(values) for values in zip(*(row.vector for row in members), strict=True))
        for _, members in sorted(grouped.items())
    ]
    labels = [members[0].source_domain for _, members in sorted(grouped.items())]
    sources = sorted(set(labels))
    if (
        len(dimensions) != 1
        or len(sources) < 2
        or any(labels.count(source) < _MIN_GROUPS_PER_SOURCE for source in sources)
    ):
        raise ValueError("source probe requires five leakage groups per source")

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
        freeze_sha256,
        model_sha256,
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
