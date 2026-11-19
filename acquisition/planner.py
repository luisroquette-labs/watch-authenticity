"""Deterministic acquisition priorities from readiness deficits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil

from dataset.readiness import MIN_CELL_ENTITIES, MIN_CELL_IMAGES, ReadinessReport
from evaluation.evaluate import CollectionWave, collection_decision


@dataclass(frozen=True)
class AcquisitionTarget:
    reference_id: str
    output: str
    angle: str
    missing_images: int
    missing_entities: int
    source: str
    seller_group_id: str
    planned_new_entities: int
    exploratory: bool
    candidate_limit: int
    eligible_source_capacity: int
    eligible_yield: tuple[int, int] | None
    observed_candidates: int | None = None


def plan_acquisition(
    report: ReadinessReport,
    waves: Sequence[CollectionWave],
    *,
    source_entity_counts: Mapping[str, int],
    seller_entity_counts: Mapping[str, int],
    eligible_yield_by_source: Mapping[str, tuple[int, int]],
    candidate_consumption_by_source: Mapping[str, int],
) -> tuple[AcquisitionTarget, ...]:
    """Rank cell deficits while selecting one non-dominant source and seller."""
    if report.ready or collection_decision(tuple(waves)).stop_same_distribution:
        return ()
    for source_name, (reviewed, admitted) in eligible_yield_by_source.items():
        consumed = candidate_consumption_by_source.get(source_name)
        if (
            any(
                isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (reviewed, admitted)
            )
            or admitted > reviewed
            or isinstance(consumed, bool) or not isinstance(consumed, int)
            or consumed < reviewed
        ):
            raise ValueError("yield must count reviewed and admitted independent entities")
    _least_used_eligible(source_entity_counts, 40)  # Validate before ranking.
    seller = _least_used_eligible(seller_entity_counts, 10)
    if seller is None:
        return ()
    seller_capacity = _additional_capacity(seller_entity_counts, seller, 10)
    viable = [
        name
        for name in source_entity_counts
        if _additional_capacity(source_entity_counts, name, 40) > 0
        and (name not in eligible_yield_by_source or eligible_yield_by_source[name][1] > 0)
    ]

    def source_rank(name: str) -> tuple[float, int, str]:
        reviewed, admitted = eligible_yield_by_source.get(name, (0, 0))
        expected_gain = (
            min(
                20 * admitted / candidate_consumption_by_source[name],
                _additional_capacity(source_entity_counts, name, 40),
                seller_capacity,
            )
            if reviewed
            else 0
        )
        return (-expected_gain, source_entity_counts[name], name)

    source = min(viable, key=source_rank) if viable else None
    if source is None:
        return ()
    source_capacity = _additional_capacity(source_entity_counts, source, 40)
    pending: list[tuple[str, str, int, int]] = []
    for cell, images, entities in report.cell_counts:
        missing_images = max(0, MIN_CELL_IMAGES - images)
        missing_entities = max(0, MIN_CELL_ENTITIES - entities)
        if not (missing_images or missing_entities):
            continue
        output, separator, angle = cell.partition("/")
        if not separator or not output or not angle:
            raise ValueError("readiness cell name must be output/angle")
        pending.append((output, angle, missing_images, missing_entities))
    ordered = sorted(
        pending,
        key=lambda target: (
            -target[3] / MIN_CELL_ENTITIES,
            -target[2] / MIN_CELL_IMAGES,
            target[0],
            target[1],
        ),
    )
    targets: list[AcquisitionTarget] = []
    remaining_capacity = min(source_capacity, seller_capacity)
    observed = eligible_yield_by_source.get(source)
    exploratory = observed is None
    candidate_limit = 20
    observed_candidates = candidate_consumption_by_source[source] if observed is not None else None
    expected = 0 if observed is None or observed_candidates is None else ceil(candidate_limit * observed[1] / observed_candidates)
    for output, angle, missing_images, missing_entities in ordered:
        if remaining_capacity <= 0:
            break
        planned = min(missing_entities, remaining_capacity, expected)
        remaining_capacity -= planned
        targets.append(
            AcquisitionTarget(
                report.reference_id,
                output,
                angle,
                missing_images,
                missing_entities,
                source,
                seller,
                planned,
                exploratory,
                candidate_limit,
                min(source_capacity, seller_capacity),
                None if exploratory else observed,
                observed_candidates,
            )
        )
        if exploratory:
            break
    return tuple(targets)


def _least_used_eligible(counts: Mapping[str, int], maximum_percent: int) -> str | None:
    if not counts or any(
        not isinstance(name, str)
        or not name.strip()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for name, count in counts.items()
    ):
        raise ValueError("entity counts must be non-negative and keyed by name")
    eligible = [
        name
        for name in counts
        if _additional_capacity(counts, name, maximum_percent) > 0
    ]
    return min(eligible, key=lambda name: (counts[name], name)) if eligible else None


def _additional_capacity(counts: Mapping[str, int], name: str, maximum_percent: int) -> int:
    total = sum(counts.values())
    current = counts[name]
    # Seed one new group until percentage limits can be met; readiness still enforces them.
    if current == 0 and sum(value > 0 for value in counts.values()) < ceil(100 / maximum_percent):
        return 1
    capacity = 0
    while (current + capacity + 1) * 100 <= (total + capacity + 1) * maximum_percent:
        capacity += 1
    return capacity
