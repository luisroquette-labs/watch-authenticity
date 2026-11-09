"""Deterministic acquisition priorities from readiness deficits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dataset.readiness import MIN_CELL_ENTITIES, MIN_CELL_IMAGES, ReadinessReport
from evaluation.evaluate import CollectionWave, collection_decision


@dataclass(frozen=True)
class AcquisitionTarget:
    output: str
    angle: str
    missing_images: int
    missing_entities: int
    source: str
    seller_group_id: str


def plan_acquisition(
    report: ReadinessReport,
    waves: Sequence[CollectionWave],
    *,
    source_entity_counts: Mapping[str, int],
    seller_entity_counts: Mapping[str, int],
) -> tuple[AcquisitionTarget, ...]:
    """Rank cell deficits while selecting one non-dominant source and seller."""
    if report.ready or collection_decision(tuple(waves)).stop_same_distribution:
        return ()
    source = _least_used_eligible(source_entity_counts, 40)
    seller = _least_used_eligible(seller_entity_counts, 10)
    if source is None or seller is None:
        return ()
    targets: list[AcquisitionTarget] = []
    for cell, images, entities in report.cell_counts:
        missing_images = max(0, MIN_CELL_IMAGES - images)
        missing_entities = max(0, MIN_CELL_ENTITIES - entities)
        if not (missing_images or missing_entities):
            continue
        output, separator, angle = cell.partition("/")
        if not separator or not output or not angle:
            raise ValueError("readiness cell name must be output/angle")
        targets.append(
            AcquisitionTarget(
                output,
                angle,
                missing_images,
                missing_entities,
                source,
                seller,
            )
        )
    return tuple(
        sorted(
            targets,
            key=lambda target: (
                -target.missing_entities / MIN_CELL_ENTITIES,
                -target.missing_images / MIN_CELL_IMAGES,
                target.output,
                target.angle,
            ),
        )
    )


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
    total = sum(counts.values())
    eligible = [
        name
        for name, count in counts.items()
        if total == 0 or (count + 1) * 100 <= (total + 1) * maximum_percent
    ]
    return min(eligible, key=lambda name: (counts[name], name)) if eligible else None
