"""Compare authoritative reference facts with explicit visual observations."""

from __future__ import annotations

import re
from datetime import date

from reference_intelligence.types import (
    AssessmentStatus,
    CheckStatus,
    ReferenceAssessment,
    TechnicalCheck,
    TechnicalClaim,
    VisualObservation,
)


def assess_reference(
    reference_id: str,
    claims: tuple[TechnicalClaim, ...],
    observations: tuple[VisualObservation, ...],
    *,
    minimum_source_date: date,
) -> ReferenceAssessment:
    if not isinstance(minimum_source_date, date):
        raise ValueError("minimum_source_date must be explicit")
    if any(claim.reference_id != reference_id for claim in claims):
        raise ValueError("claims belong to another exact reference")
    fields = [observation.field for observation in observations]
    if len(fields) != len(set(fields)):
        raise ValueError("visual observations must contain one value per field")

    authoritative: dict[str, list[TechnicalClaim]] = {}
    corroboration_count = 0
    lead_count = 0
    for claim in claims:
        if claim.source_class.authoritative:
            authoritative.setdefault(claim.field, []).append(claim)
        elif claim.source_class.corroborating:
            corroboration_count += 1
        else:
            lead_count += 1

    unknown_observations = set(fields) - set(authoritative)
    if unknown_observations:
        raise ValueError(
            "visual observations contain unknown technical fields: "
            + ", ".join(sorted(unknown_observations))
        )

    selected: dict[str, TechnicalClaim] = {}
    source_conflicts: list[str] = []
    stale_fields: list[str] = []
    for field, candidates in authoritative.items():
        values = {_normalize(candidate.value) for candidate in candidates}
        if len(values) != 1:
            source_conflicts.append(field)
            continue
        latest = max(candidates, key=lambda candidate: candidate.observed_at)
        if latest.observed_at < minimum_source_date:
            stale_fields.append(field)
            continue
        selected[field] = latest

    observed_by_field = {observation.field: observation for observation in observations}
    checks: list[TechnicalCheck] = []
    for field, claim in sorted(selected.items()):
        observation = observed_by_field.get(field)
        applicable = (
            observation is not None
            and observation.visible
            and bool(claim.applicable_angles)
            and observation.angle in claim.applicable_angles
        )
        if not applicable:
            checks.append(_check(claim, observation, CheckStatus.UNOBSERVABLE))
        else:
            assert observation is not None
            check_status = (
                CheckStatus.MATCHED
                if _normalize(observation.value or "") == _normalize(claim.value)
                else CheckStatus.CONFLICTING
            )
            checks.append(_check(claim, observation, check_status))

    if source_conflicts or stale_fields or not selected:
        status = AssessmentStatus.INCONCLUSIVE
    elif any(check.status is CheckStatus.CONFLICTING for check in checks):
        status = AssessmentStatus.CONFLICTING
    elif not any(check.status is CheckStatus.MATCHED for check in checks):
        status = AssessmentStatus.INCONCLUSIVE
    else:
        status = AssessmentStatus.CONSISTENT
    return ReferenceAssessment(
        reference_id=reference_id,
        status=status,
        checks=tuple(checks),
        source_conflicts=tuple(sorted(source_conflicts)),
        stale_fields=tuple(sorted(stale_fields)),
        corroboration_count=corroboration_count,
        lead_count=lead_count,
    )


def _check(
    claim: TechnicalClaim,
    observation: VisualObservation | None,
    status: CheckStatus,
) -> TechnicalCheck:
    return TechnicalCheck(
        field=claim.field,
        expected=claim.value,
        observed=observation.value if observation else None,
        status=status,
        source_url=claim.source_url,
        citation=claim.citation,
        source_observed_at=claim.observed_at,
        angle=(
            observation.angle
            if observation is not None
            else (claim.applicable_angles[0] if claim.applicable_angles else None)
        ),
    )


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())
