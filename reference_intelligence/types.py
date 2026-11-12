"""Validated contracts for sourced watch specifications and visual checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from urllib.parse import urlsplit

from acquisition.types import Angle

_REFERENCE_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")
_FIELD = re.compile(r"[a-z][a-z0-9_]*")


def _single_line(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a non-empty single-line string")


class SourceClass(Enum):
    OFFICIAL_BRAND = "official_brand"
    OFFICIAL_RETAILER = "official_retailer"
    NAMED_EXPERT = "named_expert"
    FORUM = "forum"
    MARKETPLACE = "marketplace"

    @property
    def authoritative(self) -> bool:
        """Only the manufacturer defines a reference's technical specification."""
        return self is SourceClass.OFFICIAL_BRAND

    @property
    def corroborating(self) -> bool:
        return self in {SourceClass.OFFICIAL_RETAILER, SourceClass.NAMED_EXPERT}


class CheckStatus(Enum):
    MATCHED = "matched"
    CONFLICTING = "conflicting"
    UNOBSERVABLE = "unobservable"


class AssessmentStatus(Enum):
    CONSISTENT = "consistent"
    CONFLICTING = "conflicting"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class TechnicalClaim:
    reference_id: str
    field: str
    value: str
    source_class: SourceClass
    source_url: str
    citation: str
    observed_at: date
    applicable_angles: tuple[Angle, ...] = ()

    def __post_init__(self) -> None:
        if _REFERENCE_ID.fullmatch(self.reference_id) is None:
            raise ValueError("reference_id must be a normalized exact reference")
        if _FIELD.fullmatch(self.field) is None:
            raise ValueError("field must be lower snake_case")
        _single_line(self.value, "value")
        _single_line(self.citation, "citation")
        if not isinstance(self.source_class, SourceClass):
            raise ValueError("source_class must be SourceClass")
        parsed = urlsplit(self.source_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("source_url must be an absolute HTTPS URL without credentials")
        if not isinstance(self.observed_at, date) or self.observed_at > date.today():
            raise ValueError("observed_at must be a date not in the future")
        if not isinstance(self.applicable_angles, tuple) or any(
            not isinstance(angle, Angle) for angle in self.applicable_angles
        ):
            raise ValueError("applicable_angles must contain Angle values")
        if len(set(self.applicable_angles)) != len(self.applicable_angles):
            raise ValueError("applicable_angles must not contain duplicates")


@dataclass(frozen=True)
class VisualObservation:
    field: str
    value: str | None
    angle: Angle
    visible: bool

    def __post_init__(self) -> None:
        if _FIELD.fullmatch(self.field) is None:
            raise ValueError("field must be lower snake_case")
        if not isinstance(self.angle, Angle):
            raise ValueError("angle must be Angle")
        if not isinstance(self.visible, bool):
            raise ValueError("visible must be bool")
        if self.visible:
            if self.value is None:
                raise ValueError("visible observations require a value")
            _single_line(self.value, "value")
        elif self.value is not None:
            raise ValueError("unobservable observations cannot claim a value")


@dataclass(frozen=True)
class TechnicalCheck:
    field: str
    expected: str
    observed: str | None
    status: CheckStatus
    source_url: str
    citation: str
    source_observed_at: date
    angle: Angle | None


@dataclass(frozen=True)
class ReferenceAssessment:
    reference_id: str
    status: AssessmentStatus
    checks: tuple[TechnicalCheck, ...]
    source_conflicts: tuple[str, ...]
    stale_fields: tuple[str, ...]
    corroboration_count: int
    lead_count: int
    authenticates_watch: bool = False

    def __post_init__(self) -> None:
        if self.authenticates_watch:
            raise ValueError("technical specifications cannot authenticate a photographed watch")
