"""Load the small, versioned technical-claim knowledge file."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from acquisition.types import Angle
from reference_intelligence.types import SourceClass, TechnicalClaim


def load_knowledge(path: Path, reference_id: str) -> tuple[TechnicalClaim, ...]:
    """Load one exact reference and fail closed on malformed or mixed data."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("reference knowledge is unreadable") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("reference knowledge schema_version must be 1")
    if raw.get("reference_id") != reference_id:
        raise ValueError("reference knowledge belongs to another exact reference")
    claims = raw.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("reference knowledge requires claims")
    parsed = tuple(_claim(reference_id, value) for value in claims)
    identities = [(claim.field, claim.source_url, claim.citation) for claim in parsed]
    if len(identities) != len(set(identities)):
        raise ValueError("reference knowledge contains duplicate claims")
    return parsed


def _claim(reference_id: str, raw: Any) -> TechnicalClaim:
    if not isinstance(raw, dict):
        raise ValueError("technical claim must be an object")
    required = {
        "field",
        "value",
        "source_class",
        "source_url",
        "citation",
        "observed_at",
        "applicable_angles",
    }
    if set(raw) != required:
        raise ValueError("technical claim fields differ from schema")
    try:
        source_class = SourceClass(raw["source_class"])
        observed_at = date.fromisoformat(raw["observed_at"])
        angles = tuple(Angle(value) for value in raw["applicable_angles"])
    except (TypeError, ValueError) as exc:
        raise ValueError("technical claim enum or date is invalid") from exc
    return TechnicalClaim(
        reference_id=reference_id,
        field=raw["field"],
        value=raw["value"],
        source_class=source_class,
        source_url=raw["source_url"],
        citation=raw["citation"],
        observed_at=observed_at,
        applicable_angles=angles,
    )
