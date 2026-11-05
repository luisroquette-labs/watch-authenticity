"""Framework-free domain vocabulary shared by every pipeline stage.

This module is the inward-most layer (L1) of the architecture: it imports
only the Python standard library, so it carries zero awareness of torch,
opencv, boto3, or any other adapter-layer dependency. Every later stage
(scrape, train, calibrate, evaluate, publish) and every safety rule in
`core/rules.py` builds on these enums and value objects instead of each
re-declaring its own copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum, IntEnum
from typing import Any


class Grade(IntEnum):
    """Market-taxonomy fake grade, 1 (camelô/street fake) to 4 (superclone).

    Backed by `int` so a `Grade` member compares equal to the plain `int`
    stored in `ManifestRecord.grade` / `Prediction.grade` (both typed
    `int | None` per the Contracts section, not `Grade | None`).
    """

    STREET_FAKE = 1
    COMMON_REPLICA = 2
    HIGH_GRADE_REP = 3
    SUPERCLONE = 4


class Label(Enum):
    """Source-trust-derived training label — never a human judgment call.

    The label comes from where/how a photo was sourced (e.g. a dealer
    listing vs. a forum post naming a replica factory), not from someone
    eyeballing the watch in the photo.
    """

    GENUINE = "genuine"
    FAKE = "fake"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ManifestRecord:
    """One auditable row per training photo (Contracts: "Manifest record").

    `grade` is required but nullable: genuine photos carry no fake grade.
    `blur_score` is optional because quality-gate scoring may run as a
    separate, later stage than manifest creation.
    """

    path: str
    source: str
    url: str
    label: Label
    grade: int | None
    phash: str
    scraped_at: date
    blur_score: float | None = None


@dataclass(frozen=True)
class Prediction:
    """A calibrated authenticity score plus its explainability artifact.

    `heatmap` is typed `Any` rather than a concrete torch/numpy array type:
    keeping the inward-most core framework-free means the adapter layer
    owns the concrete array type (see this step's Blockers & Risks note).
    """

    confidence: float
    genuine: bool
    grade: int | None
    ceiling_applied: bool
    heatmap: Any


@dataclass(frozen=True)
class Rejection:
    """A short-circuit result for out-of-distribution or low-quality input.

    Returned instead of a `Prediction` whenever the quality gate or the
    out-of-distribution check fails — the pipeline must never fabricate a
    genuine/fake score for input it cannot evaluate.
    """

    reason: str
    instruction: str
