"""Human angle labels and advisory suggestions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from acquisition.types import Angle

MIN_REVIEWED_PER_ANGLE = 100


@dataclass(frozen=True)
class AngleReview:
    candidate_id: str
    image_sha256: str
    angle: Angle | None
    reviewer_id: str
    reviewed_at: datetime
    review_sha256: str


@dataclass(frozen=True)
class AngleSuggestion:
    image_sha256: str
    angle: Angle | None
    confidence: float


def record_angle_review(
    candidate_id: str,
    image_sha256: str,
    angle: Angle | None,
    reviewer_id: str,
    reviewed_at: datetime,
) -> AngleReview:
    for value, field in ((candidate_id, "candidate_id"), (reviewer_id, "reviewer_id")):
        if not isinstance(value, str) or not value.strip() or "\n" in value:
            raise ValueError(f"{field} must be a non-empty single-line string")
    if not _is_sha256(image_sha256):
        raise ValueError("image_sha256 must be a lowercase SHA-256")
    if angle is not None and not isinstance(angle, Angle):
        raise ValueError("angle must be a known Angle or unknown")
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("reviewed_at must be timezone-aware")
    payload = {
        "candidate_id": candidate_id,
        "image_sha256": image_sha256,
        "angle": None if angle is None else angle.value,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at.isoformat(),
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    return AngleReview(
        candidate_id=candidate_id,
        image_sha256=image_sha256,
        angle=angle,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        review_sha256=digest,
    )


def suggest_angle(
    image_sha256: str,
    scores: Mapping[Angle, float],
    reviewed: Sequence[AngleReview],
    *,
    min_confidence: float = 0.8,
) -> AngleSuggestion:
    if not _is_sha256(image_sha256):
        raise ValueError("image_sha256 must be a lowercase SHA-256")
    counts = Counter(review.angle for review in reviewed if review.angle is not None)
    if any(counts[angle] < MIN_REVIEWED_PER_ANGLE for angle in Angle):
        return AngleSuggestion(image_sha256, None, 0.0)
    if set(scores) != set(Angle) or any(
        isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1
        for score in scores.values()
    ):
        raise ValueError("scores must contain one probability for every angle")
    angle, confidence = max(scores.items(), key=lambda item: (item[1], item[0].value))
    if confidence < min_confidence:
        return AngleSuggestion(image_sha256, None, float(confidence))
    return AngleSuggestion(image_sha256, angle, float(confidence))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
