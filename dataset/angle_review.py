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
    if not isinstance(min_confidence, (int, float)) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between zero and one")
    reviewed_angles: dict[str, set[Angle | None]] = {}
    for review in reviewed:
        if _review_digest(review) != review.review_sha256:
            raise ValueError("angle review hash differs from content")
        reviewed_angles.setdefault(review.image_sha256, set()).add(review.angle)
    counts = Counter(
        next(iter(angles))
        for angles in reviewed_angles.values()
        if len(angles) == 1 and next(iter(angles)) is not None
    )
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


def _review_digest(review: AngleReview) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "candidate_id": review.candidate_id,
                "image_sha256": review.image_sha256,
                "angle": None if review.angle is None else review.angle.value,
                "reviewer_id": review.reviewer_id,
                "reviewed_at": review.reviewed_at.isoformat(),
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class AngleUsabilityReview:
    candidate_id: str
    image_sha256: str
    angle: Angle
    detail_visible: bool
    usable: bool
    reviewer_id: str
    reviewed_at: datetime
    review_sha256: str


def record_usability_review(
    candidate_id: str,
    image_sha256: str,
    angle: Angle,
    detail_visible: bool,
    usable: bool,
    reviewer_id: str,
    reviewed_at: datetime,
) -> AngleUsabilityReview:
    record_angle_review(candidate_id, image_sha256, angle, reviewer_id, reviewed_at)
    if not isinstance(angle, Angle):
        raise ValueError("usability review requires a known angle")
    if not isinstance(detail_visible, bool) or not isinstance(usable, bool):
        raise ValueError("visibility and usability must be bool")
    if usable and not detail_visible:
        raise ValueError("usable detail must be visible")
    payload = {
        "candidate_id": candidate_id,
        "image_sha256": image_sha256,
        "angle": angle.value,
        "detail_visible": detail_visible,
        "usable": usable,
        "reviewer_id": reviewer_id,
        "reviewed_at": reviewed_at.isoformat(),
    }
    return AngleUsabilityReview(
        candidate_id, image_sha256, angle, detail_visible, usable, reviewer_id,
        reviewed_at, hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def verify_usability_review(
    review: AngleUsabilityReview, candidate_id: str, image_sha256: str, angle: Angle
) -> None:
    if not isinstance(review, AngleUsabilityReview):
        raise ValueError("immutable usability review is required")
    expected = record_usability_review(
        review.candidate_id, review.image_sha256, review.angle,
        review.detail_visible, review.usable, review.reviewer_id, review.reviewed_at,
    )
    if expected != review:
        raise ValueError("usability review hash differs from content")
    if (review.candidate_id, review.image_sha256, review.angle) != (candidate_id, image_sha256, angle):
        raise ValueError("usability review candidate, image or angle differs")
    if not review.detail_visible or not review.usable:
        raise ValueError("angle requires visible usable detail")
