from datetime import datetime, timezone

from acquisition.types import Angle
from dataset.angle_review import record_angle_review, suggest_angle

NOW = datetime(2026, 11, 9, 8, 0, tzinfo=timezone.utc)


def _reviews(per_angle: int):
    return tuple(
        record_angle_review(
            f"candidate-{angle.value}-{index}",
            f"{index + list(Angle).index(angle) * per_angle:064x}",
            angle,
            "reviewer-1",
            NOW,
        )
        for angle in Angle
        for index in range(per_angle)
    )


def _scores(dial: float = 0.9):
    return {angle: dial if angle is Angle.DIAL else 0.02 for angle in Angle}


def test_suggestions_stay_unknown_until_every_angle_has_100_reviews():
    suggestion = suggest_angle("f" * 64, _scores(), _reviews(99))

    assert suggestion.angle is None
    assert suggestion.confidence == 0.0


def test_reviewed_bootstrap_allows_advisory_suggestion_at_boundary():
    suggestion = suggest_angle("f" * 64, _scores(), _reviews(100))

    assert suggestion.angle is Angle.DIAL
    assert suggestion.confidence == 0.9


def test_low_confidence_result_is_unknown():
    suggestion = suggest_angle("f" * 64, _scores(0.7), _reviews(100))

    assert suggestion.angle is None
    assert suggestion.confidence == 0.7


def test_human_review_binds_reviewer_and_image_hash():
    review = record_angle_review("candidate-1", "a" * 64, None, "reviewer-1", NOW)
    changed = record_angle_review("candidate-1", "b" * 64, None, "reviewer-1", NOW)

    assert review.angle is None
    assert review.review_sha256 != changed.review_sha256
    assert review.reviewer_id == "reviewer-1"
