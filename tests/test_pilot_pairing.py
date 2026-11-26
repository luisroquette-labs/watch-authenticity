from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from evaluation.pilot_pairing import ReplayInput, verify_paired_inputs


@pytest.fixture
def capture() -> ReplayInput:
    return ReplayInput("specimen-1", ("1" * 64, "2" * 64), "3" * 64, "4" * 64,
                       "google/gemini-3.8-flash", "5" * 64, "response-1", "9" * 64)


def test_both_candidate_arms_reuse_exact_capture(capture: ReplayInput) -> None:
    previous = replace(capture, prompt_sha256="6" * 64)
    assert verify_paired_inputs(previous, capture, capture) == {
        "case_id": "specimen-1",
        "photo_count": 2,
        "baseline_uses_same_gemini_capture": False,
        "gemini_response_reused": True,
    }


@pytest.mark.parametrize("changes", [
    {"photo_sha256s": ("2" * 64, "1" * 64)},
    {"photo_sha256s": ("1" * 64, "7" * 64)},
    {"technical_facts_sha256": "7" * 64},
    {"prepared_request_sha256": "7" * 64},
    {"prompt_sha256": "7" * 64},
    {"gemini_response_sha256": "7" * 64},
    {"gemini_response_id": "response-2"},
    {"gemini_model": "google/other-model"},
    {"case_id": "other-specimen"},
])
def test_review_arm_rejects_different_case_or_input(capture: ReplayInput, changes: dict) -> None:
    with pytest.raises(ValueError, match="same Gemini capture"):
        verify_paired_inputs(capture, capture, replace(capture, **changes))


def test_previous_policy_must_use_same_photos(capture: ReplayInput) -> None:
    previous = replace(capture, photo_sha256s=("8" * 64,))
    with pytest.raises(ValueError, match="ordered photos"):
        verify_paired_inputs(previous, capture, capture)


def test_surface_parity_rejects_reordered_media_and_changed_question() -> None:
    from evaluation.pilot_pairing import verify_surface_inputs

    photos, prompt = ("a" * 64, "b" * 64), "c" * 64
    verify_surface_inputs((photos, prompt), (photos, prompt))
    with pytest.raises(ValueError, match="same ordered photos and question"):
        verify_surface_inputs((photos, prompt), (photos[::-1], prompt))
    with pytest.raises(ValueError, match="same ordered photos and question"):
        verify_surface_inputs((photos, prompt), (photos, "d" * 64))


@pytest.mark.parametrize("photos", [(), ("bad-digest",)])
def test_surface_parity_rejects_unusable_photo_digests(photos: tuple[str, ...]) -> None:
    from evaluation.pilot_pairing import verify_surface_inputs

    with pytest.raises(ValueError, match="ordered photo and prompt digests"):
        verify_surface_inputs((photos, "a" * 64), (photos, "a" * 64))


def test_surface_parity_rejects_unordered_photo_inputs() -> None:
    from evaluation.pilot_pairing import verify_surface_inputs

    unordered: Any = {"a" * 64, "b" * 64}
    with pytest.raises(ValueError, match="ordered photo and prompt digests"):
        verify_surface_inputs((unordered, "c" * 64), (unordered, "c" * 64))
