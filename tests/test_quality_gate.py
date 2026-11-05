"""Tests for the pre-inference quality gate in `quality_gate/blur_light_check.py`.

Covers this step's Success Criteria and the CK-5 Test Cases to Cover:
`quality_gate` passes a sharp, well-lit fixture; rejects a blurry fixture
with a recapture instruction; and the pass/fail boundary is exercised at
threshold-1, threshold, and threshold+1 (BVA).

The [integration] CK-5 case — "a blurry photo results in zero calls to the
authenticity model" — belongs to `inference/predict.py` (step 09, not yet
built): it is that step's orchestration being tested, not this gate in
isolation, so it is out of this step's scope.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image, ImageDraw, ImageFilter

from core.types import Rejection
from quality_gate.blur_light_check import DEFAULT_SHARPNESS_THRESHOLD, Pass, quality_gate


def _sharp_fixture(size: int = 128, seed: int = 0) -> Image.Image:
    """A high-frequency, in-focus fixture: random noise plus crisp shapes.

    Random per-pixel noise (rather than a few flat-colored shapes alone)
    guarantees a Laplacian response well above every threshold in
    `SHARPNESS_THRESHOLD_PER_ANGLE`, so this fixture reads as "sharp" the
    same way a real in-focus phone photo of a textured watch dial would.
    """
    rng = random.Random(seed)
    pixels = [rng.randint(0, 255) for _ in range(size * size * 3)]
    image = Image.frombytes("RGB", (size, size), bytes(pixels))
    draw = ImageDraw.Draw(image)
    draw.ellipse((size // 4, size // 4, 3 * size // 4, 3 * size // 4), outline=(0, 0, 0), width=4)
    return image


def _blurry_fixture(size: int = 128, seed: int = 0) -> Image.Image:
    """The same sharp fixture smoothed by a heavy Gaussian blur.

    A large enough blur radius smooths away the noise and the ellipse edge
    alike, driving the Laplacian-variance score well below every per-angle
    threshold — the photographic equivalent of camera shake or missed focus.
    """
    return _sharp_fixture(size=size, seed=seed).filter(ImageFilter.GaussianBlur(radius=8))


def test_quality_gate_passes_a_sharp_well_lit_fixture():
    result = quality_gate(_sharp_fixture())

    assert isinstance(result, Pass)
    assert result.sharpness >= DEFAULT_SHARPNESS_THRESHOLD


def test_quality_gate_rejects_a_blurry_fixture_with_a_recapture_instruction():
    result = quality_gate(_blurry_fixture())

    assert isinstance(result, Rejection)
    assert result.reason == "blurry"
    assert "recapture" in result.instruction.lower()


def test_quality_gate_applies_the_lower_caseback_threshold_for_that_angle(
    monkeypatch: pytest.MonkeyPatch,
):
    # A sharpness score between the caseback floor (80.0) and the default/
    # other-angle floor (150.0) must pass for "caseback" but fail for an
    # angle using the higher default — the concrete case the per-angle
    # threshold table exists to handle (Blockers & Risks: a polished
    # caseback is sharp but scores low on Laplacian variance).
    monkeypatch.setattr("quality_gate.blur_light_check.sharpness_score", lambda _gray: 100.0)

    caseback_result = quality_gate(_sharp_fixture(), angle="caseback")
    dial_result = quality_gate(_sharp_fixture(), angle="dial")

    assert isinstance(caseback_result, Pass)
    assert isinstance(dial_result, Rejection)


def test_quality_gate_boundary_values_around_the_threshold(monkeypatch: pytest.MonkeyPatch):
    # BVA: drive the documented pass/fail boundary directly by controlling
    # the sharpness signal `quality_gate` decides on, rather than hunting
    # for a real image that happens to score exactly at the threshold —
    # `quality_gate`'s own contract ("< threshold fails, >= passes") is
    # what this test verifies, independent of image-fixture construction.
    threshold = DEFAULT_SHARPNESS_THRESHOLD

    for score, expect_pass in ((threshold - 1, False), (threshold, True), (threshold + 1, True)):
        monkeypatch.setattr(
            "quality_gate.blur_light_check.sharpness_score", lambda _gray, _score=score: _score
        )

        result = quality_gate(_sharp_fixture())

        if expect_pass:
            assert isinstance(result, Pass), f"score={score} should pass"
        else:
            assert isinstance(result, Rejection), f"score={score} should reject"
