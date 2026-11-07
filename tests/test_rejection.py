"""Tests for the OOD/rejection gate (CK-4): `model.rejection.classify`.

Deviation note (same sandbox limitation as `tests/test_model_build.py`):
optional training dependencies may be absent from a lightweight local
environment, and `model.rejection` imports
`model.heads`, which imports `torch` unconditionally — so every test
below is gated behind `pytest.importorskip`, exactly like
`test_model_build.py`. None of these tests need the network or a real
checkpoint even when torch *is* installed: `classify()`'s own `predict`
injection seam (mirroring `model.heads.build_model(backbone=...)`) lets
every test supply crafted, deterministic per-angle logits directly, so
these are not placeholders — they run for real wherever the pinned
requirements.txt actually installs (confirmed in a disposable venv with
real `torch==2.15.0`/`transformers==5.22.1`, see this step's sub-task
file for the verification record).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip(
        "torch", reason="optional training dependency torch is not installed"
    )
    pytest.importorskip(
        "transformers", reason="model.heads imports transformers via adapters.backbone"
    )

from core.types import Rejection  # noqa: E402
from model.heads import GENUINE_CLASS_INDEX, GUIDED_ANGLES, NUM_CLASSES  # noqa: E402
from model.rejection import REJECTION_CLASS_INDEX, classify  # noqa: E402


def _logits_for_class(class_index: int, batch_size: int = 1) -> torch.Tensor:
    """Logits with `class_index` as the unambiguous argmax for every row."""
    logits = torch.full((batch_size, NUM_CLASSES), -10.0)
    logits[:, class_index] = 10.0
    return logits


def _stub_predict(class_index_per_angle: dict[str, int]):
    """A `predict` stub returning crafted, deterministic per-angle logits.

    Stands in for a real `AngleHeadModel` the same way `test_model_build.
    py`'s `_StubBackbone` stands in for `facebook/dinov2-base` — the
    scenario under test is `classify()`'s own decision logic, not the
    (already-tested, step 06) backbone/head forward pass.
    """

    def predict(pixel_values_per_angle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            angle: _logits_for_class(class_index_per_angle[angle])
            for angle in pixel_values_per_angle
        }

    return predict


def _fixture_photo() -> dict[str, torch.Tensor]:
    """A structurally-valid per-angle pixel batch.

    Pixel values are irrelevant to every test here: the stub `predict`
    function ignores them and returns crafted logits instead.
    """
    return {angle: torch.rand(1, 3, 8, 8) for angle in GUIDED_ANGLES}


# --- CK-4: wrong-watch-model input --------------------------------------


def test_classify_returns_not_a_submariner_for_a_different_watch_model():
    """photo_of_different_watch_model: some angles reject, others don't —
    a watch-shaped object is present but doesn't match Submariner
    features consistently across every guided angle.
    """
    photo_of_different_watch_model = _fixture_photo()
    angles = list(photo_of_different_watch_model)
    mixed_votes = {
        angle: REJECTION_CLASS_INDEX if index % 2 == 0 else GENUINE_CLASS_INDEX
        for index, angle in enumerate(angles)
    }

    result = classify(photo_of_different_watch_model, predict=_stub_predict(mixed_votes))

    assert isinstance(result, Rejection)
    assert result.reason == "not-a-submariner"


# --- CK-4: no-watch-in-frame input ---------------------------------------


def test_classify_returns_not_evaluable_when_no_watch_is_in_frame():
    """no_watch_in_frame: every guided angle votes rejection — nothing
    watch-like detected anywhere in the photo.
    """
    no_watch_in_frame = _fixture_photo()
    all_angles_reject = {angle: REJECTION_CLASS_INDEX for angle in no_watch_in_frame}

    result = classify(no_watch_in_frame, predict=_stub_predict(all_angles_reject))

    assert isinstance(result, Rejection)
    assert result.reason == "not-evaluable"


# --- Non-OOD input must proceed to scoring, never fabricate a verdict ---


def test_classify_returns_none_for_a_genuine_submariner_photo():
    """No angle votes rejection -> not OOD -> caller proceeds to scoring.

    `classify()` returns bare `None` here, never anything resembling a
    genuine/fake score — it has no code path that constructs a
    `core.types.Prediction` at all.
    """
    genuine_photo = _fixture_photo()
    no_angle_rejects = {angle: GENUINE_CLASS_INDEX for angle in genuine_photo}

    result = classify(genuine_photo, predict=_stub_predict(no_angle_rejects))

    assert result is None
