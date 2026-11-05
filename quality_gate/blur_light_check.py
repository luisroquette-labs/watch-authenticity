"""Pre-inference photo-quality gate (CK-5): reject a bad photo before scoring.

A blurry or badly-framed photo must never reach the authenticity model —
Requirements item 15 and the Out-of-Distribution/Quality rubric both treat
"never fabricate a score for input you cannot evaluate" as a hard rule, and
the quality gate is the first of the two short-circuits that enforce it
(`model/rejection.py`'s OOD check is the second, applied only after a photo
already passed this gate — see Architecture's runtime scenarios).

`sharpness_score` is the SKILL.md:174-175 Laplacian-variance signal as-is:
a sharp photo's edges keep their high-frequency detail, so the Laplacian
response has high variance; a blurry photo's edges are smoothed away, so
the response flattens toward zero variance.

The pass/fail threshold is deliberately per-angle rather than one universal
constant — SKILL.md's own comment on this signal is "flat/polished
casebacks score low even in focus," which is exactly this step's Blockers
& Risks entry: a single global threshold would false-reject a genuinely
sharp caseback photo while under-rejecting a blurry dial photo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from core.types import Rejection

# ponytail: empirically-tuned per-angle Laplacian-variance floor, not a
# universal constant (SKILL.md: practitioner range 100-1000). "caseback" is
# deliberately lower than the rest: a flat, polished caseback is sharp but
# has little high-frequency texture for the Laplacian to respond to, so the
# same floor used for a textured dial/rehaut would false-reject it even in
# focus (this step's Blockers & Risks). Re-tune every value once real
# guided-capture photos exist — the capture app is a separate, later task.
SHARPNESS_THRESHOLD_PER_ANGLE: dict[str, float] = {
    "rehaut": 150.0,
    "cyclops": 150.0,
    "crown": 150.0,
    "clasp": 150.0,
    "dial": 150.0,
    "caseback": 80.0,
}

# Fallback for a call that doesn't name a guided angle, or names one not yet
# in the table above — mirrors `dataset.crop.DEFAULT_CROP_FRACTION`'s same
# "still gate something, don't silently skip" fallback for an unmapped key.
DEFAULT_SHARPNESS_THRESHOLD = 150.0


@dataclass(frozen=True)
class Pass:
    """Sentinel: the photo cleared the quality gate and may reach the model.

    `sharpness` is carried only for logging/debugging. A caller decides
    whether a photo may proceed by checking for `Rejection` (`core.types.
    Rejection`), not by reading any field here — mirroring how `model.
    rejection.classify` returns `None` rather than a dedicated "ok" value.
    """

    sharpness: float


def sharpness_score(gray_image: np.ndarray) -> float:
    """Laplacian-variance sharpness signal (SKILL.md:174-175, as-is).

    `gray_image` must already be single-channel — this function is the
    bare signal, with no framing/color-conversion policy of its own.
    `quality_gate` owns that conversion plus the pass/fail decision.
    """
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def quality_gate(image: Any, *, angle: str | None = None) -> Pass | Rejection:
    """Reject a blurry photo before the authenticity model ever runs on it.

    `angle` selects the per-angle threshold in `SHARPNESS_THRESHOLD_PER_ANGLE`
    (falling back to `DEFAULT_SHARPNESS_THRESHOLD` when omitted or unmapped —
    the guided-capture caller is expected to pass the angle it just asked the
    user to photograph, but a threshold still applies when it does not).

    A score strictly below the threshold fails; a score at or above it
    passes — so the threshold value itself is the first passing score, not
    a boundary rejection.
    """
    threshold = (
        SHARPNESS_THRESHOLD_PER_ANGLE.get(angle, DEFAULT_SHARPNESS_THRESHOLD)
        if angle is not None
        else DEFAULT_SHARPNESS_THRESHOLD
    )
    score = sharpness_score(_to_grayscale(image))

    if score < threshold:
        return Rejection(
            reason="blurry",
            instruction="Photo is too blurry to evaluate — hold steady and recapture.",
        )

    return Pass(sharpness=score)


def _to_grayscale(image: Any) -> np.ndarray:
    """Convert this project's PIL-image fixture/capture convention to the
    single-channel `numpy` array `sharpness_score`'s `cv2.Laplacian` call
    needs (the same PIL-in/`numpy`-for-cv2-out shape `dataset.augment`
    already uses for its own `cv2.GaussianBlur` call)."""
    return np.array(image.convert("L"), dtype=np.uint8)
