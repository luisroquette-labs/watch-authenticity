"""OOD / rejection gate for input the model was never built to score (CK-4).

`classify()` is the short-circuit this task's credibility rests on:
input that is not a Rolex Submariner at all, or shows no watch
whatsoever, must never reach the genuine/fake scoring path (Requirements
item 14; Architecture's Rejection/OOD component lists dependencies as
`core` and `heads` only — no separate, independently-trained presence
detector). Both failure modes read off the per-angle heads' own
dedicated rejection class (`model.heads.NUM_CLASSES` already reserves
one class index for it, SKILL.md:150's "genuine + grade1-4 + rejection"),
not a second model.

One signal, two thresholds on it, per guided angle's predicted class:
  - every angle votes rejection -> nothing watch-like anywhere in the
    frame -> "not-evaluable".
  - at least one angle votes rejection while at least one other votes a
    scoreable class -> a watch-shaped object is present, but the
    Submariner-specific angle features (rehaut engraving, cyclops lens,
    crown logo, ...) are not consistently recognized -> "not-a-submariner".
  - no angle votes rejection -> not OOD; `classify()` returns `None` so
    the caller proceeds to genuine/fake scoring (Architecture's "OOD
    input must short-circuit BEFORE any genuine/fake scoring").
"""

from __future__ import annotations

from typing import Protocol, cast

import torch

from core.types import ModelClass, Rejection
from model.heads import build_model

# The per-angle heads reserve their highest class index for "rejection"
# (SKILL.md:150's "genuine + grade1-4 + rejection" ordering places it
# last; `model.heads.GENUINE_CLASS_INDEX` is 0, the opposite end).
REJECTION_CLASS_INDEX = ModelClass.REJECTION.value


class AnglePredictor(Protocol):
    """Duck-typed interface `classify()` needs from a per-angle model.

    `model.heads.build_model()`'s return value satisfies this (an
    `nn.Module`'s `forward` is its `__call__`); a test substitutes a
    plain stub function instead, the same testability seam
    `model.heads.ImageBackbone` gives `build_model()` itself.
    """

    def __call__(self, pixel_values_per_angle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


def classify(
    pixel_values_per_angle: dict[str, torch.Tensor],
    *,
    predict: AnglePredictor | None = None,
) -> Rejection | None:
    """Return a `Rejection` for OOD input, or `None` to proceed to scoring.

    Runs BEFORE any genuine/fake scoring: a caller checks this function's
    return value first and only reaches the scoring path once it returns
    `None` — `classify()` itself never constructs a `core.types.
    Prediction`, so OOD input structurally cannot carry a fabricated
    genuine/fake score.
    """
    predict_fn = predict if predict is not None else cast(AnglePredictor, build_model())
    return classify_logits(predict_fn(pixel_values_per_angle))


def classify_logits(logits_per_angle: dict[str, torch.Tensor]) -> Rejection | None:
    """Apply the OOD decision to logits already produced by inference."""
    if not logits_per_angle:
        raise ValueError("classify_logits requires at least one angle")

    rejected_angles = [_predicted_class(logits) == REJECTION_CLASS_INDEX for logits in logits_per_angle.values()]

    if not any(rejected_angles):
        return None  # every angle recognizes a scoreable class — not OOD

    if all(rejected_angles):
        return Rejection(
            reason="not-evaluable",
            instruction="No watch detected in any guided angle — recapture needed.",
        )

    return Rejection(
        reason="not-a-submariner",
        instruction="A watch is visible, but it does not match the trained Submariner angle features.",
    )


def _predicted_class(logits: torch.Tensor) -> int:
    """Majority-vote predicted class across a batch of one angle's logits.

    Real inference always passes a batch of one (a single photo); the
    majority vote only matters for multi-sample test fixtures, where it
    keeps this function correct without a separate single-photo code path.
    """
    return int(torch.mode(logits.argmax(dim=-1)).values.item())
