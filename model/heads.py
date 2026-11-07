"""Per-angle linear-probe heads over the frozen DINOv2 backbone (CK-14, CK-19).

`AngleHeadModel` is the architecture's trainable surface: a frozen
`adapters.backbone` feature extractor plus one `nn.Linear(768, NUM_CLASSES)`
per guided angle (SKILL.md:150, Requirements item 6's "linear probe"
decision). Nothing here re-implements angle aggregation — `combine_angle_
scores` calls `core.rules.aggregate_angles` directly, the same function
object `core/rules.py` already defines and already tests (CK-19: the
combination step has zero trainable parameters because it is not a second
model, just that one plain-mean rule).

Guided angles are read from `dataset.crop.ANGLE_CROP_FRACTIONS` rather than
re-declared here as a second hardcoded list — "which angles exist" stays a
single fact (CK-22) instead of two lists that could silently drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from adapters.backbone import BACKBONE_CHECKPOINT, BACKBONE_EMBED_DIM, load_frozen_backbone
from core.rules import aggregate_angles, require_all_model_classes
from core.types import ModelClass
from dataset.crop import ANGLE_CROP_FRACTIONS

GUIDED_ANGLES: tuple[str, ...] = tuple(ANGLE_CROP_FRACTIONS)

# The canonical vocabulary contains genuine + grades 1-4 + rejection.
# Deriving the size from `ModelClass` keeps indices and head width together.
NUM_CLASSES = len(ModelClass)

# Class-index convention the per-angle heads' output follows; `model/
# rejection.py` (step 07a) and `calibration` (step 07b) read this same
# position instead of each re-deriving their own layout.
GENUINE_CLASS_INDEX = ModelClass.GENUINE.value


class ImageBackbone(Protocol):
    """Duck-typed interface `build_model()` needs from a backbone.

    `load_frozen_backbone()` returns a real `transformers` model that
    satisfies this; a test may substitute a tiny stub instead (this
    step's own Blockers & Risks mitigation: "smoke test may use a tiny
    stub backbone"), so the 1-epoch fixture smoke run never has to reach
    the network or download the real ~350MB checkpoint.
    """

    def parameters(self) -> object: ...

    def __call__(self, pixel_values: torch.Tensor) -> Any: ...

    def eval(self) -> Any: ...


class AngleHeadModel(nn.Module):
    """Frozen backbone + one trainable `nn.Linear` head per guided angle.

    `self.backbone` and `self.heads` are kept as separate attributes
    (rather than one flat parameter list) specifically so a caller — or
    the CK-14 test — can walk each independently: zero trainable backbone
    parameters, fully trainable head parameters.
    """

    def __init__(self, backbone: ImageBackbone, angles: tuple[str, ...] = GUIDED_ANGLES) -> None:
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {angle: nn.Linear(BACKBONE_EMBED_DIM, NUM_CLASSES) for angle in angles}
        )

    def forward(self, pixel_values_per_angle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return per-angle class logits.

        The backbone forward pass runs under `torch.no_grad()` as a cheap
        correctness belt-and-braces on top of `requires_grad=False` — it
        guarantees no autograd graph is built through frozen weights even
        if a caller forgets to freeze them first, without duplicating the
        freeze itself.
        """
        logits_per_angle: dict[str, torch.Tensor] = {}
        for angle, pixel_values in pixel_values_per_angle.items():
            with torch.no_grad():
                features = self.backbone(pixel_values).pooler_output
            logits_per_angle[angle] = self.heads[angle](features)
        return logits_per_angle


def build_model(
    *, backbone: ImageBackbone | None = None, checkpoint: str = BACKBONE_CHECKPOINT
) -> AngleHeadModel:
    """Build the frozen-backbone + per-angle-heads model (CK-14).

    Loads the real `facebook/dinov2-base` checkpoint unless a caller
    injects its own `backbone` — tests use that seam to stay
    network-independent (see `ImageBackbone`).
    """
    resolved_backbone = backbone if backbone is not None else load_frozen_backbone(checkpoint)
    return AngleHeadModel(resolved_backbone)


def build_trainable_model(
    class_counts: Mapping[str, Mapping[ModelClass, int]],
    *,
    backbone: ImageBackbone | None = None,
    checkpoint: str = BACKBONE_CHECKPOINT,
) -> AngleHeadModel:
    """Build training heads only after every angle proves six-class coverage."""
    if set(class_counts) != set(GUIDED_ANGLES):
        raise ValueError("training class counts must cover every guided angle")
    for counts in class_counts.values():
        require_all_model_classes(counts)
    return build_model(backbone=backbone, checkpoint=checkpoint)


def trainable_head_parameters(model: AngleHeadModel) -> list[nn.Parameter]:
    """Return only the per-angle head parameters `train_one_epoch` updates.

    Isolated as its own function so the training optimizer and the CK-14
    test share one definition of "what's trainable" instead of each
    re-deriving it from `model.heads` separately.
    """
    return list(model.heads.parameters())


def angle_genuine_probabilities(logits_per_angle: dict[str, torch.Tensor]) -> dict[str, float]:
    """Reduce each angle's class logits to a scalar genuine-probability.

    `core.rules.aggregate_angles` takes `dict[str, float]` (Contracts:
    "simple rule, 0 trainable params"), not raw per-class logit tensors —
    this is the non-learned (plain softmax, no parameters) bridge between
    the heads' output and that rule, not a replacement for it.
    """
    return {
        angle: torch.softmax(logits, dim=-1)[..., GENUINE_CLASS_INDEX].mean().item()
        for angle, logits in logits_per_angle.items()
    }


def combine_angle_scores(logits_per_angle: dict[str, torch.Tensor]) -> dict[str, float]:
    """Combine per-angle head outputs via the core non-learned rule (CK-14/19).

    Calls `core.rules.aggregate_angles` directly — aggregation is
    structurally forbidden from becoming a second trained model
    (Requirements item 6, CK-19); calibration (step 07b) consumes this
    function's output, which stays un-calibrated ("raw_genuine") on
    purpose.
    """
    return aggregate_angles(angle_genuine_probabilities(logits_per_angle))


def train_one_epoch(
    model: AngleHeadModel,
    batches: list[dict[str, torch.Tensor]],
    labels_per_angle: list[dict[str, torch.Tensor]],
) -> float:
    """Run one training epoch, updating only the per-angle head parameters.

    The optimizer is built over `trainable_head_parameters(model)` only —
    the frozen backbone is never part of it, which is what keeps this a
    linear-probe training step rather than a fine-tune (Requirements item
    6). Returns the mean per-batch loss, mostly so a smoke test can assert
    it is a finite number rather than NaN/inf.
    """
    optimizer = torch.optim.Adam(trainable_head_parameters(model), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    for pixel_values_per_angle, batch_labels in zip(batches, labels_per_angle, strict=True):
        optimizer.zero_grad()
        logits_per_angle = model(pixel_values_per_angle)
        loss = sum(
            criterion(logits_per_angle[angle], batch_labels[angle]) for angle in logits_per_angle
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(batches), 1)


def run_smoke_training(
    model: AngleHeadModel,
    batches: list[dict[str, torch.Tensor]],
    labels_per_angle: list[dict[str, torch.Tensor]],
    checkpoint_path: Path,
) -> Path:
    """Run the 1-epoch fixture-split smoke train and write a checkpoint.

    Saves only `model.heads.state_dict()` — the frozen backbone's weights
    never change during training, so persisting them in every checkpoint
    would be redundant disk usage with no training signal in it.
    """
    train_one_epoch(model, batches, labels_per_angle)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.heads.state_dict(), checkpoint_path)
    return checkpoint_path
