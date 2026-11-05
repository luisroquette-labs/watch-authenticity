"""Grad-CAM explainability heatmap for every authenticity prediction (CK-6).

Dual role per SKILL.md's own framing: a user-trust signal (which part of
the photo drove the score) and a source-bias audit (a heatmap lighting up
the photo's *background* instead of the watch is flagrant evidence of the
item-13 source-style confound this task's own risk register names as
risk #1). `grad_cam()` is deliberately generic over any plain ViT image
classifier — `pixel_values -> logits`, the exact shape SKILL.md's own
code example assumes — rather than coupled to `model.heads.
AngleHeadModel`'s per-angle dict interface: wiring a per-angle classifier
view and calling this function once per guided angle is step 09
(inference/predict)'s job, not this step's.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from pytorch_grad_cam import GradCAM


class ViTClassifier(Protocol):
    """Duck-typed interface `grad_cam()` needs: a plain image classifier.

    `__call__` returns per-class logits for a batch of images — the same
    contract any `torch.nn.Module` classifier already satisfies, so a
    real DINOv2-backed model and a tiny test stub both work unchanged.
    """

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor: ...


def grad_cam(model: ViTClassifier, image: torch.Tensor) -> np.ndarray:
    """Return an image-dimensioned Grad-CAM heatmap for `model`'s top class.

    Targets the last ViT transformer block's first layer norm (SKILL.md:
    81-82, `_resolve_target_layer`'s own docstring covers the DINOv2
    attribute-path wrinkle). `image` is cloned with `requires_grad_(True)`
    rather than mutated in place or required of the caller: the backbone
    underneath is frozen (`requires_grad=False` on every parameter, per
    `adapters.backbone.load_frozen_backbone`), and without an
    autograd-tracked input, PyTorch builds no backward path at all through
    a subgraph where every parameter is also frozen — Grad-CAM's backward
    pass would silently see no gradient at the target layer.
    """
    target_layers = [_resolve_target_layer(model)]
    gradient_tracked_image = image.clone().requires_grad_(True)

    # `torch.enable_grad()` guards against the realistic caller shape
    # (`model.heads.AngleHeadModel.forward` runs its backbone call inside
    # `torch.no_grad()`): that context forces gradients off for every op
    # inside it regardless of a tensor's own `requires_grad` flag, so a
    # caller computing a heatmap from within that same ambient context
    # would otherwise silently get no gradient at the target layer.
    with torch.enable_grad(), GradCAM(
        model=model, target_layers=target_layers, reshape_transform=_vit_tokens_to_grid
    ) as cam:
        # `targets=None` makes Grad-CAM auto-pick each sample's own
        # highest-scoring class (`pytorch_grad_cam`'s own documented
        # `BaseCAM.forward` behavior) — the model's own predicted class,
        # not a caller-chosen one; `grad_cam()` has no notion of "which
        # class to explain" of its own to pass instead.
        heatmap = cam(input_tensor=gradient_tracked_image, targets=None)[0]

    return heatmap


def _resolve_target_layer(model: object) -> torch.nn.Module:
    """Return `model.blocks[-1].norm1`, or its DINOv2-shaped equivalent.

    SKILL.md's own target-layer snippet (`model.blocks[-1].norm1`)
    assumes a timm-style ViT attribute layout. The frozen backbone this
    project actually loads (`adapters.backbone.load_frozen_backbone`,
    HF `transformers.AutoModel.from_pretrained("facebook/dinov2-base")`)
    is a `Dinov2Model`, whose equivalent path is `encoder.layer[-1].
    norm1` — exactly the risk this step's Blockers & Risks table names
    ("Grad-CAM target layer incompatible with DINOv2... fall back to an
    alternate late-block norm"). Preferring `.blocks` first keeps a
    future timm-style backbone swap working with zero change here.

    Resolved via `getattr` rather than static attribute access: the
    `ViTClassifier` protocol only promises `__call__` (the one contract
    every caller — real backbone or test stub — actually shares), not
    which of these two mutually-exclusive attribute layouts it has.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        return blocks[-1].norm1
    return getattr(model, "encoder").layer[-1].norm1


def _vit_tokens_to_grid(tokens: torch.Tensor) -> torch.Tensor:
    """Reshape a ViT block's `(batch, 1+N, dim)` token output to `(batch, dim, h, w)`.

    `pytorch_grad_cam.GradCAM` assumes a convolutional `(batch, C, H, W)`
    activation by default; a transformer block instead emits one
    embedding per patch token plus a leading CLS token. Dropping the CLS
    token and reshaping the remaining square patch grid is jacobgil's own
    documented `reshape_transform` for ViT target layers — SKILL.md's
    code snippet omits it, but without it Grad-CAM on a ViT block
    produces a shape error rather than a spatially-meaningful map.
    """
    patch_tokens = tokens[:, 1:, :]
    grid_size = int(patch_tokens.shape[1] ** 0.5)
    grid = patch_tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[2])
    return grid.permute(0, 3, 1, 2)
