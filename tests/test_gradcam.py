"""Tests for Grad-CAM explainability (CK-6): `explainability.gradcam.grad_cam`.

Deviation note (same class of environment limitation as `test_model_build.
py`/`test_rejection.py`): every test here is gated behind
`pytest.importorskip` because optional training dependencies may be absent
from a lightweight local environment. None
of these tests need the network or a real DINOv2 checkpoint even where
torch *is* installed: each builds a tiny, ViT-shaped stub classifier
(patch-embed + one transformer block + linear head) instead of loading
`facebook/dinov2-base` — the same "small-scale"/fixture-level smoke
training the Test Matrix's integration row calls for ("small fixture
image set for a 1-epoch smoke training run"), not the real ~350MB
checkpoint. So these are not placeholders: wherever the pinned
requirements.txt actually installs, they run for real against real
`grad_cam()` code (independently verified for real in a disposable venv
with real `torch`/`grad-cam` installed — see this step's sub-task file
for the verification record).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip(
        "torch", reason="optional training dependency torch is not installed"
    )
    pytest.importorskip(
        "pytorch_grad_cam", reason="optional training dependency grad-cam is not installed"
    )
    from torch import nn  # noqa: E402

from explainability.gradcam import grad_cam  # noqa: E402

_PATCH_SIZE = 4
_EMBED_DIM = 8
_NUM_CLASSES = 3


class _Norm1Block(nn.Module):
    """One ViT transformer block's shape: `.norm1` is Grad-CAM's target layer.

    Shared by both stub classifiers below so the "one block with a
    `norm1`" shape is declared once, not duplicated per attribute layout.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm1(tokens)


def _tokenize(pixel_values: torch.Tensor, patch_embed: nn.Conv2d, cls_token: torch.Tensor) -> torch.Tensor:
    """Patch-embed `pixel_values` and prepend the CLS token (shared helper, CK-22).

    Both stub classifiers below need the identical tokenization step
    (patchify -> flatten -> prepend CLS); factoring it out keeps this
    file free of writing that logic twice.
    """
    patches = patch_embed(pixel_values)
    batch_size = patches.shape[0]
    patch_tokens = patches.flatten(2).transpose(1, 2)
    cls = cls_token.expand(batch_size, -1, -1)
    return torch.cat([cls, patch_tokens], dim=1)


class _StubVitClassifier(nn.Module):
    """A minimal timm-shaped ViT classifier: exposes `.blocks[-1].norm1`.

    Shaped like a real ViT (patch-grid tokens + CLS, one block) so
    `grad_cam()`'s target-layer resolution and token-to-grid reshape
    exercise real tensor shapes, not a flat MLP that would skip the
    ViT-specific code path this step exists to test. Deliberately NOT
    wrapped in `torch.no_grad()` anywhere (unlike `model.heads.
    AngleHeadModel.forward`): Grad-CAM needs an autograd path through
    this classifier's own forward pass.
    """

    def __init__(self, embed_dim: int = _EMBED_DIM, num_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([_Norm1Block(embed_dim)])
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = _tokenize(pixel_values, self.patch_embed, self.cls_token)
        tokens = self.blocks[-1](tokens)
        # Mean-pool every token (CLS + patches), not CLS alone: a real ViT
        # block's self-attention makes the CLS output depend on every
        # patch; this stub's `_Norm1Block` is a plain per-token LayerNorm
        # with no cross-token mixing, so CLS-only pooling would make the
        # patch tokens structurally unreachable by backprop — a
        # degenerate Grad-CAM map for a reason that has nothing to do
        # with `grad_cam()`'s own correctness.
        return self.head(tokens.mean(dim=1))


class _StubDinov2StyleClassifier(nn.Module):
    """Mimics HF `Dinov2Model`'s `.encoder.layer[-1].norm1` path (no `.blocks`).

    Exercises `grad_cam()`'s fallback target-layer resolution — the real
    backbone this project loads (`adapters.backbone.load_frozen_backbone`,
    `facebook/dinov2-base` via `transformers.AutoModel`) has this shape,
    not the timm-style `.blocks` attribute SKILL.md's snippet assumes.
    """

    def __init__(self, embed_dim: int = _EMBED_DIM, num_classes: int = _NUM_CLASSES) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=_PATCH_SIZE, stride=_PATCH_SIZE)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder = nn.ModuleDict({"layer": nn.ModuleList([_Norm1Block(embed_dim)])})
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        tokens = _tokenize(pixel_values, self.patch_embed, self.cls_token)
        layer = cast(nn.ModuleList, self.encoder["layer"])
        tokens = layer[-1](tokens)
        # Mean-pool every token (CLS + patches), not CLS alone: a real ViT
        # block's self-attention makes the CLS output depend on every
        # patch; this stub's `_Norm1Block` is a plain per-token LayerNorm
        # with no cross-token mixing, so CLS-only pooling would make the
        # patch tokens structurally unreachable by backprop — a
        # degenerate Grad-CAM map for a reason that has nothing to do
        # with `grad_cam()`'s own correctness.
        return self.head(tokens.mean(dim=1))


def _fixture_image(batch_size: int = 1) -> torch.Tensor:
    """An 8x8 RGB fixture image batch — `_PATCH_SIZE=4` gives a square 2x2 patch grid."""
    return torch.rand(batch_size, 3, 8, 8)


# --- CK-6: heatmap dimensions match the input image ------------------------


def test_grad_cam_returns_heatmap_matching_image_dimensions():
    model = _StubVitClassifier()
    image = _fixture_image()

    heatmap = grad_cam(model, image)

    assert heatmap.shape == tuple(image.shape[-2:])


def test_grad_cam_resolves_dinov2_style_target_layer_without_blocks_attribute():
    """ViT-compatible means both attribute layouts, not just timm's `.blocks`."""
    model = _StubDinov2StyleClassifier()
    image = _fixture_image()

    heatmap = grad_cam(model, image)

    assert heatmap.shape == tuple(image.shape[-2:])


def test_grad_cam_works_when_every_model_parameter_is_frozen():
    """Mirrors the real deployment shape: a frozen backbone (CK-14).

    `adapters.backbone.load_frozen_backbone` sets `requires_grad=False`
    on every backbone parameter. Without `grad_cam()`'s own
    `image.clone().requires_grad_(True)`, a forward pass through an
    entirely-frozen model builds no autograd graph at all, and Grad-CAM's
    backward pass would silently find nothing at the target layer.
    """
    model = _StubVitClassifier()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    image = _fixture_image()

    heatmap = grad_cam(model, image)

    assert heatmap.shape == tuple(image.shape[-2:])


def test_grad_cam_works_inside_an_ambient_no_grad_context():
    """Mirrors how `model.heads.AngleHeadModel.forward` actually runs inference.

    `AngleHeadModel.forward` wraps its backbone call in `torch.no_grad()`
    for efficiency; a real caller computing a heatmap alongside that same
    inference call plausibly does so from within that same ambient
    context. `torch.no_grad()` disables gradient tracking for every op
    inside it regardless of a tensor's own `requires_grad` flag, so
    `grad_cam()` must re-enable it itself (`torch.enable_grad()`) rather
    than trusting the caller's context.
    """
    model = _StubVitClassifier()
    image = _fixture_image()

    with torch.no_grad():
        heatmap = grad_cam(model, image)

    assert heatmap.shape == tuple(image.shape[-2:])


# --- CK-6: non-degenerate heatmap on a small-scale trained checkpoint ------


def _train_fixture_classifier(images: torch.Tensor, labels: torch.Tensor) -> _StubVitClassifier:
    """Run a short real training loop — the "small-scale trained checkpoint"
    the Test Matrix's integration row calls for, at fixture scale.
    """
    model = _StubVitClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()

    for _ in range(20):
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    return model


def test_grad_cam_is_non_degenerate_on_a_small_scale_trained_checkpoint(tmp_path: Path):
    torch.manual_seed(0)
    images = _fixture_image(batch_size=4)
    labels = torch.tensor([0, 1, 2, 0])
    trained_model = _train_fixture_classifier(images, labels)

    checkpoint_path = tmp_path / "gradcam_smoke.pt"
    torch.save(trained_model.state_dict(), checkpoint_path)

    loaded_model = _StubVitClassifier()
    loaded_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    loaded_model.eval()

    heatmap = grad_cam(loaded_model, images[:1])

    assert heatmap.shape == tuple(images.shape[-2:])
    # Non-degenerate: a real, spatially-varying map, not an all-zero or
    # uniformly-flat one — a flat `pytorch_grad_cam` output normalizes to
    # all zeros (see `explainability/gradcam.py`'s `grad_cam` docstring),
    # so a nonzero spread here is exactly what "non-degenerate" means.
    assert heatmap.std() > 0.0
