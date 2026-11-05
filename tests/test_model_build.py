"""Tests for the frozen-DINOv2 + per-angle-heads model (CK-14, CK-19).

Deviation note (flagged explicitly, not worked around silently): every
test in this module is gated behind `pytest.importorskip`. This sandbox's
Python (3.14) is newer than the pinned `torch==2.5.1` / `transformers==
4.46.0` wheels (requirements.txt) support, so neither package installs
here — the same class of environment limitation hit in step
`05c-augment-splits` (see `dataset/augment.py`'s module docstring: "torch
is not installed here"), one level more severe (torch itself, not only
the DINOv2 checkpoint download). Per this step's own Blockers & Risks
mitigation ("smoke test may use a tiny stub backbone; freeze-assertion
test does not need real download"), every test below is also written to
need neither the network nor the real ~350MB checkpoint even when torch
*is* installed — `_StubBackbone` stands in for `facebook/dinov2-base`,
and the freeze-assertion test monkeypatches `AutoModel.from_pretrained`.
So these tests are not placeholders: wherever the pinned requirements.txt
actually installs (the AWS training box, or CI once wired), they run for
real against real backbone/head code with zero network dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

# `TYPE_CHECKING` gives mypy a real `torch`/`transformers` module to resolve
# annotations like `torch.Tensor` against; `pytest.importorskip` is what
# actually runs at test time and produces the module-or-skip behavior this
# file's docstring documents. Neither branch executes both statements.
if TYPE_CHECKING:
    import torch
    import transformers
else:
    torch = pytest.importorskip(
        "torch", reason="torch==2.5.1 has no wheel for this sandbox's Python 3.14"
    )
    transformers = pytest.importorskip(
        "transformers", reason="transformers==4.46.0 requires the unavailable torch==2.5.1"
    )

from torch import nn  # noqa: E402 (after importorskip, by design)

from adapters.backbone import BACKBONE_EMBED_DIM, load_frozen_backbone  # noqa: E402
from core.rules import aggregate_angles as core_aggregate_angles  # noqa: E402
from model.heads import (  # noqa: E402
    GENUINE_CLASS_INDEX,
    GUIDED_ANGLES,
    NUM_CLASSES,
    angle_genuine_probabilities,
    build_model,
    combine_angle_scores,
    run_smoke_training,
)
from model.heads import (  # noqa: E402
    aggregate_angles as heads_aggregate_angles,
)


class _StubBackbone(nn.Module):
    """Network-free stand-in for `facebook/dinov2-base`.

    A real (but already-frozen) parameter, so build_model's CK-14 test is
    checking an actual `requires_grad=False` flag rather than an absence
    of parameters altogether. The pooled output is deterministic and
    sized to `BACKBONE_EMBED_DIM`, matching the shape a real DINOv2
    pooler_output would have.
    """

    def __init__(self, embed_dim: int = BACKBONE_EMBED_DIM) -> None:
        super().__init__()
        self._embed_dim = embed_dim
        self._probe = nn.Parameter(torch.zeros(embed_dim), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> object:
        batch_size = pixel_values.shape[0]
        pooled = self._probe.unsqueeze(0).expand(batch_size, self._embed_dim)
        return type("BackboneOutput", (), {"pooler_output": pooled})()


def _fixture_batch(angles: tuple[str, ...], batch_size: int = 2) -> dict[str, torch.Tensor]:
    """A tiny, structurally-valid per-angle pixel batch (no real photos needed)."""
    return {angle: torch.rand(batch_size, 3, 8, 8) for angle in angles}


def _fixture_labels(angles: tuple[str, ...], batch_size: int = 2) -> dict[str, torch.Tensor]:
    """All-genuine labels — the smoke test only needs the loop to run, not to converge."""
    return {
        angle: torch.full((batch_size,), GENUINE_CLASS_INDEX, dtype=torch.long)
        for angle in angles
    }


# --- CK-14: load_frozen_backbone -------------------------------------------


def test_load_frozen_backbone_freezes_every_parameter(monkeypatch):
    """requires_grad=False on all backbone params, with zero network calls."""
    fake_checkpoint = nn.Linear(BACKBONE_EMBED_DIM, BACKBONE_EMBED_DIM)
    assert any(p.requires_grad for p in fake_checkpoint.parameters())  # starts trainable

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        staticmethod(lambda *_args, **_kwargs: fake_checkpoint),
    )

    backbone = load_frozen_backbone("facebook/dinov2-base")

    assert all(not p.requires_grad for p in backbone.parameters())


# --- CK-14: build_model ------------------------------------------------


def test_build_model_backbone_frozen_heads_trainable():
    model = build_model(backbone=_StubBackbone())

    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.heads.parameters())


def test_build_model_has_one_linear_head_per_guided_angle():
    model = build_model(backbone=_StubBackbone())

    assert set(model.heads.keys()) == set(GUIDED_ANGLES)
    for head in model.heads.values():
        assert isinstance(head, nn.Linear)
        assert head.in_features == BACKBONE_EMBED_DIM
        assert head.out_features == NUM_CLASSES


# --- CK-14/CK-19: aggregation stays the core non-learned rule ---------


def test_heads_module_reuses_the_core_aggregate_angles_object():
    """No reimplementation: the imported name IS core.rules.aggregate_angles (CK-22)."""
    assert heads_aggregate_angles is core_aggregate_angles


def test_combine_angle_scores_delegates_to_core_aggregate_angles():
    model = build_model(backbone=_StubBackbone())
    logits_per_angle = model(_fixture_batch(GUIDED_ANGLES))

    scores = angle_genuine_probabilities(logits_per_angle)
    expected = core_aggregate_angles(scores)

    assert combine_angle_scores(logits_per_angle) == expected


# --- 1-epoch fixture smoke training + checkpoint -----------------------


def test_run_smoke_training_completes_and_writes_a_checkpoint(tmp_path: Path):
    model = build_model(backbone=_StubBackbone())
    batches = [_fixture_batch(GUIDED_ANGLES) for _ in range(2)]
    labels = [_fixture_labels(GUIDED_ANGLES) for _ in range(2)]
    checkpoint_path = tmp_path / "smoke_heads.pt"

    result_path = run_smoke_training(model, batches, labels, checkpoint_path)

    assert result_path == checkpoint_path
    assert checkpoint_path.exists()

    saved_state = torch.load(checkpoint_path, weights_only=True)
    assert set(saved_state.keys()) == set(model.heads.state_dict().keys())
