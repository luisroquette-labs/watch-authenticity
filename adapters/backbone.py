"""Frozen DINOv2 ViT-B/14 backbone loader (SKILL.md:146-150, CK-14).

`load_frozen_backbone()` is the only place `facebook/dinov2-base` gets
loaded: every later stage that needs backbone features reaches it through
this one function instead of each re-instantiating its own copy — the
same single-source-of-truth guarantee `core/rules.py` gives the safety
rules, applied here to the backbone adapter itself.

Freezing every parameter (`requires_grad=False`) is this architecture's
non-negotiable: Requirements item 6 and CK-14 both specify a linear-probe
backbone, so this adapter's only responsibility is "load it, then nail
every parameter shut." Training the per-angle heads on top of the frozen
features is `model/heads.py`'s job, not this module's.
"""

from __future__ import annotations

from typing import Any

from transformers import AutoModel

# Apache-2.0-licensed standard checkpoint (SKILL.md Recommended Libraries
# table) — deliberately not one of the non-commercial XRay-DINO/Cell-DINO
# specialty checkpoints the skill warns against.
BACKBONE_CHECKPOINT = "facebook/dinov2-base"
BACKBONE_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"

# ViT-B/14 pooled-output width (SKILL.md Configuration section). Per-angle
# heads in `model/heads.py` are sized against this constant so the two
# files can never drift out of sync with each other.
BACKBONE_EMBED_DIM = 768


def load_frozen_backbone(checkpoint: str = BACKBONE_CHECKPOINT) -> Any:
    """Load `checkpoint` and freeze every one of its parameters (CK-14).

    Mirrors SKILL.md's "Loading frozen DINOv2 ViT-B/14 + linear probe
    head" example verbatim (AD-1: adapters reuse the Recommended Stack
    unchanged) — the only difference is that the checkpoint name is a
    parameter rather than a hardcoded literal, so a test can substitute
    `AutoModel.from_pretrained` without this function's body changing.
    """
    backbone = AutoModel.from_pretrained(
        checkpoint,
        revision=BACKBONE_REVISION,
        local_files_only=True,
        use_safetensors=True,
    )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone
