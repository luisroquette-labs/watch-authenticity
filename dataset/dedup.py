"""Perceptual-hash near-duplicate removal and studio/catalog-photo filtering.

Dataset op (Expected Changes: `dataset/dedup.py`). Removes two confounds
before any photo reaches a split (Requirements items 5 and 13; CK-11):

1. Near-duplicates: the same physical photo reposted across multiple
   listings would let one real watch dominate a class through pure
   repetition instead of genuine diversity.
2. Studio/catalog photos (near-perfect white backdrop, repeated across
   listings): a photographic-style tell that lets the model learn "which
   marketplace this came from" instead of "is this watch fake" — the
   source-bias risk this task treats as its #1 named risk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import imagehash
import numpy as np

# ponytail: no literature-backed universal constant exists for a pHash
# Hamming-distance dedup threshold (SKILL.md: "tune empirically on this
# dataset"). 10 is a first-pass value measured against this module's own
# fixtures: a JPEG-recompressed or resize-roundtripped copy of the same
# photo lands at distance ~2-8, while two genuinely different photos land
# around 30 (64-bit pHash). Re-tune once real repost pairs from the scraped
# dataset are available.
HAMMING_DEDUP_THRESHOLD = 10

# ponytail: same empirically-tuned, not-universal caveat as above. A studio
# backdrop is "near-perfect white," not literally (255,255,255) — JPEG
# compression and studio lighting add noise. These two values are a
# first-pass estimate; re-tune against real scraped catalog-photo samples.
STUDIO_WHITE_RGB_FLOOR = 235
STUDIO_BORDER_WHITE_FRACTION = 0.8

# Checking only the outermost ring of the frame (rather than the whole
# image) avoids the centered product itself diluting the white-pixel
# fraction — catalog photos keep the backdrop visible at the edges even
# when the product fills most of the center.
_BORDER_MARGIN_FRACTION = 0.05


def hash_distance(image_a: Any, image_b: Any) -> int:
    """Hamming distance between two images' pHash fingerprints.

    `imagehash.ImageHash.__sub__` already counts differing bits between two
    equal-length hashes, which is inherently symmetric — the property the
    CK-11 symmetry test asserts (`hash_distance(a, b) == hash_distance(b, a)`).
    """
    return imagehash.phash(image_a) - imagehash.phash(image_b)


def dedup_filter(images: list[Any]) -> list[Any]:
    """Drop near-duplicate photos, keeping the first occurrence of each.

    A photo counts as a near-duplicate of an already-kept photo when their
    pHash distance is below `HAMMING_DEDUP_THRESHOLD` (CK-11).
    """
    kept: list[Any] = []
    for image in images:
        is_near_duplicate = any(
            hash_distance(image, kept_image) < HAMMING_DEDUP_THRESHOLD for kept_image in kept
        )
        if not is_near_duplicate:
            kept.append(image)
    return kept


def classify_capture_style(image: Any, metadata: Mapping[str, object]) -> str:
    """Classify photographic style without using the source or assigning a label."""
    del metadata
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border_pixels = _border_ring(pixels, _BORDER_MARGIN_FRACTION)
    near_white = np.all(border_pixels >= STUDIO_WHITE_RGB_FLOOR, axis=-1)
    return "studio" if near_white.mean() >= STUDIO_BORDER_WHITE_FRACTION else "casual"


def route_capture(style: str, metadata: Mapping[str, object]) -> str:
    """Route catalog-like captures without discarding confirmed studio replicas."""
    if style not in {"studio", "casual"}:
        raise ValueError("unknown capture style")
    if style == "casual":
        return "eligible"
    if (
        metadata.get("rights_cleared") is True
        and metadata.get("evidence_confirmed") is True
        and metadata.get("accepted_label") == "fake"
    ):
        return "cross_source_evaluation"
    if metadata.get("repeated_catalog") is True:
        return "exclude"
    return "review"


def _border_ring(pixels: np.ndarray, margin_fraction: float) -> np.ndarray:
    """Return the outermost ring of `pixels`, flattened to shape (N, channels)."""
    height, width, channels = pixels.shape
    margin_h = max(1, round(height * margin_fraction))
    margin_w = max(1, round(width * margin_fraction))
    top, bottom = pixels[:margin_h, :, :], pixels[-margin_h:, :, :]
    left, right = pixels[:, :margin_w, :], pixels[:, -margin_w:, :]
    return np.concatenate([side.reshape(-1, channels) for side in (top, bottom, left, right)])
