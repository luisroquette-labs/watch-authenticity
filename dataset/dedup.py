"""Exact-pixel removal and studio/catalog-photo filtering.

Dataset op (Expected Changes: `dataset/dedup.py`). Removes two confounds
before any photo reaches a split (Requirements items 5 and 13; CK-11):

1. Identical images: remove repeated pixels without discarding distinct
   photos that collide on perceptual hash. Near matches remain available
   for split-family grouping and review.
2. Studio/catalog photos (near-perfect white backdrop, repeated across
   listings): a photographic-style tell that lets the model learn "which
   marketplace this came from" instead of "is this watch fake" — the
   source-bias risk this task treats as its #1 named risk.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import imagehash
import numpy as np

# pHash proposes split-family review pairs. It does not prove image identity:
# distinct archived watch photos have been observed with distance zero.
HAMMING_DEDUP_THRESHOLD = 10

# Studio thresholds are provisional; calibrate them against reviewed photos.
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
    """Drop pixel-identical images; retain near matches for entity review."""
    kept: list[Any] = []
    seen: set[bytes] = set()
    for image in images:
        rgba = image.convert("RGBA")
        digest = hashlib.sha256(f"{rgba.size}:".encode() + rgba.tobytes()).digest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(image)
    return kept


def classify_capture_style(image: Any, metadata: Mapping[str, object]) -> str:
    """Detect white-backdrop studio photos; other styles require review."""
    del metadata
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    border_pixels = _border_ring(pixels, _BORDER_MARGIN_FRACTION)
    near_white = np.all(border_pixels >= STUDIO_WHITE_RGB_FLOOR, axis=-1)
    return "studio" if near_white.mean() >= STUDIO_BORDER_WHITE_FRACTION else "unknown"


def route_capture(style: str, metadata: Mapping[str, object]) -> str:
    """Route catalog-like captures without discarding confirmed studio replicas."""
    if style not in {"studio", "casual", "unknown"}:
        raise ValueError("unknown capture style")
    if style == "unknown":
        return "review"
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
