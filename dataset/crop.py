"""Per-angle crop: isolate the guided-capture subject from its backdrop.

Dataset op (Expected Changes: `dataset/crop.py`). A first-line defense
against the source-bias confound (Requirements item 13): classification
heads must see the watch part a guided angle targets (rehaut, cyclops,
crown, clasp, dial, caseback), not whatever backdrop surrounds it in the
raw scraped/captured photo (CK-21).
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass
from typing import Any

# ponytail: fraction of the full frame's width/height kept per guided
# angle, centered on the frame. First-pass estimates, not measured framing
# data — the guided-capture app that will produce these photos is a
# separate, later task (Requirements item 11), so there is no real sample
# yet of where each angle's subject lands in frame. Re-tune once that app
# ships and real captures exist. Smaller fractions (cyclops, crown) target
# features that are a small part of the watch even when well-framed;
# larger fractions (dial, caseback) target features meant to fill most of
# the frame.
ANGLE_CROP_FRACTIONS: dict[str, tuple[float, float]] = {
    "rehaut": (0.6, 0.6),
    "cyclops": (0.4, 0.4),
    "crown": (0.5, 0.5),
    "clasp": (0.6, 0.5),
    "dial": (0.75, 0.75),
    "caseback": (0.75, 0.75),
}

# Fallback for an angle not yet in ANGLE_CROP_FRACTIONS: still removes some
# backdrop rather than returning the untouched full frame.
DEFAULT_CROP_FRACTION = (0.6, 0.6)
CROP_ALGORITHM_VERSION = "centered-angle-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CropDerivative:
    image: Any
    payload: bytes
    sha256: str
    parent_sha256: str
    coordinates: tuple[int, int, int, int]
    algorithm_version: str


def crop_angle(image: Any, angle: str) -> Any:
    """Return a centered sub-region of `image` sized for the guided `angle`.

    `image` is a `PIL.Image`-like object exposing `.size` and `.crop()`
    (the same convention `dataset.dedup` uses for pHash/studio-filter
    input), so this stays free of an extra image-library dependency.
    """
    return image.crop(_crop_coordinates(image.size, angle))


def crop_derivative(
    image: Any,
    angle: str,
    *,
    parent_sha256: str,
    expected_watch_region: tuple[int, int, int, int],
    coordinates: tuple[int, int, int, int] | None = None,
) -> CropDerivative:
    """Return a traceable crop while preserving the caller's full image."""
    if not isinstance(parent_sha256, str) or _SHA256.fullmatch(parent_sha256) is None:
        raise ValueError("parent_sha256 must be lowercase SHA-256")
    full_width, full_height = image.size
    region = _validated_box(expected_watch_region, full_width, full_height, "watch region")
    crop_box = _validated_box(
        coordinates if coordinates is not None else _crop_coordinates(image.size, angle),
        full_width,
        full_height,
        "crop",
    )
    if not _contains(crop_box, region):
        raise ValueError("crop does not contain expected watch region")
    cropped = image.crop(crop_box).convert("RGB")
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    payload = buffer.getvalue()
    return CropDerivative(
        image=cropped,
        payload=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        parent_sha256=parent_sha256,
        coordinates=crop_box,
        algorithm_version=CROP_ALGORITHM_VERSION,
    )


def _crop_coordinates(size: tuple[int, int], angle: str) -> tuple[int, int, int, int]:
    width_fraction, height_fraction = ANGLE_CROP_FRACTIONS.get(angle, DEFAULT_CROP_FRACTION)
    full_width, full_height = size
    crop_width = round(full_width * width_fraction)
    crop_height = round(full_height * height_fraction)
    left = (full_width - crop_width) // 2
    top = (full_height - crop_height) // 2
    return left, top, left + crop_width, top + crop_height


def _validated_box(
    box: tuple[int, int, int, int], width: int, height: int, name: str
) -> tuple[int, int, int, int]:
    if (
        not isinstance(box, tuple)
        or len(box) != 4
        or any(not isinstance(value, int) for value in box)
    ):
        raise ValueError(f"{name} must contain four integer coordinates")
    left, top, right, bottom = box
    if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
        raise ValueError(f"{name} must be inside image bounds")
    return box


def _contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )
