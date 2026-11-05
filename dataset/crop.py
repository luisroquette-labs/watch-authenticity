"""Per-angle crop: isolate the guided-capture subject from its backdrop.

Dataset op (Expected Changes: `dataset/crop.py`). A first-line defense
against the source-bias confound (Requirements item 13): classification
heads must see the watch part a guided angle targets (rehaut, cyclops,
crown, clasp, dial, caseback), not whatever backdrop surrounds it in the
raw scraped/captured photo (CK-21).
"""

from __future__ import annotations

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


def crop_angle(image: Any, angle: str) -> Any:
    """Return a centered sub-region of `image` sized for the guided `angle`.

    `image` is a `PIL.Image`-like object exposing `.size` and `.crop()`
    (the same convention `dataset.dedup` uses for pHash/studio-filter
    input), so this stays free of an extra image-library dependency.
    """
    width_fraction, height_fraction = ANGLE_CROP_FRACTIONS.get(angle, DEFAULT_CROP_FRACTION)
    full_width, full_height = image.size
    crop_width = round(full_width * width_fraction)
    crop_height = round(full_height * height_fraction)
    left = (full_width - crop_width) // 2
    top = (full_height - crop_height) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))
