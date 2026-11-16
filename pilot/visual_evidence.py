"""Conservative, visible configuration clues for the Submariner pilot."""

from __future__ import annotations

from collections import Counter

import cv2
import numpy as np
from PIL import Image

EXPERT_SOURCE = "https://www.bobswatches.com/rolex-blog/editorial/rolex-hulk-vs-starbucks.html"
INSPECTION_SOURCE = "https://www.bobswatches.com/rolex-blog/editorial/authentication-process.html"
OFFICIAL_126610LN = "https://www.rolex.com/en-us/watches/submariner/m126610ln-0001"

REFERENCE_PATTERNS = {
    ("green", "green"): ("116610LV", "Hulk", EXPERT_SOURCE, "fonte especializada"),
    ("black", "green"): ("126610LV", "Starbucks", EXPERT_SOURCE, "fonte especializada"),
    ("black", "black"): ("126610LN", "Submariner Date", OFFICIAL_126610LN, "fabricante"),
}


def bezel_color_hint(face: Image.Image) -> str:
    """Read the outer ring only when color spans most of a large dial crop."""
    if min(face.size) < 160:
        return "uncertain"
    hsv = cv2.cvtColor(np.asarray(face.convert("RGB")), cv2.COLOR_RGB2HSV)
    height, width = hsv.shape[:2]
    y, x = np.mgrid[:height, :width]
    dx = (x - width / 2) / (width / 2)
    dy = (y - height / 2) / (height / 2)
    radius = np.hypot(dx, dy)
    sector = ((np.arctan2(dy, dx) + np.pi) / (np.pi / 4)).astype(int) % 8
    ring = (radius > 0.72) & (radius < 0.91)
    hue, saturation, lightness = cv2.split(hsv)
    masks = {
        "green": (hue >= 35) & (hue < 90) & (saturation > 45) & (lightness > 30),
        "blue": (hue >= 90) & (hue < 140) & (saturation > 45) & (lightness > 30),
        "black": (saturation < 45) & (lightness < 105),
    }
    coverage = {
        color: sum(
            np.mean(mask[ring & (sector == index)]) > 0.15
            for index in range(8)
        )
        for color, mask in masks.items()
    }
    ranked = sorted(coverage, key=coverage.get, reverse=True)  # type: ignore[arg-type]
    # shortcut: reflections or a tight crop can mimic a bezel; require broad, separated coverage.
    if coverage[ranked[0]] < 6 or coverage[ranked[0]] - coverage[ranked[1]] < 2:
        return "uncertain"
    return ranked[0]


def consensus(values: list[str]) -> tuple[str, int]:
    known = Counter(value for value in values if value not in {"uncertain", "other"})
    if len(known) != 1:
        return "uncertain", 0
    value, count = known.most_common(1)[0]
    return value, count


def reference_candidate(dial: str, bezel: str) -> dict[str, str] | None:
    candidate = REFERENCE_PATTERNS.get((dial, bezel))
    if candidate is None:
        return None
    reference, name, source_url, source_type = candidate
    return {
        "reference": reference,
        "name": name,
        "source_url": source_url,
        "source_type": source_type,
        "certainty": "configuracao compativel; referencia nao confirmada",
    }
