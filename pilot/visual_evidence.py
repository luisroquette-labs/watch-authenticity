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
    """Read color outside a visible dial boundary, including a partly cropped bezel."""
    if min(face.size) < 160:
        return "uncertain"
    face = face.convert("RGB")
    face.thumbnail((512, 512))
    rgb = np.asarray(face)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    size = min(height, width)
    circles = cv2.HoughCircles(
        cv2.medianBlur(gray, 5), cv2.HOUGH_GRADIENT, 1.2, size * 0.15,
        param1=100, param2=45, minRadius=int(size * 0.23), maxRadius=int(size * 0.6),
    )
    if circles is None:
        return "uncertain"
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    y, x = np.mgrid[:height, :width]
    hue, saturation, lightness = cv2.split(hsv)
    masks = {
        "green": (hue >= 35) & (hue < 90) & (saturation > 45) & (lightness > 30),
        "blue": (hue >= 90) & (hue < 140) & (saturation > 45) & (lightness > 30),
        "black": (saturation < 45) & (lightness < 105),
    }
    for cx, cy, dial_radius in circles[0]:
        if not (0.28 * size < dial_radius < 0.46 * size):
            continue
        if np.hypot(cx - width / 2, cy - height / 2) > 0.1 * size:
            continue
        dx, dy = x - cx, y - cy
        radius = np.hypot(dx, dy) / dial_radius
        sector = ((np.arctan2(dy, dx) + np.pi) / (np.pi / 4)).astype(int) % 8
        ring = (radius > 1.08) & (radius < 1.22)
        exterior = (radius > 1.45) & (radius < 1.65)
        coverage = {
            color: sum(
                1 for index in range(8)
                if np.count_nonzero(ring & (sector == index)) >= 50
                and np.mean(mask[ring & (sector == index)]) > 0.3
                and np.count_nonzero(exterior & (sector == index)) >= 50
                and np.mean(mask[exterior & (sector == index)]) < 0.2
            )
            for color, mask in masks.items()
        }
        ranked = sorted(coverage, key=coverage.get, reverse=True)  # type: ignore[arg-type]
        if coverage[ranked[0]] >= 6 and coverage[ranked[0]] - coverage[ranked[1]] >= 2:
            return ranked[0]
    return "uncertain"


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
