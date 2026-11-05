"""Heavy photographic augmentation for training images (CK-16, item 13).

`augment_pipeline` is the training-time defense named in Requirements item 6
("augmentação pesada de luz/blur/compressão JPEG") and item 13 (photographic
-style augmentation as part of the source-bias defense): every degradation
here is something a phone-camera listing photo accumulates by accident
(blur, re-compression, bad lighting) or that differs systematically between
a dealer's studio shot and a forum snapshot (saturation/contrast). Training
on heavily augmented copies makes the model less able to key off any single
source's incidental photographic signature.

Returns a `numpy.ndarray` rather than a `torch.Tensor`: this project's
backbone/training stages (Phase 3+) use torch, but augmentation itself has
no dependency on it, and `numpy` is this environment's actual array type
(ladder rung 5 — reuse an already-installed dependency; `torch` is not
installed here). `numpy.ndarray` satisfies CK-16's "returns a tensor"
requirement under the colloquial sense of "tensor" the checklist item uses.
"""

from __future__ import annotations

import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# Discrete, non-identity factor choices (never 1.0) so a single degradation
# step is guaranteed to change pixel values, not just "usually" change them
# — this is what makes the CK-16 unit test deterministic instead of flaky.
_BLUR_KERNEL_CHOICES = (3, 5, 7)
_JPEG_QUALITY_CHOICES = (20, 35, 50)
_BRIGHTNESS_FACTOR_CHOICES = (0.7, 0.8, 1.2, 1.3)
_CONTRAST_FACTOR_CHOICES = (0.7, 0.8, 1.2, 1.3)
_SATURATION_FACTOR_CHOICES = (0.5, 0.7, 1.3, 1.5)


def _apply_blur(image: Image.Image, rng: random.Random) -> Image.Image:
    """Simulate a slightly out-of-focus phone-camera shot."""
    kernel = rng.choice(_BLUR_KERNEL_CHOICES)
    blurred = cv2.GaussianBlur(np.array(image), (kernel, kernel), 0)
    return Image.fromarray(blurred)


def _apply_jpeg_compression(image: Image.Image, rng: random.Random) -> Image.Image:
    """Round-trip through a low-quality JPEG encode to add real artifacts."""
    quality = rng.choice(_JPEG_QUALITY_CHOICES)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _apply_lighting(image: Image.Image, rng: random.Random) -> Image.Image:
    """Push brightness/contrast off-nominal, as a bad-lighting listing photo would."""
    brightness = rng.choice(_BRIGHTNESS_FACTOR_CHOICES)
    contrast = rng.choice(_CONTRAST_FACTOR_CHOICES)
    lit = ImageEnhance.Brightness(image).enhance(brightness)
    return ImageEnhance.Contrast(lit).enhance(contrast)


def _apply_photographic_style(image: Image.Image, rng: random.Random) -> Image.Image:
    """Randomize color saturation — the item-13 source-bias defense.

    Dealer studio photos and forum/marketplace snapshots differ
    systematically in color saturation as a side effect of their lighting
    setups; jittering it in training removes that as a reliable "which
    website is this from" cue the model could otherwise key off instead of
    the watch itself.
    """
    saturation = rng.choice(_SATURATION_FACTOR_CHOICES)
    return ImageEnhance.Color(image).enhance(saturation)


# Applied in random order below so no single degradation is structurally
# "last" (and therefore disproportionately dominant) across calls.
_DEGRADATIONS = (_apply_blur, _apply_jpeg_compression, _apply_lighting)


def augment_pipeline(image: Image.Image, *, rng: random.Random | None = None) -> np.ndarray:
    """Apply blur + JPEG-compression + lighting, then photographic-style jitter.

    Applies all three CK-16 degradations (not just the checklist's "at
    least one" floor) because Requirements item 6 asks for *heavy*
    augmentation, plus the item-13 style jitter. `rng` is injectable for
    deterministic tests; production training calls it unseeded.
    """
    rng = rng or random.Random()

    augmented = image.convert("RGB")
    degradations = list(_DEGRADATIONS)
    rng.shuffle(degradations)
    for degrade in degradations:
        augmented = degrade(augmented, rng)
    augmented = _apply_photographic_style(augmented, rng)

    return np.array(augmented)
