"""Tests for `dataset/dedup.py`.

Covers this step's Success Criteria and the CK-11 Test Cases to Cover:
`dedup_filter` drops a near-duplicate, `studio_photo_filter` flags a
catalog photo, and `hash_distance` is symmetric.
"""

import io

import numpy as np
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image, ImageDraw

from dataset.dedup import dedup_filter, hash_distance, studio_photo_filter


def _textured_photo_fixture(seed: int, size: int = 128) -> Image.Image:
    """Build a deterministic, photo-like (not flat-color) fixture image.

    pHash anchors on an image's low-frequency structure, so a flat-color or
    purely random-noise image is a poor stand-in for "a real photo" — it
    either hashes identically to everything or to nothing. Placing the
    shapes at seed-dependent positions gives different seeds genuinely
    different structure (so distinct photos hash far apart), while a JPEG
    recompression of the same seed keeps that same structure (so a
    near-duplicate still hashes close).
    """
    rng = np.random.default_rng(seed)
    base_color = tuple(int(value) for value in rng.integers(40, 220, size=3))
    image = Image.new("RGB", (size, size), base_color)
    draw = ImageDraw.Draw(image)
    circle_x, circle_y = (int(value) for value in rng.integers(30, size - 30, size=2))
    radius = int(rng.integers(15, 35))
    draw.ellipse((circle_x - radius, circle_y - radius, circle_x + radius, circle_y + radius), fill=(30, 30, 30))
    square_x, square_y = (int(value) for value in rng.integers(10, size - 10, size=2))
    draw.rectangle((square_x - 10, square_y - 10, square_x + 10, square_y + 10), fill=(220, 50, 50))
    return image


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _studio_catalog_fixture(size: int = 64, product_box: int = 20) -> Image.Image:
    """A near-perfect-white-background catalog photo with a dark center product."""
    canvas = np.full((size, size, 3), 250, dtype=np.uint8)
    center, half_box = size // 2, product_box // 2
    canvas[center - half_box : center + half_box, center - half_box : center + half_box] = (20, 20, 20)
    return Image.fromarray(canvas, mode="RGB")


def test_dedup_filter_drops_near_duplicate_photo():
    original = _textured_photo_fixture(seed=1)
    near_duplicate = _jpeg_recompress(original, quality=60)

    deduped = dedup_filter([original, near_duplicate])

    assert len(deduped) == 1
    assert deduped[0] is original


def test_dedup_filter_keeps_two_genuinely_different_photos():
    photo_a = _textured_photo_fixture(seed=1)
    photo_b = _textured_photo_fixture(seed=777)

    deduped = dedup_filter([photo_a, photo_b])

    assert deduped == [photo_a, photo_b]


def test_studio_photo_filter_flags_white_background_catalog_photo():
    assert studio_photo_filter(_studio_catalog_fixture()) is True


def test_studio_photo_filter_does_not_flag_a_textured_photo():
    assert studio_photo_filter(_textured_photo_fixture(seed=1)) is False


@given(st.integers(min_value=0, max_value=2**32 - 1), st.integers(min_value=0, max_value=2**32 - 1))
def test_hash_distance_is_symmetric(seed_a: int, seed_b: int):
    image_a = _textured_photo_fixture(seed_a)
    image_b = _textured_photo_fixture(seed_b)

    assert hash_distance(image_a, image_b) == hash_distance(image_b, image_a)
