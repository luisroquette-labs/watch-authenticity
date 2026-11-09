"""Tests for `dataset/crop.py`.

Covers this step's Success Criteria and the CK-21 Test Cases to Cover:
`crop_angle` returns a region smaller than the full frame, centered on
the frame, for a guided angle.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from dataset.crop import (
    ANGLE_CROP_FRACTIONS,
    CROP_ALGORITHM_VERSION,
    DEFAULT_CROP_FRACTION,
    crop_angle,
    crop_derivative,
)


def _full_frame_fixture(size: int = 128) -> Image.Image:
    return Image.new("RGB", (size, size), (120, 100, 80))


def _coordinate_encoded_frame(width: int = 160, height: int = 100) -> Image.Image:
    """A frame whose pixel value at (x, y) encodes its own coordinates.

    Lets a test verify *where* `crop_angle` cut the frame from, rather than
    only the resulting size — content at local (0, 0) of the crop must
    equal the source frame's pixel at the expected global (left, top).
    """
    xs, ys = np.meshgrid(np.arange(width) % 256, np.arange(height) % 256)
    pixels = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_crop_angle_rehaut_returns_smaller_centered_region():
    full_frame = _full_frame_fixture()

    cropped = crop_angle(full_frame, angle="rehaut")

    assert cropped.size[0] < full_frame.size[0]
    assert cropped.size[1] < full_frame.size[1]

    width_fraction, height_fraction = ANGLE_CROP_FRACTIONS["rehaut"]
    assert cropped.size == (
        round(full_frame.size[0] * width_fraction),
        round(full_frame.size[1] * height_fraction),
    )


def test_crop_angle_is_centered_on_the_full_frame():
    # "clasp" has unequal width/height fractions (0.6, 0.5): if crop_angle
    # ever swapped left/top or width/height, this fixture's per-pixel
    # coordinate encoding would catch it where a square crop could not.
    full_frame = _coordinate_encoded_frame(width=160, height=100)
    width_fraction, height_fraction = ANGLE_CROP_FRACTIONS["clasp"]
    expected_width = round(full_frame.size[0] * width_fraction)
    expected_height = round(full_frame.size[1] * height_fraction)
    expected_left = (full_frame.size[0] - expected_width) // 2
    expected_top = (full_frame.size[1] - expected_height) // 2

    cropped = crop_angle(full_frame, angle="clasp")

    top_left_pixel = np.asarray(cropped)[0, 0]
    assert tuple(top_left_pixel) == tuple(np.asarray(full_frame)[expected_top, expected_left])


def test_crop_angle_covers_every_documented_guided_angle():
    full_frame = _full_frame_fixture()

    for angle in ("rehaut", "cyclops", "crown", "clasp", "dial", "caseback"):
        cropped = crop_angle(full_frame, angle=angle)
        assert cropped.size[0] <= full_frame.size[0]
        assert cropped.size[1] <= full_frame.size[1]
        assert cropped.size[0] < full_frame.size[0] or cropped.size[1] < full_frame.size[1]


def test_crop_angle_falls_back_to_default_fraction_for_unmapped_angle():
    full_frame = _full_frame_fixture()

    cropped = crop_angle(full_frame, angle="some-future-angle")

    width_fraction, height_fraction = DEFAULT_CROP_FRACTION
    assert cropped.size == (
        round(full_frame.size[0] * width_fraction),
        round(full_frame.size[1] * height_fraction),
    )


def test_crop_derivative_records_parent_coordinates_and_algorithm_version():
    full_frame = _coordinate_encoded_frame()

    derivative = crop_derivative(
        full_frame,
        angle="clasp",
        parent_sha256="a" * 64,
        expected_watch_region=(50, 25, 110, 75),
    )

    assert derivative.parent_sha256 == "a" * 64
    assert derivative.algorithm_version == CROP_ALGORITHM_VERSION
    assert derivative.coordinates == (32, 25, 128, 75)
    assert derivative.image.size == (96, 50)
    assert derivative.sha256 == hashlib.sha256(derivative.payload).hexdigest()


@pytest.mark.parametrize(
    "region",
    [(-1, 0, 10, 10), (0, 0, 161, 10), (20, 20, 20, 30), (20, 20, 30, 20)],
)
def test_crop_derivative_rejects_watch_region_outside_image_bounds(region):
    with pytest.raises(ValueError, match="watch region"):
        crop_derivative(
            _coordinate_encoded_frame(),
            angle="clasp",
            parent_sha256="a" * 64,
            expected_watch_region=region,
        )


def test_crop_derivative_rejects_crop_missing_expected_watch_region():
    with pytest.raises(ValueError, match="does not contain expected watch region"):
        crop_derivative(
            _coordinate_encoded_frame(),
            angle="cyclops",
            parent_sha256="a" * 64,
            expected_watch_region=(0, 0, 10, 10),
        )


def test_crop_derivative_rejects_empty_explicit_coordinates():
    with pytest.raises(ValueError, match="crop must contain four integer coordinates"):
        crop_derivative(
            _coordinate_encoded_frame(),
            angle="clasp",
            parent_sha256="a" * 64,
            expected_watch_region=(50, 25, 110, 75),
            coordinates=(),
        )
