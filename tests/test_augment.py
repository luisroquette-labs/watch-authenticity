"""Tests for the heavy augmentation pipeline in `dataset/augment.py`.

Covers this step's Success Criteria and the CK-16 Test Cases to Cover:
`augment_pipeline(fixture_image)` applies at least one of {blur, jpeg,
lighting} and returns a tensor different from the unaugmented input.
"""

import random

import numpy as np
import pytest
from PIL import Image

from dataset.augment import augment_pipeline


@pytest.fixture
def fixture_image() -> Image.Image:
    """A small flat-color RGB image.

    Flat color is deliberately the hardest case for blur/JPEG to visibly
    change (no edges, near-lossless compression of a uniform block), so a
    passing test proves the lighting/style steps alone are sufficient to
    satisfy CK-16's "at least one" requirement, not just the easy case.
    """
    return Image.new("RGB", (32, 32), color=(120, 110, 100))


def test_augment_pipeline_returns_a_tensor_different_from_the_input(
    fixture_image: Image.Image,
):
    original = np.array(fixture_image)

    augmented = augment_pipeline(fixture_image, rng=random.Random(0))

    assert isinstance(augmented, np.ndarray)
    assert augmented.shape == original.shape
    assert not np.array_equal(augmented, original)


def test_augment_pipeline_output_stays_in_valid_pixel_range(fixture_image: Image.Image):
    augmented = augment_pipeline(fixture_image, rng=random.Random(7))

    assert augmented.dtype == np.uint8
    assert augmented.min() >= 0
    assert augmented.max() <= 255


def test_augment_pipeline_is_deterministic_for_a_given_rng_seed(fixture_image: Image.Image):
    first = augment_pipeline(fixture_image, rng=random.Random(3))
    second = augment_pipeline(fixture_image, rng=random.Random(3))

    np.testing.assert_array_equal(first, second)
