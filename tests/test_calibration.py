"""Tests for temperature-scaling calibration (CK-15) + ceiling/clip wrap (CK-3, CK-1, CK-22).

Deviation note (same class of environment limitation as `test_model_build.
py`): every test here is gated behind `pytest.importorskip` because this
sandbox's Python (3.14) has no `torch==2.5.1` wheel. The tests are written
to need neither the network nor a real DINOv2 download, so wherever the
pinned `requirements.txt` actually installs, they run for real.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip(
        "torch", reason="torch==2.5.1 has no wheel for this sandbox's Python 3.14"
    )
    pytest.importorskip(
        "transformers", reason="transformers==4.46.0 requires the unavailable torch==2.5.1"
    )

from calibration.temperature_scaling import (  # noqa: E402
    apply_grade4_ceiling as calibration_apply_grade4_ceiling,
)
from calibration.temperature_scaling import (  # noqa: E402
    calibrate,
    fit_temperature,
    fit_temperature_on_validation_split,
    temperature_scale,
)
from calibration.temperature_scaling import (  # noqa: E402
    clip_confidence as calibration_clip_confidence,
)
from core.rules import CONFIDENCE_CEILING, CONFIDENCE_FLOOR, GRADE4_GENUINE_CEILING
from core.rules import apply_grade4_ceiling as core_apply_grade4_ceiling  # noqa: E402
from core.rules import clip_confidence as core_clip_confidence  # noqa: E402
from core.types import Grade, Label, ManifestRecord  # noqa: E402
from dataset.splits import TEST, VAL, build_splits  # noqa: E402
from model.heads import GENUINE_CLASS_INDEX, NUM_CLASSES  # noqa: E402

# A dedicated marketplace source avoids the cross-source split's
# CASUAL_STYLE_SOURCES/STUDIO_STYLE_SOURCES pairing, so every fixture row
# below lands in train/val/test, never cross_source/factory_holdout/gold.
_FIXTURE_SOURCE = "chrono24"


def _manifest_record(path: str) -> ManifestRecord:
    """A minimal, well-formed manifest row — only `path` varies per call."""
    return ManifestRecord(
        path=path,
        source=_FIXTURE_SOURCE,
        url=f"https://chrono24.example/listing/{path}",
        label=Label.GENUINE,
        grade=None,
        phash="deadbeefcafefeed",
        scraped_at=date(2026, 11, 3),
    )


def _logits_favoring_genuine(num_classes: int = NUM_CLASSES) -> "torch.Tensor":
    """A logit vector whose softmax strongly favors the genuine class."""
    values = torch.full((num_classes,), -5.0)
    values[GENUINE_CLASS_INDEX] = 10.0
    return values


# --- CK-15: temperature_scale -----------------------------------------------


def test_temperature_scale_is_flatter_at_higher_temperature():
    logits = torch.tensor([2.0, 0.5, -1.0, 0.0, 1.0, -0.5])
    baseline = temperature_scale(logits, 1.0)
    flattened = temperature_scale(logits, 2.0)
    assert flattened.max().item() < baseline.max().item()


@given(
    raw_logits=st.lists(
        st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=8,
    ),
    temperature=st.floats(min_value=1.0001, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_temperature_scale_never_exceeds_the_t_one_max(
    raw_logits: list[float], temperature: float
):
    logits = torch.tensor(raw_logits)
    baseline_max = temperature_scale(logits, 1.0).max().item()
    scaled_max = temperature_scale(logits, temperature).max().item()
    # Floating-point LBFGS/softmax arithmetic, not a strict rule violation,
    # is why this needs a tiny epsilon rather than a bare `<=`.
    assert scaled_max <= baseline_max + 1e-6


# --- CK-15: fit_temperature --------------------------------------------


def test_fit_temperature_returns_a_positive_scalar():
    # Mixed-confidence logits across a few "samples" — fit_temperature
    # must converge to some positive scalar, not merely not-crash.
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 2.0],
            [3.5, 0.5, 0.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 0.0


# --- CK-15 integration: validation-only fit -----------------------------


def test_fit_temperature_on_validation_split_never_reads_the_test_split():
    """Proves the split boundary by construction, not by inspection.

    `logits_by_path`/`labels_by_path` below contain entries ONLY for the
    validation split's paths. If `fit_temperature_on_validation_split`
    ever indexed `splits[TEST]` instead of (or in addition to)
    `splits[VAL]`, it would raise `KeyError` here — there is no TEST-path
    entry for it to find.
    """
    records = [_manifest_record(f"photo_{i:02d}") for i in range(10)]
    splits = build_splits(records, val_fraction=0.3, test_fraction=0.3, seed=42)

    assert len(splits[VAL]) == 3
    assert len(splits[TEST]) == 3
    val_paths = {record.path for record in splits[VAL]}
    test_paths = {record.path for record in splits[TEST]}
    assert val_paths.isdisjoint(test_paths)

    logits_by_path = {path: _logits_favoring_genuine() for path in val_paths}
    labels_by_path = {path: GENUINE_CLASS_INDEX for path in val_paths}

    temperature = fit_temperature_on_validation_split(splits, logits_by_path, labels_by_path)

    assert temperature > 0.0
    # Confirms the dicts handed in truly carry nothing a TEST-path lookup
    # could have silently succeeded against.
    assert test_paths.isdisjoint(logits_by_path.keys())


# --- CK-3/CK-1: calibrate wraps ceiling then clip -----------------------


def test_calibrate_caps_grade4_genuine_confidence_at_the_ceiling():
    logits = _logits_favoring_genuine()
    result = calibrate(logits, grade=int(Grade.SUPERCLONE), temperature=1.0)
    assert result <= GRADE4_GENUINE_CEILING


def test_calibrate_leaves_non_grade4_confidence_uncapped():
    logits = _logits_favoring_genuine()
    result = calibrate(logits, grade=int(Grade.STREET_FAKE), temperature=1.0)
    # Same dominant genuine logit, but grade 1 is not grade-4 — the
    # ceiling must not apply, so the near-certain softmax survives,
    # only clipped away from the exact 0.999 boundary.
    assert result > GRADE4_GENUINE_CEILING


@given(
    raw_logits=st.lists(
        st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        min_size=NUM_CLASSES,
        max_size=NUM_CLASSES,
    ),
    grade=st.sampled_from([1, 2, 3, 4]),
    temperature=st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
)
def test_calibrate_always_stays_within_confidence_bounds(
    raw_logits: list[float], grade: int, temperature: float
):
    logits = torch.tensor(raw_logits)
    result = calibrate(logits, grade=grade, temperature=temperature)
    assert CONFIDENCE_FLOOR <= result <= CONFIDENCE_CEILING


def test_calibrate_reuses_the_core_ceiling_and_clip_functions():
    # CK-22: this module must call the SAME function objects `core/rules.
    # py` defines, not local re-implementations that happen to agree.
    assert calibration_apply_grade4_ceiling is core_apply_grade4_ceiling
    assert calibration_clip_confidence is core_clip_confidence
