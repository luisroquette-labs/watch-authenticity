"""Tests for the pure safety rules in `core/rules.py`.

Covers the step's Success Criteria and the Test Cases to Cover entries for
CK-1, CK-3, CK-8, CK-12, CK-13/18, CK-19, CK-27/29. Reuses the stdlib-only
import check from `test_core_types.py` instead of redefining it (DRY).
"""

import inspect
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.rules import (
    GRADE4_GENUINE_CEILING,
    aggregate_angles,
    apply_grade4_ceiling,
    clip_confidence,
    goldset_regression_gate,
    grade_map,
    publish_gate,
    source_label_rule,
)
from core.types import Label
from tests.test_core_types import FORBIDDEN_IMPORT_ROOTS, _top_level_imports


def test_core_rules_imports_stdlib_only():
    source = Path("core/rules.py").read_text()
    leaked_frameworks = _top_level_imports(source) & FORBIDDEN_IMPORT_ROOTS

    assert not leaked_frameworks, f"core/rules.py must stay framework-free, got: {leaked_frameworks}"


# --- CK-1: clip_confidence -------------------------------------------------


def test_clip_confidence_floors_zero_to_point_zero_zero_one():
    assert clip_confidence(0.0) == 0.001


def test_clip_confidence_ceilings_one_to_point_nine_nine_nine():
    assert clip_confidence(1.0) == 0.999


def test_clip_confidence_leaves_typical_value_unchanged():
    assert clip_confidence(0.5) == 0.5


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_clip_confidence_always_stays_in_bounds(p: float):
    assert 0.001 <= clip_confidence(p) <= 0.999


# --- CK-3: apply_grade4_ceiling --------------------------------------------


def test_grade4_ceiling_caps_a_high_raw_score():
    assert apply_grade4_ceiling(0.95, grade=4) <= 0.60


def test_grade4_ceiling_leaves_a_low_raw_score_unchanged():
    assert apply_grade4_ceiling(0.40, grade=4) == 0.40


def test_grade4_ceiling_does_not_apply_to_other_grades():
    assert apply_grade4_ceiling(0.95, grade=2) == 0.95


@given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
def test_grade4_ceiling_always_bounds_grade_four(raw_genuine: float):
    assert apply_grade4_ceiling(raw_genuine, grade=4) <= GRADE4_GENUINE_CEILING


# --- grade_map ---------------------------------------------------------


def test_grade_map_named_superclone_factory_returns_grade_four():
    assert grade_map(factory="VSF") == 4


def test_grade_map_unnamed_factory_defaults_to_high_grade_rep():
    assert grade_map(factory="noob") == 3


def test_grade_map_low_price_returns_street_fake_grade():
    assert grade_map(price=20.0) == 1


def test_grade_map_higher_price_returns_common_replica_grade():
    assert grade_map(price=150.0) == 2


def test_grade_map_without_factory_or_price_raises():
    with pytest.raises(ValueError):
        grade_map()


# --- CK-13/18 & CK-27/29: source_label_rule ---------------------------------


def test_source_label_rule_excludes_olx_original_listing():
    assert source_label_rule(listing_text="original", marketplace="olx") == Label.EXCLUDED


def test_source_label_rule_excludes_reddit_without_permission():
    result = source_label_rule(source="reddit", subreddit="r/rolex", permission_on_file=False)
    assert result == Label.EXCLUDED


def test_source_label_rule_allows_reddit_with_documented_permission():
    result = source_label_rule(source="reddit", subreddit="r/rolex", permission_on_file=True)
    assert result == Label.GENUINE


def test_source_label_rule_treats_dealer_forums_as_genuine():
    assert source_label_rule(source="chrono24") == Label.GENUINE
    assert source_label_rule(source="watchuseek") == Label.GENUINE


def test_source_label_rule_treats_replica_forums_as_fake():
    assert source_label_rule(source="reptime") == Label.FAKE


def test_source_label_rule_treats_undeclared_marketplace_ad_as_fake():
    assert source_label_rule(listing_text="réplica", marketplace="olx") == Label.FAKE


def test_source_label_rule_excludes_unrecognized_source():
    assert source_label_rule(source="unknown_site") == Label.EXCLUDED


# --- CK-19: aggregate_angles ------------------------------------------------


def test_aggregate_angles_averages_per_angle_scores():
    result = aggregate_angles({"dial": 0.9, "crown": 0.7})
    assert result["raw_genuine"] == pytest.approx(0.8)


def test_aggregate_angles_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate_angles({})


def test_aggregate_angles_has_zero_trainable_parameters():
    # A plain function has no `.parameters()` the way a trained module
    # (e.g. torch.nn.Module) would — this is what "zero trainable
    # parameters" looks like for a non-learned rule (CK-19).
    assert inspect.isfunction(aggregate_angles)
    assert not hasattr(aggregate_angles, "parameters")


# --- CK-8: publish_gate ------------------------------------------------


def test_publish_gate_blocks_on_large_cross_source_degradation():
    assert publish_gate(standard_acc=0.94, cross_source_acc=0.60, max_delta=0.10) is False


def test_publish_gate_allows_small_cross_source_degradation():
    assert publish_gate(standard_acc=0.94, cross_source_acc=0.91, max_delta=0.10) is True


# --- CK-12: goldset_regression_gate ------------------------------------


def test_goldset_regression_gate_blocks_on_regression():
    assert goldset_regression_gate(current_acc=0.80, baseline_acc=0.85) is False


def test_goldset_regression_gate_allows_matching_or_improved_accuracy():
    assert goldset_regression_gate(current_acc=0.85, baseline_acc=0.85) is True
    assert goldset_regression_gate(current_acc=0.90, baseline_acc=0.85) is True
