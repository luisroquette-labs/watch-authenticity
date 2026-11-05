"""Tests for the framework-free core vocabulary in `core/types.py`.

Covers the step's Success Criteria: stdlib-only imports, required-field
presence on `ManifestRecord`, and field presence on `Prediction`/`Rejection`.
"""

import ast
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

from core.types import Grade, Label, ManifestRecord, ModelClass, Prediction, Rejection

# Frameworks the domain core must never import (dataclasses/enum/typing/
# datetime are stdlib and therefore always allowed).
FORBIDDEN_IMPORT_ROOTS = {"torch", "cv2", "opencv", "numpy", "boto3"}


def _top_level_imports(source: str) -> set[str]:
    """Return every top-level module name imported by `source`.

    Walking the AST (rather than importing the module and inspecting
    `sys.modules`) lets this test fail even if a forbidden package happens
    to be installed in the test environment — it checks what the code
    *asks for*, not what is incidentally available.
    """
    tree = ast.parse(source)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_types_imports_stdlib_only():
    source = Path("core/types.py").read_text()
    imported_roots = _top_level_imports(source)

    leaked_frameworks = imported_roots & FORBIDDEN_IMPORT_ROOTS

    assert not leaked_frameworks, (
        f"core/types.py must stay framework-free, but imports: {leaked_frameworks}"
    )


def test_grade_enum_has_the_four_market_grades():
    assert {grade.value for grade in Grade} == {1, 2, 3, 4}
    assert Grade.SUPERCLONE == 4


def test_label_enum_has_genuine_fake_rejection_excluded():
    assert {label.name for label in Label} == {"GENUINE", "FAKE", "REJECTION", "EXCLUDED"}


def test_model_class_is_the_canonical_six_class_target_vocabulary():
    assert list(ModelClass) == [
        ModelClass.GENUINE,
        ModelClass.GRADE_1,
        ModelClass.GRADE_2,
        ModelClass.GRADE_3,
        ModelClass.GRADE_4,
        ModelClass.REJECTION,
    ]


def test_manifest_record_genuine_photo_has_null_grade():
    record = ManifestRecord(
        path="img001.jpg",
        source="chrono24",
        url="https://chrono24.example/listing/1",
        label=Label.GENUINE,
        grade=None,
        phash="a1b2c3d4e5f6",
        scraped_at=date(2026, 11, 3),
    )

    assert record.source == "chrono24"
    assert record.url == "https://chrono24.example/listing/1"
    assert record.label is Label.GENUINE
    assert record.grade is None
    assert record.phash == "a1b2c3d4e5f6"
    assert record.scraped_at == date(2026, 11, 3)
    assert record.blur_score is None


def test_manifest_record_fake_photo_carries_a_grade():
    record = ManifestRecord(
        path="img002.jpg",
        source="reptime_forum",
        url="https://reptime.example/thread/1234",
        label=Label.FAKE,
        grade=3,
        phash="deadbeef0000",
        scraped_at=date(2026, 11, 3),
        blur_score=0.12,
    )

    assert record.grade == 3
    assert record.blur_score == 0.12


def test_manifest_record_missing_required_field_raises():
    constructor = cast(Any, ManifestRecord)
    with pytest.raises(TypeError):
        constructor(
            path="img003.jpg",
            label=Label.GENUINE,
            grade=None,
            phash="abc123",
            scraped_at=date(2026, 11, 3),
        )


def test_prediction_carries_confidence_grade_ceiling_and_heatmap():
    prediction = Prediction(
        confidence=0.94,
        genuine=False,
        grade=2,
        ceiling_applied=False,
        heatmap=[[0.0, 0.1], [0.2, 0.3]],
    )

    assert prediction.confidence == 0.94
    assert prediction.genuine is False
    assert prediction.grade == 2
    assert prediction.ceiling_applied is False
    assert prediction.heatmap == [[0.0, 0.1], [0.2, 0.3]]


def test_rejection_carries_reason_and_instruction():
    rejection = Rejection(reason="blurry", instruction="recapture needed")

    assert rejection.reason == "blurry"
    assert rejection.instruction == "recapture needed"
