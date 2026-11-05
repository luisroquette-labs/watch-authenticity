"""Tests for the versioned manifest I/O in `labeling/manifest.py`.

Covers this step's Success Criteria: complete non-empty fields on a written
row (CK-10), grade assignment routed through `core.grade_map` only (CK-22),
and versioning that never mutates a prior version's file in place.
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from core.rules import grade_map
from core.types import Label, ManifestRecord
from labeling.manifest import append, build_fake_row, read_manifest, write_row


@pytest.fixture
def fixture_photo() -> dict:
    """A raw scraped-photo signal, as a scraper would hand it to the manifest."""
    return {
        "path": "img001.jpg",
        "source": "reptime_forum",
        "url": "https://reptime.example/thread/1234",
        "phash": "a1b2c3d4e5f6",
        "scraped_at": date(2026, 11, 3),
        "factory": "vsf",
    }


# --- CK-10: a written row has complete, non-empty fields -------------------


def test_manifest_row_has_complete_fields_for_fixture_photo(fixture_photo: dict):
    row = build_fake_row(**fixture_photo)

    assert row.source == fixture_photo["source"]
    assert row.url == fixture_photo["url"]
    assert row.label is Label.FAKE
    assert row.grade  # non-empty/non-falsy: a real grade (1-4), not None or 0
    assert row.phash == fixture_photo["phash"]


def test_append_then_read_round_trips_every_field(tmp_path: Path, fixture_photo: dict):
    row = build_fake_row(**fixture_photo)

    append(row, manifest_dir=tmp_path, version=1)
    rows = read_manifest(tmp_path, version=1)

    assert rows == [row]


# --- Grade assignment routes through core.grade_map only (CK-22) ----------


def test_build_fake_row_grade_matches_core_grade_map_for_the_same_factory(
    fixture_photo: dict,
):
    row = build_fake_row(**fixture_photo)

    assert row.grade == grade_map(factory=fixture_photo["factory"])


def test_build_fake_row_delegates_grade_assignment_to_core_grade_map(
    fixture_photo: dict,
):
    # Patching `core.grade_map` (the single permitted call site, per CK-22)
    # and asserting it fires proves `build_fake_row` has no inline fallback
    # grade logic of its own.
    with patch("labeling.manifest.grade_map", return_value=3) as mock_grade_map:
        row = build_fake_row(**fixture_photo)

    mock_grade_map.assert_called_once_with(factory=fixture_photo["factory"], price=None)
    assert row.grade == 3


def test_write_row_is_the_same_implementation_as_append():
    assert write_row is append


# --- Versioning: a new version never mutates a prior version's file -------


def test_appending_a_new_version_does_not_change_a_prior_version(
    tmp_path: Path, fixture_photo: dict
):
    v1_row = build_fake_row(**fixture_photo)
    append(v1_row, manifest_dir=tmp_path, version=1)
    v1_path_before = read_manifest(tmp_path, version=1)

    v2_row = ManifestRecord(
        path="img002.jpg",
        source="chrono24",
        url="https://chrono24.example/listing/9",
        label=Label.GENUINE,
        grade=None,
        phash="deadbeef0000",
        scraped_at=date(2026, 11, 3),
    )
    append(v2_row, manifest_dir=tmp_path, version=2)

    assert read_manifest(tmp_path, version=1) == v1_path_before == [v1_row]
    assert read_manifest(tmp_path, version=2) == [v2_row]


def test_appending_twice_within_the_same_version_is_additive_not_overwriting(
    tmp_path: Path, fixture_photo: dict
):
    first_row = build_fake_row(**fixture_photo)
    second_row = build_fake_row(**{**fixture_photo, "path": "img002.jpg"})

    append(first_row, manifest_dir=tmp_path, version=1)
    append(second_row, manifest_dir=tmp_path, version=1)

    assert read_manifest(tmp_path, version=1) == [first_row, second_row]


def test_reading_a_version_never_written_returns_an_empty_list(tmp_path: Path):
    assert read_manifest(tmp_path, version=99) == []
