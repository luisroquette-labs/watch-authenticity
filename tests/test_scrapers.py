"""Tests for the seven per-source scrapers in `scrapers/` (Step 05a).

Covers this step's Success Criteria: every scraper derives its label via
`core.source_label_rule` (never inline), Mercado Livre's official-API-only
gate (CK-28), Reddit's permission gate (CK-27/29), and the OLX/ML
"original"-listing exclusion (CK-13/18). No test makes a real network
call — every `*_ingest` is exercised with pre-fetched `listings` fixtures.
"""

from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from core.types import Label
from labeling.manifest import read_manifest
from scrapers.chinese_replica import chinese_replica_ingest
from scrapers.chrono24 import chrono24_ingest
from scrapers.mercadolivre import mercadolivre_ingest
from scrapers.olx import olx_ingest
from scrapers.reddit import reddit_ingest
from scrapers.reptime import reptime_ingest
from scrapers.watchuseek import watchuseek_ingest


def _listing(**overrides) -> dict:
    """A minimal valid fetched-photo record shared across scraper tests."""
    base = {
        "path": "photo001.jpg",
        "url": "https://example.test/listing/1",
        "phash": "a1b2c3d4",
        "scraped_at": date(2026, 11, 3),
    }
    base.update(overrides)
    return base


# --- CK-13/18: OLX/ML "original" listings are excluded, never GENUINE -----


def test_olx_original_listing_is_excluded_not_genuine(tmp_path: Path):
    listing = _listing(listing_text="Rolex Submariner original, caixa e manual", price=30.0)

    written = olx_ingest([listing], manifest_dir=tmp_path, version=1)

    assert written == []
    assert read_manifest(tmp_path, 1) == []


def test_olx_replica_listing_is_written_as_fake_not_genuine(tmp_path: Path):
    listing = _listing(listing_text="Rolex Submariner réplica automática", price=30.0)

    olx_ingest([listing], manifest_dir=tmp_path, version=1)

    [record] = read_manifest(tmp_path, 1)
    assert record.label == Label.FAKE


def test_mercadolivre_original_listing_is_excluded_not_genuine(tmp_path: Path):
    listing = _listing(listing_text="Rolex Submariner original com garantia", price=40.0)

    written = mercadolivre_ingest("official_api", [listing], manifest_dir=tmp_path, version=1)

    assert written == []
    assert read_manifest(tmp_path, 1) == []


# --- CK-28: Mercado Livre sourcing is official-API-only, never raw scrape --


def test_mercadolivre_scrape_method_refuses_to_ingest():
    with pytest.raises(PermissionError):
        mercadolivre_ingest("scrape")


def test_mercadolivre_scrape_method_never_touches_the_manifest(tmp_path: Path):
    with pytest.raises(PermissionError):
        mercadolivre_ingest("scrape", [_listing()], manifest_dir=tmp_path, version=1)

    assert read_manifest(tmp_path, 1) == []


def test_mercadolivre_official_api_method_proceeds_to_ingest(tmp_path: Path):
    listing = _listing(listing_text="réplica automática", price=35.0)

    written = mercadolivre_ingest("official_api", [listing], manifest_dir=tmp_path, version=1)

    assert len(written) == 1
    [record] = read_manifest(tmp_path, 1)
    assert record.label == Label.FAKE


# --- CK-27/29: Reddit is EXCLUDED without a documented permission record --


def test_reddit_without_permission_is_excluded_by_default(tmp_path: Path):
    written = reddit_ingest([_listing()], manifest_dir=tmp_path, version=1)

    assert written == []
    assert read_manifest(tmp_path, 1) == []


def test_reddit_default_argument_is_no_permission(tmp_path: Path):
    # Calling without `permission_on_file` at all proves the pitfall-safe
    # default (False) is what actually ships, not just what a kwarg test
    # happens to pass (CK-29).
    written = reddit_ingest([_listing()], manifest_dir=tmp_path, version=1)

    assert written == []


def test_reddit_with_documented_permission_is_written_as_genuine(tmp_path: Path):
    written = reddit_ingest(
        [_listing()], manifest_dir=tmp_path, version=1, permission_on_file=True
    )

    assert len(written) == 1
    [record] = read_manifest(tmp_path, 1)
    assert record.label == Label.GENUINE


# --- Every scraper derives its label via core.source_label_rule -----------
# (no scraper may hardcode GENUINE/FAKE/EXCLUDED itself)


def test_olx_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.olx.source_label_rule", return_value=Label.FAKE) as mock_rule:
        olx_ingest([_listing(price=10.0)], manifest_dir=tmp_path, version=1)
    mock_rule.assert_called_once()


def test_mercadolivre_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.mercadolivre.source_label_rule", return_value=Label.FAKE) as mock_rule:
        mercadolivre_ingest(
            "official_api", [_listing(price=10.0)], manifest_dir=tmp_path, version=1
        )
    mock_rule.assert_called_once()


def test_chrono24_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.chrono24.source_label_rule", return_value=Label.GENUINE) as mock_rule:
        chrono24_ingest([_listing()], manifest_dir=tmp_path, version=1)
    mock_rule.assert_called_once()


def test_watchuseek_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.watchuseek.source_label_rule", return_value=Label.GENUINE) as mock_rule:
        watchuseek_ingest([_listing()], manifest_dir=tmp_path, version=1)
    mock_rule.assert_called_once()


def test_reptime_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.reptime.source_label_rule", return_value=Label.FAKE) as mock_rule:
        reptime_ingest([_listing(factory="clean")], manifest_dir=tmp_path, version=1)
    mock_rule.assert_called_once()


def test_chinese_replica_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch(
        "scrapers.chinese_replica.source_label_rule", return_value=Label.FAKE
    ) as mock_rule:
        chinese_replica_ingest([_listing(price=10.0)], manifest_dir=tmp_path, version=1)
    mock_rule.assert_called_once()


def test_reddit_delegates_label_derivation_to_source_label_rule(tmp_path: Path):
    with patch("scrapers.reddit.source_label_rule", return_value=Label.EXCLUDED) as mock_rule:
        reddit_ingest([_listing()], manifest_dir=tmp_path, version=1, permission_on_file=True)

    mock_rule.assert_called_once_with(
        source="reddit", subreddit="r/rolex", permission_on_file=True
    )
