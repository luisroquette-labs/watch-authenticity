"""Chrono24 verified-dealer scraper — routine GENUINE source photos.

Chrono24's verified-dealer listings are the brainstorm's baseline genuine
source (Requirements item 5): "Chrono24 dealers verificados = GENUÍNO".
This adapter still routes every photo's label through
`core.source_label_rule` rather than hardcoding GENUINE, so a future rule
change (e.g. an unverified-dealer carve-out) only needs to change
`core/rules.py`, never this file.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label, ManifestRecord
from labeling.manifest import append

CHRONO24_SEARCH_URL = "https://www.chrono24.com/rolex/submariner--mod.htm"


def fetch_chrono24_listings(search_url: str = CHRONO24_SEARCH_URL) -> list[dict]:
    """Fetches verified-dealer listing photos."""
    response = requests.get(search_url, timeout=10)
    response.raise_for_status()
    return response.json()


def chrono24_ingest(listings: list[dict], *, manifest_dir: Path, version: int) -> list[Path]:
    """Labels and writes Chrono24 verified-dealer listing photos."""
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(source="chrono24")
        if label == Label.EXCLUDED:
            continue  # unrecognized-source fallback — never fabricate a label

        record = ManifestRecord(
            path=listing["path"],
            source="chrono24",
            url=listing["url"],
            label=label,
            grade=None,  # genuine photos carry no fake grade
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
