"""Watchuseek forum scraper — owner-forum GENUINE source (wrist photos).

Brainstorm Requirements item 5: "Fóruns de donos (Watchuseek, r/rolex) =
GENUÍNO (foto de pulso, usuário real)" — Watchuseek carries no Reddit-style
permission gate (unlike r/rolex, see `scrapers/reddit.py`), so it routes
straight through `core.source_label_rule` as a routine forum source.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label, ManifestRecord
from labeling.manifest import append

WATCHUSEEK_FORUM_URL = "https://www.watchuseek.com/forums/rolex-submariner-5.html"


def fetch_watchuseek_threads(forum_url: str = WATCHUSEEK_FORUM_URL) -> list[dict]:
    """Fetches owner-forum thread photos."""
    response = requests.get(forum_url, timeout=10)
    response.raise_for_status()
    return response.json()


def watchuseek_ingest(listings: list[dict], *, manifest_dir: Path, version: int) -> list[Path]:
    """Labels and writes Watchuseek owner-forum thread photos."""
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(source="watchuseek")
        if label == Label.EXCLUDED:
            continue  # unrecognized-source fallback — never fabricate a label

        record = ManifestRecord(
            path=listing["path"],
            source="watchuseek",
            url=listing["url"],
            label=label,
            grade=None,  # genuine photos carry no fake grade
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
