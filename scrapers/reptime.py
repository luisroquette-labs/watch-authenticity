"""RepTime replica-forum scraper — named-factory FAKE source (grade 3-4).

RepTime threads name a specific replica factory (brainstorm Requirements
item 5: "Fóruns de réplica (RepTime etc.) com factory nomeada = FAKE grau
3-4"); the factory name flows into `core.grade_map` via `build_fake_row` so
grade assignment stays centralized (CK-22), never guessed here.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label
from labeling.manifest import append, build_fake_row

REPTIME_FORUM_URL = "https://www.reptime.com/forums/rolex-submariner-replica.123/"


def fetch_reptime_threads(forum_url: str = REPTIME_FORUM_URL) -> list[dict]:
    """Fetches replica-forum thread photos naming a factory."""
    response = requests.get(forum_url, timeout=10)
    response.raise_for_status()
    return response.json()


def reptime_ingest(listings: list[dict], *, manifest_dir: Path, version: int) -> list[Path]:
    """Labels and writes RepTime thread photos (named-factory FAKE grading)."""
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(source="reptime")
        if label == Label.EXCLUDED:
            continue  # unrecognized-source fallback — never fabricate a label

        record = build_fake_row(
            path=listing["path"],
            source="reptime",
            url=listing["url"],
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            factory=listing.get("factory"),
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
