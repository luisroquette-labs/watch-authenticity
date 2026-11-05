"""Chinese replica-site scraper — studio-photo FAKE source (filtered in 05b).

Brainstorm Requirements item 5: "Sites chineses de réplica = FAKE (filtrar
foto de estúdio/roubada de genuíno)" — the studio-photo filter itself is a
dataset-ops concern (`dataset/dedup.py`, step 05b), not this adapter's job;
this adapter only fetches, labels, and writes.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label
from labeling.manifest import append, build_fake_row

CHINESE_REPLICA_SEARCH_URL = "https://www.example-replica-site.com/search?q=submariner"


def fetch_chinese_replica_listings(search_url: str = CHINESE_REPLICA_SEARCH_URL) -> list[dict]:
    """Fetches replica-site listing photos."""
    response = requests.get(search_url, timeout=10)
    response.raise_for_status()
    return response.json()


def chinese_replica_ingest(
    listings: list[dict], *, manifest_dir: Path, version: int
) -> list[Path]:
    """Labels and writes Chinese replica-site listing photos."""
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(source="chinese_replica")
        if label == Label.EXCLUDED:
            continue  # unrecognized-source fallback — never fabricate a label

        record = build_fake_row(
            path=listing["path"],
            source="chinese_replica",
            url=listing["url"],
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            factory=listing.get("factory"),
            price=listing.get("price"),
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
