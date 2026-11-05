"""Reddit (r/rolex) scraper — default-EXCLUDED without permission (CK-27/29).

Reddit's Data API Terms explicitly ban using User Content to train ML
models without Reddit's written permission (SKILL.md: Data API Terms,
rev. 2026-07-20); Reddit has litigated this exact pattern. Every call here
passes `source="reddit"` through `core.source_label_rule`, so the default
EXCLUDED behavior lives in exactly one place — this adapter never
second-guesses it.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label, ManifestRecord
from labeling.manifest import append

REDDIT_SUBREDDIT = "r/rolex"
REDDIT_JSON_URL = f"https://www.reddit.com/{REDDIT_SUBREDDIT}/top.json"


def fetch_reddit_posts(json_url: str = REDDIT_JSON_URL) -> list[dict]:
    """Fetches r/rolex post photos."""
    response = requests.get(
        json_url, timeout=10, headers={"User-Agent": "watch-authenticity/0.1"}
    )
    response.raise_for_status()
    return response.json().get("data", {}).get("children", [])


def reddit_ingest(
    listings: list[dict],
    *,
    manifest_dir: Path,
    version: int,
    permission_on_file: bool = False,
) -> list[Path]:
    """Labels and writes r/rolex post photos; EXCLUDED without permission on file.

    `permission_on_file` defaults to `False` — a documented Reddit-
    permission and owner sign-off record is required before any caller may
    flip it (CK-27). This is the one knob the pitfall test CK-29 exists to
    make sure nobody bypasses by accident.
    """
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(
            source="reddit",
            subreddit=REDDIT_SUBREDDIT,
            permission_on_file=permission_on_file,
        )
        if label == Label.EXCLUDED:
            continue  # default posture — no documented permission record on file

        record = ManifestRecord(
            path=listing["path"],
            source="reddit",
            url=listing["url"],
            label=label,
            grade=None,  # genuine photos carry no fake grade
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
