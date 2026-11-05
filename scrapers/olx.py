"""OLX marketplace scraper (FAKE-labeled replica listings only; CK-13).

OLX sellers plainly offer réplica/inspirado watches at a declared price,
and some lie and call a replica "original" — the latter must never enter
training as GENUINE (CK-13/18). This adapter fetches listing photos and
routes every listing's label through `core.source_label_rule`; it never
derives GENUINE/FAKE itself.
"""

from __future__ import annotations

from pathlib import Path

import requests

from core.rules import source_label_rule
from core.types import Label
from labeling.manifest import append, build_fake_row

OLX_SEARCH_URL = "https://www.olx.com.br/brasil?q=rolex+submariner"


def fetch_olx_listings(search_url: str = OLX_SEARCH_URL) -> list[dict]:
    """Fetches raw OLX listing photos for the search query.

    Real HTML-scraping mechanics are adapter-specific and out of this
    step's test scope (the Blockers & Risks note: "per-source ToS/anti-bot
    variance... mechanics isolated per adapter"); tests call
    `olx_ingest(listings=...)` directly and never exercise this network path.
    """
    response = requests.get(search_url, timeout=10)
    response.raise_for_status()
    return response.json()  # adapter-specific parsing of the OLX response


def olx_ingest(listings: list[dict], *, manifest_dir: Path, version: int) -> list[Path]:
    """Labels and writes OLX listing photos; "original" ads are excluded (CK-13).

    Every listing's label comes from `core.source_label_rule` — never a
    literal GENUINE/FAKE assigned here — so an "original" claim is excluded
    by the shared rule, not by adapter-local logic that could drift.
    """
    written: list[Path] = []
    for listing in listings:
        label = source_label_rule(
            source="olx",
            marketplace="olx",
            listing_text=listing.get("listing_text"),
        )
        if label == Label.EXCLUDED:
            continue  # seller claims "original" (or unrecognized) — never enters the dataset

        record = build_fake_row(
            path=listing["path"],
            source="olx",
            url=listing["url"],
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            price=listing.get("price"),
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
