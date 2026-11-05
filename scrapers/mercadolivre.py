"""Mercado Livre scraper — official Developer API only (CK-28).

Mercado Livre's Developer ToS explicitly prohibits scraping/robots outside
its official API (SKILL.md: developers.mercadolivre.com.br ToS). This
adapter has exactly one legal ingestion path — `method="official_api"`.
`method="scrape"` is refused before any HTTP call is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import requests

from core.rules import source_label_rule
from core.types import Label
from labeling.manifest import append, build_fake_row

MERCADOLIVRE_OFFICIAL_API_URL = "https://api.mercadolibre.com/sites/MLB/search?q=rolex+submariner"

IngestMethod = Literal["scrape", "official_api"]


def fetch_mercadolivre_listings(api_url: str = MERCADOLIVRE_OFFICIAL_API_URL) -> list[dict]:
    """Fetches listing photos via the official Developer API only."""
    response = requests.get(api_url, timeout=10)
    response.raise_for_status()
    return response.json().get("results", [])


def mercadolivre_ingest(
    method: IngestMethod,
    listings: list[dict] | None = None,
    *,
    manifest_dir: Path | None = None,
    version: int = 1,
) -> list[Path]:
    """Labels and writes Mercado Livre listing photos (CK-28, CK-13).

    `method="scrape"` is refused outright — Mercado Livre's ToS bans it,
    and this adapter is the sole ingestion point, so the refusal happens
    before any HTTP call, manifest write, or `listings` lookup occurs.
    `method="official_api"` is the only path that proceeds.
    """
    if method == "scrape":
        raise PermissionError(
            "Mercado Livre scraping is prohibited by its Developer ToS; "
            'use mercadolivre_ingest(method="official_api") instead (CK-28)'
        )
    if method != "official_api":
        raise ValueError(f"Unknown Mercado Livre ingestion method: {method!r}")
    if manifest_dir is None:
        raise ValueError("manifest_dir is required for method='official_api'")

    resolved_listings = listings if listings is not None else fetch_mercadolivre_listings()

    written: list[Path] = []
    for listing in resolved_listings:
        label = source_label_rule(
            source="mercadolivre",
            marketplace="mercadolivre",
            listing_text=listing.get("listing_text"),
        )
        if label == Label.EXCLUDED:
            continue  # seller claims "original" (or unrecognized) — never enters the dataset

        record = build_fake_row(
            path=listing["path"],
            source="mercadolivre",
            url=listing["url"],
            phash=listing["phash"],
            scraped_at=listing["scraped_at"],
            price=listing.get("price"),
            blur_score=listing.get("blur_score"),
        )
        written.append(append(record, manifest_dir=manifest_dir, version=version))
    return written
