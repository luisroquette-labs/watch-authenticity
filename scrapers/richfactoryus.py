"""Metadata-only discovery for vendor-declared RICH Factory replicas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from acquisition.discovery import DiscoveryPage
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.replica_sitemap import ReplicaSitemapConfig, discover_replica_sitemap

_CONFIG = ReplicaSitemapConfig(
    source="richfactoryus",
    sitemap_url="https://richfactoryus.com/sitemap.xml",
    host="richfactoryus.com",
    seller_group_id="richfactoryus",
    factory_claim="RICH Factory",
    required_path_markers=("qc-sets", "product"),
)


def discover_richfactoryus(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: object | None = None,
) -> DiscoveryPage:
    return discover_replica_sitemap(
        http_client, policy, access, _CONFIG, now=now, previous=previous
    )
