"""Public-image discovery for the standard VSF 126610LN product."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from acquisition.discovery import DiscoveryPage
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.replica_sitemap import ReplicaSitemapConfig, discover_replica_sitemap

_CONFIG = ReplicaSitemapConfig(
    source="vsfwatcheshub",
    sitemap_url="https://vsfwatcheshub.com/wpkj_product-sitemap.xml",
    host="vsfwatcheshub.com",
    seller_group_id="vsfwatcheshub",
    factory_claim="VS Factory (VSF)",
    exact_path="/products/rolex-126610ln-vsf-2/",
    expected_image_count=10,
    required_path_markers=("products/rolex-126610ln-vsf-2",),
    required_page_title_markers=("super clone",),
)


def discover_vsfwatcheshub(
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
