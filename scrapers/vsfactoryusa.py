"""Public-image discovery for vendor-declared VSFactoryUSA replicas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from acquisition.discovery import DiscoveryPage
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.replica_sitemap import (
    ReplicaSitemapConfig,
    ReplicaSitemapError,
    discover_replica_sitemap,
)

VSFactoryAPIError = ReplicaSitemapError

_CONFIG = ReplicaSitemapConfig(
    source="vsfactoryusa",
    sitemap_url="https://vsfactoryusa.com/sitemap.xml",
    host="vsfactoryusa.com",
    seller_group_id="vsfactoryusa",
    factory_claim="VS Factory (VSF)",
    required_path_markers=("replica", "super-clone"),
    required_title_markers=("clone",),
)


def discover_vsfactoryusa(
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
