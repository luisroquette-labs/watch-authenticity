from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from scrapers.replica_sitemap import ReplicaSitemapError
from scrapers.richfactoryus import discover_richfactoryus

NOW = datetime(2026, 11, 12, 16, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, body: str, status_code: int = 200):
        self.body = body.encode()
        self.status_code = status_code
        self.closed = False

    def iter_content(self, chunk_size: int):
        del chunk_size
        yield self.body

    def close(self):
        self.closed = True


def _registry():
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))
    return get_source_policy(policies, "richfactoryus"), access["richfactoryus"]


def _sitemap(*, image_host: str = "richfactoryus.com", title: str | None = None):
    image_title = title or "Submariner Date 126610LN - Front, dial and bezel"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://richfactoryus.com/qc-sets/rolex-submariner-126610ln-black/</loc>
    <lastmod>2026-11-12</lastmod>
    <image:image>
      <image:loc>https://{image_host}/images/qc-126610ln-01.webp</image:loc>
      <image:title>{image_title}</image:title>
    </image:image>
  </url>
  <url>
    <loc>https://richfactoryus.com/product/rf-submariner-126610ln/</loc>
    <lastmod>2026-11-12</lastmod>
  </url>
</urlset>"""


def test_discovers_rich_factory_qc_images_for_private_download_only():
    policy, access = _registry()
    http = MagicMock()
    response = _Response(_sitemap())
    http.get.return_value = response

    page = discover_richfactoryus(http, policy, access, now=NOW)

    assert policy.may_download
    assert not policy.may_train
    assert response.closed
    assert len(page.records) == 1
    record = page.records[0]
    assert record.image_urls == (
        "https://richfactoryus.com/images/qc-126610ln-01.webp",
    )
    assert dict(record.metadata) == {
        "classification": "replica_candidate_vendor_declared",
        "evidence_tier": "c",
        "factory_claim": "RICH Factory",
        "last_modified": "2026-11-12",
        "rights_status": "owner_private_download_only",
    }


def test_rich_factory_unapproved_image_host_fails_closed():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response(_sitemap(image_host="evil.example"))

    with pytest.raises(ReplicaSitemapError, match="approved host"):
        discover_richfactoryus(http, policy, access, now=NOW)


def test_rich_factory_image_title_must_name_exact_reference():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response(_sitemap(title="Submariner Date 126610LV"))

    with pytest.raises(ReplicaSitemapError, match="lacks the reference"):
        discover_richfactoryus(http, policy, access, now=NOW)


def test_rich_factory_rate_limit_is_recoverable():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response("", status_code=429)

    with pytest.raises(ReplicaSitemapError, match="resume later"):
        discover_richfactoryus(http, policy, access, now=NOW)
