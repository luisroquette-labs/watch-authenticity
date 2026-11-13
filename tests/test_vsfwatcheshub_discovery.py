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
from scrapers.vsfwatcheshub import discover_vsfwatcheshub

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)
SITEMAP_URL = "https://vsfwatcheshub.com/wpkj_product-sitemap.xml"
PRODUCT_URL = "https://vsfwatcheshub.com/products/rolex-126610ln-vsf-2/"


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
    return get_source_policy(policies, "vsfwatcheshub"), access["vsfwatcheshub"]


def _sitemap(*, image_host: str = "vsfwatcheshub.com") -> str:
    images = "".join(
        f"<image:image><image:loc>https://{image_host}/wp-content/uploads/126610ln-{index}.webp</image:loc></image:image>"
        for index in range(1, 11)
    )
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url><loc>{PRODUCT_URL}</loc><lastmod>2026-07-30T12:40:23+00:00</lastmod>{images}</url>
  <url><loc>https://vsfwatcheshub.com/products/rolex-126610ln-vsf/</loc><lastmod>2026-04-20T05:13:22+00:00</lastmod><image:image><image:loc>https://vsfwatcheshub.com/wp-content/uploads/modified.webp</image:loc></image:image></url>
</urlset>"""


def _page(*, canonical: str = PRODUCT_URL, title: str | None = None) -> str:
    product_title = title or "VSF Rolex Submariner Date 126610LN Black Dial 41mm Super Clone"
    return (
        f"<html><head><title>{product_title}</title>"
        f'<link rel="canonical" href="{canonical}" /></head></html>'
    )


def _http(*, sitemap: str | None = None, page: str | None = None):
    http = MagicMock()
    bodies = {
        SITEMAP_URL: _Response(sitemap or _sitemap()),
        PRODUCT_URL: _Response(page or _page()),
    }
    http.get.side_effect = lambda url, **_kwargs: bodies[url]
    return http


def test_discovers_ten_exact_standard_vsf_images_for_private_download_only():
    policy, access = _registry()
    http = _http()

    page = discover_vsfwatcheshub(http, policy, access, now=NOW)

    assert policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "products/rolex-126610ln-vsf-2"
    assert len(record.image_urls) == 10
    assert dict(record.metadata) == {
        "classification": "replica_candidate_vendor_declared",
        "evidence_tier": "c",
        "factory_claim": "VS Factory (VSF)",
        "last_modified": "2026-07-30T12:40:23+00:00",
        "rights_status": "owner_private_download_only",
    }
    assert [call.args[0] for call in http.get.call_args_list] == [
        SITEMAP_URL,
        PRODUCT_URL,
    ]


def test_rejects_unapproved_image_host_or_unproven_replica_declaration():
    policy, access = _registry()
    with pytest.raises(ReplicaSitemapError, match="approved host"):
        discover_vsfwatcheshub(
            _http(sitemap=_sitemap(image_host="evil.example")),
            policy,
            access,
            now=NOW,
        )
    with pytest.raises(ReplicaSitemapError, match="replica declaration"):
        discover_vsfwatcheshub(
            _http(page=_page(title="Rolex Submariner Date 126610LN")),
            policy,
            access,
            now=NOW,
        )


def test_rejects_product_page_canonical_mismatch():
    policy, access = _registry()

    with pytest.raises(ReplicaSitemapError, match="canonical"):
        discover_vsfwatcheshub(
            _http(page=_page(canonical="https://vsfwatcheshub.com/products/other/")),
            policy,
            access,
            now=NOW,
        )
