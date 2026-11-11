from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from scrapers.watchesofswitzerland import (
    WatchesOfSwitzerlandAPIError,
    discover_watchesofswitzerland,
)

NOW = datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc)
INDEX_URL = "https://www.watchesofswitzerland.com/sitemap.xml"
PRODUCT_SITEMAP = (
    "https://www.watchesofswitzerland.com/sitemap_products_3.xml?from=1&to=2"
)


class _Response:
    def __init__(self, body: str, status_code: int = 200):
        self.body = body.encode()
        self.headers = {}
        self.raw = self
        self.offset = 0
        self.status_code = status_code
        self.closed = False

    def read1(self, amount: int, *, decode_content: bool):
        del decode_content
        chunk = self.body[self.offset : self.offset + amount]
        self.offset += len(chunk)
        return chunk

    def close(self):
        self.closed = True


def _registry():
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))
    return (
        get_source_policy(policies, "watchesofswitzerland"),
        access["watchesofswitzerland"],
    )


def _index(product_url: str = PRODUCT_SITEMAP):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>{product_url.replace('&', '&amp;')}</loc></sitemap>
</sitemapindex>"""


def _products(*, image_host: str = "cdn.shopify.com", suffix: str = "126610ln"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://www.watchesofswitzerland.com/products/submariner-date-m{suffix}-0001-40615391</loc>
    <lastmod>2026-11-11T10:48:28-05:00</lastmod>
    <image:image>
      <image:loc>https://{image_host}/s/files/1/watch/40615391_whiteHero.jpg?v=1</image:loc>
      <image:title>Submariner Date</image:title>
      <image:caption>Rolex Certified Pre-owned Submariner Date M{suffix.upper()}-0001 front on white background</image:caption>
    </image:image>
  </url>
  <url>
    <loc>https://www.watchesofswitzerland.com/products/submariner-m126610lv-0002-40615392</loc>
    <lastmod>2026-11-11</lastmod>
    <image:image>
      <image:loc>https://cdn.shopify.com/s/files/1/watch/other.jpg</image:loc>
      <image:title>Submariner</image:title>
      <image:caption>Rolex Certified Pre-owned Submariner M126610LV-0002</image:caption>
    </image:image>
  </url>
</urlset>"""


def _http(products: str | None = None):
    http = MagicMock()
    bodies = {
        "https://www.watchesofswitzerland.com/robots.txt": _Response(
            f"User-agent: *\nAllow: /\nSitemap: {INDEX_URL}\n"
        ),
        INDEX_URL: _Response(_index()),
        PRODUCT_SITEMAP: _Response(products or _products()),
    }
    http.get.side_effect = lambda url, **_kwargs: bodies[url]
    return http


def _discover(http, policy, access):
    return discover_watchesofswitzerland(
        http,
        policy,
        access,
        now=NOW,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )


def test_discovers_exact_cpo_reference_as_metadata_only_original_candidate():
    policy, access = _registry()

    page = _discover(_http(), policy, access)

    assert not policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.seller_group_id == "wos-group-cpo"
    assert record.source_record_id == "40615391"
    assert record.source_url.endswith("m126610ln-0001-40615391")
    assert record.declared_reference == "rolex-submariner-126610ln"
    assert record.image_urls == (
        "https://cdn.shopify.com/s/files/1/watch/40615391_whiteHero.jpg?v=1",
    )
    assert dict(record.metadata) == {
        "classification": "original_candidate_cpo_certified",
        "evidence_tier": "a",
        "physical_unit_id": "40615391",
        "authenticity_claim": "Rolex Certified Pre-Owned",
        "last_modified": "2026-11-11T10:48:28-05:00",
        "rights_status": "metadata_only",
    }


def test_reads_only_public_discovery_surfaces_and_does_not_fetch_images():
    policy, access = _registry()
    http = _http()

    _discover(http, policy, access)

    requested = [call.args[0] for call in http.get.call_args_list]
    assert requested == [
        "https://www.watchesofswitzerland.com/robots.txt",
        INDEX_URL,
        PRODUCT_SITEMAP,
    ]
    assert all("cdn.shopify.com" not in url for url in requested)


def test_near_reference_and_other_submariner_are_not_mislabeled():
    policy, access = _registry()

    page = _discover(_http(_products(suffix="126610ln999")), policy, access)

    assert page.records == ()


def test_unapproved_product_or_image_host_fails_closed():
    policy, access = _registry()
    bad_index = _http()
    bad_index.get.side_effect = [
        _Response(f"Sitemap: {INDEX_URL}\n"),
        _Response(_index("https://evil.example/sitemap_products_1.xml")),
    ]
    with pytest.raises(WatchesOfSwitzerlandAPIError, match="approved host"):
        _discover(bad_index, policy, access)

    with pytest.raises(WatchesOfSwitzerlandAPIError, match="approved host"):
        _discover(_http(_products(image_host="evil.example")), policy, access)


def test_missing_cpo_declaration_fails_closed():
    policy, access = _registry()
    products = _products().replace("Rolex Certified Pre-owned ", "")

    with pytest.raises(WatchesOfSwitzerlandAPIError, match="CPO declaration"):
        _discover(_http(products), policy, access)


def test_rate_limit_is_recoverable_and_response_is_closed():
    policy, access = _registry()
    response = _Response("", status_code=429)
    http = MagicMock()
    http.get.return_value = response

    with pytest.raises(WatchesOfSwitzerlandAPIError, match="resume later"):
        _discover(http, policy, access)

    assert response.closed


def test_access_record_must_bind_the_exact_image_host():
    policy, access = _registry()

    with pytest.raises(PermissionError, match="image hosts"):
        _discover(
            _http(), policy, replace(access, allowed_image_hosts=("images.example",))
        )


def test_oversized_response_stops_stream_and_closes(monkeypatch):
    policy, access = _registry()
    response = _Response("")
    chunks = iter((b"123456789", b"must-not-be-consumed"))
    response.read1 = MagicMock(side_effect=lambda *_args, **_kwargs: next(chunks, b""))
    http = MagicMock()
    http.get.return_value = response
    monkeypatch.setattr("scrapers.watchesofswitzerland._MAX_ROBOTS_BYTES", 8)

    with pytest.raises(WatchesOfSwitzerlandAPIError, match="too large"):
        _discover(http, policy, access)

    assert response.closed
    assert next(chunks) == b"must-not-be-consumed"


def test_requests_are_paced_by_the_access_record_limit():
    policy, access = _registry()

    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

        def sleep(self, seconds):
            self.value += seconds

    clock = Clock()
    discover_watchesofswitzerland(
        _http(),
        policy,
        access,
        now=NOW,
        clock=clock,
        sleeper=clock.sleep,
    )

    assert clock.value == 12.0
