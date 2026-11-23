from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.discovery import DiscoveryCommit
from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from scrapers.swisswatchexpo import SwissWatchExpoDiscoveryError, discover_swisswatchexpo

NOW = datetime(2026, 11, 23, 3, 0, tzinfo=timezone.utc)
HOST = "https://www.swisswatchexpo.com"
CDN = "https://cdn.swisswatchexpo.com/productphotos/1/2"
PRODUCT = f"{HOST}/watches/cartier-santos-large-watch-wssa0018-53951/"


class _Response:
    def __init__(self, body: str):
        self.body = body.encode()
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.raw = self
        self.offset = 0

    def read1(self, amount: int, *, decode_content: bool):
        assert decode_content is False
        chunk = self.body[self.offset:self.offset + amount]
        self.offset += len(chunk)
        return chunk

    def close(self):
        pass


def _discover(*, image_host: str = CDN, duplicate: bool = False):
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))
    policy = get_source_policy(policies, "swisswatchexpo")
    sitemap = f"{HOST}/sitemaps/in-stock-product-detail-sitemap-1.xml"
    entry = f'''<url><loc>{PRODUCT}</loc><lastmod>2026-11-22</lastmod>
    <image:image><image:loc>{image_host}/front.jpg</image:loc></image:image>
    <image:image><image:loc>{image_host}/back.jpg</image:loc></image:image></url>'''
    bodies = {
        f"{HOST}/robots.txt": f"User-agent: *\nSitemap: {HOST}/sitemap.xml\n",
        f"{HOST}/sitemap.xml": f'''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>{sitemap}</loc></sitemap></sitemapindex>''',
        sitemap: f'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
          xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
          {entry}{entry if duplicate else ''}</urlset>''',
    }
    client = MagicMock()
    client.get.side_effect = lambda url, **kwargs: _Response(bodies[url])
    page = discover_swisswatchexpo(client, policy, access["swisswatchexpo"], now=NOW,
                                   clock=lambda: 0.0, sleeper=lambda _: None)
    return page, policy, client


def test_sitemap_groups_angles_by_stock_and_keeps_seller_claim_weak():
    page, policy, client = _discover(duplicate=True)
    assert not policy.may_download and not policy.may_train
    assert page.next_cursor is None and len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "53951"
    assert record.declared_reference == "cartier-santos-wssa0018"
    assert record.image_urls == (f"{CDN}/front.jpg", f"{CDN}/back.jpg")
    assert dict(record.metadata)["independent_confirmation"] == "not_established"
    assert client.get.call_count == 3


def test_rejects_off_host_image_before_persisting_page():
    with pytest.raises(SwissWatchExpoDiscoveryError, match="outside approved public host"):
        _discover(image_host="https://untrusted.example/productphotos")


def test_continuation_reads_only_next_sitemap_and_skips_prior_stock():
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))["swisswatchexpo"]
    policy = get_source_policy(policies, "swisswatchexpo")
    first = f"{HOST}/sitemaps/in-stock-product-detail-sitemap-1.xml"
    second = f"{HOST}/sitemaps/out-of-stock-product-detail-sitemap-1.xml"
    entry = f'''<url><loc>{PRODUCT}</loc><image:image><image:loc>{CDN}/front.jpg</image:loc>
      </image:image></url>'''
    other = entry.replace("53951", "53952")
    bodies = {
        f"{HOST}/robots.txt": f"Sitemap: {HOST}/sitemap.xml",
        f"{HOST}/sitemap.xml": f'''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>{first}</loc></sitemap><sitemap><loc>{second}</loc></sitemap></sitemapindex>''',
        first: f'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
          xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">{entry}</urlset>''',
        second: f'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
          xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">{entry}{other}</urlset>''',
    }
    client = MagicMock()
    client.get.side_effect = lambda url, **kwargs: _Response(bodies[url])
    kwargs = {"now": NOW, "clock": lambda: 0.0, "sleeper": lambda _: None}
    page = discover_swisswatchexpo(client, policy, access, **kwargs)
    assert page.next_cursor == first
    client.get.reset_mock()
    bodies[f"{HOST}/sitemap.xml"] = f'''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>{second}</loc></sitemap><sitemap><loc>{first}</loc></sitemap></sitemapindex>'''
    previous = DiscoveryCommit("swisswatchexpo", access.access_record_id, None,
                               "page-sha", page.next_cursor, (("53951", "record-sha"),), "commit-sha")
    continued = discover_swisswatchexpo(client, policy, access, previous=previous, **kwargs)
    assert continued.cursor == page.next_cursor and continued.next_cursor is None
    assert sorted(record.source_record_id.split("#")[0] for record in continued.records) == [
        "53951", "53952"
    ]
    assert all(dict(record.metadata)["listing_stock_id"] in {"53951", "53952"}
               for record in continued.records)
    assert all("physical_stock_id" not in dict(record.metadata) for record in continued.records)
    assert second in [call.args[0] for call in client.get.call_args_list]
    assert first not in [call.args[0] for call in client.get.call_args_list]

    bodies[f"{HOST}/sitemap.xml"] = f'''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <sitemap><loc>{first}</loc></sitemap></sitemapindex>'''
    terminal = discover_swisswatchexpo(client, policy, access, previous=previous, **kwargs)
    assert terminal.records == () and terminal.next_cursor is None


def test_reference_must_be_a_slug_token():
    import xml.etree.ElementTree as ET

    from scrapers.swisswatchexpo import _parse_entry

    element = ET.fromstring(f'''<url xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
      xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
      <loc>{HOST}/watches/example-1263345-53951/</loc>
      <image:image><image:loc>{CDN}/front.jpg</image:loc></image:image></url>''')
    assert _parse_entry(element) is None
