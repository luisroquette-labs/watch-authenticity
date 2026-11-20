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
from scrapers.bobswatches import BobsWatchesDiscoveryError, discover_bobswatches

NOW = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)
HOST = "https://www.bobswatches.com"
PRODUCT = f"{HOST}/used-rolex-submariner-126610ln-black-dial.html"
IMAGE = f"{HOST}/images/zUsed-Rolex-Submariner-126610-SKU195033.jpg"
SITEMAP = f"{HOST}/products-4-sitemap.xml"


class _Response:
    def __init__(self, body: str):
        self.body = body.encode()
        self.status_code = 200
        self.headers: dict[str, str] = {}
        self.raw = self
        self.offset = 0

    def read1(self, amount: int, *, decode_content: bool):
        assert decode_content is False
        part = self.body[self.offset:self.offset + amount]
        self.offset += len(part)
        return part

    def close(self):
        pass


def _registry():
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))
    return get_source_policy(policies, "bobswatches"), access["bobswatches"]


def _http(*, image: str = IMAGE, duplicate: bool = False, product: str = PRODUCT,
          title: str = "Used Rolex Submariner 126610LN Black Dial",
          lastmod: str = "<lastmod>2026-11-20T00:00:00Z</lastmod>"):
    index = f'''<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <sitemap><loc>{SITEMAP}</loc></sitemap></sitemapindex>'''
    entry = f'''<url><loc>{product}</loc>{lastmod}
      <image:image><image:loc>{image}</image:loc>
      <image:title>{title}</image:title></image:image></url>'''
    products = f'''<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
    xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
    {entry}{entry if duplicate else ''}
    <url><loc>{HOST}/used-rolex-submariner-126610lv.html</loc>
    <image:image><image:loc>{HOST}/images/SKU195034.jpg</image:loc>
    <image:title>Rolex Submariner 126610LV</image:title></image:image></url></urlset>'''
    bodies = {
        f"{HOST}/robots.txt": f"User-agent: *\nSitemap: {HOST}/sitemap.xml\n",
        f"{HOST}/sitemap.xml": index,
        SITEMAP: products,
    }
    client = MagicMock()
    client.get.side_effect = lambda url, **kwargs: _Response(bodies[url])
    return client


def _discover(client, policy=None, access=None):
    p, a = _registry()
    return discover_bobswatches(client, policy or p, access or a, now=NOW,
                                clock=lambda: 0.0, sleeper=lambda _seconds: None)


def test_discovers_exact_reference_as_metadata_only_seller_claim():
    client = _http()
    page = _discover(client)
    policy, _ = _registry()
    assert not policy.may_download and not policy.may_train
    assert page.next_cursor is None and len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "195033"
    assert record.declared_reference == "rolex-submariner-126610ln"
    assert record.image_urls == (IMAGE,)
    assert dict(record.metadata)["independent_confirmation"] == "not_established"
    assert {call.args[0] for call in client.get.call_args_list} == {
        f"{HOST}/robots.txt", f"{HOST}/sitemap.xml", SITEMAP,
    }


def test_rejects_cross_host_image_and_duplicate_sku():
    with pytest.raises(BobsWatchesDiscoveryError, match="approved public host"):
        _discover(_http(image="https://evil.example/images/SKU195033.jpg"))
    with pytest.raises(BobsWatchesDiscoveryError, match="SKU appears"):
        _discover(_http(duplicate=True))


def test_skips_non_product_paths_and_accepts_missing_lastmod():
    assert _discover(_http(product=f"{HOST}/rolex-blog/126610ln.html")).records == ()
    record = _discover(_http(lastmod="")).records[0]
    assert "last_modified" not in dict(record.metadata)


@pytest.mark.parametrize(("product", "image", "title", "expected"), [
    (f"{HOST}/193694-pre-owned-rolex-datejust-41-ref-126334.html",
     f"{HOST}/images/zUsed-Rolex-Datejust-41-126334-SKU193694.jpg",
     "193694 Pre-Owned Rolex Datejust 41 Ref 126334 Black Diamond Dial",
     "rolex-datejust-126334"),
    (f"{HOST}/cartier/pre-owned-santos-de-cartier-wssa0061.html",
     f"{HOST}/cartier/images/zUsed-Cartier-Santos-WSSA0061-SKU185936.jpg",
     "Pre-Owned Santos De Cartier WSSA0061 Medium Model Green Dial",
     "cartier-santos-wssa0061"),
])
def test_discovers_other_catalog_references_without_authentication(
    product, image, title, expected,
):
    record = _discover(_http(product=product, image=image, title=title)).records[0]
    assert record.declared_reference == expected
    assert record.source_record_id in {"193694", "185936"}
    assert dict(record.metadata)["independent_confirmation"] == "not_established"


def test_denies_wrong_access_host_and_continuation():
    policy, access = _registry()
    with pytest.raises(PermissionError):
        _discover(_http(), access=replace(access, allowed_image_hosts=("evil.example",)))
    with pytest.raises(BobsWatchesDiscoveryError, match="no continuation"):
        discover_bobswatches(_http(), policy, access, now=NOW, previous=object(),
                             clock=lambda: 0.0, sleeper=lambda _seconds: None)
