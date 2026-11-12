from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from scrapers.superclonerolex import (
    SuperCloneRolexAPIError,
    discover_superclonerolex,
)

NOW = datetime(2026, 11, 12, 17, 0, tzinfo=timezone.utc)
SITEMAP_URL = "https://superclonerolex.io/sitemap.xml"
PRODUCT_URL = "https://superclonerolex.io/watches/126610ln/"


class _Response:
    def __init__(self, body: str):
        self.body = body.encode()
        self.headers: dict[str, str] = {}
        self.raw = self
        self.offset = 0
        self.status_code = 200

    def read1(self, amount: int, *, decode_content: bool):
        del decode_content
        chunk = self.body[self.offset : self.offset + amount]
        self.offset += len(chunk)
        return chunk

    def close(self):
        pass


def _registry():
    policies = load_source_policies(Path("config/source_policies.json"))
    access = load_source_access_registry(Path("config/source_access.json"))
    return get_source_policy(policies, "superclonerolex"), access["superclonerolex"]


def _sitemap(product_url: str = PRODUCT_URL):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{product_url}</loc><lastmod>2026-10-22T04:55:53+00:00</lastmod></url>
  <url><loc>https://superclonerolex.io/watches/126610lv/</loc><lastmod>2026-10-22T04:55:53+00:00</lastmod></url>
</urlset>"""


def _page(*, canonical: str = PRODUCT_URL, image_host: str = "superclonerolex.io"):
    images = [
        {
            "@type": "ImageObject",
            "contentUrl": f"https://{image_host}/images/products/Super-Clone-Submariner-126610LN-{index}-miyota.webp",
            "caption": f"Declared replica view {index}",
        }
        for index in range(1, 5)
    ]
    product = {
        "@type": "Product",
        "name": "Rolex Submariner 126610LN 41mm Black Dial",
        "sku": "JP-126610LN",
        "category": "Super Clone Rolex, Submariner",
        "image": images,
    }
    return (
        "<html><head><title>Black Cerachrom Super Clone: Rolex Submariner 126610LN, $359</title>"
        f'<link rel="canonical" href="{canonical}" />'
        f'<script type="application/ld+json">{json.dumps(product)}</script>'
        "</head></html>"
    )


def _http(*, sitemap: str | None = None, page: str | None = None):
    http = MagicMock()
    bodies = {
        "https://superclonerolex.io/robots.txt": _Response(
            f"User-Agent: *\nAllow: /\nSitemap: {SITEMAP_URL}\n"
        ),
        SITEMAP_URL: _Response(sitemap or _sitemap()),
        PRODUCT_URL: _Response(page or _page()),
    }
    http.get.side_effect = lambda url, **_kwargs: bodies[url]
    return http


def _discover(http):
    policy, access = _registry()
    page = discover_superclonerolex(
        http,
        policy,
        access,
        now=NOW,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
        deadline=100.0,
    )
    return policy, page


def test_discovers_vendor_declared_product_images_for_private_download_only():
    http = _http()

    policy, page = _discover(http)

    assert policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "JP-126610LN"
    assert len(record.image_urls) == 4
    assert dict(record.metadata) == {
        "classification": "replica_candidate_vendor_declared",
        "evidence_tier": "c",
        "seller_claim": "Super Clone Rolex",
        "last_modified": "2026-10-22T04:55:53+00:00",
        "rights_status": "owner_private_download_only",
    }
    assert [call.args[0] for call in http.get.call_args_list] == [
        "https://superclonerolex.io/robots.txt",
        SITEMAP_URL,
        PRODUCT_URL,
    ]


def test_unapproved_product_or_image_host_fails_closed():
    with pytest.raises(SuperCloneRolexAPIError, match="approved host"):
        _discover(_http(sitemap=_sitemap("https://evil.example/watches/126610ln/")))
    with pytest.raises(SuperCloneRolexAPIError, match="approved host"):
        _discover(_http(page=_page(image_host="evil.example")))


def test_product_requires_exact_replica_identity_and_canonical():
    with pytest.raises(SuperCloneRolexAPIError, match="replica declaration"):
        _discover(_http(page=_page().replace("Super Clone", "Watch")))
    with pytest.raises(SuperCloneRolexAPIError, match="canonical"):
        _discover(_http(page=_page(canonical="https://superclonerolex.io/watches/other/")))


def test_duplicate_or_incomplete_image_inventory_fails_closed():
    duplicate = _page().replace("126610LN-4-miyota", "126610LN-3-miyota")
    with pytest.raises(SuperCloneRolexAPIError, match="duplicated"):
        _discover(_http(page=duplicate))
    short = _page().replace(
        '"contentUrl": "https://superclonerolex.io/images/products/Super-Clone-Submariner-126610LN-4-miyota.webp", "caption": "Declared replica view 4"',
        '"contentUrl": 4, "caption": ""',
    )
    with pytest.raises(SuperCloneRolexAPIError, match="incomplete"):
        _discover(_http(page=short))
