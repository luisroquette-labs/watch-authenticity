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
from scrapers.precisionwatches import PrecisionWatchesAPIError, discover_precisionwatches

NOW = datetime(2026, 11, 11, 17, 30, tzinfo=timezone.utc)
INDEX_URL = "https://precisionwatches.com/sitemap_index.xml"
PRODUCT_SITEMAP = "https://precisionwatches.com/product-sitemap.xml"
PRODUCT_URL = (
    "https://precisionwatches.com/rolex-certified-pre-owned/rolex-submariner/"
    "submariner-date-126610ln-black-cerachrom/"
)
UNIT_ID = "PA0000495763"


class _Response:
    def __init__(self, body: str, status_code: int = 200):
        self.body = body.encode()
        self.headers: dict[str, str] = {}
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
    return get_source_policy(policies, "precisionwatches"), access["precisionwatches"]


def _index(product_url: str = PRODUCT_SITEMAP):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <sitemap><loc>{product_url}</loc></sitemap>
</sitemapindex>"""


def _sitemap(product_url: str = PRODUCT_URL):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{product_url}</loc><lastmod>2026-11-11T16:25:06+00:00</lastmod></url>
  <url><loc>https://precisionwatches.com/product/rolex-submariner-126610ln/</loc><lastmod>2026-11-11T16:00:00+00:00</lastmod></url>
</urlset>"""


def _page(*, title: str | None = None, image_host: str = "precisionwatches.com"):
    product_title = title or (
        "Rolex Certified Pre-Owned Submariner Date 41mm Steel Black Dial 2021 "
        "| Ref. 126610LN"
    )
    images = "".join(
        f'<img src="https://{image_host}/wp-content/uploads/2026/11/'
        f'R_3120.{UNIT_ID}_{view}-scaled.jpg">'
        for view in ("SOLDIER_BLACK_01", "V_360M_06", "V_360M_17", "V_360M_32")
    )
    return (
        f"<html><head><title>{product_title}</title>"
        f'<link rel="canonical" href="{PRODUCT_URL}" /></head><body>'
        f'<img src="https://{image_host}/wp-content/uploads/logo.png">'
        f"{images}</body></html>"
    )


def _http(*, index: str | None = None, sitemap: str | None = None, page: str | None = None):
    http = MagicMock()
    bodies = {
        "https://precisionwatches.com/robots.txt": _Response(
            f"User-agent: *\nSitemap: {INDEX_URL}\n"
        ),
        INDEX_URL: _Response(index or _index()),
        PRODUCT_SITEMAP: _Response(sitemap or _sitemap()),
        PRODUCT_URL: _Response(page or _page()),
    }
    http.get.side_effect = lambda url, **_kwargs: bodies[url]
    return http


def _discover(http, policy, access):
    return discover_precisionwatches(
        http,
        policy,
        access,
        now=NOW,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
        deadline=100.0,
    )


def test_discovers_independent_cpo_unit_and_four_unique_image_links():
    policy, access = _registry()

    page = _discover(_http(), policy, access)

    assert not policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == UNIT_ID
    assert record.seller_group_id == "precision-watches-cpo"
    assert record.declared_reference == "rolex-submariner-126610ln"
    assert len(record.image_urls) == len(set(record.image_urls)) == 4
    assert dict(record.metadata) == {
        "classification": "original_candidate_cpo_certified",
        "evidence_tier": "a",
        "physical_unit_id": UNIT_ID,
        "authenticity_claim": "Rolex Certified Pre-Owned",
        "last_modified": "2026-11-11T16:25:06+00:00",
        "rights_status": "metadata_only",
    }


def test_reads_only_metadata_pages_and_never_fetches_image_bytes():
    policy, access = _registry()
    http = _http()

    _discover(http, policy, access)

    requested = [call.args[0] for call in http.get.call_args_list]
    assert requested == [
        "https://precisionwatches.com/robots.txt",
        INDEX_URL,
        PRODUCT_SITEMAP,
        PRODUCT_URL,
    ]
    assert all("wp-content/uploads" not in url for url in requested)


def test_near_reference_and_non_cpo_products_are_not_mislabeled():
    policy, access = _registry()
    near = PRODUCT_URL.replace("126610ln", "126610ln999")

    page = _discover(_http(sitemap=_sitemap(near)), policy, access)

    assert page.records == ()


def test_unapproved_sitemap_product_and_image_hosts_fail_closed():
    policy, access = _registry()
    with pytest.raises(PrecisionWatchesAPIError, match="approved host"):
        _discover(_http(index=_index("https://evil.example/product-sitemap.xml")), policy, access)

    evil_product = PRODUCT_URL.replace("precisionwatches.com", "evil.example")
    with pytest.raises(PrecisionWatchesAPIError, match="approved host"):
        _discover(_http(sitemap=_sitemap(evil_product)), policy, access)

    with pytest.raises(PrecisionWatchesAPIError, match="approved host"):
        _discover(_http(page=_page(image_host="evil.example")), policy, access)


def test_product_page_requires_exact_cpo_title_canonical_and_one_physical_unit():
    policy, access = _registry()
    with pytest.raises(PrecisionWatchesAPIError, match="CPO declaration"):
        _discover(_http(page=_page(title="Rolex Submariner 126610LN")), policy, access)

    with pytest.raises(PrecisionWatchesAPIError, match="canonical"):
        _discover(_http(page=_page().replace(PRODUCT_URL, "https://example.test/")), policy, access)

    mixed = _page().replace(f"{UNIT_ID}_V_360M_32", "PA9999999999_V_360M_32")
    with pytest.raises(PrecisionWatchesAPIError, match="physical watch"):
        _discover(_http(page=mixed), policy, access)
