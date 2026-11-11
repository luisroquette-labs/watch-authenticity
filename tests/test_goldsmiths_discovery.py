from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.run import run_metadata_discovery_once
from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from scrapers.goldsmiths import GoldsmithsAPIError, discover_goldsmiths
from scrapers.metadata_http import HTTPDeadlineExceeded, PacedBoundedHTTP

NOW = datetime(2026, 11, 11, 16, 30, tzinfo=timezone.utc)
INDEX_URL = "https://www.goldsmiths.co.uk/sitemap.xml"
PRODUCT_SITEMAP = "https://www.goldsmiths.co.uk/sitemap/product-en-gbp-0.xml"
PRODUCT_URL = (
    "https://www.goldsmiths.co.uk/"
    "Rolex-Certified-Pre+Owned-Submariner-Date-126610LN/p/406107907490"
)


class _Response:
    def __init__(self, body: str, status_code: int = 200):
        self.body = body.encode()
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
    return get_source_policy(policies, "goldsmiths"), access["goldsmiths"]


def _index(product_url: str = PRODUCT_SITEMAP):
    return json.dumps({"sitemap": [{"loc": product_url}]})


def _sitemap(product_url: str = PRODUCT_URL):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{product_url}</loc></url>
  <url><loc>https://www.goldsmiths.co.uk/Rolex/Submariner-Date-M126610LN+0001/p/17304447</loc></url>
  <url><loc>https://www.goldsmiths.co.uk/Rolex-Certified-Pre+Owned-Submariner-Date-126610LV/p/406107882490</loc></url>
</urlset>"""


def _product_page(
    *,
    title: str | None = None,
    image_host: str = "content.thewosgroup.com",
    product_url: str = PRODUCT_URL,
):
    unit_id = product_url.rsplit("/", 1)[-1]
    product_title = title or "Rolex Certified Pre-Owned Submariner Date 126610LN | Goldsmiths"
    images = "".join(
        f'<img src="https://{image_host}/productimage/{unit_id}/'
        f'{unit_id}_{position}.png?impolicy=lister">'
        for position in (1, 2, 3)
    )
    return (
        f"<html><head><title>{product_title}</title>"
        f'<link rel="canonical" href="{product_url}"></head><body>{images}</body></html>'
    )


def _http(*, index: str | None = None, sitemap: str | None = None, page: str | None = None):
    http = MagicMock()
    bodies = {
        "https://www.goldsmiths.co.uk/robots.txt": _Response(
            f"User-agent: *\nSitemap: {INDEX_URL}\n"
        ),
        INDEX_URL: _Response(index or _index()),
        PRODUCT_SITEMAP: _Response(sitemap or _sitemap()),
        PRODUCT_URL: _Response(page or _product_page()),
    }
    http.get.side_effect = lambda url, **_kwargs: bodies[url]
    return http


def _discover(http, policy, access):
    return discover_goldsmiths(
        http,
        policy,
        access,
        now=NOW,
        clock=lambda: 0.0,
        sleeper=lambda _seconds: None,
    )


def test_discovers_physical_cpo_unit_and_all_unique_image_links():
    policy, access = _registry()

    page = _discover(_http(), policy, access)

    assert not policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id == "406107907490"
    assert record.seller_group_id == "wos-group-cpo"
    assert record.declared_reference == "rolex-submariner-126610ln"
    assert len(record.image_urls) == 3
    assert record.image_urls[-1].endswith("406107907490_3.png?impolicy=lister")
    assert dict(record.metadata) == {
        "classification": "original_candidate_cpo_certified",
        "evidence_tier": "a",
        "physical_unit_id": "406107907490",
        "authenticity_claim": "Rolex Certified Pre-Owned",
        "rights_status": "metadata_only",
    }


def test_fetches_pages_but_never_fetches_image_bytes():
    policy, access = _registry()
    http = _http()

    _discover(http, policy, access)

    requested = [call.args[0] for call in http.get.call_args_list]
    assert requested == [
        "https://www.goldsmiths.co.uk/robots.txt",
        INDEX_URL,
        PRODUCT_SITEMAP,
        PRODUCT_URL,
    ]
    assert all("content.thewosgroup.com" not in url for url in requested)


def test_non_cpo_and_near_reference_products_are_not_mislabeled():
    policy, access = _registry()
    near = PRODUCT_URL.replace("126610LN", "126610LN999")

    page = _discover(_http(sitemap=_sitemap(near)), policy, access)

    assert page.records == ()


def test_unapproved_sitemap_and_product_hosts_fail_closed():
    policy, access = _registry()
    with pytest.raises(GoldsmithsAPIError, match="approved host"):
        _discover(_http(index=_index("https://evil.example/product.xml")), policy, access)

    evil_product = PRODUCT_URL.replace("www.goldsmiths.co.uk", "evil.example")
    with pytest.raises(GoldsmithsAPIError, match="approved host"):
        _discover(_http(sitemap=_sitemap(evil_product)), policy, access)


def test_product_page_must_repeat_exact_cpo_claim_and_canonical_url():
    policy, access = _registry()
    with pytest.raises(GoldsmithsAPIError, match="CPO declaration"):
        _discover(_http(page=_product_page(title="Rolex Submariner 126610LN")), policy, access)

    with pytest.raises(GoldsmithsAPIError, match="canonical"):
        _discover(_http(page=_product_page().replace(PRODUCT_URL, "https://example.test")), policy, access)


def test_exact_m_prefixed_reference_title_is_accepted():
    policy, access = _registry()
    page = _product_page(
        title="Rolex Certified Pre-Owned Submariner Date M126610LN-0001 | Goldsmiths"
    )

    assert len(_discover(_http(page=page), policy, access).records) == 1


def test_oversized_response_stops_stream_and_closes(monkeypatch):
    policy, access = _registry()
    response = _Response("")
    chunks = iter((b"123456789", b"must-not-be-consumed"))
    response.read1 = MagicMock(side_effect=lambda *_args, **_kwargs: next(chunks, b""))
    http = MagicMock()
    http.get.return_value = response
    monkeypatch.setattr("scrapers.goldsmiths._MAX_ROBOTS_BYTES", 8)

    with pytest.raises(GoldsmithsAPIError, match="too large"):
        _discover(http, policy, access)

    assert response.closed
    assert next(chunks) == b"must-not-be-consumed"


def test_product_inventory_above_job_budget_fails_before_product_page(monkeypatch):
    policy, access = _registry()
    http = _http()
    monkeypatch.setattr("scrapers.goldsmiths._MAX_EXACT_PRODUCTS", 0)

    with pytest.raises(GoldsmithsAPIError, match="inventory is invalid"):
        _discover(http, policy, access)

    assert PRODUCT_URL not in [call.args[0] for call in http.get.call_args_list]


def test_deadline_checkpoint_persists_and_resumes_without_repeating_product(cos):
    policy, access = _registry()
    second_url = PRODUCT_URL.replace(
        "Date-126610LN/p/406107907490", "Date-M126610LN/p/406107907489"
    )
    sitemap = _sitemap().replace(
        "</urlset>", f"  <url><loc>{second_url}</loc></url>\n</urlset>"
    )
    pages = []
    requests_by_run = []

    def discover(previous, deadline):
        elapsed = [0.0]
        bodies = {
            "https://www.goldsmiths.co.uk/robots.txt": _Response(
                f"User-agent: *\nSitemap: {INDEX_URL}\n"
            ),
            INDEX_URL: _Response(_index()),
            PRODUCT_SITEMAP: _Response(sitemap),
            PRODUCT_URL: _Response(_product_page()),
            second_url: _Response(_product_page(product_url=second_url)),
        }
        http = MagicMock()
        http.get.side_effect = lambda url, **_kwargs: bodies[url]
        page = discover_goldsmiths(
            http,
            policy,
            access,
            now=NOW,
            previous=previous,
            clock=lambda: elapsed[0],
            sleeper=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
            deadline=deadline,
        )
        pages.append(page)
        requests_by_run.append([call.args[0] for call in http.get.call_args_list])
        return page

    first = run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "goldsmiths",
        now=NOW,
        discover=lambda previous: discover(previous, 38.0),
    )
    second = run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "goldsmiths",
        now=NOW,
        discover=lambda previous: discover(previous, 100.0),
    )

    assert first["state"] == "page_complete"
    assert second["state"] == "complete"
    assert [record.source_record_id for record in pages[0].records] == ["406107907489"]
    assert pages[0].next_cursor == "after:406107907489"
    assert pages[1].cursor == pages[0].next_cursor
    assert [record.source_record_id for record in pages[1].records] == ["406107907490"]
    assert second_url in requests_by_run[0]
    assert second_url not in requests_by_run[1]


def test_streaming_response_is_stopped_at_absolute_deadline():
    elapsed = [0.0]
    response = _Response("unused")
    consumed = []

    chunks = iter((b"first", b"second"))

    def slow_read(*_args, **_kwargs):
        elapsed[0] += 2
        chunk = next(chunks, b"")
        consumed.append(chunk)
        return chunk

    response.read1 = slow_read
    http = MagicMock()
    http.get.return_value = response
    fetcher = PacedBoundedHTTP(
        http,
        10,
        GoldsmithsAPIError,
        clock=lambda: elapsed[0],
        sleeper=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
    )

    with pytest.raises(HTTPDeadlineExceeded, match="deadline exhausted"):
        fetcher.get_bytes("https://example.test", 100, "slow", deadline=16.0)

    assert consumed == [b"first"]
    assert response.closed
