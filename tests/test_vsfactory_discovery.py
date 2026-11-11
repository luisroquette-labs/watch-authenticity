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
from scrapers.vsfactoryusa import VSFactoryAPIError, discover_vsfactoryusa

NOW = datetime(2026, 11, 11, 16, 0, tzinfo=timezone.utc)


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
    return get_source_policy(policies, "vsfactoryusa"), access["vsfactoryusa"]


def _sitemap(*, product_host: str = "vsfactoryusa.com", image_host: str = "vsfactoryusa.com"):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">
  <url>
    <loc>https://{product_host}/vs-factory-rolex-submariner-date-black-41mm-126610ln-super-clone/</loc>
    <lastmod>2026-11-09</lastmod>
    <image:image>
      <image:loc>https://{image_host}/images/126610ln-01.webp</image:loc>
      <image:title>Rolex Submariner Date 126610LN Super Clone</image:title>
    </image:image>
    <image:image>
      <image:loc>https://{image_host}/images/126610ln-02.webp</image:loc>
      <image:title>Rolex Submariner Date 126610LN Super Clone</image:title>
    </image:image>
  </url>
  <url>
    <loc>https://vsfactoryusa.com/vs-factory-rolex-submariner-116610ln-super-clone/</loc>
    <lastmod>2026-11-09</lastmod>
    <image:image><image:loc>https://vsfactoryusa.com/images/116610ln.webp</image:loc></image:image>
  </url>
</urlset>"""


def test_discovers_exact_replica_reference_as_candidate_only():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response(_sitemap())

    page = discover_vsfactoryusa(http, policy, access, now=NOW)

    assert not policy.may_download
    assert not policy.may_train
    assert len(page.records) == 1
    record = page.records[0]
    assert record.source_record_id.endswith("126610ln-super-clone")
    assert record.declared_reference == "rolex-submariner-126610ln"
    assert record.image_urls == (
        "https://vsfactoryusa.com/images/126610ln-01.webp",
        "https://vsfactoryusa.com/images/126610ln-02.webp",
    )
    assert dict(record.metadata) == {
        "classification": "replica_candidate_vendor_declared",
        "evidence_tier": "c",
        "factory_claim": "VS Factory (VSF)",
        "last_modified": "2026-11-09",
        "rights_status": "metadata_only",
    }
    http.get.assert_called_once_with(
        "https://vsfactoryusa.com/sitemap.xml",
        headers={"User-Agent": "watch-authenticity/0.1 (research collector)"},
        timeout=30,
        allow_redirects=False,
        stream=True,
    )


@pytest.mark.parametrize(
    ("product_host", "image_host"),
    [("evil.example", "vsfactoryusa.com"), ("vsfactoryusa.com", "evil.example")],
)
def test_unapproved_hosts_fail_closed(product_host, image_host):
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response(
        _sitemap(product_host=product_host, image_host=image_host)
    )

    with pytest.raises(VSFactoryAPIError, match="approved host"):
        discover_vsfactoryusa(http, policy, access, now=NOW)


def test_rate_limit_is_recoverable():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response("", status_code=429)

    with pytest.raises(VSFactoryAPIError, match="resume later"):
        discover_vsfactoryusa(http, policy, access, now=NOW)


def test_reference_near_match_is_not_mislabeled():
    policy, access = _registry()
    http = MagicMock()
    http.get.return_value = _Response(_sitemap().replace("126610ln", "126610ln999"))

    with pytest.raises(VSFactoryAPIError, match="no exact-reference"):
        discover_vsfactoryusa(http, policy, access, now=NOW)


def test_oversized_response_stops_stream_and_closes(monkeypatch):
    policy, access = _registry()
    response = _Response("")
    chunks = iter((b"123456789", b"must-not-be-consumed"))
    response.iter_content = MagicMock(return_value=chunks)
    http = MagicMock()
    http.get.return_value = response
    monkeypatch.setattr("scrapers.vsfactoryusa._MAX_SITEMAP_BYTES", 8)

    with pytest.raises(VSFactoryAPIError, match="too large"):
        discover_vsfactoryusa(http, policy, access, now=NOW)

    assert response.closed
    assert next(chunks) == b"must-not-be-consumed"
