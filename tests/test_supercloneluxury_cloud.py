"""Cloud discovery must keep seller claims weak and resume by API page."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from acquisition.run import run_public_image_collection_resilient
from acquisition.source_policy import load_source_access_registry, load_source_policies
from acquisition.supercloneluxury_archive import CatalogError
from scrapers.supercloneluxury import discover_supercloneluxury

NOW = datetime(2026, 11, 19, 5, tzinfo=timezone.utc)


class _Raw:
    def __init__(self, body: bytes) -> None:
        self._buffer = BytesIO(body)

    def read1(self, size: int, *, decode_content: bool = False) -> bytes:
        assert not decode_content
        return self._buffer.read(size)


class _Session:
    def __init__(self, products: list[dict[str, Any]], *, total: int | None = None):
        self.body = json.dumps(products).encode()
        self.total = len(products) if total is None else total
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: Any):
        assert kwargs["allow_redirects"] is False
        self.urls.append(url)
        return SimpleNamespace(
            status_code=200,
            headers={
                "X-WP-Total": str(self.total),
                "X-WP-TotalPages": str((self.total + 24) // 25),
            },
            raw=_Raw(self.body),
            close=lambda: None,
        )


def _product(product_id: int, *, replica: bool = True) -> dict[str, Any]:
    return {
        "id": product_id,
        "name": "Super Clone watch" if replica else "Watch",
        "permalink": f"https://supercloneluxury.com/product/watch-{product_id}/",
        "description": "",
        "images": [
            {"src": f"https://supercloneluxury.com/wp-content/uploads/watch-{product_id}.webp"}
        ],
    }


def _access():
    policies = load_source_policies(Path("config/source_policies.json"))
    accesses = load_source_access_registry(Path("config/source_access.json"))
    return policies["supercloneluxury"], accesses["supercloneluxury"]


def test_cloud_discovery_paginates_without_claiming_exact_reference_or_authenticity():
    policy, access = _access()
    first = _Session([_product(i) for i in range(1, 26)], total=26)
    page = discover_supercloneluxury(first, policy, access, now=NOW)

    assert page.next_cursor == "2"
    assert len(page.records) == 25
    assert all(record.declared_reference == "unverified" for record in page.records)
    assert dict(page.records[0].metadata)["classification"] == "vendor_declared_replica"
    assert first.urls == [
        "https://supercloneluxury.com/wp-json/wc/store/v1/products?per_page=25&page=1&order=asc"
    ]
    assert not policy.may_train

    last = _Session([_product(26, replica=False)], total=26)
    resumed = discover_supercloneluxury(
        last, policy, access, now=NOW, previous=SimpleNamespace(next_cursor="2")
    )
    assert resumed.cursor == "2" and resumed.next_cursor is None
    assert dict(resumed.records[0].metadata)["classification"] == "unlabeled"


def test_exact_multiple_of_page_size_finishes_without_invalid_extra_page():
    policy, access = _access()
    page = discover_supercloneluxury(
        _Session([_product(i) for i in range(1, 26)]), policy, access, now=NOW
    )
    assert page.next_cursor is None


def test_cloud_discovery_rejects_unapproved_image_host():
    policy, access = _access()
    product = _product(1)
    product["images"][0]["src"] = "https://other.example/watch.webp"
    with pytest.raises(CatalogError, match="outside the public uploads host"):
        discover_supercloneluxury(_Session([product]), policy, access, now=NOW)


def test_cloud_discovery_rejects_product_url_with_credentials():
    policy, access = _access()
    product = _product(1)
    product["permalink"] = "https://user:secret@supercloneluxury.com/product/watch-1/"
    with pytest.raises(CatalogError, match="outside the catalog"):
        discover_supercloneluxury(_Session([product]), policy, access, now=NOW)


def test_cloud_metadata_failure_is_persisted_without_advancing_cursor(cos):
    policy, access = _access()
    result = run_public_image_collection_resilient(
        cos,
        "private",
        "datasets",
        policy,
        access,
        now=NOW,
        discover=lambda _previous: (_ for _ in ()).throw(CatalogError("invalid page")),
    )

    assert result["state"] == "failed"
    assert result["error_type"] == "CatalogError"
    assert not any("/discovery/" in key for key in cos.bodies)
