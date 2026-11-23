from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from acquisition.source_policy import load_source_access_registry, load_source_policies
from scrapers.replica_sitemap import ReplicaSitemapError
from scrapers.richardmillereplica import SITEMAP, discover_richardmillereplica

NOW = datetime(2026, 11, 23, 13, 0, tzinfo=timezone.utc)
PRODUCT = "https://www.richardmillereplica.io/products/richard-mille-rm-88-smiley-replica"
IMAGE = "https://cdn.gfimage.com/product/30/image/watch.jpg"


class Response:
    status_code = 200

    def __init__(self, body: str):
        self.body = body.encode()
        self.closed = False

    def iter_content(self, chunk_size: int):
        yield self.body

    def close(self):
        self.closed = True


class Client:
    def __init__(self, product: str, *, second: bool = False):
        self.calls: list[str] = []
        second_url = PRODUCT + "-blue"
        self.responses = {
            SITEMAP: Response(
                '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                f"<url><loc>{PRODUCT}</loc></url>"
                + (f"<url><loc>{second_url}</loc></url>" if second else "")
                + "</urlset>"
            ),
            PRODUCT: Response(product),
        }
        if second:
            self.responses[second_url] = Response(
                product.replace(PRODUCT, second_url)
            )

    def get(self, url: str, **kwargs):
        assert kwargs["allow_redirects"] is False
        self.calls.append(url)
        return self.responses[url]


def _access():
    return (
        load_source_policies(Path("config/source_policies.json"))["richardmillereplica"],
        load_source_access_registry(Path("config/source_access.json"))["richardmillereplica"],
    )


def _product(image: str = IMAGE, title: str = "Richard Mille RM 88 Smiley Replica") -> str:
    return (
        "<html><head>"
        f"<title>{title}</title><link rel='canonical' href='{PRODUCT}'>"
        "</head><body><img src='https://evil.example/outside.jpg'>"
        f"<media-gallery><div data-media-src='{image}'></div>"
        f"<div data-media-src='{image}'></div></media-gallery></body></html>"
    )


def test_discovers_only_product_gallery_with_weak_vendor_label():
    policy, access = _access()
    client = Client(_product())
    pauses: list[float] = []

    page = discover_richardmillereplica(
        client, policy, access, now=NOW, sleep=pauses.append
    )

    assert not policy.may_train
    assert client.calls == [SITEMAP, PRODUCT]
    assert all(response.closed for response in client.responses.values())
    assert pauses == [10.0, 10.0]
    assert len(page.records) == 1
    record = page.records[0]
    assert record.image_urls == (IMAGE,)
    assert record.declared_reference == "richardmille-rm88-smiley"
    assert dict(record.metadata)["classification"] == "replica_candidate_vendor_declared"
    assert dict(record.metadata)["evidence_tier"] == "c"


def test_one_product_per_run_resumes_from_committed_cursor():
    policy, access = _access()
    client = Client(_product(), second=True)
    first = discover_richardmillereplica(
        client, policy, access, now=NOW, sleep=lambda _: None
    )
    assert len(first.records) == 1
    assert first.next_cursor == PRODUCT
    assert client.calls == [SITEMAP, PRODUCT]

    client.calls.clear()
    second = discover_richardmillereplica(
        client, policy, access, now=NOW,
        previous=SimpleNamespace(next_cursor=first.next_cursor), sleep=lambda _: None,
    )
    assert second.cursor == first.next_cursor
    assert second.next_cursor is None
    assert client.calls == [SITEMAP, PRODUCT + "-blue"]


def test_removed_pending_product_does_not_strand_cursor():
    policy, access = _access()
    client = Client(_product(), second=True)
    first = discover_richardmillereplica(
        client, policy, access, now=NOW, sleep=lambda _: None
    )
    client.responses[SITEMAP] = Response(
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{PRODUCT}</loc></url></urlset>"
    )

    finished = discover_richardmillereplica(
        client, policy, access, now=NOW,
        previous=SimpleNamespace(next_cursor=first.next_cursor), sleep=lambda _: None,
    )

    assert finished.cursor == PRODUCT
    assert finished.next_cursor is None
    assert finished.records == ()


@pytest.mark.parametrize(
    "product",
    [
        _product(image="https://evil.example/product/watch.jpg"),
        _product(title="Richard Mille RM 88 Smiley Original"),
        _product(image="https://cdn.gfimage.com/common/logo.jpg"),
    ],
)
def test_rejects_unsafe_gallery_or_missing_replica_declaration(product):
    policy, access = _access()
    with pytest.raises(ReplicaSitemapError):
        discover_richardmillereplica(
            Client(product), policy, access, now=NOW, sleep=lambda _: None
        )
