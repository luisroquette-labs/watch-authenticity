"""CBP collection must preserve source evidence without inventing references."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from acquisition.source_policy import load_source_access_registry, load_source_policies
from scrapers import cbp

NOW = datetime(2026, 11, 19, 5, tzinfo=timezone.utc)
NS = "http://www.sitemaps.org/schemas/sitemap/0.9"


def _xml(root: str, urls: list[str]) -> bytes:
    child = "sitemap" if root == "sitemapindex" else "url"
    return (
        f'<{root} xmlns="{NS}">' +
        "".join(f"<{child}><loc>{url}</loc></{child}>" for url in urls) +
        f"</{root}>"
    ).encode()


class _Read:
    def __init__(self, bodies: dict[str, bytes]):
        self.bodies = bodies
        self.urls: list[str] = []

    def get_bytes(self, url, limit, _surface, *, deadline=None):
        self.urls.append(url)
        body = self.bodies[url]
        assert len(body) <= limit
        return body


def _access():
    policies = load_source_policies(Path("config/source_policies.json"))
    accesses = load_source_access_registry(Path("config/source_access.json"))
    return policies["cbp"], accesses["cbp"]


def test_cbp_discovers_five_articles_then_resumes_without_training_claim(monkeypatch):
    policy, access = _access()
    urls = [f"https://www.cbp.gov/newsroom/local-media-release/fake-watch-seized-{i}" for i in range(6)]
    sitemap = "https://www.cbp.gov/sitemap.xml?page=1"
    article = (
        '<div class="field--name-body"><p>Counterfeit watches were seized.</p>'
        '<div class="embedded-entity">'
        '<img src="/sites/default/files/styles/300px_newsroom_body_image/public/watch.jpg.webp?itok=x" alt="Watch">'
        '<img src="https://other.example/evil.jpg" alt="Watch"></div>'
        '<p><img src="/sites/default/files/other-watch.jpg" alt="Casio"></p>'
        '<div class="usa-card__container"><img src="/sites/default/files/unrelated.jpg" alt="Media contacts"></div></div>'
    ).encode()
    reader = _Read({cbp._INDEX: _xml("sitemapindex", [sitemap]),
                    sitemap: _xml("urlset", urls), **dict.fromkeys(urls, article)})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)

    first = cbp.discover_cbp(object(), policy, access, now=NOW)
    assert len(first.records) == 5
    assert json.loads(first.next_cursor) == {"page": 0, "after": urls[4], "offset": 0}
    assert all(record.image_urls == (
        "https://www.cbp.gov/sites/default/files/watch.jpg",
        "https://www.cbp.gov/sites/default/files/other-watch.jpg",
    )
               and record.declared_reference == "unverified" for record in first.records)
    assert not policy.may_train

    second = cbp.discover_cbp(
        object(), policy, access, now=NOW,
        previous=SimpleNamespace(next_cursor=first.next_cursor),
    )
    assert len(second.records) == 1 and second.next_cursor is None
    assert second.records[0].source_url == urls[5]


def test_cbp_resumes_after_sitemap_deletion_and_rejects_foreign_cursor(monkeypatch):
    policy, access = _access()
    sitemap = "https://www.cbp.gov/sitemap.xml?page=1"
    remaining = "https://www.cbp.gov/newsroom/local-media-release/fake-watch-z"
    reader = _Read({cbp._INDEX: _xml("sitemapindex", [sitemap]),
                    sitemap: _xml("urlset", [remaining]),
                    remaining: b'<div class="field--name-body">Counterfeit watch<img src="/sites/default/files/watch.jpg"></div>'})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)
    resumed = cbp.discover_cbp(object(), policy, access, now=NOW,
                               previous=SimpleNamespace(next_cursor=json.dumps({"page": 0, "after": "https://www.cbp.gov/newsroom/local-media-release/fake-watch-a", "offset": 0})))
    assert resumed.records[0].source_url == remaining
    with pytest.raises(cbp.CBPDiscoveryError, match="resume cursor is invalid"):
        cbp.discover_cbp(object(), policy, access, now=NOW,
                         previous=SimpleNamespace(next_cursor=json.dumps({"page": 0, "after": "https://other.example/newsroom/fake-watch", "offset": 0})))
    assert cbp._image_url("https://other.example/watch.jpg") is None
    assert cbp._image_url("//other.example/watch.jpg") is None


def test_cbp_splits_large_image_batch_without_losing_articles(monkeypatch):
    policy, access = _access()
    urls = [f"https://www.cbp.gov/newsroom/local-media-release/fake-watch-{i}" for i in range(5)]
    sitemap = "https://www.cbp.gov/sitemap.xml?page=1"
    images = "".join(f'<img src="/sites/default/files/watch-{i}.jpg">' for i in range(3))
    article = f'<div class="field--name-body">Counterfeit watch {images}</div>'.encode()
    reader = _Read({cbp._INDEX: _xml("sitemapindex", [sitemap]),
                    sitemap: _xml("urlset", urls), **dict.fromkeys(urls, article)})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)
    first = cbp.discover_cbp(object(), policy, access, now=NOW)
    assert sum(len(record.image_urls) for record in first.records) == 10
    assert json.loads(first.next_cursor) == {"page": 0, "after": urls[3], "offset": 1}
    second = cbp.discover_cbp(object(), policy, access, now=NOW,
                              previous=SimpleNamespace(next_cursor=first.next_cursor))
    assert sum(len(record.image_urls) for record in second.records) == 5
    assert second.next_cursor is None
    assert {record.source_record_id for record in first.records}.isdisjoint(
        record.source_record_id for record in second.records
    )


def test_cbp_scans_one_sitemap_page_per_run(monkeypatch):
    policy, access = _access()
    pages = [f"https://www.cbp.gov/sitemap.xml?page={i}" for i in range(1, 4)]
    reader = _Read({cbp._INDEX: _xml("sitemapindex", pages),
                    **{page: _xml("urlset", []) for page in pages}})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)
    first = cbp.discover_cbp(object(), policy, access, now=NOW)
    assert json.loads(first.next_cursor)["page"] == 1
    assert reader.urls == [cbp._INDEX, pages[0]]
    second = cbp.discover_cbp(object(), policy, access, now=NOW,
                              previous=SimpleNamespace(next_cursor=first.next_cursor))
    assert json.loads(second.next_cursor)["page"] == 2
    assert reader.urls[-2:] == [cbp._INDEX, pages[1]]
    shrunk = _Read({cbp._INDEX: _xml("sitemapindex", pages[:1])})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: shrunk)
    terminal = cbp.discover_cbp(object(), policy, access, now=NOW,
                                previous=SimpleNamespace(next_cursor=second.next_cursor))
    assert terminal.records == () and terminal.next_cursor is None


def test_cbp_advances_after_five_examined_articles_without_images(monkeypatch):
    policy, access = _access()
    urls = [f"https://www.cbp.gov/newsroom/local-media-release/fake-watch-{i}" for i in range(6)]
    sitemap = "https://www.cbp.gov/sitemap.xml?page=1"
    reader = _Read({cbp._INDEX: _xml("sitemapindex", [sitemap]),
                    sitemap: _xml("urlset", urls),
                    **dict.fromkeys(urls, b'<div class="field--name-body">Counterfeit watch</div>')})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)
    first = cbp.discover_cbp(object(), policy, access, now=NOW)
    assert first.records == ()
    assert json.loads(first.next_cursor) == {"page": 0, "after": urls[4], "offset": 0}
    assert reader.urls == [cbp._INDEX, sitemap, *urls[:5]]
    second = cbp.discover_cbp(object(), policy, access, now=NOW,
                              previous=SimpleNamespace(next_cursor=first.next_cursor))
    assert second.records == () and second.next_cursor is None
    assert reader.urls[-3:] == [cbp._INDEX, sitemap, urls[5]]
