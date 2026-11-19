"""CBP collection must preserve source evidence without inventing references."""

from __future__ import annotations

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
    assert first.next_cursor == urls[4]
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
                               previous=SimpleNamespace(next_cursor="https://www.cbp.gov/newsroom/local-media-release/fake-watch-a"))
    assert resumed.records[0].source_url == remaining
    with pytest.raises(cbp.CBPDiscoveryError, match="resume cursor is invalid"):
        cbp.discover_cbp(object(), policy, access, now=NOW,
                         previous=SimpleNamespace(next_cursor="https://other.example/newsroom/fake-watch"))
    assert cbp._image_url("https://other.example/watch.jpg") is None
    assert cbp._image_url("//other.example/watch.jpg") is None


def test_cbp_refuses_unbounded_image_batch(monkeypatch):
    policy, access = _access()
    urls = [f"https://www.cbp.gov/newsroom/local-media-release/fake-watch-{i}" for i in range(5)]
    sitemap = "https://www.cbp.gov/sitemap.xml?page=1"
    images = "".join(f'<img src="/sites/default/files/watch-{i}.jpg">' for i in range(3))
    article = f'<div class="field--name-body">Counterfeit watch {images}</div>'.encode()
    reader = _Read({cbp._INDEX: _xml("sitemapindex", [sitemap]),
                    sitemap: _xml("urlset", urls), **dict.fromkeys(urls, article)})
    monkeypatch.setattr(cbp, "PacedBoundedHTTP", lambda *_args: reader)
    with pytest.raises(cbp.CBPDiscoveryError, match="exceeds ten images"):
        cbp.discover_cbp(object(), policy, access, now=NOW)
