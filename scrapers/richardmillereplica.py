"""Discover vendor-declared RM 88 galleries from a public product sitemap."""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryCommit, DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.replica_sitemap import ReplicaSitemapError, _read_bounded

SOURCE = "richardmillereplica"
SITEMAP = "https://www.richardmillereplica.io/sitemap_products_1.xml"
_PRODUCT_HOST = "www.richardmillereplica.io"
_IMAGE_HOST = "cdn.gfimage.com"
_REFERENCE = "richardmille-rm88-smiley"
_RM88 = re.compile(r"\brm[\s-]?0?88\b", re.IGNORECASE)
_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class _ProductGallery(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.canonical = ""
        self.images: list[str] = []
        self._in_title = False
        self._gallery_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "title":
            self._in_title = True
        if tag == "link" and values.get("rel") == "canonical":
            self.canonical = values.get("href") or ""
        if tag == "media-gallery":
            self._gallery_depth += 1
        if self._gallery_depth and tag == "div" and values.get("data-media-src"):
            self.images.append(values["data-media-src"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag == "media-gallery" and self._gallery_depth:
            self._gallery_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def _approved_url(value: str, host: str, path_prefix: str) -> bool:
    try:
        url = urlsplit(value)
        return (
            url.scheme == "https"
            and url.hostname == host
            and url.username is None
            and url.password is None
            and url.port in {None, 443}
            and not url.query
            and not url.fragment
            and url.path.startswith(path_prefix)
        )
    except ValueError:
        return False


def _get(http_client: Any, url: str, *, limit: int) -> bytes:
    response = http_client.get(
        url,
        headers={"User-Agent": "watch-authenticity/0.1 (research collector)"},
        timeout=30,
        allow_redirects=False,
        stream=True,
    )
    try:
        if response.status_code == 429:
            raise ReplicaSitemapError(f"{SOURCE} rate limited; resume later")
        if response.status_code != 200:
            raise ReplicaSitemapError(f"{SOURCE} returned HTTP {response.status_code}")
        return _read_bounded(response, SOURCE, max_bytes=limit, label="source")
    finally:
        response.close()


def discover_richardmillereplica(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: DiscoveryCommit | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> DiscoveryPage:
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None or access.requests_per_minute is None:
        raise PermissionError("source rate limit missing")
    try:
        root = ET.fromstring(_get(http_client, SITEMAP, limit=5 * 1024 * 1024))
    except ET.ParseError as exc:
        raise ReplicaSitemapError(f"{SOURCE} sitemap is invalid XML") from exc
    urls: list[str] = []
    for entry in root.findall(f"{_NS}url"):
        loc = entry.find(f"{_NS}loc")
        url = loc.text.strip() if loc is not None and loc.text else ""
        if not _approved_url(url, _PRODUCT_HOST, "/products/"):
            raise ReplicaSitemapError(f"{SOURCE} sitemap contains an unapproved product URL")
        if _RM88.search(urlsplit(url).path):
            urls.append(url)
    urls = sorted(set(urls))
    cursor = previous.next_cursor if previous else None
    if not urls:
        raise ReplicaSitemapError(f"{SOURCE} sitemap has no RM 88 products")
    index = next((i for i, item in enumerate(urls) if cursor is None or item > cursor), len(urls))
    if index == len(urls):
        return DiscoveryPage(SOURCE, access.access_record_id, cursor, None, now, ())
    url = urls[index]
    delay = 60 / access.requests_per_minute
    sleep(delay)
    gallery = _ProductGallery()
    try:
        gallery.feed(_get(http_client, url, limit=2 * 1024 * 1024).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ReplicaSitemapError(f"{SOURCE} product page is not UTF-8") from exc
    title = gallery.title.strip()
    if gallery.canonical != url or not _RM88.search(title) or "replica" not in title.lower():
        raise ReplicaSitemapError(f"{SOURCE} product identity or replica declaration missing")
    images = tuple(dict.fromkeys(gallery.images))
    if not images or len(images) > 30 or any(
        not _approved_url(image, _IMAGE_HOST, "/product/") for image in images
    ):
        raise ReplicaSitemapError(f"{SOURCE} product gallery is missing or unsafe")
    digest = hashlib.sha256("\0".join((title, *images)).encode()).hexdigest()
    record = DiscoveryRecord(
        source=SOURCE,
        source_record_id=urlsplit(url).path.removeprefix("/products/"),
        source_version=digest,
        source_url=url,
        seller_group_id=SOURCE,
        declared_reference=_REFERENCE,
        image_ids=tuple(hashlib.sha256(image.encode()).hexdigest() for image in images),
        image_urls=images,
        metadata=(
            ("classification", "replica_candidate_vendor_declared"),
            ("evidence_tier", "c"),
            ("rights_status", "owner_private_download_only"),
        ),
    )
    sleep(delay)
    next_cursor = url if index + 1 < len(urls) else None
    return DiscoveryPage(SOURCE, access.access_record_id, cursor, next_cursor, now, (record,))
