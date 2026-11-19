"""Bounded CBP sitemap discovery; seizure labels remain source evidence."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryCommit, DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.metadata_http import PacedBoundedHTTP

_HOST = "www.cbp.gov"
_INDEX = "https://www.cbp.gov/sitemap.xml"
_MAX_ARTICLES = 5
_WATCH = re.compile(r"watch|rolex", re.I)
_SEIZURE = re.compile(r"counterfeit|fake|knockoff|seiz", re.I)


class CBPDiscoveryError(RuntimeError):
    pass


class _BodyImages(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.stopped = False
        self.images: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "div":
            if self.depth:
                self.depth += 1
                if {"usa-card__container", "usa-footer"} & set((values.get("class") or "").split()):
                    self.stopped = True
            elif "field--name-body" in (values.get("class") or "").split():
                self.depth = 1
        elif tag == "img" and self.depth and not self.stopped:
            self.images.append((values.get("src") or "", values.get("alt") or ""))

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self.depth:
            self.depth -= 1


def _xml_locations(body: bytes, root_name: str) -> list[str]:
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise CBPDiscoveryError("CBP sitemap is invalid XML") from exc
    if root.tag != f"{{http://www.sitemaps.org/schemas/sitemap/0.9}}{root_name}":
        raise CBPDiscoveryError("CBP sitemap has an unexpected root")
    return [node.text or "" for node in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]


def _image_url(src: str) -> str | None:
    parts = urlsplit(src)
    if (parts.scheme or parts.netloc) and (
        parts.scheme != "https" or parts.netloc != _HOST or parts.username or parts.password
    ):
        return None
    path = parts.path
    match = re.fullmatch(r"/sites/default/files/styles/[^/]+/public/(.+\.(?:jpg|jpeg|png))\.webp", path, re.I)
    if match:
        path = "/sites/default/files/" + match.group(1)
    elif not re.fullmatch(r"/sites/default/files/[^?]+\.(?:jpg|jpeg|png|webp)", path, re.I):
        return None
    return f"https://{_HOST}{path}"


def discover_cbp(
    http_client: object,
    policy: SourcePolicy,
    access: SourceAccessRecord,
    *,
    now: datetime,
    previous: DiscoveryCommit | None = None,
    deadline: float | None = None,
) -> DiscoveryPage:
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access.allowed_image_hosts != (_HOST,) or access.requests_per_minute is None:
        raise PermissionError("CBP access boundary is invalid")
    read = PacedBoundedHTTP(http_client, access.requests_per_minute, CBPDiscoveryError)
    index = _xml_locations(read.get_bytes(_INDEX, 64 * 1024, "CBP sitemap index", deadline=deadline), "sitemapindex")
    if not 1 <= len(index) <= 100 or any(
        not re.fullmatch(r"https://www\.cbp\.gov/sitemap\.xml\?page=[1-9][0-9]*", url)
        for url in index
    ):
        raise CBPDiscoveryError("CBP sitemap page inventory is invalid")
    urls: set[str] = set()
    for page_url in index:
        for url in _xml_locations(
            read.get_bytes(page_url, 2 * 1024 * 1024, "CBP sitemap page", deadline=deadline),
            "urlset",
        ):
            if (
                re.fullmatch(r"https://www\.cbp\.gov/newsroom/[a-z0-9/_-]+", url)
                and _WATCH.search(url)
                and _SEIZURE.search(url)
            ):
                urls.add(url)
    cursor = previous.next_cursor if previous else None
    if cursor is not None and not re.fullmatch(
        r"https://www\.cbp\.gov/newsroom/[a-z0-9/_-]+", cursor
    ):
        raise CBPDiscoveryError("CBP resume cursor is invalid")
    pending = [url for url in sorted(urls) if cursor is None or url > cursor]
    selected = pending[:_MAX_ARTICLES]
    records: list[DiscoveryRecord] = []
    for url in selected:
        body = read.get_bytes(url, 2 * 1024 * 1024, "CBP article", deadline=deadline)
        text = body.decode("utf-8", errors="replace")
        if not _SEIZURE.search(text):
            continue
        parser = _BodyImages()
        parser.feed(text)
        images = tuple(dict.fromkeys(
            image for src, _alt in parser.images if (image := _image_url(src)) is not None
        ))
        if not images:
            continue
        records.append(DiscoveryRecord(
            source="cbp",
            source_record_id="article-" + hashlib.sha256(url.encode()).hexdigest()[:24],
            source_version=hashlib.sha256(body).hexdigest(),
            source_url=url,
            seller_group_id="article-" + hashlib.sha256(url.encode()).hexdigest()[:24],
            declared_reference="unverified",
            image_ids=tuple(hashlib.sha256(image.encode()).hexdigest() for image in images),
            image_urls=images,
            metadata=(("classification", "government_seizure_article"),
                      ("label_basis", "CBP article text; item linkage requires review")),
        ))
    if sum(len(record.image_urls) for record in records) > 10:
        raise CBPDiscoveryError("CBP article batch exceeds ten images")
    return DiscoveryPage(
        source="cbp",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=selected[-1] if len(pending) > len(selected) else None,
        observed_at=now,
        records=tuple(records),
    )
