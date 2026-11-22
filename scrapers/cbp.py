"""Bounded CBP sitemap discovery; seizure labels remain source evidence."""

from __future__ import annotations

import hashlib
import json
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
_MAX_EMPTY_SITEMAP_PAGES = 8
_WATCH = re.compile(r"watch|rolex|cartier|omega|patek|seiko|audemars|richard-mille", re.I)
_WATCH_CONTEXT = re.compile(r"\b(?:watches?|wristwatches?|timepieces?)\b", re.I)
_SEIZURE = re.compile(r"counterfeit|fake|knockoff|seiz", re.I)


class CBPDiscoveryError(RuntimeError):
    pass


class _BodyImages(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.stopped = False
        self.images: list[tuple[str, str]] = []
        self.body_text: list[str] = []

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

    def handle_data(self, data: str) -> None:
        if self.depth and not self.stopped:
            self.body_text.append(data)


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
    if not index or any(
        not re.fullmatch(r"https://www\.cbp\.gov/sitemap\.xml\?page=[1-9][0-9]*", url)
        for url in index
    ):
        raise CBPDiscoveryError("CBP sitemap page inventory is invalid")
    cursor = previous.next_cursor if previous else None
    try:
        position = json.loads(cursor) if cursor is not None else {"page": 0, "after": "", "offset": 0}
    except json.JSONDecodeError as exc:
        raise CBPDiscoveryError("CBP resume cursor is invalid") from exc
    if (
        not isinstance(position, dict)
        or set(position) != {"page", "after", "offset"}
        or not isinstance(position["page"], int)
        or isinstance(position["page"], bool)
        or not 0 <= position["page"] <= 1000
        or not isinstance(position["offset"], int)
        or isinstance(position["offset"], bool)
        or not 0 <= position["offset"] <= 1000
        or not isinstance(position["after"], str)
        or (position["after"] and not re.fullmatch(
            r"https://www\.cbp\.gov/newsroom/[a-z0-9/_-]+", position["after"]
        ))
        or (position["offset"] and not position["after"])
    ):
        raise CBPDiscoveryError("CBP resume cursor is invalid")
    page_index = position["page"]
    after = position["after"]
    offset = position["offset"]
    pending: list[str] = []
    empty_pages = 0
    while page_index < len(index) and empty_pages < _MAX_EMPTY_SITEMAP_PAGES:
        urls = sorted({url for url in _xml_locations(
            read.get_bytes(index[page_index], 2 * 1024 * 1024, "CBP sitemap page", deadline=deadline),
            "urlset",
        ) if (
            re.fullmatch(r"https://www\.cbp\.gov/newsroom/[a-z0-9/_-]+", url)
            and _WATCH.search(url)
            and _SEIZURE.search(url)
        )})
        pending = [url for url in urls if url > after or (offset and url == after)]
        if pending:
            break
        page_index += 1
        after = ""
        offset = 0
        empty_pages += 1
    if not pending:
        next_cursor = (json.dumps({"page": page_index, "after": "", "offset": 0}, separators=(",", ":"))
                       if page_index < len(index) else None)
        return DiscoveryPage("cbp", access.access_record_id, cursor, next_cursor, now, ())
    records: list[DiscoveryRecord] = []
    last_url = after
    next_offset = 0
    image_count = 0
    examined = 0
    for url in pending:
        if examined >= _MAX_ARTICLES or image_count >= 10:
            break
        examined += 1
        body = read.get_bytes(url, 2 * 1024 * 1024, "CBP article", deadline=deadline)
        text = body.decode("utf-8", errors="replace")
        if not _SEIZURE.search(text):
            last_url = url
            continue
        parser = _BodyImages()
        parser.feed(text)
        if not re.search(r"watch|rolex", url, re.I) and not _WATCH_CONTEXT.search(" ".join(parser.body_text)):
            last_url = url
            continue
        images = tuple(dict.fromkeys(
            image for src, _alt in parser.images if (image := _image_url(src)) is not None
        ))
        start = offset if url == after else 0
        if start >= len(images):
            last_url = url
            continue
        selected = images[start:start + 10 - image_count]
        next_offset = start + len(selected) if start + len(selected) < len(images) else 0
        image_count += len(selected)
        last_url = url
        records.append(DiscoveryRecord(
            source="cbp",
            source_record_id="article-" + hashlib.sha256(url.encode()).hexdigest()[:24] + f"-part-{start}",
            source_version=hashlib.sha256(body).hexdigest(),
            source_url=url,
            seller_group_id="article-" + hashlib.sha256(url.encode()).hexdigest()[:24],
            declared_reference="unverified",
            image_ids=tuple(hashlib.sha256(image.encode()).hexdigest() for image in selected),
            image_urls=selected,
            metadata=(("classification", "government_seizure_article"),
                      ("label_basis", "CBP article text; item linkage requires review")),
        ))
        if next_offset:
            break
    if next_offset or (pending and last_url != pending[-1]):
        next_cursor = json.dumps({"page": page_index, "after": last_url, "offset": next_offset}, separators=(",", ":"))
    elif page_index + 1 < len(index):
        next_cursor = json.dumps({"page": page_index + 1, "after": "", "offset": 0}, separators=(",", ":"))
    else:
        next_cursor = None
    return DiscoveryPage(
        source="cbp",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )
