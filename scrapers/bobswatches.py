"""Metadata-only discovery of seller-claimed Rolex watches from public sitemaps."""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.metadata_http import PacedBoundedHTTP

_HOST = "www.bobswatches.com"
_ROBOTS = f"https://{_HOST}/robots.txt"
_INDEX = f"https://{_HOST}/sitemap.xml"
_SITEMAP_PATH = re.compile(r"/products-[1-9][0-9]*-sitemap\.xml$")
_PRODUCT_PATH = re.compile(r"/[^/]+\.html$")
_REFERENCE = re.compile(r"(?<![A-Z0-9])126610LN(?![A-Z0-9])", re.IGNORECASE)
_SKU = re.compile(r"SKU([0-9]{5,8})(?![0-9])", re.IGNORECASE)
_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}


class BobsWatchesDiscoveryError(RuntimeError):
    """A public metadata surface changed or cannot be read safely."""


def discover_bobswatches(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: object | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    deadline: float | None = None,
) -> DiscoveryPage:
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None or access.allowed_image_hosts != (_HOST,):
        raise PermissionError("approved source image host is missing")
    if access.requests_per_minute is None:
        raise PermissionError("source request limit is missing")
    if previous is not None:
        raise BobsWatchesDiscoveryError("sitemap discovery has no continuation cursor")
    fetcher = PacedBoundedHTTP(
        http_client, access.requests_per_minute, BobsWatchesDiscoveryError,
        clock=clock, sleeper=sleeper,
    )
    robots = fetcher.get_bytes(_ROBOTS, 256 * 1024, "robots.txt", deadline=deadline)
    if _INDEX not in robots.decode("utf-8", errors="strict"):
        raise BobsWatchesDiscoveryError("robots.txt does not advertise the sitemap")
    index = _xml(fetcher.get_bytes(_INDEX, 1024 * 1024, "sitemap index", deadline=deadline))
    sitemap_urls: list[str] = []
    for entry in index.findall("sm:sitemap", _NS):
        url = _text(entry.find("sm:loc", _NS), "sitemap URL")
        if _SITEMAP_PATH.fullmatch(_approved_url(url).path):
            sitemap_urls.append(url)
    if not sitemap_urls or len(sitemap_urls) > 6 or len(sitemap_urls) != len(set(sitemap_urls)):
        raise BobsWatchesDiscoveryError("product sitemap inventory is missing or invalid")

    records: dict[str, DiscoveryRecord] = {}
    for url in sitemap_urls:
        sitemap = _xml(fetcher.get_bytes(url, 4 * 1024 * 1024, "product sitemap", deadline=deadline))
        for entry in sitemap.findall("sm:url", _NS):
            record = _parse_entry(entry)
            if record is None:
                continue
            if record.source_record_id in records:
                raise BobsWatchesDiscoveryError("SKU appears in multiple product entries")
            records[record.source_record_id] = record
    return DiscoveryPage(
        source="bobswatches", access_record_id=access.access_record_id,
        cursor=None, next_cursor=None, observed_at=now,
        records=tuple(records.values()),
    )


def _parse_entry(entry: ET.Element) -> DiscoveryRecord | None:
    source_url = _text(entry.find("sm:loc", _NS), "product URL")
    path = _approved_url(source_url).path
    if _PRODUCT_PATH.fullmatch(path) is None:
        return None
    images = entry.findall("image:image", _NS)
    if len(images) != 1:
        return None
    title = images[0].findtext("image:title", default="", namespaces=_NS).strip()
    if _REFERENCE.search(title) is None or "submariner" not in title.lower():
        return None
    image_url = _text(images[0].find("image:loc", _NS), "image URL")
    if not _approved_url(image_url).path.startswith("/images/"):
        raise BobsWatchesDiscoveryError("product image is outside the approved image path")
    sku = _SKU.search(image_url)
    if sku is None:
        return None
    lastmod = entry.findtext("sm:lastmod", default="", namespaces=_NS).strip()
    version = hashlib.sha256("\0".join((lastmod, title, image_url)).encode()).hexdigest()
    metadata: tuple[tuple[str, str], ...] = (("classification", "original_candidate_seller_claim"),
                ("label_basis", "sitemap_title_and_seller_claim"),
                ("rights_status", "metadata_only"),
                ("image_title", title),
                ("independent_confirmation", "not_established"))
    if lastmod:
        metadata += (("last_modified", lastmod),)
    return DiscoveryRecord(
        source="bobswatches", source_record_id=sku.group(1),
        source_version=version, source_url=source_url,
        seller_group_id="bobswatches", declared_reference="rolex-submariner-126610ln",
        image_ids=(hashlib.sha256(image_url.encode()).hexdigest(),),
        image_urls=(image_url,),
        metadata=metadata,
    )


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise BobsWatchesDiscoveryError("sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise BobsWatchesDiscoveryError(f"{field} is missing")
    return value


def _approved_url(url: str):
    try:
        parsed = urlsplit(url)
        authority = (parsed.scheme, parsed.hostname, parsed.port,
                     parsed.username, parsed.password, parsed.query, parsed.fragment)
    except ValueError as exc:
        raise BobsWatchesDiscoveryError("URL is invalid") from exc
    if authority != ("https", _HOST, None, None, None, "", ""):
        raise BobsWatchesDiscoveryError("URL is outside the approved public host")
    return parsed
