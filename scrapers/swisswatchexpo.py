"""Seller-claimed watch metadata from SwissWatchExpo's public product sitemaps."""

from __future__ import annotations

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryCommit, DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.metadata_http import PacedBoundedHTTP

_HOST = "www.swisswatchexpo.com"
_CDN = "cdn.swisswatchexpo.com"
_ROOT = f"https://{_HOST}"
_INDEX = f"{_ROOT}/sitemap.xml"
_SITEMAP = re.compile(r"/sitemaps/(?:in-stock|out-of-stock)-product-detail-sitemap-[1-9][0-9]*\.xml$")
_STOCK = re.compile(r"-([0-9]{4,8})/$")
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
       "image": "http://www.google.com/schemas/sitemap-image/1.1"}
_REFERENCES = (
    (re.compile(r"(?<![a-z0-9])wssa0018(?![a-z0-9])"), "cartier-santos-wssa0018"),
    (re.compile(r"(?<![a-z0-9])wssa0061(?![a-z0-9])"), "cartier-santos-wssa0061"),
    (re.compile(r"(?<![a-z0-9])126334(?![a-z0-9])"), "rolex-datejust-126334"),
    (re.compile(r"(?<![a-z0-9])(?:31030425001001|310-30-42-50-01-001)(?![a-z0-9])"),
     "omega-speedmaster-31030425001001"),
    (re.compile(r"(?<![a-z0-9])(?:21030422001001|210-30-42-20-01-001)(?![a-z0-9])"),
     "omega-seamaster-21030422001001"),
)


class SwissWatchExpoDiscoveryError(RuntimeError):
    """A public sitemap changed or cannot be read within the approved bounds."""


def discover_swisswatchexpo(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: DiscoveryCommit | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
    deadline: float | None = None,
) -> DiscoveryPage:
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None or access.allowed_image_hosts != (_CDN,):
        raise PermissionError("approved image host is missing")
    if access.requests_per_minute is None:
        raise PermissionError("source request limit is missing")
    fetcher = PacedBoundedHTTP(http_client, access.requests_per_minute,
                               SwissWatchExpoDiscoveryError, clock=clock, sleeper=sleeper)
    robots = fetcher.get_bytes(f"{_ROOT}/robots.txt", 256 * 1024, "robots.txt", deadline=deadline)
    if _INDEX not in robots.decode("utf-8", errors="strict"):
        raise SwissWatchExpoDiscoveryError("robots.txt does not advertise sitemap")
    index = _xml(fetcher.get_bytes(_INDEX, 256 * 1024, "sitemap index", deadline=deadline))
    sitemap_urls = []
    for element in index.findall("sm:sitemap", _NS):
        url = _text(element.find("sm:loc", _NS), "sitemap URL")
        if _SITEMAP.fullmatch(_approved(url, _HOST).path):
            sitemap_urls.append(url)
    if not 1 <= len(sitemap_urls) <= 16 or len(sitemap_urls) != len(set(sitemap_urls)):
        raise SwissWatchExpoDiscoveryError("product sitemap inventory is invalid")
    cursor = previous.next_cursor if previous else None
    processed = tuple(cursor.split("|")) if cursor else ()
    if previous is not None and (previous.source != "swisswatchexpo"
                                 or previous.access_record_id != access.access_record_id
                                 or not processed or len(processed) > 16
                                 or len(processed) != len(set(processed))
                                 or any(_SITEMAP.fullmatch(_approved(item, _HOST).path) is None
                                        for item in processed)):
        raise SwissWatchExpoDiscoveryError("sitemap continuation cursor is invalid")
    remaining = [item for item in sitemap_urls if item not in processed]
    if not remaining:
        return DiscoveryPage(source="swisswatchexpo", access_record_id=access.access_record_id,
                             cursor=cursor, next_cursor=None, observed_at=now, records=())
    url = remaining[0]
    processed_after = (*processed, url)
    next_cursor = ("|".join(processed_after)
                   if any(item not in processed_after for item in sitemap_urls) else None)
    root = _xml(fetcher.get_bytes(url, 32 * 1024 * 1024, "product sitemap", deadline=deadline))
    records: dict[str, DiscoveryRecord] = {}
    seen = dict(previous.record_hashes) if previous else {}
    for entry in root.findall("sm:url", _NS):
        record = _parse_entry(entry)
        if record is None:
            continue
        if record.source_record_id in seen:
            record = replace(record, source_record_id=(
                record.source_record_id + "#" + hashlib.sha256(url.encode()).hexdigest()[:12]
            ))
        if record.source_record_id in seen:
            continue
        existing = records.get(record.source_record_id)
        if existing is None or (len(record.image_urls), record.source_url) > (
            len(existing.image_urls), existing.source_url
        ):
            records[record.source_record_id] = record
    return DiscoveryPage(source="swisswatchexpo", access_record_id=access.access_record_id,
                         cursor=cursor, next_cursor=next_cursor, observed_at=now,
                         records=tuple(records.values()))


def _parse_entry(entry: ET.Element) -> DiscoveryRecord | None:
    url = _text(entry.find("sm:loc", _NS), "product URL")
    path = _approved(url, _HOST).path
    match = _STOCK.search(path)
    if not path.startswith("/watches/") or match is None:
        return None
    slug = path.rstrip("/").rsplit("/", 1)[-1].lower()
    reference = next((value for pattern, value in _REFERENCES if pattern.search(slug)), None)
    if reference is None:
        return None
    urls = []
    for image in entry.findall("image:image", _NS):
        image_url = _text(image.find("image:loc", _NS), "image URL")
        if not _approved(image_url, _CDN).path.startswith("/productphotos/"):
            raise SwissWatchExpoDiscoveryError("image is outside approved CDN path")
        if image_url not in urls:
            urls.append(image_url)
    if not urls:
        return None
    lastmod = entry.findtext("sm:lastmod", default="", namespaces=_NS).strip()
    version = hashlib.sha256("\0".join((lastmod, url, *urls)).encode()).hexdigest()
    return DiscoveryRecord(
        source="swisswatchexpo", source_record_id=match.group(1),
        source_version=version, source_url=url, seller_group_id="swisswatchexpo",
        declared_reference=reference,
        image_ids=tuple(hashlib.sha256(image_url.encode()).hexdigest() for image_url in urls),
        image_urls=tuple(urls),
        metadata=(("classification", "original_candidate_seller_claim"),
                  ("label_basis", "seller_listing"),
                  ("rights_status", "metadata_only"),
                  ("independent_confirmation", "not_established"),
                  ("physical_stock_id", match.group(1)),
                  ("last_modified", lastmod or "unknown")),
    )


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise SwissWatchExpoDiscoveryError("sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise SwissWatchExpoDiscoveryError(f"{field} is missing")
    return value


def _approved(url: str, host: str):
    try:
        parsed = urlsplit(url)
        authority = (parsed.scheme, parsed.hostname, parsed.port,
                     parsed.username, parsed.password, parsed.query, parsed.fragment)
    except ValueError as exc:
        raise SwissWatchExpoDiscoveryError("URL is invalid") from exc
    if authority != ("https", host, None, None, None, "", ""):
        raise SwissWatchExpoDiscoveryError("URL is outside approved public host")
    return parsed
