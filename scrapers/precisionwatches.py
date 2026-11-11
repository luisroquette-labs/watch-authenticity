"""Metadata-only discovery for Precision Watches Rolex CPO watches."""

from __future__ import annotations

import hashlib
import html
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

_HOST = "precisionwatches.com"
_ROBOTS_URL = f"https://{_HOST}/robots.txt"
_SITEMAP_URL = f"https://{_HOST}/sitemap_index.xml"
_PRODUCT_SITEMAP_PATH = re.compile(r"/product-sitemap(?:[0-9]+)?\.xml$")
_PRODUCT_PATH = re.compile(
    r"/rolex-certified-pre-owned/rolex-submariner/"
    r"submariner-date-126610ln-black-cerachrom/$",
    re.IGNORECASE,
)
_UNIT_ID = re.compile(r"(?<![A-Z0-9])PA[0-9]+(?![A-Z0-9])")
_MAX_ROBOTS_BYTES = 256 * 1024
_MAX_INDEX_BYTES = 1024 * 1024
_MAX_PRODUCT_SITEMAP_BYTES = 8 * 1024 * 1024
_MAX_PRODUCT_PAGE_BYTES = 8 * 1024 * 1024
_MAX_PRODUCT_SITEMAPS = 4
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


class PrecisionWatchesAPIError(RuntimeError):
    """The public CPO discovery surface is unavailable or unsafe to commit."""


def discover_precisionwatches(
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
    """Discover the exact-reference CPO page and links without image downloads."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:
        raise PermissionError("source access evidence missing")
    if access.allowed_image_hosts != (_HOST,):
        raise PermissionError("source access image hosts do not match the approved host")
    if access.requests_per_minute is None:
        raise PermissionError("source access request limit is missing")
    if previous is not None:
        raise PrecisionWatchesAPIError("CPO sitemap discovery has no continuation cursor")

    fetcher = PacedBoundedHTTP(
        http_client,
        access.requests_per_minute,
        PrecisionWatchesAPIError,
        clock=clock,
        sleeper=sleeper,
    )
    robots = fetcher.get_bytes(
        _ROBOTS_URL, _MAX_ROBOTS_BYTES, "robots.txt", deadline=deadline
    )
    if f"Sitemap: {_SITEMAP_URL}" not in robots.decode("utf-8", errors="strict"):
        raise PrecisionWatchesAPIError("robots.txt does not advertise the approved sitemap")

    index = _xml(
        fetcher.get_bytes(
            _SITEMAP_URL, _MAX_INDEX_BYTES, "sitemap index", deadline=deadline
        )
    )
    records: list[DiscoveryRecord] = []
    for sitemap_url in _product_sitemap_urls(index):
        sitemap = _xml(
            fetcher.get_bytes(
                sitemap_url,
                _MAX_PRODUCT_SITEMAP_BYTES,
                "product sitemap",
                deadline=deadline,
            )
        )
        for entry in sitemap.findall("sm:url", _NS):
            product_url = _text(entry.find("sm:loc", _NS), "product URL")
            parsed = _approved_url(product_url)
            if _PRODUCT_PATH.fullmatch(parsed.path) is None:
                continue
            last_modified = _text(entry.find("sm:lastmod", _NS), "last modified")
            body = fetcher.get_bytes(
                product_url,
                _MAX_PRODUCT_PAGE_BYTES,
                "product page",
                deadline=deadline,
            )
            records.append(_parse_product(product_url, last_modified, body))
    if len(records) != len({record.source_record_id for record in records}):
        raise PrecisionWatchesAPIError("exact-reference product inventory is duplicated")
    return DiscoveryPage(
        source="precisionwatches",
        access_record_id=access.access_record_id,
        cursor=None,
        next_cursor=None,
        observed_at=now,
        records=tuple(records),
    )


def _product_sitemap_urls(root: ET.Element) -> tuple[str, ...]:
    urls: list[str] = []
    for sitemap in root.findall("sm:sitemap", _NS):
        url = _text(sitemap.find("sm:loc", _NS), "sitemap URL")
        parsed = _approved_url(url)
        if _PRODUCT_SITEMAP_PATH.fullmatch(parsed.path):
            urls.append(url)
    if not urls or len(urls) > _MAX_PRODUCT_SITEMAPS or len(urls) != len(set(urls)):
        raise PrecisionWatchesAPIError("product sitemap inventory is missing or invalid")
    return tuple(urls)


def _parse_product(source_url: str, last_modified: str, body: bytes) -> DiscoveryRecord:
    try:
        document = html.unescape(body.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise PrecisionWatchesAPIError("product page is not UTF-8") from exc
    title_match = re.search(r"<title>\s*([^<]+?)\s*</title>", document, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    if "Rolex Certified Pre-Owned" not in title or re.search(
        r"(?<![A-Z0-9])126610LN(?![A-Z0-9])", title, re.IGNORECASE
    ) is None:
        raise PrecisionWatchesAPIError("product page lacks the exact Rolex CPO declaration")
    canonical = re.search(
        r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']\s*/?>',
        document,
        re.IGNORECASE,
    )
    if canonical is None or canonical.group(1) != source_url:
        raise PrecisionWatchesAPIError("product page canonical URL is missing or invalid")

    page_images = re.findall(
        r'<img\b[^>]*\bsrc=["\'](https://[^"\'<>\s]+)["\']',
        document,
        re.IGNORECASE,
    )
    candidates = [url for url in page_images if _UNIT_ID.search(url)]
    images = tuple(dict.fromkeys(candidates))
    if not images:
        raise PrecisionWatchesAPIError("exact-reference product images are missing")
    unit_ids: set[str] = set()
    for image_url in images:
        _approved_url(image_url, image=True)
        unit_ids.update(_UNIT_ID.findall(image_url))
    if len(unit_ids) != 1:
        raise PrecisionWatchesAPIError("product images do not identify one physical watch")
    unit_id = unit_ids.pop()
    if any(unit_id not in image_url for image_url in images):
        raise PrecisionWatchesAPIError("product images do not identify one physical watch")

    version_input = "\0".join((last_modified, title, source_url, *images)).encode()
    return DiscoveryRecord(
        source="precisionwatches",
        source_record_id=unit_id,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id="precision-watches-cpo",
        declared_reference="rolex-submariner-126610ln",
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=images,
        metadata=(
            ("classification", "original_candidate_cpo_certified"),
            ("evidence_tier", "a"),
            ("physical_unit_id", unit_id),
            ("authenticity_claim", "Rolex Certified Pre-Owned"),
            ("last_modified", last_modified),
            ("rights_status", "metadata_only"),
        ),
    )


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise PrecisionWatchesAPIError("sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise PrecisionWatchesAPIError(f"{field} is missing")
    return value


def _approved_url(value: str, *, image: bool = False):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise PrecisionWatchesAPIError("URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or (image and not parsed.path.startswith("/wp-content/uploads/"))
    ):
        raise PrecisionWatchesAPIError("URL is not on the approved host")
    return parsed
