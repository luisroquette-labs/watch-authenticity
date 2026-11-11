"""Metadata-only discovery for Goldsmiths Rolex CPO watches."""

from __future__ import annotations

import hashlib
import html
import json
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

_HOST = "www.goldsmiths.co.uk"
_IMAGE_HOST = "content.thewosgroup.com"
_ROBOTS_URL = f"https://{_HOST}/robots.txt"
_SITEMAP_URL = f"https://{_HOST}/sitemap.xml"
_PRODUCT_SITEMAP_PATH = re.compile(r"/sitemap/product-en-gbp-[0-9]+\.xml$")
_PRODUCT_PATH = re.compile(
    r"/Rolex-Certified-Pre\+Owned-Submariner-Date-M?126610LN(?:\+0001)?/p/"
    r"(?P<unit_id>[0-9]+)$",
    re.IGNORECASE,
)
_MAX_ROBOTS_BYTES = 256 * 1024
_MAX_INDEX_BYTES = 1024 * 1024
_MAX_PRODUCT_SITEMAP_BYTES = 16 * 1024 * 1024
_MAX_PRODUCT_PAGE_BYTES = 8 * 1024 * 1024
_MAX_PRODUCT_SITEMAPS = 8
_MAX_EXACT_PRODUCTS = 50
_SOURCE_BUDGET_SECONDS = 420
_CURSOR = re.compile(r"after:(?P<unit_id>[0-9]+)")
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


class GoldsmithsAPIError(RuntimeError):
    """The public CPO discovery surface is unavailable or unsafe to commit."""


def discover_goldsmiths(
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
    """Discover exact-reference CPO pages and image links without downloading images."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:
        raise PermissionError("source access evidence missing")
    if access.allowed_image_hosts != (_IMAGE_HOST,):
        raise PermissionError("source access image hosts do not match the approved host")
    if access.requests_per_minute is None:
        raise PermissionError("source access request limit is missing")
    cursor = getattr(previous, "next_cursor", None) if previous is not None else None
    cursor_match = _CURSOR.fullmatch(cursor) if cursor is not None else None
    if cursor is not None and cursor_match is None:
        raise GoldsmithsAPIError("CPO discovery continuation cursor is invalid")
    after_unit = cursor_match.group("unit_id") if cursor_match is not None else None
    source_deadline = deadline if deadline is not None else clock() + _SOURCE_BUDGET_SECONDS

    fetcher = PacedBoundedHTTP(
        http_client,
        access.requests_per_minute,
        GoldsmithsAPIError,
        clock=clock,
        sleeper=sleeper,
    )
    robots = fetcher.get_bytes(_ROBOTS_URL, _MAX_ROBOTS_BYTES, "robots.txt")
    if f"Sitemap: {_SITEMAP_URL}" not in robots.decode("utf-8", errors="strict"):
        raise GoldsmithsAPIError("robots.txt does not advertise the approved sitemap")

    sitemap_urls = _product_sitemap_urls(
        fetcher.get_bytes(_SITEMAP_URL, _MAX_INDEX_BYTES, "sitemap index")
    )
    products: list[tuple[str, str]] = []
    for sitemap_url in sitemap_urls:
        root = _xml(
            fetcher.get_bytes(
                sitemap_url,
                _MAX_PRODUCT_SITEMAP_BYTES,
                "product sitemap",
            )
        )
        for entry in root.findall("sm:url", _NS):
            source_url = _text(entry.find("sm:loc", _NS), "product URL")
            parsed = _approved_source_url(source_url)
            match = _PRODUCT_PATH.fullmatch(parsed.path)
            if match is not None:
                products.append((source_url, match.group("unit_id")))
    if len(products) > _MAX_EXACT_PRODUCTS or len(products) != len(set(products)):
        raise GoldsmithsAPIError("exact-reference product inventory is invalid")

    remaining = sorted(
        (source_url, unit_id)
        for source_url, unit_id in products
        if after_unit is None or unit_id > after_unit
    )
    records: list[DiscoveryRecord] = []
    for source_url, unit_id in remaining:
        if not fetcher.can_complete_before(source_deadline):
            break
        records.append(
            _parse_product_page(
                source_url,
                unit_id,
                fetcher.get_bytes(source_url, _MAX_PRODUCT_PAGE_BYTES, "product page"),
            )
        )
    if remaining and not records:
        raise GoldsmithsAPIError("CPO discovery budget exhausted before progress")
    next_cursor = None
    if len(records) < len(remaining):
        next_cursor = f"after:{records[-1].source_record_id}"
    return DiscoveryPage(
        source="goldsmiths",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )


def _product_sitemap_urls(body: bytes) -> tuple[str, ...]:
    try:
        document = json.loads(body)
        entries = document["sitemap"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GoldsmithsAPIError("sitemap index is invalid JSON") from exc
    if not isinstance(entries, list):
        raise GoldsmithsAPIError("sitemap index entries are invalid")
    urls: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("loc"), str):
            raise GoldsmithsAPIError("sitemap index entry is invalid")
        url = entry["loc"]
        parsed = _approved_source_url(url)
        if _PRODUCT_SITEMAP_PATH.fullmatch(parsed.path):
            urls.append(url)
    if not urls or len(urls) > _MAX_PRODUCT_SITEMAPS or len(urls) != len(set(urls)):
        raise GoldsmithsAPIError("product sitemap inventory is missing or invalid")
    return tuple(urls)


def _parse_product_page(source_url: str, unit_id: str, body: bytes) -> DiscoveryRecord:
    try:
        document = html.unescape(body.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise GoldsmithsAPIError("product page is not UTF-8") from exc
    title_match = re.search(r"<title>\s*([^<]+?)\s*</title>", document, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match is not None else ""
    if re.fullmatch(
        r"Rolex Certified Pre-Owned Submariner Date (?:M126610LN-0001|126610LN) "
        r"\| Goldsmiths",
        title,
    ) is None:
        raise GoldsmithsAPIError("product page lacks the exact Rolex CPO declaration")
    canonical = f'<link rel="canonical" href="{source_url}">'
    if canonical not in document:
        raise GoldsmithsAPIError("product page canonical URL is missing")

    image_pattern = re.compile(
        rf"https://{re.escape(_IMAGE_HOST)}/productimage/{unit_id}/"
        rf"{unit_id}_(?P<position>[0-9]+)\.png\?impolicy=lister"
    )
    positioned = {
        int(match.group("position")): match.group(0)
        for match in image_pattern.finditer(document)
    }
    images = tuple(positioned[position] for position in sorted(positioned))
    if not images:
        raise GoldsmithsAPIError("exact-reference product images are missing")
    for image_url in images:
        _approved_image_url(image_url, unit_id)

    version_input = "\0".join((title, source_url, *images)).encode()
    return DiscoveryRecord(
        source="goldsmiths",
        source_record_id=unit_id,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id="wos-group-cpo",
        declared_reference="rolex-submariner-126610ln",
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=images,
        metadata=(
            ("classification", "original_candidate_cpo_certified"),
            ("evidence_tier", "a"),
            ("physical_unit_id", unit_id),
            ("authenticity_claim", "Rolex Certified Pre-Owned"),
            ("rights_status", "metadata_only"),
        ),
    )


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise GoldsmithsAPIError("product sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise GoldsmithsAPIError(f"{field} is missing")
    return value


def _approved_source_url(value: str):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise GoldsmithsAPIError("source URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise GoldsmithsAPIError("source URL is not on the approved host")
    return parsed


def _approved_image_url(value: str, unit_id: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != _IMAGE_HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.fragment
        or not parsed.path.startswith(f"/productimage/{unit_id}/{unit_id}_")
        or parsed.query != "impolicy=lister"
    ):
        raise GoldsmithsAPIError("image URL is not on the approved host")
