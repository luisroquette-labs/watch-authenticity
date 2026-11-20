"""Metadata-only discovery for exact-reference Goldsmiths Rolex CPO watches."""

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
from scrapers.metadata_http import HTTPDeadlineExceeded, PacedBoundedHTTP

_HOST = "www.goldsmiths.co.uk"
_IMAGE_HOST = "content.thewosgroup.com"
_ROBOTS_URL = f"https://{_HOST}/robots.txt"
_SITEMAP_URL = f"https://{_HOST}/sitemap.xml"
_PRODUCT_SITEMAP_PATH = re.compile(r"/sitemap/product-en-gbp-[0-9]+\.xml$")
_PRODUCT_PATH = re.compile(
    r"/Rolex-Certified-Pre\+Owned-(?P<model>Submariner-Date-M?126610LN(?:\+0001)?|Datejust-41-126334)/p/"
    r"(?P<unit_id>[0-9]+)$",
    re.IGNORECASE,
)
_DATEJUST_MODEL = "datejust-41-126334"
_DATEJUST_REFERENCE = "rolex-datejust-126334"
_SUBMARINER_REFERENCE = "rolex-submariner-126610ln"
_MAX_ROBOTS_BYTES = 256 * 1024
_MAX_INDEX_BYTES = 1024 * 1024
_MAX_PRODUCT_SITEMAP_BYTES = 16 * 1024 * 1024
_MAX_PRODUCT_PAGE_BYTES = 8 * 1024 * 1024
_MAX_PRODUCT_SITEMAPS = 8
_MAX_EXACT_PRODUCTS = 1000
_SOURCE_BUDGET_SECONDS = 420
_CURSOR = re.compile(r"after:(?P<unit_id>[0-9]+)")
_CURSOR_V2 = re.compile(r"after2:(?P<sub>[0-9]+|-):(?P<datejust>[0-9]+|-)")
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
    after: dict[str, int | None] = {
        _SUBMARINER_REFERENCE: None,
        _DATEJUST_REFERENCE: None,
    }
    if cursor is not None:
        legacy = _CURSOR.fullmatch(cursor)
        current = _CURSOR_V2.fullmatch(cursor)
        if legacy is not None:
            after[_SUBMARINER_REFERENCE] = int(legacy.group("unit_id"))
        elif current is not None:
            after[_SUBMARINER_REFERENCE] = _cursor_unit(current.group("sub"))
            after[_DATEJUST_REFERENCE] = _cursor_unit(current.group("datejust"))
        else:
            raise GoldsmithsAPIError("CPO discovery continuation cursor is invalid")
    source_deadline = deadline if deadline is not None else clock() + _SOURCE_BUDGET_SECONDS

    fetcher = PacedBoundedHTTP(
        http_client,
        access.requests_per_minute,
        GoldsmithsAPIError,
        clock=clock,
        sleeper=sleeper,
    )
    robots = fetcher.get_bytes(
        _ROBOTS_URL, _MAX_ROBOTS_BYTES, "robots.txt", deadline=source_deadline
    )
    if f"Sitemap: {_SITEMAP_URL}" not in robots.decode("utf-8", errors="strict"):
        raise GoldsmithsAPIError("robots.txt does not advertise the approved sitemap")

    sitemap_urls = _product_sitemap_urls(
        fetcher.get_bytes(
            _SITEMAP_URL, _MAX_INDEX_BYTES, "sitemap index", deadline=source_deadline
        )
    )
    products: list[tuple[str, str, str]] = []
    for sitemap_url in sitemap_urls:
        root = _xml(
            fetcher.get_bytes(
                sitemap_url,
                _MAX_PRODUCT_SITEMAP_BYTES,
                "product sitemap",
                deadline=source_deadline,
            )
        )
        for entry in root.findall("sm:url", _NS):
            source_url = _text(entry.find("sm:loc", _NS), "product URL")
            parsed = _approved_source_url(source_url)
            match = _PRODUCT_PATH.fullmatch(parsed.path)
            if match is not None:
                reference_id = (
                    _DATEJUST_REFERENCE
                    if match.group("model").lower() == _DATEJUST_MODEL
                    else _SUBMARINER_REFERENCE
                )
                products.append((source_url, match.group("unit_id"), reference_id))
    if (len(products) > _MAX_EXACT_PRODUCTS
            or len(products) != len({unit_id for _, unit_id, _ in products})):
        raise GoldsmithsAPIError("exact-reference product inventory is invalid")

    remaining = sorted(
        (
            (source_url, unit_id, reference_id)
            for source_url, unit_id, reference_id in products
            if (marker := after[reference_id]) is None or int(unit_id) > marker
        ),
        key=lambda item: int(item[1]),
    )
    records: list[DiscoveryRecord] = []
    for source_url, unit_id, reference_id in remaining:
        if not fetcher.can_complete_before(source_deadline):
            break
        try:
            body = fetcher.get_bytes(
                source_url,
                _MAX_PRODUCT_PAGE_BYTES,
                "product page",
                deadline=source_deadline,
            )
        except HTTPDeadlineExceeded:
            break
        records.append(_parse_product_page(source_url, unit_id, reference_id, body))
        after[reference_id] = int(unit_id)
    if remaining and not records:
        raise GoldsmithsAPIError("CPO discovery budget exhausted before progress")
    next_cursor = None
    if len(records) < len(remaining):
        next_cursor = f"after2:{after[_SUBMARINER_REFERENCE] or '-'}:{after[_DATEJUST_REFERENCE] or '-'}"
    return DiscoveryPage(
        source="goldsmiths",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )


def _cursor_unit(value: str) -> int | None:
    return None if value == "-" else int(value)


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


def _parse_product_page(source_url: str, unit_id: str, reference_id: str, body: bytes) -> DiscoveryRecord:
    try:
        document = html.unescape(body.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise GoldsmithsAPIError("product page is not UTF-8") from exc
    title_match = re.search(r"<title>\s*([^<]+?)\s*</title>", document, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match is not None else ""
    expected_title = (
        r"Rolex Certified Pre-Owned Datejust 41 126334 \| Goldsmiths"
        if reference_id == _DATEJUST_REFERENCE
        else r"Rolex Certified Pre-Owned Submariner Date (?:M126610LN-0001|126610LN) \| Goldsmiths"
    )
    if re.fullmatch(expected_title, title) is None:
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
        declared_reference=reference_id,
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
