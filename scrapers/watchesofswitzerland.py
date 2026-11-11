"""Metadata-only discovery for Watches of Switzerland Rolex CPO watches."""

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

_HOST = "www.watchesofswitzerland.com"
_IMAGE_HOST = "cdn.shopify.com"
_ROBOTS_URL = f"https://{_HOST}/robots.txt"
_SITEMAP_URL = f"https://{_HOST}/sitemap.xml"
_USER_AGENT = "watch-authenticity/0.1 (research collector)"
_REFERENCE = re.compile(r"(?<![a-z0-9])m?126610ln(?:-0001)?(?![a-z0-9])", re.IGNORECASE)
_PRODUCT_PATH = re.compile(
    r"/products/submariner(?:-date)?-m126610ln-0001-(?P<unit_id>[0-9]+)$",
    re.IGNORECASE,
)
_PRODUCT_SITEMAP = re.compile(r"/sitemap_products_[0-9]+\.xml$")
_MAX_ROBOTS_BYTES = 256 * 1024
_MAX_INDEX_BYTES = 1024 * 1024
_MAX_PRODUCT_SITEMAP_BYTES = 8 * 1024 * 1024
_MAX_PRODUCT_SITEMAPS = 12
_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}


class WatchesOfSwitzerlandAPIError(RuntimeError):
    """The public CPO discovery surface is unavailable or unsafe to commit."""


def discover_watchesofswitzerland(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: object | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> DiscoveryPage:
    """Discover exact-reference CPO metadata without downloading image bytes."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:
        raise PermissionError("source access evidence missing")
    if access.allowed_image_hosts != (_IMAGE_HOST,):
        raise PermissionError("source access image hosts do not match the approved host")
    if access.requests_per_minute is None:
        raise PermissionError("source access request limit is missing")
    requests_per_minute = access.requests_per_minute
    if previous is not None:
        raise WatchesOfSwitzerlandAPIError("CPO sitemap discovery has no continuation cursor")

    next_request_at = clock()

    def get_bounded(url: str, limit: int, surface: str) -> bytes:
        nonlocal next_request_at
        delay = next_request_at - clock()
        if delay > 0:
            sleeper(delay)
        body = _get_bounded(http_client, url, limit, surface)
        next_request_at = clock() + 60.0 / requests_per_minute
        return body

    robots = get_bounded(_ROBOTS_URL, _MAX_ROBOTS_BYTES, "robots.txt")
    if f"Sitemap: {_SITEMAP_URL}" not in robots.decode("utf-8", errors="strict"):
        raise WatchesOfSwitzerlandAPIError("robots.txt does not advertise the approved sitemap")

    index = _xml(get_bounded(_SITEMAP_URL, _MAX_INDEX_BYTES, "sitemap index"))
    sitemap_urls = _product_sitemap_urls(index)
    records: list[DiscoveryRecord] = []
    for sitemap_url in sitemap_urls:
        sitemap = _xml(
            get_bounded(
                sitemap_url,
                _MAX_PRODUCT_SITEMAP_BYTES,
                "product sitemap",
            )
        )
        records.extend(
            record
            for entry in sitemap.findall("sm:url", _NS)
            if (record := _parse_entry(entry)) is not None
        )

    return DiscoveryPage(
        source="watchesofswitzerland",
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
        parsed = _approved_source_url(url, allow_query=True)
        if _PRODUCT_SITEMAP.fullmatch(parsed.path):
            urls.append(url)
    if not urls or len(urls) > _MAX_PRODUCT_SITEMAPS or len(urls) != len(set(urls)):
        raise WatchesOfSwitzerlandAPIError("product sitemap inventory is missing or invalid")
    return tuple(urls)


def _parse_entry(entry: ET.Element) -> DiscoveryRecord | None:
    source_url = _text(entry.find("sm:loc", _NS), "product URL")
    parsed = _approved_source_url(source_url)
    match = _PRODUCT_PATH.fullmatch(parsed.path)
    if match is None:
        return None

    last_modified = _text(entry.find("sm:lastmod", _NS), "last modified")
    images: list[str] = []
    titles: list[str] = []
    captions: list[str] = []
    for image in entry.findall("image:image", _NS):
        image_url = _text(image.find("image:loc", _NS), "image URL")
        _approved_image_url(image_url)
        title = _text(image.find("image:title", _NS), "image title")
        caption = _text(image.find("image:caption", _NS), "image caption")
        normalized_caption = caption.lower()
        if _REFERENCE.search(caption) is None or not (
            "rolex certified pre-owned" in normalized_caption
            or "rolex cpo" in normalized_caption
        ):
            raise WatchesOfSwitzerlandAPIError(
                "exact-reference image lacks a Rolex CPO declaration"
            )
        images.append(image_url)
        titles.append(title)
        captions.append(caption)
    if not images or len(images) != len(set(images)):
        raise WatchesOfSwitzerlandAPIError("exact-reference product images are missing or duplicated")

    unit_id = match.group("unit_id")
    version_input = "\0".join((last_modified, *images, *titles, *captions)).encode()
    return DiscoveryRecord(
        source="watchesofswitzerland",
        source_record_id=unit_id,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id="watchesofswitzerland-cpo",
        declared_reference="rolex-submariner-126610ln",
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=tuple(images),
        metadata=(
            ("classification", "original_candidate_cpo_certified"),
            ("evidence_tier", "a"),
            ("physical_unit_id", unit_id),
            ("authenticity_claim", "Rolex Certified Pre-Owned"),
            ("last_modified", last_modified),
            ("rights_status", "metadata_only"),
        ),
    )


def _get_bounded(http_client: Any, url: str, limit: int, surface: str) -> bytes:
    response = http_client.get(
        url,
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
        allow_redirects=False,
        stream=True,
    )
    try:
        status = getattr(response, "status_code", None)
        if status == 429:
            raise WatchesOfSwitzerlandAPIError(f"{surface} rate limited; resume later")
        if status != 200:
            raise WatchesOfSwitzerlandAPIError(f"{surface} returned HTTP {status}")
        chunks: list[bytes] = []
        size = 0
        iterator = getattr(response, "iter_content", None)
        if not callable(iterator):
            raise WatchesOfSwitzerlandAPIError(f"{surface} response is not streamable")
        for chunk in iterator(chunk_size=64 * 1024):
            if not isinstance(chunk, bytes):
                raise WatchesOfSwitzerlandAPIError(f"{surface} stream is invalid")
            size += len(chunk)
            if size > limit:
                raise WatchesOfSwitzerlandAPIError(f"{surface} body is too large")
            chunks.append(chunk)
        body = b"".join(chunks)
        if not body:
            raise WatchesOfSwitzerlandAPIError(f"{surface} body is missing")
        return body
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise WatchesOfSwitzerlandAPIError("sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise WatchesOfSwitzerlandAPIError(f"{field} is missing")
    return value


def _approved_source_url(value: str, *, allow_query: bool = False):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise WatchesOfSwitzerlandAPIError("source URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or (parsed.query and not allow_query)
        or parsed.fragment
    ):
        raise WatchesOfSwitzerlandAPIError("source URL is not on the approved host")
    return parsed


def _approved_image_url(value: str) -> None:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise WatchesOfSwitzerlandAPIError("image URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _IMAGE_HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.fragment
        or not parsed.path.startswith("/s/files/")
    ):
        raise WatchesOfSwitzerlandAPIError("image URL is not on the approved host")
