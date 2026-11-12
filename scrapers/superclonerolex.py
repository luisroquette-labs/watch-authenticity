"""Public-image discovery for vendor-declared SuperClone Rolex products."""

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

_HOST = "superclonerolex.io"
_ROBOTS_URL = f"https://{_HOST}/robots.txt"
_SITEMAP_URL = f"https://{_HOST}/sitemap.xml"
_PRODUCT_PATH = "/watches/126610ln/"
_PRODUCT_URL = f"https://{_HOST}{_PRODUCT_PATH}"
_REFERENCE_TOKEN = re.compile(r"(?<![A-Z0-9])126610LN(?![A-Z0-9])", re.IGNORECASE)
_MAX_ROBOTS_BYTES = 256 * 1024
_MAX_SITEMAP_BYTES = 2 * 1024 * 1024
_MAX_PRODUCT_PAGE_BYTES = 4 * 1024 * 1024
_MAX_IMAGES = 32
_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


class SuperCloneRolexAPIError(RuntimeError):
    """The public product metadata is unavailable or unsafe to commit."""


def discover_superclonerolex(
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
    if access is None:
        raise PermissionError("source access evidence missing")
    if access.allowed_image_hosts != (_HOST,):
        raise PermissionError("source access image hosts do not match the approved host")
    if access.requests_per_minute is None:
        raise PermissionError("source access request limit is missing")
    if previous is not None:
        raise SuperCloneRolexAPIError("product discovery has no continuation cursor")

    fetcher = PacedBoundedHTTP(
        http_client,
        access.requests_per_minute,
        SuperCloneRolexAPIError,
        clock=clock,
        sleeper=sleeper,
    )
    robots = fetcher.get_bytes(
        _ROBOTS_URL, _MAX_ROBOTS_BYTES, "robots.txt", deadline=deadline
    ).decode("utf-8", errors="strict")
    if f"Sitemap: {_SITEMAP_URL}" not in robots:
        raise SuperCloneRolexAPIError("robots.txt does not advertise the approved sitemap")

    root = _xml(
        fetcher.get_bytes(
            _SITEMAP_URL, _MAX_SITEMAP_BYTES, "sitemap", deadline=deadline
        )
    )
    matches: list[tuple[str, str]] = []
    for entry in root.findall("sm:url", _NS):
        source_url = _text(entry.find("sm:loc", _NS), "product URL")
        parsed = _approved_url(source_url)
        if parsed.path == _PRODUCT_PATH:
            matches.append(
                (source_url, _text(entry.find("sm:lastmod", _NS), "last modified"))
            )
    if len(matches) != 1 or matches[0][0] != _PRODUCT_URL:
        raise SuperCloneRolexAPIError("exact-reference product inventory is missing or duplicated")

    source_url, last_modified = matches[0]
    body = fetcher.get_bytes(
        source_url, _MAX_PRODUCT_PAGE_BYTES, "product page", deadline=deadline
    )
    record = _parse_product(source_url, last_modified, body)
    return DiscoveryPage(
        source="superclonerolex",
        access_record_id=access.access_record_id,
        cursor=None,
        next_cursor=None,
        observed_at=now,
        records=(record,),
    )


def _parse_product(source_url: str, last_modified: str, body: bytes) -> DiscoveryRecord:
    try:
        document = html.unescape(body.decode("utf-8", errors="strict"))
    except UnicodeDecodeError as exc:
        raise SuperCloneRolexAPIError("product page is not UTF-8") from exc
    title_match = re.search(r"<title>\s*([^<]+?)\s*</title>", document, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else ""
    if "Super Clone" not in title or _REFERENCE_TOKEN.search(title) is None:
        raise SuperCloneRolexAPIError("product page lacks an exact replica declaration")
    canonical = re.search(
        r'<link\s+rel=["\']canonical["\']\s+href=["\']([^"\']+)["\']\s*/?>',
        document,
        re.IGNORECASE,
    )
    if canonical is None or canonical.group(1) != source_url:
        raise SuperCloneRolexAPIError("product page canonical URL is missing or invalid")

    products: list[dict[str, Any]] = []
    for payload in re.findall(
        r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        document,
        re.IGNORECASE | re.DOTALL,
    ):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SuperCloneRolexAPIError("product structured data is invalid") from exc
        if isinstance(value, dict) and value.get("@type") == "Product":
            products.append(value)
    if len(products) != 1:
        raise SuperCloneRolexAPIError("exact product structured data is missing or duplicated")
    product = products[0]
    name = product.get("name")
    sku = product.get("sku")
    category = product.get("category")
    if (
        not isinstance(name, str)
        or _REFERENCE_TOKEN.search(name) is None
        or not isinstance(sku, str)
        or sku != "JP-126610LN"
        or not isinstance(category, str)
        or "Super Clone" not in category
    ):
        raise SuperCloneRolexAPIError("structured product lacks the exact replica identity")

    raw_images = product.get("image")
    if not isinstance(raw_images, list) or not 4 <= len(raw_images) <= _MAX_IMAGES:
        raise SuperCloneRolexAPIError("exact product image inventory is missing or invalid")
    images: list[str] = []
    captions: list[str] = []
    for image in raw_images:
        if not isinstance(image, dict):
            raise SuperCloneRolexAPIError("structured product image is invalid")
        image_url = image.get("contentUrl")
        caption = image.get("caption")
        if not isinstance(image_url, str) or not isinstance(caption, str) or not caption.strip():
            raise SuperCloneRolexAPIError("structured product image metadata is incomplete")
        parsed = _approved_url(image_url, image=True)
        if _REFERENCE_TOKEN.search(parsed.path) is None:
            raise SuperCloneRolexAPIError("product image lacks the exact reference")
        images.append(image_url)
        captions.append(caption.strip())
    if len(images) != len(set(images)):
        raise SuperCloneRolexAPIError("exact product images are duplicated")

    version_input = "\0".join((last_modified, title, sku, *images, *captions)).encode()
    return DiscoveryRecord(
        source="superclonerolex",
        source_record_id=sku,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id="superclonerolex",
        declared_reference="rolex-submariner-126610ln",
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=tuple(images),
        metadata=(
            ("classification", "replica_candidate_vendor_declared"),
            ("evidence_tier", "c"),
            ("seller_claim", "Super Clone Rolex"),
            ("last_modified", last_modified),
            ("rights_status", "owner_private_download_only"),
        ),
    )


def _xml(body: bytes) -> ET.Element:
    try:
        return ET.fromstring(body)
    except ET.ParseError as exc:
        raise SuperCloneRolexAPIError("sitemap is invalid XML") from exc


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise SuperCloneRolexAPIError(f"{field} is missing")
    return value


def _approved_url(value: str, *, image: bool = False):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise SuperCloneRolexAPIError("URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
        or (image and not parsed.path.startswith("/images/products/"))
    ):
        raise SuperCloneRolexAPIError("URL is not on the approved host")
    return parsed
