"""Metadata-only discovery for vendor-declared VSFactoryUSA replicas."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy

_SITEMAP_URL = "https://vsfactoryusa.com/sitemap.xml"
_HOST = "vsfactoryusa.com"
_REFERENCE = "126610ln"
_REFERENCE_TOKEN = re.compile(r"(?<![a-z0-9])126610ln(?![a-z0-9])", re.IGNORECASE)
_DECLARED_REFERENCE = "rolex-submariner-126610ln"
_MAX_SITEMAP_BYTES = 5 * 1024 * 1024
_USER_AGENT = "watch-authenticity/0.1 (research collector)"
_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}


class VSFactoryAPIError(RuntimeError):
    """The public sitemap is unavailable or unsafe to commit."""


def discover_vsfactoryusa(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: object | None = None,
) -> DiscoveryPage:
    """Discover exact-reference image URLs without downloading copyrighted bytes."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:
        raise PermissionError("source access evidence missing")
    if previous is not None:
        raise VSFactoryAPIError("VSFactoryUSA sitemap discovery has no continuation cursor")

    response = http_client.get(
        _SITEMAP_URL,
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
        allow_redirects=False,
        stream=True,
    )
    status = getattr(response, "status_code", None)
    if status == 429:
        raise VSFactoryAPIError("VSFactoryUSA rate limited; resume later")
    if status != 200:
        raise VSFactoryAPIError(f"VSFactoryUSA sitemap returned HTTP {status}")
    try:
        body = _read_bounded(response)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise VSFactoryAPIError("VSFactoryUSA sitemap is invalid XML") from exc

    records: list[DiscoveryRecord] = []
    for entry in root.findall("sm:url", _NS):
        record = _parse_entry(entry)
        if record is not None:
            records.append(record)
    if not records:
        raise VSFactoryAPIError("VSFactoryUSA sitemap has no exact-reference replica records")
    return DiscoveryPage(
        source="vsfactoryusa",
        access_record_id=access.access_record_id,
        cursor=None,
        next_cursor=None,
        observed_at=now,
        records=tuple(records),
    )


def _parse_entry(entry: ET.Element) -> DiscoveryRecord | None:
    source_url = _text(entry.find("sm:loc", _NS), "product URL")
    parsed = _approved_url(source_url)
    slug = parsed.path.strip("/")
    normalized = slug.lower()
    if _REFERENCE_TOKEN.search(normalized) is None or "submariner" not in normalized:
        return None
    if "replica" not in normalized and "super-clone" not in normalized:
        return None

    last_modified = _text(entry.find("sm:lastmod", _NS), "last modified")
    images: list[str] = []
    titles: list[str] = []
    for image in entry.findall("image:image", _NS):
        image_url = _text(image.find("image:loc", _NS), "image URL")
        _approved_url(image_url)
        title_element = image.find("image:title", _NS)
        title = title_element.text.strip() if title_element is not None and title_element.text else ""
        if _REFERENCE_TOKEN.search(title) is None or "clone" not in title.lower():
            raise VSFactoryAPIError("exact-reference image lacks an explicit replica declaration")
        images.append(image_url)
        titles.append(title)
    if not images or len(images) != len(set(images)):
        raise VSFactoryAPIError("exact-reference product images are missing or duplicated")

    version_input = "\0".join((last_modified, *images, *titles)).encode()
    return DiscoveryRecord(
        source="vsfactoryusa",
        source_record_id=slug,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id="vsfactoryusa",
        declared_reference=_DECLARED_REFERENCE,
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=tuple(images),
        metadata=(
            ("classification", "replica_candidate_vendor_declared"),
            ("evidence_tier", "c"),
            ("factory_claim", "VS Factory (VSF)"),
            ("last_modified", last_modified),
            ("rights_status", "metadata_only"),
        ),
    )


def _read_bounded(response: Any) -> bytes:
    chunks: list[bytes] = []
    size = 0
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        raise VSFactoryAPIError("VSFactoryUSA sitemap response is not streamable")
    for chunk in iterator(chunk_size=64 * 1024):
        if not isinstance(chunk, bytes):
            raise VSFactoryAPIError("VSFactoryUSA sitemap stream is invalid")
        size += len(chunk)
        if size > _MAX_SITEMAP_BYTES:
            raise VSFactoryAPIError("VSFactoryUSA sitemap body is too large")
        chunks.append(chunk)
    body = b"".join(chunks)
    if not body:
        raise VSFactoryAPIError("VSFactoryUSA sitemap body is missing")
    return body


def _text(element: ET.Element | None, field: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise VSFactoryAPIError(f"VSFactoryUSA {field} is missing")
    return value


def _approved_url(value: str):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise VSFactoryAPIError("VSFactoryUSA URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _HOST
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise VSFactoryAPIError("VSFactoryUSA URL is not on the approved host")
    return parsed
