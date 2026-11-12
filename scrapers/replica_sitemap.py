"""Shared metadata-only discovery for declared replica-vendor sitemaps."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from acquisition.discovery import DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy

_REFERENCE_TOKEN = re.compile(r"(?<![a-z0-9])126610ln(?![a-z0-9])", re.IGNORECASE)
_DECLARED_REFERENCE = "rolex-submariner-126610ln"
_MAX_SITEMAP_BYTES = 5 * 1024 * 1024
_USER_AGENT = "watch-authenticity/0.1 (research collector)"
_NS = {
    "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
    "image": "http://www.google.com/schemas/sitemap-image/1.1",
}


class ReplicaSitemapError(RuntimeError):
    """A replica-vendor sitemap is unavailable or unsafe to commit."""


@dataclass(frozen=True)
class ReplicaSitemapConfig:
    source: str
    sitemap_url: str
    host: str
    seller_group_id: str
    factory_claim: str
    required_path_markers: tuple[str, ...] = ()
    required_title_markers: tuple[str, ...] = ()


def discover_replica_sitemap(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    config: ReplicaSitemapConfig,
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
        raise ReplicaSitemapError(f"{config.source} sitemap has no continuation cursor")

    response = http_client.get(
        config.sitemap_url,
        headers={"User-Agent": _USER_AGENT},
        timeout=30,
        allow_redirects=False,
        stream=True,
    )
    status = getattr(response, "status_code", None)
    if status == 429:
        raise ReplicaSitemapError(f"{config.source} rate limited; resume later")
    if status != 200:
        raise ReplicaSitemapError(f"{config.source} sitemap returned HTTP {status}")
    try:
        body = _read_bounded(response, config.source)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        raise ReplicaSitemapError(f"{config.source} sitemap is invalid XML") from exc

    records = tuple(
        record
        for entry in root.findall("sm:url", _NS)
        if (record := _parse_entry(entry, config)) is not None
    )
    if not records:
        raise ReplicaSitemapError(
            f"{config.source} sitemap has no exact-reference replica records"
        )
    return DiscoveryPage(
        source=config.source,
        access_record_id=access.access_record_id,
        cursor=None,
        next_cursor=None,
        observed_at=now,
        records=records,
    )


def _parse_entry(
    entry: ET.Element, config: ReplicaSitemapConfig
) -> DiscoveryRecord | None:
    source_url = _text(entry.find("sm:loc", _NS), "product URL", config.source)
    parsed = _approved_url(source_url, config.host, config.source)
    slug = parsed.path.strip("/")
    normalized = slug.lower()
    if _REFERENCE_TOKEN.search(normalized) is None or "submariner" not in normalized:
        return None
    if config.required_path_markers and not any(
        marker in normalized for marker in config.required_path_markers
    ):
        return None

    last_modified = _text(
        entry.find("sm:lastmod", _NS), "last modified", config.source
    )
    images: list[str] = []
    titles: list[str] = []
    for image in entry.findall("image:image", _NS):
        image_url = _text(image.find("image:loc", _NS), "image URL", config.source)
        _approved_url(image_url, config.host, config.source)
        title = _text(image.find("image:title", _NS), "image title", config.source)
        if _REFERENCE_TOKEN.search(title) is None:
            raise ReplicaSitemapError(
                f"{config.source} exact-reference image title lacks the reference"
            )
        if config.required_title_markers and not any(
            marker in title.lower() for marker in config.required_title_markers
        ):
            raise ReplicaSitemapError(
                f"{config.source} exact-reference image lacks a replica declaration"
            )
        images.append(image_url)
        titles.append(title)
    if not images:
        raise ReplicaSitemapError(
            f"{config.source} exact-reference product images are missing"
        )
    if len(images) != len(set(images)):
        raise ReplicaSitemapError(
            f"{config.source} exact-reference product images are duplicated"
        )

    version_input = "\0".join((last_modified, *images, *titles)).encode()
    return DiscoveryRecord(
        source=config.source,
        source_record_id=slug,
        source_version=hashlib.sha256(version_input).hexdigest(),
        source_url=source_url,
        seller_group_id=config.seller_group_id,
        declared_reference=_DECLARED_REFERENCE,
        image_ids=tuple(hashlib.sha256(url.encode()).hexdigest() for url in images),
        image_urls=tuple(images),
        metadata=(
            ("classification", "replica_candidate_vendor_declared"),
            ("evidence_tier", "c"),
            ("factory_claim", config.factory_claim),
            ("last_modified", last_modified),
            ("rights_status", "metadata_only"),
        ),
    )


def _read_bounded(response: Any, source: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        raise ReplicaSitemapError(f"{source} sitemap response is not streamable")
    for chunk in iterator(chunk_size=64 * 1024):
        if not isinstance(chunk, bytes):
            raise ReplicaSitemapError(f"{source} sitemap stream is invalid")
        size += len(chunk)
        if size > _MAX_SITEMAP_BYTES:
            raise ReplicaSitemapError(f"{source} sitemap body is too large")
        chunks.append(chunk)
    body = b"".join(chunks)
    if not body:
        raise ReplicaSitemapError(f"{source} sitemap body is missing")
    return body


def _text(element: ET.Element | None, field: str, source: str) -> str:
    value = element.text.strip() if element is not None and element.text else ""
    if not value or "\n" in value or "\r" in value:
        raise ReplicaSitemapError(f"{source} {field} is missing")
    return value


def _approved_url(value: str, host: str, source: str):
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ReplicaSitemapError(f"{source} URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ReplicaSitemapError(f"{source} URL is not on the approved host")
    return parsed
