"""Bounded WooCommerce discovery for the private cloud archive."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import cast
from urllib.parse import urlencode

from acquisition.discovery import DiscoveryCommit, DiscoveryPage, DiscoveryRecord
from acquisition.source_policy import authorize_discovery
from acquisition.supercloneluxury_archive import API, HOST, CatalogError, _product
from acquisition.types import SourceAccessRecord, SourcePolicy
from scrapers.metadata_http import PacedBoundedHTTP

_PAGE_SIZE = 25
_MAX_PAGE_BYTES = 8 * 1024 * 1024


def discover_supercloneluxury(
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
    if access.allowed_image_hosts != (HOST,) or access.requests_per_minute is None:
        raise PermissionError("Super Clone Luxury access boundary is invalid")
    cursor = previous.next_cursor if previous else None
    try:
        page = int(cursor) if cursor is not None else 1
    except ValueError as exc:
        raise CatalogError("invalid WooCommerce page cursor") from exc
    if not 1 <= page <= 1000 or (cursor is not None and str(page) != cursor):
        raise CatalogError("invalid WooCommerce page cursor")
    url = API + "?" + urlencode({"per_page": _PAGE_SIZE, "page": page, "order": "asc"})
    body, headers = PacedBoundedHTTP(
        http_client, access.requests_per_minute, CatalogError
    ).get_response(url, _MAX_PAGE_BYTES, "WooCommerce product page", deadline=deadline)
    try:
        products = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CatalogError("WooCommerce product page is invalid JSON") from exc
    if not isinstance(products, list) or len(products) > _PAGE_SIZE:
        raise CatalogError("WooCommerce product page is invalid")
    normalized_headers = {key.lower(): value for key, value in headers.items()}
    try:
        total = int(normalized_headers["x-wp-total"])
        total_pages = int(normalized_headers["x-wp-totalpages"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CatalogError("WooCommerce pagination headers are invalid") from exc
    if (
        total < 1
        or total_pages != (total + _PAGE_SIZE - 1) // _PAGE_SIZE
        or page > total_pages
        or len(products) != min(_PAGE_SIZE, total - (page - 1) * _PAGE_SIZE)
    ):
        raise CatalogError("WooCommerce pagination is inconsistent")
    records = []
    for raw in products:
        product = _product(raw)
        urls = cast(list[str], product["image_urls"])
        if not urls:
            continue
        product_id = product["product_id"]
        records.append(
            DiscoveryRecord(
                source="supercloneluxury",
                source_record_id=f"product-{product_id}",
                source_version=hashlib.sha256(
                    json.dumps(raw, sort_keys=True).encode()
                ).hexdigest(),
                source_url=cast(str, product["permalink"]),
                seller_group_id=f"listing-{product_id}",
                declared_reference="unverified",
                image_ids=tuple(hashlib.sha256(u.encode()).hexdigest() for u in urls),
                image_urls=tuple(urls),
                metadata=(
                    ("classification", cast(str, product["source_label"])),
                    ("listing_title", cast(str, product["name"])),
                ),
            )
        )
    return DiscoveryPage(
        source="supercloneluxury",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=str(page + 1) if page < total_pages else None,
        observed_at=now,
        records=tuple(records),
    )
