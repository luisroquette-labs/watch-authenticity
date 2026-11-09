"""Mercado Livre official-API discovery and metadata normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Literal
from urllib.parse import urlsplit

from acquisition.collect import (
    CandidateMetadata,
    SourceUnavailable,
    normalize_candidate_metadata,
)
from acquisition.discovery import (
    DiscoveryCommit,
    DiscoveryPage,
    DiscoveryRecord,
    append_discovery_page,
)
from acquisition.source_policy import authorize_discovery
from acquisition.types import SourceAccessRecord, SourcePolicy

IngestMethod = Literal["scrape", "official_api"]
_SEARCH_URL = "https://api.mercadolibre.com/sites/MLB/search"
_ITEM_URL = "https://api.mercadolibre.com/items/{item_id}"
_QUERY = "Rolex Submariner 126610LN"
_ITEM_ID = re.compile(r"MLB[0-9]+")


class MercadoLivreAPIError(RuntimeError):
    """Official API returned a response that cannot be committed."""


def discover_mercadolivre(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    cos_client: Any,
    bucket: str,
    prefix: str,
    token: str,
    now: datetime,
    previous: DiscoveryCommit | None = None,
    reference_id: str = "rolex-submariner-126610ln",
) -> DiscoveryCommit:
    """Persist one validated page of official API metadata."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:  # narrowed by authorize_discovery; keeps type checkers honest
        raise PermissionError("source access evidence missing")
    if not isinstance(token, str) or not token.strip():
        raise PermissionError("Mercado Livre API token is required")
    if access.requests_per_minute is None or access.requests_per_minute < 2:
        raise PermissionError("source rate limit cannot cover search and item detail")

    cursor = previous.next_cursor if previous is not None else None
    if previous is not None and cursor is None:
        raise MercadoLivreAPIError("discovery ledger is already complete")
    offset = int(cursor) if cursor is not None and cursor.isdecimal() else 0
    if cursor is not None and str(offset) != cursor:
        raise MercadoLivreAPIError("invalid Mercado Livre cursor")
    limit = min(20, access.requests_per_minute - 1)
    headers = {"Authorization": f"Bearer {token.strip()}"}
    search = _json_response(
        http_client.get(
            _SEARCH_URL,
            params={"q": _QUERY, "offset": offset, "limit": limit},
            headers=headers,
            timeout=30,
            allow_redirects=False,
        ),
        "search",
    )
    results = search.get("results")
    paging = search.get("paging")
    if not isinstance(results, list) or not results or not isinstance(paging, Mapping):
        raise MercadoLivreAPIError("search payload has no valid results page")
    if len(results) > limit:
        raise MercadoLivreAPIError("search payload exceeds requested page limit")

    records: list[DiscoveryRecord] = []
    for result in results:
        result_id = _result_id(result)
        records.append(
            _item_record(
                _json_response(
                    http_client.get(
                        _ITEM_URL.format(item_id=result_id),
                        headers=headers,
                        timeout=30,
                        allow_redirects=False,
                    ),
                    "item",
                ),
                access,
                reference_id,
                expected_item_id=result_id,
            )
        )
    payload_offset = _nonnegative_int(paging.get("offset"), "paging.offset")
    payload_total = _nonnegative_int(paging.get("total"), "paging.total")
    if payload_offset != offset:
        raise MercadoLivreAPIError("search payload offset differs from request")
    next_offset = offset + len(results)
    next_cursor = str(next_offset) if next_offset < payload_total else None
    page = DiscoveryPage(
        source="mercadolivre",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )
    return append_discovery_page(
        cos_client,
        bucket,
        prefix,
        page,
        previous=previous,
    )


def _json_response(response: Any, operation: str) -> Mapping[str, Any]:
    status = getattr(response, "status_code", None)
    if status == 429:
        raise MercadoLivreAPIError(f"Mercado Livre {operation} rate limited; resume later")
    if status != 200:
        raise MercadoLivreAPIError(f"Mercado Livre {operation} returned HTTP {status}")
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise MercadoLivreAPIError(f"Mercado Livre {operation} returned invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise MercadoLivreAPIError(f"Mercado Livre {operation} payload must be an object")
    return payload


def _result_id(result: object) -> str:
    if not isinstance(result, Mapping):
        raise MercadoLivreAPIError("search result must be an object")
    item_id = _nonempty_string(result.get("id"), "result.id")
    if _ITEM_ID.fullmatch(item_id) is None:
        raise MercadoLivreAPIError("result.id must be a Mercado Livre Brazil item ID")
    return item_id


def _item_record(
    item: Mapping[str, Any],
    access: SourceAccessRecord,
    reference_id: str,
    *,
    expected_item_id: str,
) -> DiscoveryRecord:
    item_id = _nonempty_string(item.get("id"), "item.id")
    if item_id != expected_item_id:
        raise MercadoLivreAPIError("item detail ID differs from search result")
    seller_id = item.get("seller_id")
    if isinstance(seller_id, bool) or not isinstance(seller_id, (str, int)):
        raise MercadoLivreAPIError("item.seller_id must be a string or integer")
    permalink = _https_url(item.get("permalink"), "item.permalink")
    pictures = item.get("pictures")
    if not isinstance(pictures, list) or not pictures:
        raise MercadoLivreAPIError("item.pictures must be a non-empty array")
    image_ids: list[str] = []
    image_urls: list[str] = []
    for picture in pictures:
        if not isinstance(picture, Mapping):
            raise MercadoLivreAPIError("item picture must be an object")
        image_id = _nonempty_string(picture.get("id"), "picture.id")
        image_url = _https_url(picture.get("secure_url"), "picture.secure_url")
        if urlsplit(image_url).hostname not in access.allowed_image_hosts:
            raise MercadoLivreAPIError("picture host is not approved by source access")
        image_ids.append(image_id)
        image_urls.append(image_url)
    if len(set(image_ids)) != len(image_ids) or len(set(image_urls)) != len(image_urls):
        raise MercadoLivreAPIError("item pictures contain duplicates")
    return DiscoveryRecord(
        source="mercadolivre",
        source_record_id=item_id,
        source_version=str(item.get("last_updated") or item.get("date_created") or "current"),
        source_url=permalink,
        seller_group_id=str(seller_id),
        declared_reference=_nonempty_string(reference_id, "reference_id"),
        image_ids=tuple(image_ids),
        image_urls=tuple(image_urls),
    )


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise MercadoLivreAPIError(f"{field} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MercadoLivreAPIError(f"{field} must be a non-negative integer")
    return value


def _https_url(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise MercadoLivreAPIError(f"{field} must be an HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise MercadoLivreAPIError(
            f"{field} must be an HTTPS URL without credentials or custom port"
        )
    return value


def mercadolivre_ingest(
    method: IngestMethod,
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if method == "scrape":
        raise PermissionError("Mercado Livre raw scraping is prohibited; use its official API")
    if method != "official_api":
        raise ValueError(f"unknown Mercado Livre ingestion method: {method!r}")
    if records is None:
        return SourceUnavailable("mercadolivre", "no approved official API feed was supplied")
    return tuple(normalize_candidate_metadata("mercadolivre", record) for record in records)
