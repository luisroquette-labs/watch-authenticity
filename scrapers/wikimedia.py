"""Licensed Wikimedia Commons discovery for unverified watch-family candidates."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from acquisition.collect import CandidateMetadata
from acquisition.discovery import (
    DiscoveryCommit,
    DiscoveryPage,
    DiscoveryRecord,
)
from acquisition.source_policy import authorize_discovery
from acquisition.types import (
    EvidenceTier,
    RightsRecord,
    RightsStatus,
    SourceAccessRecord,
    SourcePolicy,
)

_API_URL = "https://commons.wikimedia.org/w/api.php"
_FAMILY_SEARCHES = (
    ("Seiko Prospex watch", "seiko", "prospex"),
    ("Cartier Santos wristwatch", "cartier", "santos"),
    ("Omega Seamaster watch", "omega", "seamaster"),
    ("Patek Philippe Nautilus", "patek philippe", "nautilus"),
    ("Rolex Submariner wristwatch", "rolex", "submariner"),
    ("Omega Speedmaster watch", "omega", "speedmaster"),
    ("replica Omega Speedmaster", "omega", "speedmaster"),
    ("fake Rolex Datejust", "rolex", "datejust"),
    ("1953 Rolex Oyster Perpetual Auckland Museum", "rolex", "oyster perpetual"),
    ("Omega Speedmaster Professional Moonwatch", "omega", "speedmaster"),
    ("replica Rolex Daytona", "rolex", "daytona"),
    ("fake Rolex Daytona watch", "rolex", "daytona"),
    ("Audemars Piguet Royal Oak", "audemars piguet", "royal oak"),
    ('incategory:"Illegal copies of watches of quality brands"', "unverified", "unverified"),
    ('incategory:"Omega Speedmaster in Apollo, Gemini and other space programs" incategory:"PD NASA"', "unverified", "unverified"),
    ('incategory:"Counterfeit watches"', "unverified", "unverified"),
)
_REPLICA_SEARCHES = frozenset({6, 7, 10, 11})
_COUNTERFEIT_CATEGORY_QUERY = 13
_NASA_SPEEDMASTER_QUERY = 14
_BROAD_COUNTERFEIT_CATEGORY_QUERY = 15
_CATEGORY_QUERIES = frozenset({_COUNTERFEIT_CATEGORY_QUERY, _BROAD_COUNTERFEIT_CATEGORY_QUERY})
_DEFAULT_SEARCH_STOP = 8
_REPLICA_RESEARCH = "commons_replica_keyword_research_unverified"
_CATEGORY_RESEARCH = "commons_counterfeit_category_research_unverified"
_CLASSIFICATION = "brand_family_candidate_unverified"
_CURSOR = re.compile(r"^q:(\d+):(\d+)$")
_IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})
_ALLOWED_LICENSES = frozenset(
    {
        "CC0",
        "Public domain",
        "CC BY 2.0",
        "CC BY 3.0",
        "CC BY 4.0",
        "CC BY-SA 3.0",
        "CC BY-SA 4.0",
    }
)
_CC0_LICENSE_URLS = frozenset({
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
})
_CBP_FLICKR_CREDIT = re.compile(
    r'href="(?P<url>https://www\.flickr\.com/photos/cbpphotos/(?P<id>[0-9]+)/)"'
)
_CBP_ARTIST_URL = 'href="https://www.flickr.com/people/54593278@N03"'
_CBP_PUBLIC_DOMAIN_TEMPLATE = "Template:PD-USGov-DHS-CBP"
_CBP_VERIFIED_FILE = (
    57419144,
    "c3088f52d34c3bc28cd3bae0dbdf2329af8b2929",
    "2017-03-26T14:48:07Z",
    "32231527976",
    "https://commons.wikimedia.org/wiki/File:San_Juan_CBP_seizes_20_counterfeit_luxury_watches_(32231527976).jpg",
)
_TAG = re.compile(r"<[^>]*>")


class WikimediaAPIError(RuntimeError):
    """Commons API response is unsafe or recoverably unavailable."""


@dataclass(frozen=True)
class WikimediaCandidate:
    metadata: CandidateMetadata
    rights: RightsRecord
    rights_evidence: bytes
    observed_at: datetime
    author: str
    license_name: str
    license_url: str
    classification: str = _CLASSIFICATION
    evidence_tier: EvidenceTier = EvidenceTier.INCONCLUSIVE


@dataclass(frozen=True)
class WikimediaDiscovery:
    page: DiscoveryPage
    candidates: tuple[WikimediaCandidate, ...]


def discover_wikimedia(
    http_client: Any,
    policy: SourcePolicy,
    access: SourceAccessRecord | None,
    *,
    now: datetime,
    previous: DiscoveryCommit | None = None,
    query_range: tuple[int, int] | None = None,
) -> WikimediaDiscovery:
    """Return one validated page; the caller commits it after collecting every file."""
    decision = authorize_discovery(policy, access, now=now)
    if not decision.allowed:
        raise PermissionError("; ".join(decision.reasons))
    if access is None:
        raise PermissionError("source access evidence missing")
    if access.requests_per_minute is None or access.requests_per_minute < 4:
        raise PermissionError("source rate limit cannot cover API and image retries")

    if query_range is not None and not (0 <= query_range[0] < query_range[1] <= len(_FAMILY_SEARCHES)):
        raise ValueError("Wikimedia query range is invalid")
    start, stop = query_range or (0, _DEFAULT_SEARCH_STOP)
    cursor = previous.next_cursor if previous is not None else None
    if previous is not None and cursor is None:
        raise WikimediaAPIError("Wikimedia discovery ledger is already complete")
    if cursor is not None and not cursor.startswith("q:"):
        # An older categorymembers checkpoint must be closed without losing its chain.
        return WikimediaDiscovery(
            page=DiscoveryPage(
                source="wikimedia", access_record_id=access.access_record_id,
                cursor=cursor, next_cursor=f"q:{start}:0", observed_at=now, records=(),
            ),
            candidates=(),
        )
    query_index, offset = _parse_cursor(cursor) if cursor is not None else (start, 0)
    if not start <= query_index < stop:
        raise WikimediaAPIError("Wikimedia cursor belongs to another query range")
    query, brand, family = _FAMILY_SEARCHES[query_index]
    params: dict[str, object] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        # One API request plus up to three bounded attempts per image stays below the source cap.
        "gsrlimit": max(1, min(50, (access.requests_per_minute - 1) // 3)),
        "prop": "imageinfo|info|templates",
        "inprop": "url",
        "iiprop": "url|sha1|timestamp|mime|extmetadata",
        "iiurlwidth": 1280,
        "tltemplates": _CBP_PUBLIC_DOMAIN_TEMPLATE,
    }
    if offset:
        params["gsroffset"] = offset
    response = http_client.get(
        _API_URL,
        params=params,
        headers={"User-Agent": "watch-authenticity/0.1 (licensed research collector)"},
        timeout=30,
        allow_redirects=False,
    )
    payload = _json_response(response)
    query_data = payload.get("query")
    pages = query_data.get("pages", []) if isinstance(query_data, Mapping) else [] if query_data is None else None
    if not isinstance(pages, list):
        raise WikimediaAPIError("Wikimedia payload has no file pages")

    records: list[DiscoveryRecord] = []
    candidates: list[WikimediaCandidate] = []
    for page in pages:
        parsed = _parse_page(page, access, now, brand, family, query_index)
        if parsed is None:
            continue
        record, candidate = parsed
        records.append(record)
        candidates.append(candidate)
    continuation = payload.get("continue")
    if continuation is not None and not isinstance(continuation, Mapping):
        raise WikimediaAPIError("Wikimedia continuation must be an object")
    next_offset = continuation.get("gsroffset") if continuation else None
    if next_offset is not None and (
        isinstance(next_offset, bool) or not isinstance(next_offset, int) or next_offset <= offset
    ):
        raise WikimediaAPIError("Wikimedia continuation offset is invalid")
    next_cursor = (
        f"q:{query_index}:{next_offset}" if next_offset is not None
        else f"q:{query_index + 1}:0" if query_index + 1 < stop
        else None
    )
    discovery_page = DiscoveryPage(
        source="wikimedia",
        access_record_id=access.access_record_id,
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )
    return WikimediaDiscovery(page=discovery_page, candidates=tuple(candidates))


def _parse_cursor(cursor: str | None) -> tuple[int, int]:
    if cursor is None:
        return 0, 0
    match = _CURSOR.fullmatch(cursor)
    if match is None or int(match[1]) >= len(_FAMILY_SEARCHES):
        raise WikimediaAPIError("Wikimedia continuation cursor is invalid")
    return int(match[1]), int(match[2])


def _json_response(response: Any) -> Mapping[str, Any]:
    status = getattr(response, "status_code", None)
    if status == 429:
        raise WikimediaAPIError("Wikimedia rate limited; resume later")
    if status != 200:
        raise WikimediaAPIError(f"Wikimedia API returned HTTP {status}")
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise WikimediaAPIError("Wikimedia API returned invalid JSON") from exc
    if not isinstance(payload, Mapping) or "error" in payload:
        raise WikimediaAPIError("Wikimedia API returned an error payload")
    return payload


def _parse_page(
    page: object,
    access: SourceAccessRecord,
    now: datetime,
    brand: str,
    family: str,
    query_index: int,
) -> tuple[DiscoveryRecord, WikimediaCandidate] | None:
    if not isinstance(page, Mapping):
        raise WikimediaAPIError("Wikimedia file page must be an object")
    page_id = page.get("pageid")
    title = _text(page.get("title"), "page.title")
    source_url = _https_url(page.get("canonicalurl"), "page.canonicalurl", "commons.wikimedia.org")
    imageinfo = page.get("imageinfo")
    if isinstance(page_id, bool) or not isinstance(page_id, int) or page_id < 1:
        raise WikimediaAPIError("page.pageid must be a positive integer")
    if not isinstance(imageinfo, list) or len(imageinfo) != 1 or not isinstance(imageinfo[0], Mapping):
        raise WikimediaAPIError("page.imageinfo must contain one object")
    info = imageinfo[0]
    if info.get("mime") not in _IMAGE_MIMES:
        return None
    metadata = info.get("extmetadata")
    if not isinstance(metadata, Mapping):
        raise WikimediaAPIError("image metadata is missing")
    searchable = re.sub(r"[-_]+", " ", " ".join(
        (
            title,
            _optional_metadata_text(metadata, "ObjectName"),
            _optional_metadata_text(metadata, "ImageDescription"),
        )
    ))
    if query_index not in _CATEGORY_QUERIES | {_NASA_SPEEDMASTER_QUERY} and any(
        re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", searchable, re.I) is None
        for word in (brand, family)
    ):
        return None
    if query_index in _REPLICA_SEARCHES and re.search(r"\b(fake|replica|counterfeit|illegal cop(?:y|ies))\b", searchable, re.I) is None:
        return None
    license_name = _metadata_value(metadata, "LicenseShortName")
    if license_name not in _ALLOWED_LICENSES:
        return None
    author = _plain_text(_metadata_value(metadata, "Artist"))
    license_url = _license_url(metadata, license_name, source_url)
    image_url = _https_url(
        info.get("thumburl") or info.get("url"),
        "image URL",
        *access.allowed_image_hosts,
    )
    revision = _text(info.get("timestamp"), "image timestamp")
    try:
        observed_at = datetime.fromisoformat(revision.replace("Z", "+00:00"))
    except ValueError as exc:
        raise WikimediaAPIError("image timestamp must be ISO-8601") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise WikimediaAPIError("image timestamp must be timezone-aware")
    sha1 = _text(info.get("sha1"), "image sha1")
    license_url_entry = metadata.get("LicenseUrl")
    raw_license_url = license_url_entry.get("value") if isinstance(license_url_entry, Mapping) else None
    cc0_train = license_name == "CC0" and raw_license_url in _CC0_LICENSE_URLS
    cbp_evidence = _cbp_public_domain_evidence(
        page, metadata, license_name, author, sha1, revision, source_url
    )
    allowed_purposes = ("download", "train") if cc0_train or cbp_evidence else ("download",)
    rights_evidence: dict[str, object] = {
        "author": author,
        "license": license_name,
        "license_url": license_url,
        "source_url": source_url,
        "title": title,
    }
    if cc0_train:
        rights_evidence.update({
            "allowed_purposes": allowed_purposes,
            "license_url_literal": raw_license_url,
            "image_sha1": sha1,
            "image_timestamp": revision,
        })
    elif cbp_evidence is not None:
        rights_evidence.update({
            "allowed_purposes": allowed_purposes,
            "commons_page_id": page_id,
            "image_sha1": sha1,
            "image_timestamp": revision,
            **cbp_evidence,
        })
    rights_evidence_bytes = json.dumps(
        rights_evidence, sort_keys=True, separators=(",", ":")
    ).encode()
    rights_sha256 = hashlib.sha256(rights_evidence_bytes).hexdigest()
    rights = RightsRecord(
        rights_record_id=f"wikimedia-{rights_sha256[:24]}",
        source="wikimedia",
        acquisition_method="official-api",
        allowed_purposes=allowed_purposes,
        effective_at=now,
        expires_at=now + timedelta(days=3650),
        required_disposition="retain",
        disposition_deadline=now + timedelta(days=3650),
        rightsholder=author,
        evidence_sha256=rights_sha256,
        status=RightsStatus.ACTIVE,
    )
    record_id = f"q{query_index}:{page_id}"
    research_metadata = (
        (("research_keyword_context", searchable[:2000]),)
        if query_index in _REPLICA_SEARCHES else
        (("research_category", "Illegal copies of watches of quality brands"),)
        if query_index == _COUNTERFEIT_CATEGORY_QUERY else
        (("research_category", "Counterfeit watches"),)
        if query_index == _BROAD_COUNTERFEIT_CATEGORY_QUERY else
        (("research_category", "Omega Speedmaster in Apollo, Gemini and other space programs; PD NASA"),)
        if query_index == _NASA_SPEEDMASTER_QUERY else ()
    )
    classification = _CATEGORY_RESEARCH if query_index in _CATEGORY_QUERIES else _CLASSIFICATION
    record = DiscoveryRecord(
        source="wikimedia",
        source_record_id=record_id,
        source_version=f"{revision}|{sha1}",
        source_url=source_url,
        seller_group_id=author,
        declared_reference="unverified",
        image_ids=(sha1,),
        image_urls=(image_url,),
        metadata=(
            ("author", author),
            ("brand_claim_from_text", brand),
            ("classification", classification),
            ("evidence_tier", EvidenceTier.INCONCLUSIVE.value),
            ("family_claim_from_text", family),
            ("image_timestamp", revision),
            ("license", license_name),
            ("license_url", license_url),
            *research_metadata,
        ),
    )
    candidate = WikimediaCandidate(
        metadata=CandidateMetadata(
            source="wikimedia",
            source_record_id=record_id,
            source_url=source_url,
            image_url=image_url,
            declared_reference="unverified",
            listing_text=(
                _CATEGORY_RESEARCH if query_index in _CATEGORY_QUERIES else
                _REPLICA_RESEARCH if query_index in _REPLICA_SEARCHES else _CLASSIFICATION
            ),
        ),
        rights=rights,
        rights_evidence=rights_evidence_bytes,
        observed_at=observed_at,
        author=author,
        license_name=license_name,
        license_url=license_url,
        classification=classification,
    )
    return record, candidate


def _cbp_public_domain_evidence(
    page: Mapping[str, object], metadata: Mapping[str, object],
    license_name: str, author: str, sha1: str, revision: str, source_url: str,
) -> dict[str, str] | None:
    if (
        license_name != "Public domain" or author != "U.S. Customs and Border Protection"
        or (page.get("pageid"), sha1, revision, source_url)
        != (_CBP_VERIFIED_FILE[0], _CBP_VERIFIED_FILE[1], _CBP_VERIFIED_FILE[2],
            _CBP_VERIFIED_FILE[4])
    ):
        return None
    templates = page.get("templates")
    if not isinstance(templates, list) or not any(
        isinstance(item, Mapping) and item.get("ns") == 10
        and item.get("title") == _CBP_PUBLIC_DOMAIN_TEMPLATE for item in templates
    ):
        return None
    values = {
        name: entry.get("value") if isinstance(entry, Mapping) else None
        for name, entry in metadata.items()
        if name in {"Artist", "Credit", "UsageTerms", "Copyrighted", "AttributionRequired"}
    }
    credit = values.get("Credit")
    artist = values.get("Artist")
    match = _CBP_FLICKR_CREDIT.search(credit) if isinstance(credit, str) else None
    if (
        match is None or match["id"] != _CBP_VERIFIED_FILE[3]
        or not isinstance(artist, str) or _CBP_ARTIST_URL not in artist
        or values.get("UsageTerms") != "Public domain"
        or values.get("Copyrighted") != "False"
        or values.get("AttributionRequired") != "false"
        or metadata.get("LicenseUrl") is not None
    ):
        return None
    return {
        "public_domain_template": _CBP_PUBLIC_DOMAIN_TEMPLATE,
        "flickr_photo_id": match["id"],
        "credit_url_literal": match["url"],
        "artist_url_literal": "https://www.flickr.com/people/54593278@N03",
        "usage_terms": "Public domain",
        "copyrighted": "False",
        "attribution_required": "false",
    }


def _metadata_value(metadata: Mapping[str, object], name: str) -> str:
    entry = metadata.get(name)
    if not isinstance(entry, Mapping):
        raise WikimediaAPIError(f"{name} metadata is missing")
    value = entry.get("value")
    if not isinstance(value, str) or not value.strip():
        raise WikimediaAPIError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_metadata_text(metadata: Mapping[str, object], name: str) -> str:
    entry = metadata.get(name)
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, str):
        return ""
    return " ".join(_TAG.sub(" ", html.unescape(value)).split())


def _license_url(
    metadata: Mapping[str, object], license_name: str, source_url: str
) -> str:
    entry = metadata.get("LicenseUrl")
    value = entry.get("value") if isinstance(entry, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        if license_name == "Public domain":
            return source_url
        raise WikimediaAPIError("LicenseUrl must be a non-empty string")
    try:
        parsed = urlsplit(value.strip())
    except ValueError as exc:
        raise WikimediaAPIError("LicenseUrl is invalid") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname != "creativecommons.org"
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 80, 443)
    ):
        raise WikimediaAPIError("LicenseUrl host is not approved")
    return urlunsplit(("https", parsed.netloc.split(":", 1)[0], parsed.path, parsed.query, ""))


def _plain_text(value: str) -> str:
    normalized = " ".join(_TAG.sub(" ", html.unescape(value)).split())
    if not normalized:
        raise WikimediaAPIError("image author is empty")
    return normalized


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise WikimediaAPIError(f"{field} must be a non-empty single-line string")
    return value.strip()


def _https_url(value: object, field: str, *allowed_hosts: str) -> str:
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise WikimediaAPIError(f"{field} must be an HTTPS URL")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise WikimediaAPIError(f"{field} host is not approved")
    return value
