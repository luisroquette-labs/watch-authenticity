"""Content-addressed, resumable source-discovery ledger."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit


class DiscoveryConflict(ValueError):
    pass


@dataclass(frozen=True)
class DiscoveryRecord:
    source: str
    source_record_id: str
    source_version: str
    source_url: str
    seller_group_id: str
    declared_reference: str
    image_ids: tuple[str, ...]
    image_urls: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "source",
            "source_record_id",
            "source_version",
            "seller_group_id",
            "declared_reference",
        ):
            _single_line(getattr(self, field), field)
        _https_url(self.source_url, "source_url")
        if not isinstance(self.image_ids, tuple) or not self.image_ids:
            raise ValueError("image_ids must be a non-empty tuple")
        if len(set(self.image_ids)) != len(self.image_ids):
            raise ValueError("image_ids must not contain duplicates")
        for image_id in self.image_ids:
            _single_line(image_id, "image_id")
        if len(self.image_ids) != len(self.image_urls):
            raise ValueError("image_ids must align with image_urls")
        if not isinstance(self.image_urls, tuple) or not self.image_urls:
            raise ValueError("image_urls must be a non-empty tuple")
        if len(set(self.image_urls)) != len(self.image_urls):
            raise ValueError("image_urls must not contain duplicates")
        for image_url in self.image_urls:
            _https_url(image_url, "image_url")


@dataclass(frozen=True)
class DiscoveryPage:
    source: str
    access_record_id: str
    cursor: str | None
    next_cursor: str | None
    observed_at: datetime
    records: tuple[DiscoveryRecord, ...]

    def __post_init__(self) -> None:
        _single_line(self.source, "source")
        _single_line(self.access_record_id, "access_record_id")
        for name, value in (("cursor", self.cursor), ("next_cursor", self.next_cursor)):
            if value is not None:
                _single_line(value, name)
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("observed_at must be timezone-aware")
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("discovery page requires records")
        if any(record.source != self.source for record in self.records):
            raise ValueError("discovery record source differs from page")
        ids = [record.source_record_id for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("discovery page contains duplicate record IDs")


@dataclass(frozen=True)
class DiscoveryCommit:
    source: str
    access_record_id: str
    previous_sha256: str | None
    page_sha256: str
    next_cursor: str | None
    record_hashes: tuple[tuple[str, str], ...]
    sha256: str


def append_discovery_page(
    cos_client: Any,
    bucket: str,
    prefix: str,
    page: DiscoveryPage,
    *,
    previous: DiscoveryCommit | None = None,
) -> DiscoveryCommit:
    """Persist a complete page before returning its resumable cursor commit."""
    root = prefix.rstrip("/")
    if previous is not None:
        if previous.source != page.source or previous.access_record_id != page.access_record_id:
            raise DiscoveryConflict("discovery page belongs to another ledger")
        if previous.next_cursor is None or page.cursor != previous.next_cursor:
            raise DiscoveryConflict("discovery cursor fork")
        parent = previous.previous_sha256 or "root"
        authoritative = _read_if_present(
            cos_client,
            bucket,
            f"{root}/{page.source}/children/{parent}.json",
        )
        if authoritative != _commit_bytes(previous):
            raise DiscoveryConflict("previous discovery commit is not authoritative")
    elif page.cursor is not None:
        raise DiscoveryConflict("first discovery page must use an empty cursor")

    records = tuple(sorted(page.records, key=lambda record: record.source_record_id))
    record_hashes = dict(previous.record_hashes) if previous is not None else {}
    record_documents: list[dict[str, object]] = []
    for record in records:
        document = _record_document(record)
        digest = _sha256(document)
        known = record_hashes.get(record.source_record_id)
        if known is not None and known != digest:
            raise DiscoveryConflict(
                f"divergent discovery record: {record.source_record_id}"
            )
        record_hashes[record.source_record_id] = digest
        record_documents.append(document)

    page_document = {
        "source": page.source,
        "access_record_id": page.access_record_id,
        "cursor": page.cursor,
        "next_cursor": page.next_cursor,
        "observed_at": page.observed_at.isoformat(),
        "records": record_documents,
    }
    page_bytes = _canonical(page_document)
    page_sha256 = hashlib.sha256(page_bytes).hexdigest()
    _put_immutable(
        cos_client,
        bucket,
        f"{root}/{page.source}/pages/{page_sha256}.json",
        page_bytes,
        page_sha256,
    )

    commit_document = {
        "source": page.source,
        "access_record_id": page.access_record_id,
        "previous_sha256": previous.sha256 if previous else None,
        "page_sha256": page_sha256,
        "next_cursor": page.next_cursor,
        "record_hashes": sorted(record_hashes.items()),
    }
    commit_bytes = _canonical(commit_document)
    commit_sha256 = hashlib.sha256(commit_bytes).hexdigest()
    _put_immutable(
        cos_client,
        bucket,
        f"{root}/{page.source}/commits/{commit_sha256}.json",
        commit_bytes,
        commit_sha256,
    )
    parent = previous.sha256 if previous else "root"
    _put_immutable(
        cos_client,
        bucket,
        f"{root}/{page.source}/children/{parent}.json",
        commit_bytes,
        commit_sha256,
    )
    return DiscoveryCommit(
        source=page.source,
        access_record_id=page.access_record_id,
        previous_sha256=previous.sha256 if previous else None,
        page_sha256=page_sha256,
        next_cursor=page.next_cursor,
        record_hashes=tuple(sorted(record_hashes.items())),
        sha256=commit_sha256,
    )


def _commit_bytes(commit: DiscoveryCommit) -> bytes:
    return _canonical(
        {
            "source": commit.source,
            "access_record_id": commit.access_record_id,
            "previous_sha256": commit.previous_sha256,
            "page_sha256": commit.page_sha256,
            "next_cursor": commit.next_cursor,
            "record_hashes": list(commit.record_hashes),
        }
    )


def _record_document(record: DiscoveryRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "source_record_id": record.source_record_id,
        "source_version": record.source_version,
        "source_url": record.source_url,
        "seller_group_id": record.seller_group_id,
        "declared_reference": record.declared_reference,
        "image_ids": list(record.image_ids),
        "image_urls": list(record.image_urls),
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _put_immutable(client: Any, bucket: str, key: str, body: bytes, digest: str) -> None:
    existing = _read_if_present(client, bucket, key)
    if existing is not None:
        if existing != body:
            raise DiscoveryConflict(f"immutable discovery object diverged: {key}")
        return
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        Metadata={"sha256": digest},
        ContentType="application/json",
    )
    if _read_if_present(client, bucket, key) != body:
        raise DiscoveryConflict(f"discovery object readback failed: {key}")


def _read_if_present(client: Any, bucket: str, key: str) -> bytes | None:
    try:
        client.head_object(Bucket=bucket, Key=key)
        response = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _not_found(exc):
            return None
        raise
    stream = response["Body"]
    if not isinstance(stream, io.BytesIO) and not hasattr(stream, "read"):
        raise DiscoveryConflict("discovery object body is unreadable")
    body = stream.read()
    if not isinstance(body, bytes):
        raise DiscoveryConflict("discovery object body is not bytes")
    return body


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return isinstance(response, dict) and response.get("Error", {}).get("Code") in {
        "NoSuchKey",
        "404",
        "NotFound",
    }


def _single_line(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\n" in value or "\r" in value:
        raise ValueError(f"{field} must be a non-empty single-line string")


def _https_url(value: object, field: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an absolute HTTPS URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{field} must be an absolute HTTPS URL without credentials")
