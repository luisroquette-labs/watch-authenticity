"""Authorized, bounded image collection with immutable COS receipts."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

from acquisition.source_policy import authorize_candidate
from acquisition.types import BatchReceipt, CandidateRecord, RightsRecord, SourcePolicy
from core.rules import propose_source_label
from infra.ibm_cos import IntegrityError, upload_verified

MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
HTTP_TIMEOUT = (10, 30)
MAX_HTTP_ATTEMPTS = 3
_CONTENT_TYPES = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}


class AcquisitionRefused(PermissionError):
    """Policy or rights rejected collection before any external call."""


class UnsafeImage(ValueError):
    """Downloaded bytes are not a safe, complete single-frame image."""


@dataclass(frozen=True)
class CandidateMetadata:
    source: str
    source_record_id: str
    source_url: str
    image_url: str
    declared_reference: str
    listing_text: str | None = None
    factory: str | None = None
    price_usd: Decimal | None = None
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in ("source", "source_record_id", "declared_reference"):
            value = getattr(self, field)
            if not value or "\n" in value or "\r" in value:
                raise ValueError(f"{field} must be a non-empty single-line string")
        for field in ("source_url", "image_url"):
            parsed = urlsplit(getattr(self, field))
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"{field} must be an absolute HTTP(S) URL")
        if self.price_usd is not None and (
            not isinstance(self.price_usd, Decimal)
            or not self.price_usd.is_finite()
            or self.price_usd < 0
        ):
            raise ValueError("price_usd must be finite and non-negative")
        if self.expected_sha256 is not None and (
            len(self.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.expected_sha256)
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256")


@dataclass(frozen=True)
class SourceUnavailable:
    source: str
    reason: str


def normalize_candidate_metadata(source: str, record: Mapping[str, object]) -> CandidateMetadata:
    """Normalize one pre-fetched API/feed/export record without doing I/O."""

    def required(name: str) -> str:
        value = record.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    price = record.get("price_usd", record.get("price"))
    record_source = record.get("source", source)
    if not isinstance(record_source, str) or not record_source.strip():
        raise ValueError("source must be a non-empty string")
    expected = record.get("expected_sha256")
    if expected is not None and (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("expected_sha256 must be a lowercase SHA-256")
    return CandidateMetadata(
        source=record_source.strip().lower(),
        source_record_id=required("source_record_id"),
        source_url=required("source_url"),
        image_url=required("image_url"),
        declared_reference=required("declared_reference"),
        listing_text=_optional_string(record.get("listing_text")),
        factory=_optional_string(record.get("factory")),
        price_usd=None if price is None else Decimal(str(price)),
        expected_sha256=expected,
    )


def collect_batch(
    records: Iterable[CandidateMetadata | Mapping[str, object]],
    policy: SourcePolicy,
    rights: RightsRecord | None,
    *,
    http_client: Any,
    cos_client: Any,
    bucket: str,
    batch_id: str,
    now: datetime,
) -> BatchReceipt:
    """Collect an authorized batch; refuse policy/rights before HTTP or COS."""

    decision = authorize_candidate(policy, rights, now=now)
    if not decision.may_download:
        raise AcquisitionRefused(", ".join(decision.reasons))
    assert rights is not None  # authorize_candidate proved this before any I/O
    _safe_id(batch_id, "batch_id")

    normalized_records = tuple(
        record
        if isinstance(record, CandidateMetadata)
        else normalize_candidate_metadata(policy.source, record)
        for record in records
    )
    authorized_source = policy.source.strip().lower()
    if any(record.source != authorized_source for record in normalized_records):
        raise AcquisitionRefused("record source differs from authorized policy")

    accepted: list[str] = []
    receipt_hashes: list[str] = []
    for metadata in normalized_records:
        payload, raw_sha256 = _download_image(http_client, metadata)
        if metadata.expected_sha256 is not None and raw_sha256 != metadata.expected_sha256:
            raise IntegrityError("downloaded image hash does not match expected_sha256")
        derivative, phash = sanitize_image(payload)
        derivative_sha256 = hashlib.sha256(derivative).hexdigest()
        candidate_id = "candidate-" + hashlib.sha256(
            f"{batch_id}\0{policy.policy_id}\0{metadata.source_record_id}".encode()
        ).hexdigest()[:24]
        candidate = CandidateRecord(
            candidate_id=candidate_id,
            acquisition_batch_id=batch_id,
            source_policy_id=policy.policy_id,
            source_record_id=metadata.source_record_id,
            source_url=metadata.source_url,
            collected_at=now,
            declared_reference=metadata.declared_reference,
            proposed_label=propose_source_label(
                source=metadata.source,
                listing_text=metadata.listing_text,
            ),
            proposed_grade=None,
            factory=metadata.factory,
            price_usd=metadata.price_usd,
            raw_sha256=raw_sha256,
            phash=phash,
            rights_record_id=rights.rights_record_id,
        )
        receipt = _receipt_bytes(candidate, derivative_sha256)
        receipt_key = f"datasets/receipts/{batch_id}/{candidate_id}.json"
        existing = _read_receipt(cos_client, bucket, receipt_key)
        if existing is not None and existing != receipt:
            raise IntegrityError(f"receipt collision: {receipt_key}")

        _upload_bytes(cos_client, bucket, f"datasets/raw/{raw_sha256}", payload, raw_sha256)
        _upload_bytes(
            cos_client,
            bucket,
            f"datasets/derivatives/{derivative_sha256}",
            derivative,
            derivative_sha256,
        )
        if existing is None:
            _write_receipt(cos_client, bucket, receipt_key, receipt)
        accepted.append(candidate_id)
        receipt_hashes.append(hashlib.sha256(receipt).hexdigest())

    batch_sha256 = hashlib.sha256(
        json.dumps(receipt_hashes, separators=(",", ":")).encode()
    ).hexdigest()
    return BatchReceipt(
        batch_id=batch_id,
        batch_sha256=batch_sha256,
        policy_sha256=policy.policy_evidence_sha256,
        rights_sha256=rights.evidence_sha256,
        accepted_candidate_ids=tuple(accepted),
        refused_candidate_ids=(),
    )


def sanitize_image(payload: bytes) -> tuple[bytes, str]:
    """Validate image structure and return metadata-free RGB PNG plus pHash."""

    image_format = _magic_format(payload)
    _require_exact_container(payload, image_format)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as probe:
                if probe.format != image_format:
                    raise UnsafeImage("decoded format differs from magic bytes")
                if getattr(probe, "n_frames", 1) != 1:
                    raise UnsafeImage("multi-frame images are not accepted")
                width, height = probe.size
                if width * height > MAX_IMAGE_PIXELS:
                    raise UnsafeImage("image exceeds 100 megapixels")
                probe.verify()
            with Image.open(io.BytesIO(payload)) as decoded:
                decoded.load()
                rgb = decoded.convert("RGB")
                phash = _average_hash(rgb)
                output = io.BytesIO()
                rgb.save(output, format="PNG", optimize=False)
                return output.getvalue(), phash
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise UnsafeImage("decompression bomb rejected") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        if isinstance(exc, UnsafeImage):
            raise
        raise UnsafeImage("image decode or verification failed") from exc


def _download_image(http_client: Any, metadata: CandidateMetadata) -> tuple[bytes, str]:
    last_status: int | None = None
    for attempt in range(1, MAX_HTTP_ATTEMPTS + 1):
        try:
            response = http_client.get(
                metadata.image_url,
                stream=True,
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            if attempt == MAX_HTTP_ATTEMPTS:
                raise IntegrityError(f"image download failed after {attempt} attempts") from exc
            continue
        try:
            last_status = int(response.status_code)
            if last_status == 429 or 500 <= last_status <= 599:
                if attempt == MAX_HTTP_ATTEMPTS:
                    raise IntegrityError(f"image download failed after {attempt} attempts")
                continue
            if not 200 <= last_status <= 299:
                raise IntegrityError(f"image download returned HTTP {last_status}")
            content_type = str(response.headers.get("Content-Type", "")).split(";", 1)[0].lower()
            if content_type not in _CONTENT_TYPES:
                raise UnsafeImage("response content type is not JPEG, PNG or WebP")
            content_length = _content_length(response.headers.get("Content-Length"))
            if content_length is not None and content_length > MAX_IMAGE_BYTES:
                raise UnsafeImage("image exceeds 25 MiB")
            body = bytearray()
            digest = hashlib.sha256()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not isinstance(chunk, bytes):
                    raise IntegrityError("HTTP stream yielded non-bytes")
                body.extend(chunk)
                digest.update(chunk)
                if len(body) > MAX_IMAGE_BYTES:
                    raise UnsafeImage("image exceeds 25 MiB")
            if content_length is not None and len(body) != content_length:
                raise IntegrityError("HTTP body length differs from Content-Length")
            image_format = _magic_format(bytes(body))
            if _CONTENT_TYPES[content_type] != image_format:
                raise UnsafeImage("content type does not match image magic")
            return bytes(body), digest.hexdigest()
        finally:
            response.close()
    raise IntegrityError(f"image download failed after {MAX_HTTP_ATTEMPTS} attempts: {last_status}")


def _content_length(value: object) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise IntegrityError("invalid Content-Length")
    try:
        length = int(value)
    except (TypeError, ValueError) as exc:
        raise IntegrityError("invalid Content-Length") from exc
    if length < 0:
        raise IntegrityError("invalid Content-Length")
    return length


def _magic_format(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "WEBP"
    raise UnsafeImage("unsupported or missing image magic")


def _require_exact_container(payload: bytes, image_format: str) -> None:
    if image_format == "JPEG":
        _require_exact_jpeg(payload)
        return
    if image_format == "WEBP":
        if len(payload) < 12 or int.from_bytes(payload[4:8], "little") + 8 != len(payload):
            raise UnsafeImage("truncated or polyglot WebP")
        return

    offset = 8
    found_iend = False
    while offset + 12 <= len(payload):
        length = int.from_bytes(payload[offset : offset + 4], "big")
        chunk_type = payload[offset + 4 : offset + 8]
        offset += 12 + length
        if offset > len(payload):
            raise UnsafeImage("truncated PNG")
        if chunk_type == b"IEND":
            found_iend = length == 0
            break
    if not found_iend or offset != len(payload):
        raise UnsafeImage("truncated or polyglot PNG")


def _require_exact_jpeg(payload: bytes) -> None:
    offset = 2
    in_scan = False
    while offset < len(payload):
        if payload[offset] != 0xFF:
            if not in_scan:
                raise UnsafeImage("invalid JPEG marker structure")
            offset += 1
            continue
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker == 0x00 and in_scan:
            continue
        if marker == 0xD9:
            if offset != len(payload):
                raise UnsafeImage("polyglot JPEG")
            return
        if in_scan and 0xD0 <= marker <= 0xD7:
            continue
        in_scan = False
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if offset + 2 > len(payload):
            break
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            break
        if marker == 0xDA:
            in_scan = True
        offset += segment_length
    raise UnsafeImage("truncated JPEG")


def _average_hash(image: Image.Image) -> str:
    grayscale = image.resize((8, 8)).convert("L")
    pixels = cast(list[int], list(grayscale.get_flattened_data()))
    average = sum(pixels) / len(pixels)
    bits = sum((pixel >= average) << index for index, pixel in enumerate(pixels))
    return f"{bits:016x}"


def _receipt_bytes(candidate: CandidateRecord, derivative_sha256: str) -> bytes:
    document = {
        "batch_id": candidate.acquisition_batch_id,
        "candidate_id": candidate.candidate_id,
        "collected_at": candidate.collected_at.isoformat(),
        "declared_reference": candidate.declared_reference,
        "derivative_key": f"datasets/derivatives/{derivative_sha256}",
        "derivative_sha256": derivative_sha256,
        "factory": candidate.factory,
        "phash": candidate.phash,
        "policy_id": candidate.source_policy_id,
        "price_usd": None if candidate.price_usd is None else str(candidate.price_usd),
        "proposed_grade": candidate.proposed_grade,
        "proposed_label": (
            None if candidate.proposed_label is None else candidate.proposed_label.value
        ),
        "raw_key": f"datasets/raw/{candidate.raw_sha256}",
        "raw_sha256": candidate.raw_sha256,
        "rights_record_id": candidate.rights_record_id,
        "source_record_id": candidate.source_record_id,
        "source_url": candidate.source_url,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


def _upload_bytes(
    cos_client: Any,
    bucket: str,
    key: str,
    body: bytes,
    sha256: str,
) -> None:
    with tempfile.NamedTemporaryFile(prefix="watch-auth-collect-", delete=False) as handle:
        path = Path(handle.name)
        handle.write(body)
    try:
        upload_verified(cos_client, path, bucket, key, sha256)
    finally:
        path.unlink(missing_ok=True)


def _read_cos_bytes(cos_client: Any, bucket: str, key: str) -> bytes | None:
    try:
        response = cos_client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    body = response.get("Body") if isinstance(response, dict) else None
    read = getattr(body, "read", None)
    result = read() if callable(read) else body
    if not isinstance(result, bytes):
        raise IntegrityError("COS receipt body is unavailable")
    return result


def _read_receipt(cos_client: Any, bucket: str, key: str) -> bytes | None:
    try:
        head = cos_client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _is_not_found(exc):
            return None
        raise
    body = _read_cos_bytes(cos_client, bucket, key)
    metadata = head.get("Metadata") if isinstance(head, dict) else None
    if body is None or not isinstance(metadata, dict):
        raise IntegrityError("receipt metadata is unavailable")
    if metadata.get("sha256") != hashlib.sha256(body).hexdigest():
        raise IntegrityError("receipt metadata hash differs from body")
    if metadata.get("length") != str(len(body)) or head.get("ContentLength") != len(body):
        raise IntegrityError("receipt metadata length differs from body")
    return body


def _write_receipt(cos_client: Any, bucket: str, key: str, body: bytes) -> None:
    digest = hashlib.sha256(body).hexdigest()
    try:
        cos_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={"sha256": digest, "length": str(len(body))},
            ContentType="application/json",
            IfNoneMatch="*",
        )
    except Exception as exc:
        if not _is_precondition_conflict(exc):
            raise
        if _read_receipt(cos_client, bucket, key) != body:
            raise IntegrityError(f"receipt collision: {key}") from exc
        return
    if _read_receipt(cos_client, bucket, key) != body:
        raise IntegrityError(f"receipt collision: {key}")


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, dict) else None
    return isinstance(error, dict) and str(error.get("Code")) in {"404", "NoSuchKey", "NotFound"}


def _is_precondition_conflict(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    metadata = response.get("ResponseMetadata")
    error = response.get("Error")
    return (
        isinstance(metadata, dict)
        and metadata.get("HTTPStatusCode") == 412
        or isinstance(error, dict)
        and str(error.get("Code")) in {"412", "PreconditionFailed"}
    )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("optional text fields must be non-empty strings")
    return value.strip()


def _safe_id(value: str, field: str) -> None:
    if not value or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in value):
        raise ValueError(f"{field} must be a safe identifier")
