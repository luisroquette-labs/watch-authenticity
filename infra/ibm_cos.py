"""Verified IBM COS storage for immutable training inputs and run events."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from infra.ibm_types import ObjectRef, RunEvent, RunSpec


class IntegrityError(RuntimeError):
    """Stored bytes or metadata differ from their immutable contract."""


class EventChainFork(IntegrityError):
    """More than one immutable event exists at the same sequence."""


def create_cos_client(spec: RunSpec) -> Any:
    """Create an OAuth COS client backed only by the VSI trusted profile."""

    import ibm_boto3
    from ibm_botocore.client import Config
    from ibm_cloud_sdk_core.authenticators import VPCInstanceAuthenticator

    auth = VPCInstanceAuthenticator(iam_profile_id=spec.trusted_profile_id)
    return ibm_boto3.client(
        "s3",
        ibm_service_instance_id=spec.cos_service_instance_id,
        token_manager=auth.token_manager,
        config=Config(signature_version="oauth"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    return isinstance(error, dict) and str(error.get("Code")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


def _head(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _not_found(exc):
            return None
        raise
    if not isinstance(response, dict):
        raise IntegrityError("COS head response must be a mapping")
    return response


def _object_ref(
    bucket: str,
    key: str,
    head: dict[str, Any],
    expected_sha256: str,
    expected_length: int,
    *,
    context: str,
) -> ObjectRef:
    metadata = head.get("Metadata")
    if not isinstance(metadata, dict):
        raise IntegrityError(f"{context} metadata is missing")
    try:
        metadata_length = int(metadata.get("length", ""))
        content_length = int(head["ContentLength"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError(f"{context} length metadata is invalid") from exc
    if (
        metadata.get("sha256") != expected_sha256
        or metadata_length != expected_length
        or content_length != expected_length
    ):
        raise IntegrityError(f"{context} hash or length does not match")
    etag = str(head.get("ETag", "")).strip('"')
    if not etag:
        raise IntegrityError(f"{context} ETag is missing")
    return ObjectRef(
        bucket=bucket,
        key=key,
        sha256=expected_sha256,
        length=expected_length,
        etag=etag,
    )


def _content_key(key: str, digest: str) -> str:
    base = key.strip("/")
    if not base:
        raise ValueError("key must be non-empty")
    return base if base.rsplit("/", 1)[-1] == digest else f"{base}/{digest}"


def upload_verified(
    client: Any,
    path: Path,
    bucket: str,
    key: str,
    expected_sha256: str,
) -> ObjectRef:
    """Upload through a temporary key and publish only verified content."""

    local_length = path.stat().st_size
    if _sha256(path) != expected_sha256:
        raise IntegrityError("local SHA-256 does not match expected_sha256")
    final_key = _content_key(key, expected_sha256)
    existing = _head(client, bucket, final_key)
    if existing is not None:
        try:
            return _object_ref(
                bucket,
                final_key,
                existing,
                expected_sha256,
                local_length,
                context="existing object",
            )
        except IntegrityError as exc:
            raise IntegrityError("existing object does not match immutable content") from exc

    temporary_key = f"{final_key}.upload-{uuid4()}"
    metadata = {"sha256": expected_sha256, "length": str(local_length)}
    final_written = False
    try:
        client.upload_file(
            str(path),
            bucket,
            temporary_key,
            ExtraArgs={"Metadata": metadata},
        )
        temporary_head = _head(client, bucket, temporary_key)
        if temporary_head is None:
            raise IntegrityError("temporary upload is missing")
        _object_ref(
            bucket,
            temporary_key,
            temporary_head,
            expected_sha256,
            local_length,
            context="temporary object",
        )
        client.copy_object(
            Bucket=bucket,
            Key=final_key,
            CopySource={"Bucket": bucket, "Key": temporary_key},
            MetadataDirective="COPY",
        )
        final_written = True
        final_head = _head(client, bucket, final_key)
        if final_head is None:
            raise IntegrityError("final object is missing after copy")
        return _object_ref(
            bucket,
            final_key,
            final_head,
            expected_sha256,
            local_length,
            context="remote object metadata",
        )
    except BaseException:
        if final_written:
            client.delete_object(Bucket=bucket, Key=final_key)
        raise
    finally:
        client.delete_object(Bucket=bucket, Key=temporary_key)


def download_verified(
    client: Any,
    bucket: str,
    key: str,
    destination: Path,
    expected_sha256: str,
) -> Path:
    """Download to a temporary path and publish only verified bytes."""

    head = _head(client, bucket, key)
    if head is None:
        raise FileNotFoundError(f"COS object not found: {bucket}/{key}")
    metadata = head.get("Metadata")
    if not isinstance(metadata, dict) or metadata.get("sha256") != expected_sha256:
        raise IntegrityError("remote object metadata hash does not match")
    try:
        expected_length = int(metadata.get("length", ""))
    except (TypeError, ValueError) as exc:
        raise IntegrityError("remote object metadata length is invalid") from exc
    _object_ref(
        bucket,
        key,
        head,
        expected_sha256,
        expected_length,
        context="remote object metadata",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.download-{uuid4()}")
    try:
        client.download_file(bucket, key, str(temporary))
        if temporary.stat().st_size != expected_length or _sha256(temporary) != expected_sha256:
            raise IntegrityError("downloaded object hash or length does not match")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def _event_bytes(event: RunEvent) -> bytes:
    def encode(value: UUID | datetime | str | int | None) -> str | int | None:
        if isinstance(value, UUID):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    payload = {
        name: encode(getattr(event, name))
        for name in (
            "event_id",
            "run_id",
            "sequence",
            "state",
            "observed_at",
            "instance_id",
            "lineage_sha256",
            "previous_event_sha256",
            "payload_sha256",
        )
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def append_event(
    client: Any,
    bucket: str,
    prefix: str,
    event: RunEvent,
) -> ObjectRef:
    """Write an event once under its collision-resistant immutable UUID key."""

    event_prefix = prefix.strip("/")
    if not event_prefix:
        raise ValueError("prefix must be non-empty")
    key = f"{event_prefix}/events/{event.sequence:06d}-{event.event_id}.json"
    body = _event_bytes(event)
    digest = hashlib.sha256(body).hexdigest()
    length = len(body)
    existing = _head(client, bucket, key)
    if existing is not None:
        try:
            return _object_ref(
                bucket,
                key,
                existing,
                digest,
                length,
                context="existing event",
            )
        except IntegrityError as exc:
            raise IntegrityError("event key already exists with different content") from exc

    written = False
    try:
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            Metadata={"sha256": digest, "length": str(length)},
            ContentType="application/json",
        )
        written = True
        head = _head(client, bucket, key)
        if head is None:
            raise IntegrityError("event is missing after write")
        return _object_ref(
            bucket,
            key,
            head,
            digest,
            length,
            context="event object",
        )
    except BaseException:
        if written:
            client.delete_object(Bucket=bucket, Key=key)
        raise


def validate_event_chain(events: tuple[RunEvent, ...], spec: RunSpec) -> None:
    """Reject duplicate sequence branches; later tasks extend full chain checks."""

    sequences: dict[int, UUID] = {}
    for event in events:
        if event.run_id != spec.run_id:
            raise IntegrityError("event run_id does not match RunSpec")
        previous = sequences.get(event.sequence)
        if previous is not None and previous != event.event_id:
            raise EventChainFork(f"multiple event IDs found at sequence {event.sequence}")
        sequences[event.sequence] = event.event_id
