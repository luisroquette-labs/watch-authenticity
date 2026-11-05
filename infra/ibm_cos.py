"""Verified IBM COS storage for immutable training inputs and run events."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from infra.ibm_types import ObjectRef, RunEvent, RunSpec, validate_run_event_chain


class IntegrityError(RuntimeError):
    """Stored bytes or metadata differ from their immutable contract."""


class EventChainFork(IntegrityError):
    """More than one immutable event exists at the same sequence."""


def verify_cos_configuration(client: Any, spec: RunSpec) -> None:
    """Fail closed unless the bound bucket is private and immutable."""

    _require_bound_destination(client, spec.bucket, prefix=spec.prefix)
    public = client.get_public_access_block(Bucket=spec.bucket)
    if not isinstance(public, dict):
        raise IntegrityError("COS public-access response must be a mapping")
    configuration = public.get("PublicAccessBlockConfiguration", {})
    required = {
        "BlockPublicAcls",
        "IgnorePublicAcls",
        "BlockPublicPolicy",
        "RestrictPublicBuckets",
    }
    if not isinstance(configuration, dict) or any(
        configuration.get(name) is not True for name in required
    ):
        raise IntegrityError("COS bucket does not block all public access")
    versioning = client.get_bucket_versioning(Bucket=spec.bucket)
    if not isinstance(versioning, dict) or versioning.get("Status") != "Enabled":
        raise IntegrityError("COS bucket versioning is not enabled")


def create_cos_client(spec: RunSpec) -> Any:
    """Create an OAuth COS client backed only by the VSI trusted profile."""

    import ibm_boto3
    from ibm_botocore.client import Config
    from ibm_cloud_sdk_core.authenticators import VPCInstanceAuthenticator

    auth = VPCInstanceAuthenticator(iam_profile_id=spec.trusted_profile_id)
    client = ibm_boto3.client(
        "s3",
        endpoint_url=spec.cos_endpoint,
        ibm_service_instance_id=spec.cos_service_instance_id,
        token_manager=auth.token_manager,
        config=Config(signature_version="oauth"),
    )
    setattr(client, "_watch_auth_bucket", spec.bucket)
    setattr(client, "_watch_auth_prefix", spec.prefix.strip("/"))
    return client


def _require_bound_destination(
    client: Any,
    bucket: str,
    *,
    prefix: str | None = None,
) -> None:
    if getattr(client, "_watch_auth_bucket", None) != bucket:
        raise IntegrityError("COS client is bound to a different bucket")
    if prefix is not None and getattr(client, "_watch_auth_prefix", None) != prefix.strip("/"):
        raise IntegrityError("COS client is bound to a different run prefix")


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


def _verify_remote_bytes(
    client: Any,
    bucket: str,
    key: str,
    expected_sha256: str,
    expected_length: int,
    *,
    context: str,
) -> None:
    handle = tempfile.NamedTemporaryFile(prefix="watch-auth-cos-", delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        client.download_file(bucket, key, str(temporary))
        if temporary.stat().st_size != expected_length or _sha256(temporary) != expected_sha256:
            raise IntegrityError(f"{context} hash or length does not match")
    finally:
        temporary.unlink(missing_ok=True)


def upload_verified(
    client: Any,
    path: Path,
    bucket: str,
    key: str,
    expected_sha256: str,
) -> ObjectRef:
    """Upload through a temporary key and publish only verified content."""

    _require_bound_destination(client, bucket)
    local_length = path.stat().st_size
    if _sha256(path) != expected_sha256:
        raise IntegrityError("local SHA-256 does not match expected_sha256")
    final_key = _content_key(key, expected_sha256)
    existing = _head(client, bucket, final_key)
    if existing is not None:
        ref = _object_ref(
            bucket,
            final_key,
            existing,
            expected_sha256,
            local_length,
            context="existing object",
        )
        _verify_remote_bytes(
            client,
            bucket,
            final_key,
            expected_sha256,
            local_length,
            context="existing remote object bytes",
        )
        return ref

    temporary_key = f"{final_key}.upload-{uuid4()}"
    metadata = {"sha256": expected_sha256, "length": str(local_length)}
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
        _verify_remote_bytes(
            client,
            bucket,
            temporary_key,
            expected_sha256,
            local_length,
            context="temporary object bytes",
        )
        client.copy_object(
            Bucket=bucket,
            Key=final_key,
            CopySource={"Bucket": bucket, "Key": temporary_key},
            MetadataDirective="COPY",
        )
        final_head = _head(client, bucket, final_key)
        if final_head is None:
            raise IntegrityError("final object is missing after copy")
        ref = _object_ref(
            bucket,
            final_key,
            final_head,
            expected_sha256,
            local_length,
            context="remote object metadata",
        )
        _verify_remote_bytes(
            client,
            bucket,
            final_key,
            expected_sha256,
            local_length,
            context="remote object bytes",
        )
        return ref
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

    _require_bound_destination(client, bucket)
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


def load_event_inventory(client: Any, spec: RunSpec) -> tuple[RunEvent, ...]:
    """List and verify every persisted event for a run prefix."""

    _require_bound_destination(client, spec.bucket, prefix=spec.prefix)
    event_prefix = f"{spec.prefix.strip('/')}/events/"
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": spec.bucket, "Prefix": event_prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        if not isinstance(page, dict):
            raise IntegrityError("COS event listing response must be a mapping")
        contents = page.get("Contents", [])
        if not isinstance(contents, list):
            raise IntegrityError("COS event listing contents are invalid")
        for item in contents:
            if not isinstance(item, dict) or not isinstance(item.get("Key"), str):
                raise IntegrityError("COS event listing key is invalid")
            key = item["Key"]
            if not key.startswith(event_prefix):
                raise IntegrityError("COS event listing escaped the run prefix")
            keys.append(key)
        if not page.get("IsTruncated"):
            break
        next_token = page.get("NextContinuationToken")
        if not isinstance(next_token, str) or not next_token or next_token == token:
            raise IntegrityError("COS event listing pagination is incomplete")
        token = next_token

    if len(set(keys)) != len(keys):
        raise IntegrityError("COS event listing contains duplicate keys")
    events: list[RunEvent] = []
    for key in sorted(keys):
        response = client.get_object(Bucket=spec.bucket, Key=key)
        body_stream = response.get("Body") if isinstance(response, dict) else None
        if body_stream is None or not hasattr(body_stream, "read"):
            raise IntegrityError("COS event body is unavailable")
        body = body_stream.read()
        if not isinstance(body, bytes):
            raise IntegrityError("COS event body is unavailable")
        digest = hashlib.sha256(body).hexdigest()
        if not key.endswith(f"-{digest}.json"):
            raise IntegrityError("COS event key digest does not match body")
        try:
            payload = json.loads(body)
            event = RunEvent(
                event_id=UUID(payload["event_id"]),
                run_id=UUID(payload["run_id"]),
                sequence=payload["sequence"],
                state=payload["state"],
                observed_at=datetime.fromisoformat(payload["observed_at"]),
                instance_id=payload["instance_id"],
                lineage_sha256=payload["lineage_sha256"],
                previous_event_sha256=payload["previous_event_sha256"],
                payload_sha256=payload["payload_sha256"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError("COS event payload is invalid") from exc
        events.append(event)
    snapshot = tuple(events)
    if snapshot:
        validate_event_chain(snapshot, spec)
    return snapshot


def append_event(
    client: Any,
    bucket: str,
    prefix: str,
    event: RunEvent,
) -> ObjectRef:
    """Write an event once under its collision-resistant immutable UUID key."""

    _require_bound_destination(client, bucket, prefix=prefix)
    event_prefix = prefix.strip("/")
    if not event_prefix:
        raise ValueError("prefix must be non-empty")
    body = _event_bytes(event)
    digest = hashlib.sha256(body).hexdigest()
    key = f"{event_prefix}/events/{event.sequence:06d}-{event.event_id}-{digest}.json"
    length = len(body)
    existing = _head(client, bucket, key)
    if existing is not None:
        try:
            ref = _object_ref(
                bucket,
                key,
                existing,
                digest,
                length,
                context="existing event",
            )
            _verify_remote_bytes(
                client,
                bucket,
                key,
                digest,
                length,
                context="existing event bytes",
            )
            return ref
        except IntegrityError as exc:
            raise IntegrityError("event key already exists with different content") from exc

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=body,
        Metadata={"sha256": digest, "length": str(length)},
        ContentType="application/json",
    )
    head = _head(client, bucket, key)
    if head is None:
        raise IntegrityError("event is missing after write")
    ref = _object_ref(
        bucket,
        key,
        head,
        digest,
        length,
        context="event object",
    )
    _verify_remote_bytes(
        client,
        bucket,
        key,
        digest,
        length,
        context="event object bytes",
    )
    return ref


def validate_event_chain(events: tuple[RunEvent, ...], spec: RunSpec) -> None:
    """Validate a complete linear lifecycle chain for one RunSpec."""

    from infra.ibm_evidence import run_spec_sha256

    expected_lineage = run_spec_sha256(spec)
    sequences: dict[int, RunEvent] = {}
    for event in events:
        if event.run_id != spec.run_id:
            raise IntegrityError("event run_id does not match RunSpec")
        if event.lineage_sha256 != expected_lineage:
            raise IntegrityError("event lineage does not match RunSpec")
        previous = sequences.get(event.sequence)
        if previous is not None and previous != event:
            raise EventChainFork(f"multiple event IDs found at sequence {event.sequence}")
        sequences[event.sequence] = event
    try:
        validate_run_event_chain(events, spec.run_id)
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc
