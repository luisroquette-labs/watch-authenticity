"""IBM COS integrity, authentication and immutable-event tests."""

from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest

from infra.ibm_cos import (
    EventChainFork,
    IntegrityError,
    append_event,
    create_cos_client,
    download_verified,
    upload_verified,
    validate_event_chain,
)
from infra.ibm_types import RunEvent
from tests.conftest import NOW, valid_run_spec


class _NotFound(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeCOS:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.fail_upload = False
        self.corrupt_final_head = False
        self.corrupt_download = False
        self.upload_calls = 0

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        del Bucket
        try:
            body, metadata = self.objects[Key]
        except KeyError as exc:
            raise _NotFound() from exc
        if self.corrupt_final_head and ".upload-" not in Key:
            metadata = {"sha256": "0" * 64, "length": str(len(body))}
        return {
            "Metadata": metadata,
            "ContentLength": len(body),
            "ETag": f'"etag-{len(body)}"',
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, object],
    ) -> None:
        del bucket
        self.upload_calls += 1
        if self.fail_upload:
            raise OSError("multipart interrupted")
        metadata = ExtraArgs["Metadata"]
        assert isinstance(metadata, dict)
        self.objects[key] = (Path(filename).read_bytes(), dict(metadata))

    def copy_object(
        self,
        *,
        Bucket: str,
        Key: str,
        CopySource: dict[str, str],
        MetadataDirective: str,
    ) -> None:
        del Bucket, MetadataDirective
        self.objects[Key] = self.objects[CopySource["Key"]]

    def delete_object(self, *, Bucket: str, Key: str) -> None:
        del Bucket
        self.objects.pop(Key, None)

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        del bucket
        body = self.objects[key][0]
        Path(filename).write_bytes(body + (b"corrupt" if self.corrupt_download else b""))

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata: dict[str, str],
        ContentType: str,
    ) -> dict[str, str]:
        del Bucket, ContentType
        self.objects[Key] = (Body, Metadata)
        return {"ETag": f'"etag-{len(Body)}"'}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event(event_id: UUID, sequence: int = 3) -> RunEvent:
    return RunEvent(
        event_id=event_id,
        run_id=valid_run_spec().run_id,
        sequence=sequence,
        state="running",
        observed_at=NOW,
        instance_id="vsi-1",
        lineage_sha256="a" * 64,
        previous_event_sha256="b" * 64,
        payload_sha256="c" * 64,
    )


def test_create_client_uses_trusted_profile_token_manager_only(monkeypatch):
    calls: dict[str, object] = {}
    token_manager = object()

    class Authenticator:
        def __init__(self, *, iam_profile_id: str) -> None:
            calls["profile"] = iam_profile_id
            self.token_manager = token_manager

    class Config:
        def __init__(self, *, signature_version: str) -> None:
            calls["signature"] = signature_version

    def client(service: str, **kwargs: object) -> object:
        calls["service"] = service
        calls["kwargs"] = kwargs
        return object()

    boto = ModuleType("ibm_boto3")
    boto.client = client
    auth_module = ModuleType("ibm_cloud_sdk_core.authenticators")
    auth_module.VPCInstanceAuthenticator = Authenticator
    botocore_client = ModuleType("ibm_botocore.client")
    botocore_client.Config = Config
    monkeypatch.setitem(sys.modules, "ibm_boto3", boto)
    monkeypatch.setitem(sys.modules, "ibm_cloud_sdk_core.authenticators", auth_module)
    monkeypatch.setitem(sys.modules, "ibm_botocore.client", botocore_client)

    spec = valid_run_spec()
    create_cos_client(spec)

    assert calls["profile"] == spec.trusted_profile_id
    assert calls["signature"] == "oauth"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["token_manager"] is token_manager
    assert kwargs["ibm_service_instance_id"] == spec.cos_service_instance_id
    assert kwargs["endpoint_url"] == spec.cos_endpoint
    assert not ({"api_key", "aws_access_key_id", "aws_secret_access_key"} & kwargs.keys())


def test_upload_uses_digest_key_and_verifies_remote_object(tmp_path: Path):
    source = tmp_path / "dataset.json"
    source.write_bytes(b"immutable dataset")
    digest = _sha256(source)
    cos = FakeCOS()

    ref = upload_verified(cos, source, "private", "objects/dataset", digest)

    assert ref.key == f"objects/dataset/{digest}"
    assert ref.sha256 == digest
    assert ref.length == source.stat().st_size
    assert all(".upload-" not in key for key in cos.objects)


def test_upload_rejects_local_or_remote_checksum_mismatch(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    with pytest.raises(IntegrityError, match="local SHA-256"):
        upload_verified(cos, source, "private", "objects/artifact", "0" * 64)
    assert cos.upload_calls == 0

    cos.corrupt_final_head = True
    with pytest.raises(IntegrityError, match="remote object metadata"):
        upload_verified(cos, source, "private", "objects/artifact", _sha256(source))
    assert not cos.objects


def test_upload_is_idempotent_only_for_matching_hash_and_length(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    digest = _sha256(source)
    key = f"objects/artifact/{digest}"
    cos = FakeCOS()
    cos.objects[key] = (source.read_bytes(), {"sha256": digest, "length": "8"})

    ref = upload_verified(cos, source, "private", "objects/artifact", digest)
    assert ref.key == key
    assert cos.upload_calls == 0

    cos.objects[key] = (b"short", {"sha256": digest, "length": "5"})
    with pytest.raises(IntegrityError, match="existing object"):
        upload_verified(cos, source, "private", "objects/artifact", digest)


def test_interrupted_upload_removes_temporary_object_and_has_no_final(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    cos.fail_upload = True

    with pytest.raises(OSError, match="interrupted"):
        upload_verified(cos, source, "private", "objects/artifact", _sha256(source))

    assert not cos.objects


def test_download_verifies_metadata_length_and_bytes_before_publish(tmp_path: Path):
    destination = tmp_path / "artifact.bin"
    body = b"verified"
    digest = hashlib.sha256(body).hexdigest()
    cos = FakeCOS()
    cos.objects["objects/artifact"] = (
        body,
        {"sha256": digest, "length": str(len(body))},
    )

    assert download_verified(cos, "private", "objects/artifact", destination, digest) == destination
    assert destination.read_bytes() == body


def test_corrupt_download_is_deleted_locally(tmp_path: Path):
    destination = tmp_path / "artifact.bin"
    body = b"verified"
    digest = hashlib.sha256(body).hexdigest()
    cos = FakeCOS()
    cos.objects["objects/artifact"] = (
        body,
        {"sha256": digest, "length": str(len(body))},
    )
    cos.corrupt_download = True

    with pytest.raises(IntegrityError, match="downloaded object"):
        download_verified(cos, "private", "objects/artifact", destination, digest)

    assert not destination.exists()
    assert not list(tmp_path.glob("*.download-*"))


def test_events_use_immutable_uuid_keys_and_same_sequence_is_fork():
    cos = FakeCOS()
    first = _event(UUID(int=1))
    second = replace(first, event_id=UUID(int=2), payload_sha256="d" * 64)

    first_ref = append_event(cos, "private", "runs/abc", first)
    second_ref = append_event(cos, "private", "runs/abc", second)

    assert first_ref.key.startswith(f"runs/abc/events/000003-{first.event_id}-")
    assert first_ref.key.endswith(".json")
    assert second_ref.key.startswith(f"runs/abc/events/000003-{second.event_id}-")
    assert second_ref.key.endswith(".json")
    assert len(cos.objects) == 2
    stored = json.loads(cos.objects[first_ref.key][0])
    assert stored["event_id"] == str(first.event_id)
    with pytest.raises(EventChainFork, match="sequence 3"):
        validate_event_chain((first, second), valid_run_spec())


def test_concurrent_event_writers_preserve_both_fork_branches():
    cos = FakeCOS()
    first = _event(UUID(int=1))
    second = replace(first, event_id=UUID(int=2), payload_sha256="d" * 64)

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = tuple(
            executor.map(
                lambda event: append_event(cos, "private", "runs/abc", event),
                (first, second),
            )
        )

    assert len({ref.key for ref in refs}) == 2
    assert all(ref.key.startswith("runs/abc/events/000003-") for ref in refs)
    assert len(cos.objects) == 2


def test_append_event_never_overwrites_an_existing_key():
    cos = FakeCOS()
    event = _event(UUID(int=1))
    key = append_event(cos, "private", "runs/abc", event).key
    cos.objects[key] = (b"different", {"sha256": "0" * 64, "length": "9"})

    with pytest.raises(IntegrityError, match="event key already exists"):
        append_event(cos, "private", "runs/abc", event)

    assert cos.objects[key][0] == b"different"


def test_concurrent_same_uuid_different_payloads_are_preserved_and_detected_as_fork():
    cos = FakeCOS()
    first = _event(UUID(int=1))
    second = replace(first, payload_sha256="d" * 64)

    with ThreadPoolExecutor(max_workers=2) as executor:
        refs = tuple(
            executor.map(
                lambda event: append_event(cos, "private", "runs/abc", event),
                (first, second),
            )
        )

    assert len({ref.key for ref in refs}) == 2
    assert len(cos.objects) == 2
    with pytest.raises(EventChainFork, match="sequence 3"):
        validate_event_chain((first, second), valid_run_spec())
