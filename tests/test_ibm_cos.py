"""IBM COS integrity, authentication and immutable-event tests."""

from __future__ import annotations

import hashlib
import io
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
    EventInventorySnapshot,
    IntegrityError,
    append_event,
    create_cos_client,
    download_verified,
    load_event_inventory,
    upload_verified,
    validate_event_chain,
    verify_cos_configuration,
)
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import RunEvent
from tests.conftest import NOW, valid_run_spec


class _NotFound(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeCOS:
    def __init__(self, *, bucket: str = "private", prefix: str = "runs/abc") -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.version_entries: list[dict[str, object]] | None = None
        self.version_bodies: dict[tuple[str, str], bytes] = {}
        self.delete_markers: list[dict[str, object]] = []
        self.fail_upload = False
        self.corrupt_final_head = False
        self.corrupt_download = False
        self.upload_calls = 0
        self._watch_auth_bucket = bucket
        self._watch_auth_prefix = prefix
        self._watch_auth_configuration_verified = True

    def get_public_access_block(self, *, Bucket: str) -> dict[str, object]:
        assert Bucket == valid_run_spec().bucket
        return {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            }
        }

    def get_bucket_versioning(self, *, Bucket: str) -> dict[str, str]:
        assert Bucket == valid_run_spec().bucket
        return {"Status": "Enabled"}

    def list_objects_v2(self, *, Bucket: str, Prefix: str, **_kwargs: object) -> dict[str, object]:
        del Bucket
        return {
            "Contents": [{"Key": key} for key in sorted(self.objects) if key.startswith(Prefix)],
            "IsTruncated": False,
        }

    def list_object_versions(
        self,
        *,
        Bucket: str,
        Prefix: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        del Bucket
        versions = self.version_entries
        if versions is None:
            versions = [
                {"Key": key, "VersionId": f"current:{key}", "IsLatest": True}
                for key in sorted(self.objects)
                if key.startswith(Prefix)
            ]
        return {
            "Versions": [item for item in versions if str(item["Key"]).startswith(Prefix)],
            "DeleteMarkers": [
                item for item in self.delete_markers if str(item["Key"]).startswith(Prefix)
            ],
            "IsTruncated": False,
        }

    def get_object(
        self,
        *,
        Bucket: str,
        Key: str,
        VersionId: str | None = None,
    ) -> dict[str, object]:
        del Bucket
        if VersionId is not None and not VersionId.startswith("current:"):
            body = self.version_bodies[(Key, VersionId)]
        else:
            try:
                body = self.objects[Key][0]
            except KeyError as exc:
                raise _NotFound() from exc
        return {"Body": io.BytesIO(body)}

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
        lineage_sha256=run_spec_sha256(valid_run_spec()),
        previous_event_sha256="b" * 64,
        payload_sha256="c" * 64,
    )


def test_create_client_uses_trusted_profile_token_manager_only(monkeypatch):
    calls: dict[str, object] = {}

    class TrustedProfileTokenManager:
        requests = 0

        def request_token(self) -> dict[str, object]:
            self.requests += 1
            return {
                "access_token": f"trusted-token-{self.requests}",
                "expires_in": 3600,
            }

    token_manager = TrustedProfileTokenManager()

    class Authenticator:
        def __init__(self, *, iam_profile_id: str) -> None:
            calls["profile"] = iam_profile_id
            self.token_manager = token_manager

    class Config:
        def __init__(self, *, signature_version: str) -> None:
            calls["signature"] = signature_version

    class Client:
        pass

    def client(service: str, **kwargs: object) -> object:
        calls["service"] = service
        calls["kwargs"] = kwargs
        return Client()

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
    created = create_cos_client(spec)

    assert calls["profile"] == spec.trusted_profile_id
    assert calls["signature"] == "oauth"
    kwargs = calls["kwargs"]
    assert isinstance(kwargs, dict)
    adapted = kwargs["token_manager"]
    assert adapted.get_token() == "trusted-token-1"
    assert adapted.auth_function() == {
        "access_token": "trusted-token-2",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    adapted.stop_refresh_thread()
    assert kwargs["ibm_service_instance_id"] == spec.cos_service_instance_id
    assert kwargs["endpoint_url"] == spec.cos_endpoint
    assert not ({"api_key", "aws_access_key_id", "aws_secret_access_key"} & kwargs.keys())
    assert getattr(created, "_watch_auth_bucket") == spec.bucket
    assert getattr(created, "_watch_auth_prefix") == spec.prefix
    assert getattr(created, "_watch_auth_spec") == spec
    assert getattr(created, "_watch_auth_configuration_verified") is False


def test_cos_configuration_requires_private_access_and_versioning():
    spec = valid_run_spec()
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    verify_cos_configuration(cos, spec)
    assert cos._watch_auth_configuration_verified is True

    cos.get_bucket_versioning = lambda **_kwargs: {"Status": "Suspended"}
    with pytest.raises(IntegrityError, match="versioning"):
        verify_cos_configuration(cos, spec)
    assert cos._watch_auth_configuration_verified is False


def test_upload_uses_digest_key_and_verifies_remote_object(tmp_path: Path):
    source = tmp_path / "dataset.json"
    source.write_bytes(b"immutable dataset")
    digest = _sha256(source)
    cos = FakeCOS()

    ref = upload_verified(cos, source, "private", "runs/abc/objects/dataset", digest)

    assert ref.key == f"runs/abc/objects/dataset/{digest}"
    assert ref.sha256 == digest
    assert ref.length == source.stat().st_size
    assert all(".upload-" not in key for key in cos.objects)


def test_upload_preserves_explicit_content_addressed_json_key(tmp_path: Path):
    source = tmp_path / "review.json"
    source.write_bytes(b'{"review":"accepted"}')
    digest = _sha256(source)
    cos = FakeCOS(prefix="datasets")

    ref = upload_verified(
        cos,
        source,
        "private",
        f"datasets/reviews/candidate/reviewer/{digest}.json",
        digest,
    )

    assert ref.key == f"datasets/reviews/candidate/reviewer/{digest}.json"


def test_cos_operations_reject_destinations_outside_client_binding(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    with pytest.raises(IntegrityError, match="different bucket"):
        upload_verified(cos, source, "other", "runs/abc/objects/artifact", _sha256(source))
    with pytest.raises(IntegrityError, match="different run prefix"):
        upload_verified(cos, source, "private", "runs/other/artifact", _sha256(source))
    with pytest.raises(IntegrityError, match="different run prefix"):
        download_verified(
            cos,
            "private",
            "runs/other/artifact",
            tmp_path / "download",
            "0" * 64,
        )
    with pytest.raises(IntegrityError, match="different run prefix"):
        append_event(cos, "private", "runs/other", _event(UUID(int=1)))


def test_cos_operations_require_verified_configuration(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    cos._watch_auth_configuration_verified = False

    with pytest.raises(IntegrityError, match="configuration is not verified"):
        upload_verified(
            cos,
            source,
            "private",
            "runs/abc/objects/artifact",
            _sha256(source),
        )
    with pytest.raises(IntegrityError, match="configuration is not verified"):
        append_event(cos, "private", "runs/abc", _event(UUID(int=1)))


def test_upload_rejects_local_or_remote_checksum_mismatch(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    with pytest.raises(IntegrityError, match="local SHA-256"):
        upload_verified(cos, source, "private", "runs/abc/objects/artifact", "0" * 64)
    assert cos.upload_calls == 0

    cos.corrupt_final_head = True
    with pytest.raises(IntegrityError, match="remote object metadata"):
        upload_verified(
            cos, source, "private", "runs/abc/objects/artifact", _sha256(source)
        )
    assert any(".upload-" not in key for key in cos.objects)


def test_upload_is_idempotent_only_for_matching_hash_and_length(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    digest = _sha256(source)
    key = f"runs/abc/objects/artifact/{digest}"
    cos = FakeCOS()
    cos.objects[key] = (source.read_bytes(), {"sha256": digest, "length": "8"})

    ref = upload_verified(cos, source, "private", "runs/abc/objects/artifact", digest)
    assert ref.key == key
    assert cos.upload_calls == 0

    cos.objects[key] = (b"short", {"sha256": digest, "length": "5"})
    with pytest.raises(IntegrityError, match="existing object"):
        upload_verified(cos, source, "private", "runs/abc/objects/artifact", digest)


def test_upload_rejects_remote_bytes_that_lie_through_metadata(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"good")
    digest = _sha256(source)
    key = f"runs/abc/objects/artifact/{digest}"
    cos = FakeCOS()
    cos.objects[key] = (b"evil", {"sha256": digest, "length": "4"})

    with pytest.raises(IntegrityError, match="remote object bytes"):
        upload_verified(cos, source, "private", "runs/abc/objects/artifact", digest)


def test_interrupted_upload_removes_temporary_object_and_has_no_final(tmp_path: Path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = FakeCOS()
    cos.fail_upload = True

    with pytest.raises(OSError, match="interrupted"):
        upload_verified(
            cos, source, "private", "runs/abc/objects/artifact", _sha256(source)
        )

    assert not cos.objects


def test_download_verifies_metadata_length_and_bytes_before_publish(tmp_path: Path):
    destination = tmp_path / "artifact.bin"
    body = b"verified"
    digest = hashlib.sha256(body).hexdigest()
    cos = FakeCOS()
    cos.objects["runs/abc/objects/artifact"] = (
        body,
        {"sha256": digest, "length": str(len(body))},
    )

    assert (
        download_verified(
            cos, "private", "runs/abc/objects/artifact", destination, digest
        )
        == destination
    )
    assert destination.read_bytes() == body


def test_corrupt_download_is_deleted_locally(tmp_path: Path):
    destination = tmp_path / "artifact.bin"
    body = b"verified"
    digest = hashlib.sha256(body).hexdigest()
    cos = FakeCOS()
    cos.objects["runs/abc/objects/artifact"] = (
        body,
        {"sha256": digest, "length": str(len(body))},
    )
    cos.corrupt_download = True

    with pytest.raises(IntegrityError, match="downloaded object"):
        download_verified(
            cos, "private", "runs/abc/objects/artifact", destination, digest
        )

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


def test_upload_preserves_final_object_when_post_copy_head_is_ambiguous(tmp_path: Path):
    class AmbiguousHeadCOS(FakeCOS):
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            if Key in self.objects and ".upload-" not in Key:
                raise OSError("HEAD unavailable")
            return super().head_object(Bucket=Bucket, Key=Key)

    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    cos = AmbiguousHeadCOS()

    with pytest.raises(OSError, match="HEAD unavailable"):
        upload_verified(
            cos, source, "private", "runs/abc/objects/artifact", _sha256(source)
        )

    assert len(cos.objects) == 1
    assert next(iter(cos.objects)).endswith(_sha256(source))


def test_append_event_preserves_written_object_when_head_is_ambiguous():
    class AmbiguousHeadCOS(FakeCOS):
        def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
            if Key in self.objects:
                raise OSError("HEAD unavailable")
            return super().head_object(Bucket=Bucket, Key=Key)

    cos = AmbiguousHeadCOS()

    with pytest.raises(OSError, match="HEAD unavailable"):
        append_event(cos, "private", "runs/abc", _event(UUID(int=1)))

    assert len(cos.objects) == 1


def test_validate_event_chain_checks_contiguous_hashes_lineage_and_transitions():
    spec = valid_run_spec()
    planned = replace(
        _event(UUID(int=1), sequence=1),
        sequence=0,
        state="planned",
        instance_id=None,
        previous_event_sha256=None,
    )
    from infra.ibm_types import run_event_sha256

    deleting = replace(
        planned,
        event_id=UUID(int=2),
        sequence=1,
        state="deleting",
        previous_event_sha256=run_event_sha256(planned),
    )
    deleted = replace(
        deleting,
        event_id=UUID(int=3),
        sequence=2,
        state="deleted",
        previous_event_sha256=run_event_sha256(deleting),
    )
    validate_event_chain((planned, deleting, deleted), spec)
    for broken in (
        (planned, deleted),
        (planned, replace(deleting, previous_event_sha256="f" * 64)),
        (planned, replace(deleting, lineage_sha256="f" * 64)),
        (planned, replace(deleting, state="running")),
    ):
        with pytest.raises(IntegrityError):
            validate_event_chain(broken, spec)


def test_event_inventory_lists_and_verifies_the_complete_run_prefix():
    spec = valid_run_spec()
    lineage = run_spec_sha256(spec)
    planned = RunEvent(
        event_id=UUID(int=1),
        run_id=spec.run_id,
        sequence=0,
        state="planned",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256=lineage,
        previous_event_sha256=None,
        payload_sha256="a" * 64,
    )
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    append_event(cos, spec.bucket, spec.prefix, planned)

    snapshot = load_event_inventory(cos, spec)
    assert isinstance(snapshot, EventInventorySnapshot)
    assert snapshot.events == (planned,)
    assert snapshot.authoritative_for_launch is False


def test_event_inventory_rejects_delete_markers():
    spec = valid_run_spec()
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    cos.delete_markers = [
        {
            "Key": f"{spec.prefix}/events/000000-deleted.json",
            "VersionId": "delete-1",
            "IsLatest": True,
        }
    ]

    with pytest.raises(EventChainFork, match="delete marker"):
        load_event_inventory(cos, spec)


def test_event_inventory_rejects_multiple_versions_of_one_key():
    spec = valid_run_spec()
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    event = replace(
        _event(UUID(int=1)),
        sequence=0,
        state="planned",
        instance_id=None,
        previous_event_sha256=None,
    )
    key = append_event(cos, spec.bucket, spec.prefix, event).key
    cos.version_entries = [
        {"Key": key, "VersionId": "v2", "IsLatest": True},
        {"Key": key, "VersionId": "v1", "IsLatest": False},
    ]

    with pytest.raises(EventChainFork, match="multiple versions"):
        load_event_inventory(cos, spec)


def test_event_inventory_rejects_duplicate_sequence():
    spec = valid_run_spec()
    first = replace(
        _event(UUID(int=1)),
        sequence=0,
        state="planned",
        instance_id=None,
        previous_event_sha256=None,
    )
    second = replace(first, event_id=UUID(int=2), payload_sha256="d" * 64)
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    append_event(cos, spec.bucket, spec.prefix, first)
    append_event(cos, spec.bucket, spec.prefix, second)

    with pytest.raises(EventChainFork, match="sequence 0"):
        load_event_inventory(cos, spec)


def test_event_inventory_rejects_key_payload_mismatch():
    spec = valid_run_spec()
    lineage = run_spec_sha256(spec)
    planned = RunEvent(
        event_id=UUID(int=1),
        run_id=spec.run_id,
        sequence=0,
        state="planned",
        observed_at=NOW,
        instance_id=None,
        lineage_sha256=lineage,
        previous_event_sha256=None,
        payload_sha256="a" * 64,
    )
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    ref = append_event(cos, spec.bucket, spec.prefix, planned)
    body, metadata = cos.objects.pop(ref.key)
    wrong_key = ref.key.replace("/000000-", "/000001-")
    cos.objects[wrong_key] = (body, metadata)

    with pytest.raises(IntegrityError, match="key does not match payload"):
        load_event_inventory(cos, spec)


def test_event_inventory_rejects_noncanonical_json_bytes():
    spec = valid_run_spec()
    planned = replace(
        _event(UUID(int=1)),
        sequence=0,
        state="planned",
        instance_id=None,
        previous_event_sha256=None,
    )
    source = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    ref = append_event(source, spec.bucket, spec.prefix, planned)
    canonical = source.objects[ref.key][0]
    duplicate = (
        b'{"event_id":"' + str(planned.event_id).encode() + b'",' + canonical[1:]
    )
    variants = (
        canonical[:-1] + b',"shadow_authorization":"x"}',
        b" " + canonical,
        duplicate,
    )

    for body in variants:
        cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
        digest = hashlib.sha256(body).hexdigest()
        key = f"{spec.prefix}/events/000000-{planned.event_id}-{digest}.json"
        cos.objects[key] = (body, {"sha256": digest, "length": str(len(body))})
        with pytest.raises(IntegrityError, match="canonical"):
            load_event_inventory(cos, spec)
