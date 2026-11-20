from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from pilot import model_asset_stage as stage


class Missing(Exception):
    response = {"Error": {"Code": "404"}}


class Body:
    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, chunk_size: int):
        for start in range(0, len(self.data), chunk_size):
            yield self.data[start:start + chunk_size]

    def close(self):
        pass


class COS:
    exceptions = SimpleNamespace(NoSuchKey=Missing)

    def __init__(self):
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.parts: list[bytes] = []
        self.aborted = False

    def head_object(self, *, Key, **_kwargs):
        if Key not in self.objects:
            raise Missing()
        body, metadata = self.objects[Key]
        return {"ContentLength": len(body), "Metadata": metadata}

    def get_object(self, *, Key, **_kwargs):
        return {"Body": Body(self.objects[Key][0])}

    def put_object(self, *, Key, Body, Metadata, **_kwargs):
        self.objects[Key] = (Body, Metadata)

    def create_multipart_upload(self, **_kwargs):
        return {"UploadId": "one"}

    def upload_part(self, *, Body, **_kwargs):
        self.parts.append(Body)
        return {"ETag": str(len(self.parts))}

    def complete_multipart_upload(self, *, Key, MultipartUpload, **_kwargs):
        assert len(MultipartUpload["Parts"]) == len(self.parts)
        self.objects[Key] = (b"".join(self.parts), {"source-digest": self.digest, "digest-type": "sha256"})

    def abort_multipart_upload(self, **_kwargs):
        self.aborted = True


class Response:
    def __init__(self, body: bytes):
        self.content = body
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]


def test_small_source_is_hash_verified_and_resume_reads_same_bytes(monkeypatch):
    body = b'{"model":"pinned"}'
    digest = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    spec = {"path": "config.json", "size": len(body), "digest_type": "git-sha1", "digest": digest}
    model = {"name": "smolvlm", "repository": "owner/model", "revision": "a" * 40}
    cos = COS()
    monkeypatch.setattr(stage.requests, "get", lambda *_args, **_kwargs: Response(body))

    expected = hashlib.sha256(body).hexdigest()
    assert stage.stage_file(cos, "private", model, spec) == expected
    monkeypatch.setattr(stage.requests, "get", lambda *_args, **_kwargs: pytest.fail("resume must not redownload"))
    assert stage.stage_file(cos, "private", model, spec) == expected


def test_multipart_aborts_before_commit_on_bad_source_digest(monkeypatch):
    body = b"123456789abcdefgh"
    spec = {"path": "model.safetensors", "size": len(body), "digest_type": "sha256", "digest": "0" * 64}
    model = {"name": "owlv2", "repository": "owner/model", "revision": "b" * 40}
    cos = COS()
    monkeypatch.setattr(stage, "PART_BYTES", 8)
    monkeypatch.setattr(stage.requests, "get", lambda *_args, **_kwargs: Response(body))

    with pytest.raises(RuntimeError, match="source model asset digest differs"):
        stage.stage_file(cos, "private", model, spec)
    assert cos.aborted
    assert cos.objects == {}


def test_multipart_readback_matches_pinned_bytes(monkeypatch):
    body = b"123456789abcdefgh"
    digest = hashlib.sha256(body).hexdigest()
    spec = {"path": "model.safetensors", "size": len(body), "digest_type": "sha256", "digest": digest}
    model = {"name": "owlv2", "repository": "owner/model", "revision": "b" * 40}
    cos = COS()
    cos.digest = digest
    monkeypatch.setattr(stage, "PART_BYTES", 8)
    monkeypatch.setattr(stage.requests, "get", lambda *_args, **_kwargs: Response(body))

    assert stage.stage_file(cos, "private", model, spec) == digest
    assert b"".join(cos.parts) == body
