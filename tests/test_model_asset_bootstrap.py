"""Model bootstrap rejects unpinned or incomplete COS bytes before serving."""

from __future__ import annotations

import hashlib

import pytest

from pilot.model_asset_bootstrap import download_models


class _Body:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def iter_chunks(self, chunk_size: int):
        yield self.data

    def close(self) -> None:
        pass


class _Client:
    def __init__(self, data: bytes) -> None:
        self.data = data

    def get_object(self, **kwargs):
        return {"Body": _Body(self.data)}


def _manifest(data: bytes):
    return {"models": [{"name": "smolvlm", "revision": "fixed", "files": [{
        "path": "config.json", "size": len(data), "digest_type": "sha256",
        "digest": hashlib.sha256(data).hexdigest(),
    }]}]}


def test_download_models_verifies_before_publish(tmp_path):
    download_models(_Client(b"valid"), "private", _manifest(b"valid"), tmp_path)
    assert (tmp_path / "smolvlm/config.json").read_bytes() == b"valid"


def test_download_models_rejects_corruption_without_partial_file(tmp_path):
    with pytest.raises(RuntimeError, match="differs from pinned source"):
        download_models(_Client(b"wrong"), "private", _manifest(b"valid"), tmp_path)
    assert not list(tmp_path.rglob("*.partial"))
    assert not (tmp_path / "smolvlm/config.json").exists()
