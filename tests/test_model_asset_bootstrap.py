"""Model bootstrap rejects unpinned or incomplete COS bytes before serving."""

from __future__ import annotations

import base64
import hashlib
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

import pytest

from pilot import model_asset_bootstrap, server
from pilot.model_asset_bootstrap import download_models
from pilot.server import PilotHTTPServer


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


def test_gemini_bootstrap_requires_no_model_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATCH_AUTH_PILOT_MODE", "gemini")
    monkeypatch.delenv("WATCH_AUTH_COS_BUCKET", raising=False)
    monkeypatch.delenv("WATCH_AUTH_TRUSTED_PROFILE_ID", raising=False)
    launched: list[list[str]] = []
    monkeypatch.setattr(server, "main", launched.append)
    model_asset_bootstrap.main()
    assert launched == [["--gemini-flash", "--host", "0.0.0.0", "--port", "8080"]]


def test_default_bootstrap_still_checks_models_before_serving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from ibm_cloud_sdk_core import authenticators

    from infra import ibm_cos

    monkeypatch.delenv("WATCH_AUTH_PILOT_MODE", raising=False)
    monkeypatch.setenv("WATCH_AUTH_COS_BUCKET", "protected-models")
    monkeypatch.setenv("WATCH_AUTH_COS_ENDPOINT", "https://cos.example")
    monkeypatch.setenv("WATCH_AUTH_COS_SERVICE_INSTANCE_ID", "instance")
    monkeypatch.setenv("WATCH_AUTH_TRUSTED_PROFILE_ID", "profile")
    monkeypatch.setattr(authenticators, "ContainerAuthenticator", lambda **_: object())
    monkeypatch.setattr(ibm_cos, "create_oauth_cos_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(model_asset_bootstrap, "MODEL_DIR", tmp_path)
    calls: list[object] = []
    monkeypatch.setattr(model_asset_bootstrap, "download_models", lambda client, bucket, manifest, root: calls.append((bucket, root)))
    monkeypatch.setattr(server, "main", calls.append)
    model_asset_bootstrap.main()
    assert calls == [("protected-models", tmp_path), ["--checkpoint", str(tmp_path / "smolvlm"),
                                                     "--host", "0.0.0.0", "--port", "8080"]]


def test_unrecognized_bootstrap_mode_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATCH_AUTH_PILOT_MODE", "misspelled")
    with pytest.raises(SystemExit, match="WATCH_AUTH_PILOT_MODE"):
        model_asset_bootstrap.main()


@pytest.mark.parametrize(("password", "key", "expected"), [
    ("short", "some-key", "public bind requires"),
    ("p" * 32, "", "requires WATCH_AUTH_OPENROUTER_PILOT_KEY"),
    ("p" * 32, "existing-benchmark-key", "provider or spending envelope unavailable"),
])
def test_public_gemini_bootstrap_refuses_to_bind_without_provider_envelope(
    monkeypatch: pytest.MonkeyPatch, password: str, key: str, expected: str,
) -> None:
    monkeypatch.setenv("WATCH_AUTH_PILOT_MODE", "gemini")
    monkeypatch.setenv("WATCH_AUTH_PILOT_PASSWORD", password)
    monkeypatch.setenv("WATCH_AUTH_OPENROUTER_PILOT_KEY", key)

    def unavailable(provider_key: str) -> None:
        assert provider_key == "existing-benchmark-key"
        raise RuntimeError("provider details")

    def unexpected_bind(*args: object) -> None:
        pytest.fail("public socket must not bind")

    monkeypatch.setattr(server, "check_flash_budget", unavailable)
    monkeypatch.setattr(server, "PilotHTTPServer", unexpected_bind)
    with pytest.raises(SystemExit, match=expected):
        model_asset_bootstrap.main()


def test_public_gemini_bootstrap_uses_authenticated_gemini_page(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WATCH_AUTH_PILOT_MODE", "gemini")
    monkeypatch.setenv("WATCH_AUTH_PILOT_PASSWORD", "p" * 32)
    monkeypatch.setenv("WATCH_AUTH_OPENROUTER_PILOT_KEY", "scoped-test-key")
    monkeypatch.delenv("WATCH_AUTH_COS_BUCKET", raising=False)
    checked_keys: list[str] = []
    monkeypatch.setattr(server, "check_flash_budget", checked_keys.append)

    class LocalProbe(PilotHTTPServer):
        def __init__(self, address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]) -> None:
            assert address == ("0.0.0.0", 8080)
            assert checked_keys == ["scoped-test-key"]
            super().__init__(("127.0.0.1", 0), handler_class)

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            worker = Thread(target=super().serve_forever, kwargs={"poll_interval": 0.01})
            worker.start()
            client = HTTPConnection("127.0.0.1", self.server_address[1], timeout=3)
            try:
                client.request("GET", "/")
                denied = client.getresponse()
                assert denied.status == 401
                denied.read()
                identity = base64.b64encode(("watch:" + "p" * 32).encode()).decode()
                client.request("GET", "/", headers={"Authorization": f"Basic {identity}"})
                accepted = client.getresponse()
                assert accepted.status == 200
                assert b"Piloto com Gemini Flash" in accepted.read()
            finally:
                client.close()
                self.shutdown()
                worker.join(timeout=3)
                self.server_close()

    monkeypatch.setattr(server, "PilotHTTPServer", LocalProbe)
    model_asset_bootstrap.main()
