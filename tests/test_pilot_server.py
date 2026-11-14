from __future__ import annotations

import base64
import http.client
import io
import json
import subprocess
import sys
import threading
from http.server import HTTPServer
from pathlib import Path
from typing import Any, cast

import pytest
from PIL import Image

from pilot.server import (
    _HTML,
    ANGLES,
    MAX_BATCH_DECODED_BYTES,
    MAX_BATCH_IMAGES,
    MAX_UPLOAD_BYTES,
    analyze_batch,
    checks_for_angle,
    decode_image,
    handler,
)
from reference_intelligence.knowledge import load_knowledge


def _jpeg(size: tuple[int, int] = (300, 300)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="JPEG")
    return output.getvalue()


def test_decode_image_accepts_bounded_jpeg() -> None:
    assert decode_image(_jpeg(), "image/jpeg").size == (300, 300)


def test_decode_image_rejects_mime_mismatch() -> None:
    output = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(output, format="PNG")

    with pytest.raises(ValueError, match="corresponde"):
        decode_image(output.getvalue(), "image/jpeg")


def test_importing_server_does_not_load_the_vision_model() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pilot.server; "
            "assert 'pilot.online_assessor' not in sys.modules; "
            "assert 'quality_gate.blur_light_check' not in sys.modules",
        ],
        check=True,
        timeout=10,
    )


def test_online_checks_keep_official_facts_and_forum_leads_distinct() -> None:
    claims = load_knowledge(
        Path("config/references/rolex-submariner-126610ln.json"),
        "rolex-submariner-126610ln",
    )

    checks = checks_for_angle(claims, "dial")

    assert {item["role"] for item in checks} == {"official_spec", "community_lead"}
    assert all(item["url"].startswith("https://") for item in checks)


def test_pilot_ci_is_bounded_and_does_not_build() -> None:
    workflow = Path(".github/workflows/pilot-ci.yml").read_text(encoding="utf-8")

    assert '"pilot/**"' in workflow
    assert '"reference_intelligence/**"' in workflow
    assert "timeout-minutes: 10" in workflow
    assert "group: pilot-bounded-v3-${{ github.event.pull_request.number }}" in workflow
    assert "tests/test_online_assessor.py" in workflow
    assert "tests/test_predict.py" in workflow
    assert "Lint online pilot" in workflow
    assert "Typecheck online pilot" in workflow
    assert workflow.count("timeout 120s python -m pytest") == 4
    assert "Test online assessor" in workflow
    assert "Test pilot server" in workflow
    assert "Test prediction contract" in workflow
    assert "Test reference intelligence" in workflow
    assert "docker build" not in workflow
    assert "next build" not in workflow


@pytest.mark.parametrize(
    ("body", "content_type", "message"),
    [
        (b"", "image/jpeg", "vazia"),
        (b"not-an-image", "image/jpeg", "invalido"),
        (_jpeg(), "application/pdf", "formato"),
        (_jpeg((100, 100)), "image/jpeg", "pequena"),
        (b"x" * (MAX_UPLOAD_BYTES + 1), "image/jpeg", "maior"),
    ],
    ids=["empty", "invalid-bytes", "unsupported-type", "too-small", "oversized"],
)
def test_decode_image_rejects_unsafe_input(body: bytes, content_type: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_image(body, content_type)


class _StubService:
    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = iter(verdicts)
        self.calls: list[tuple[bytes, str, str]] = []

    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, Any]:
        self.calls.append((body, content_type, angle))
        verdict = next(self.verdicts)
        return {
            "verdict": verdict,
            "message": verdict,
            "next_photo": angle,
            "online_checks": [{"field": angle, "url": f"https://example.test/{angle}"}],
        }


def _batch_item(angle: str = "dial", body: bytes | None = None) -> dict[str, str]:
    return {
        "angle": angle,
        "content_type": "image/jpeg",
        "data": base64.b64encode(body or _jpeg()).decode("ascii"),
    }


def test_batch_processes_each_image_sequentially_and_preserves_sources() -> None:
    service = _StubService(["inconclusivo", "sem_inconsistencia_visivel"])

    result = analyze_batch(
        service,
        {"images": [_batch_item("dial", b"first"), _batch_item("crown", b"second")]},
    )

    assert [call[0] for call in service.calls] == [b"first", b"second"]
    assert [item["angle"] for item in result["results"]] == ["dial", "crown"]
    assert result["results"][0]["online_checks"][0]["url"].endswith("/dial")
    assert result["joint"]["verdict"] == "inconclusivo"
    assert result["joint"]["missing_angles"] == sorted(ANGLES - {"dial", "crown"})


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["foto_invalida", "foto_invalida"], "foto_invalida"),
        (["foto_invalida", "inconclusivo"], "inconclusivo"),
        (["inconclusivo", "inconsistencia_com_referencia"], "inconsistencia_com_referencia"),
    ],
)
def test_batch_joint_verdict_fails_closed(verdicts: list[str], expected: str) -> None:
    result = analyze_batch(
        _StubService(verdicts),
        {"images": [_batch_item("dial"), _batch_item("rehaut")]},
    )

    assert result["joint"]["verdict"] == expected
    assert "original" not in result["joint"]["message"].lower()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "lista"),
        ({"images": []}, "1 a 6"),
        ({"images": [_batch_item()] * (MAX_BATCH_IMAGES + 1)}, "1 a 6"),
        ({"images": [{**_batch_item(), "angle": "top"}]}, "angulo"),
        ({"images": [{**_batch_item(), "content_type": "application/pdf"}]}, "formato"),
        ({"images": [{**_batch_item(), "data": "%%%"}]}, "base64"),
        ({"images": [{**_batch_item(), "data": "A" * (MAX_BATCH_DECODED_BYTES * 2)}]}, "codificado"),
    ],
)
def test_batch_rejects_invalid_contract(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analyze_batch(_StubService([]), payload)


def test_batch_keeps_single_image_limit() -> None:
    oversized = base64.b64encode(b"x" * (MAX_UPLOAD_BYTES + 1)).decode("ascii")

    with pytest.raises(ValueError, match="15 MB"):
        analyze_batch(
            _StubService([]),
            {"images": [{"angle": "dial", "content_type": "image/jpeg", "data": oversized}]},
        )


def test_batch_keeps_decoded_total_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pilot.server.MAX_BATCH_DECODED_BYTES", 5)

    with pytest.raises(ValueError, match="decodificado"):
        analyze_batch(
            _StubService([]),
            {"images": [_batch_item("dial", b"123"), _batch_item("crown", b"456")]},
        )


def test_individual_endpoint_remains_compatible() -> None:
    service = _StubService(["inconclusivo"])
    server = HTTPServer(("127.0.0.1", 0), handler(service))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address)
        connection = http.client.HTTPConnection(host, port, timeout=2)
        connection.request(
            "POST",
            "/predict?angle=dial",
            body=b"legacy-body",
            headers={"Content-Type": "image/jpeg"},
        )
        response = connection.getresponse()
        payload = json.loads(response.read())
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()

    assert response.status == 200
    assert service.calls == [(b"legacy-body", "image/jpeg", "dial")]
    assert payload["verdict"] == "inconclusivo"


def test_html_supports_six_files_per_image_angles_and_recovers_after_error() -> None:
    html = _HTML.decode("utf-8")

    assert 'multiple' in html
    assert 'Máximo de 6 imagens' in html
    assert 'fetch("/predict/batch"' in html
    assert 'finally' in html
    assert 'b.disabled=false' in html
    assert 'Selecione o ângulo' in html
