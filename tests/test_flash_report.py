from __future__ import annotations

import base64
import io
import json
from threading import Thread
from typing import Any, cast

import pytest
import requests
from PIL import Image

from evaluation.pilot_reports import score
from pilot.flash_report import FlashReport
from pilot.server import PilotHTTPServer, handler


def _photo(color: str) -> str:
    output = io.BytesIO()
    Image.new("RGB", (240, 240), color).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def _answer() -> dict[str, Any]:
    return {
        "candidates": [{"brand": "Rolex", "family": "Submariner", "reference": "126610LN",
                        "reference_id": "rolex-submariner-126610ln"}],
        "linkage": "consistent",
        "hypothesis": "probable_original",
        "observations": [
            {"photo": 1, "angle": "dial", "location": "centro", "detail": "Mostrador preto",
             "visibility": "visible", "field": "dial_color", "value": "black"},
            {"photo": 3, "angle": "cyclops", "location": "janela da data", "detail": "Lupa sobre a data",
             "visibility": "visible", "field": "date_magnifier", "value": "Cyclops"},
        ],
        "next_verification": "Fotografe a janela da data de frente e em foco.",
    }


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class StubOpenRouter:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.posts: list[dict[str, Any]] = []

    def get(self, url: str, **_: object) -> _Response:
        if url.endswith("/auth/key"):
            return _Response({"data": {"limit_remaining": 10, "usage": 1}})
        return _Response({"data": [{"id": "google/gemini-3.8-flash", "context_length": 1048576,
                                    "pricing": {key: 0.0000005 for key in (
                                        "prompt", "image", "completion", "internal_reasoning")}}]})

    def post(self, url: str, **kwargs: Any) -> _Response:
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        self.posts.append(kwargs["json"])
        return _Response({"model": "google/gemini-3.8-flash", "id": "fixture",
                          "usage": {"cost": 0.01}, "choices": [{"finish_reason": "stop",
                          "message": {"content": json.dumps(self.answer)}}]})


@pytest.mark.parametrize("scenario, status, conclusion", [
    ("compatible", 200, "visually_compatible"),
    ("mixed", 200, "inconclusive"),
    ("color-only", 200, "inconclusive"),
    ("weak-compatibility", 200, "inconclusive"),
    ("calibre-unseen", 200, "inconclusive"),
    ("wrong-reference-code", 200, "inconclusive"),
    ("wrong-reference-brand", 200, "inconclusive"),
    ("equivalent-spec-words", 200, "inconclusive"),
    ("structural-divergence", 200, "probable_fake"),
    ("wrong-photo", 503, None),
])
def test_flash_batch_http_route(scenario: str, status: int, conclusion: str | None) -> None:
    answer = _answer()
    if scenario == "mixed":
        answer["linkage"] = "mixed"
    elif scenario == "color-only":
        answer["hypothesis"] = "probable_fake"
        answer["observations"][0]["value"] = "blue"
    elif scenario == "weak-compatibility":
        answer["observations"][1].update(visibility="ambiguous", value=None)
    elif scenario == "calibre-unseen":
        answer["observations"][0].update(field="calibre", value="3235")
    elif scenario == "wrong-reference-code":
        answer["candidates"][0]["reference"] = "126610LV"
    elif scenario == "wrong-reference-brand":
        answer["candidates"][0].update(brand="Cartier", family="Santos")
    elif scenario in {"equivalent-spec-words", "structural-divergence"}:
        answer["candidates"] = [{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                                 "reference_id": "cartier-santos-wssa0061"}]
        answer["hypothesis"] = "probable_fake"
        hand, crown = ("sword shaped", "seven-sided") if scenario == "equivalent-spec-words" else ("baton", "round")
        answer["observations"][0].update(field="hand_shape", value=hand, detail=f"Ponteiro: {hand}")
        answer["observations"][1].update(angle="crown", field="crown_shape", value=crown, detail=f"Coroa: {crown}")
    elif scenario == "wrong-photo":
        answer["observations"][1]["photo"] = 2
    provider = StubOpenRouter(answer)
    password = "example-password-used-only-for-tests"
    server = PilotHTTPServer(("127.0.0.1", 0), handler(None, password, report=FlashReport("fixture", request=provider)))
    thread = Thread(target=server.serve_forever)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    auth = ("watch", password)
    black = _photo("black")
    body = {"images": [{"angle": angle, "content_type": "image/png", "data": image}
                      for angle, image in [("dial", black), ("rehaut", black), ("cyclops", _photo("white"))]]}
    try:
        assert requests.post(base + "/predict/batch", json=body, timeout=3).status_code == 401
        assert "OpenRouter" in requests.get(base, auth=auth, timeout=3).text
        response = requests.post(base + "/predict/batch", auth=auth, json=body, timeout=5)
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()
    assert response.status_code == status
    assert len(provider.posts) == 1
    content = provider.posts[0]["messages"][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 2
    assert response.json().get("conclusion") == conclusion
    if status == 200:
        data = response.json()
        assert data["examined_photos"] == [1, 3]
        checks = cast(list[dict[str, Any]], data["technical_checks"])
        expected_photos = {
            "compatible": [1, 3], "mixed": [1, 3], "color-only": [1, 3],
            "weak-compatibility": [1], "calibre-unseen": [3], "wrong-reference-code": [], "wrong-reference-brand": [],
            "equivalent-spec-words": [1, 3], "structural-divergence": [1, 3],
        }
        assert [check["photo"] for check in checks] == expected_photos[scenario]
        brand = "cartier" if scenario in {"equivalent-spec-words", "structural-divergence"} else "rolex"
        assert all(check["url"].startswith(f"https://www.{brand}.com/") for check in checks)
        if scenario == "equivalent-spec-words":
            assert {check["status"] for check in checks} == {"matched"}


def test_scoring_preserves_denominator_and_evidence_tier() -> None:
    labels = [("a", "fake", "independent"), ("b", "original", "independent"),
              ("c", "fake", "independent"), ("d", "original", "declared"),
              ("e", "fake", "independent")]
    cases = [{"specimen_id": name, "label": label, "evidence": evidence,
              "label_source": "test fixture", "physical_identity_confirmed": evidence == "independent"}
             for name, label, evidence in labels]
    reports = {"a": "probable_original", "b": "visually_compatible", "c": "probable_fake", "d": "probable_original"}
    report = score(cases, reports)
    independent = report["independent"]
    assert (independent["correct"], independent["incorrect"], independent["inconclusive"],
            independent["unavailable"], independent["fake_called_original"],
            independent["accuracy_all_cases"]) == (1, 1, 1, 1, 1, 0.25)
    assert report["declared"]["agrees_with_claim"] == 1
    assert independent["compatible_but_inconclusive"] == 1
    with pytest.raises(ValueError, match="unique specimen"):
        score([*cases, cases[0]], reports)
