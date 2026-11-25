from __future__ import annotations

import base64
import io
import json
from threading import Thread
from typing import Any, cast

import pytest
import requests
from PIL import Image

from evaluation.pilot_pairing import replay_with_review
from pilot.flash_report import FlashReport
from pilot.flash_research import complete_flash
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
    ("cartier-crown-ambiguous", 200, "inconclusive"),
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
        answer["next_verification"] = "Peça a abertura da caixa."
    elif scenario == "calibre-unseen":
        answer["observations"][0].update(field="calibre", value="3235")
    elif scenario == "wrong-reference-code":
        answer["candidates"][0]["reference"] = "126610LV"
    elif scenario == "wrong-reference-brand":
        answer["candidates"][0].update(brand="Cartier", family="Santos")
    elif scenario in {"equivalent-spec-words", "structural-divergence", "cartier-crown-ambiguous"}:
        answer["candidates"] = [{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                                 "reference_id": "cartier-santos-wssa0061"}]
        answer["hypothesis"] = "probable_fake"
        hand, crown = ("baton", "round") if scenario == "structural-divergence" else ("sword shaped", "seven-sided")
        answer["observations"][0].update(field="hand_shape", value=hand, detail=f"Ponteiro: {hand}")
        answer["observations"][1].update(angle="crown", field="crown_shape", value=crown, detail=f"Coroa: {crown}")
        if scenario == "cartier-crown-ambiguous":
            answer["observations"][1].update(visibility="ambiguous", value=None)
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
            "equivalent-spec-words": [1, 3], "structural-divergence": [1, 3], "cartier-crown-ambiguous": [1],
        }
        assert [check["photo"] for check in checks] == expected_photos[scenario]
        brand = answer["candidates"][0]["brand"].lower()
        if scenario == "wrong-reference-brand":
            assert not checks
        assert all(check["url"].startswith(f"https://www.{brand}.com/") for check in checks)
        if scenario == "equivalent-spec-words":
            assert {check["status"] for check in checks} == {"matched"}
        assert data["model_hypothesis"] == answer["hypothesis"]
        expected_reason = {
            "mixed": "unlinked_photos", "wrong-reference-code": "reference_unresolved",
            "wrong-reference-brand": "reference_unresolved", "structural-divergence": None,
            "equivalent-spec-words": "compatible_or_insufficient", "cartier-crown-ambiguous": "detail_not_verified",
        }.get(scenario, "catalog_gap")
        assert data["decision_reason"] == expected_reason
        if scenario == "cartier-crown-ambiguous":
            assert "coroa de perfil" in data["next_verification"]
        if scenario == "weak-compatibility":
            assert "janela de data perpendicular" in data["next_verification"]
            assert "autenticidade segue em aberto" in data["next_verification"]
        if scenario == "mixed":
            assert "um relógio por vez" in data["next_verification"]


@pytest.mark.parametrize(("scenario", "expected"), [
    ("agree", "probable_original"),
    ("different_reading", "inconclusive"),
    ("independently_confirmed_divergence", "probable_fake"),
    ("unavailable", "unavailable"),
    ("mixed", "inconclusive"),
    ("wrong_case", None),
    ("wrong_photo", None),
    ("wrong_fact", None),
])
def test_local_replay_shares_gemini_capture_and_blinds_reviewer(
    scenario: str, expected: str | None,
) -> None:
    answer = _answer()
    if scenario == "mixed":
        answer["linkage"] = "mixed"
    if scenario == "independently_confirmed_divergence":
        answer["hypothesis"] = "probable_fake"
        answer["observations"][0].update(value="blue", detail="Mostrador azul")
    provider = StubOpenRouter(answer)
    builder = FlashReport("fixture", request=provider)
    black, white = (base64.b64decode(_photo(color)) for color in ("black", "white"))
    photos = [("dial", "image/png", black), ("rehaut", "image/png", black),
              ("cyclops", "image/png", white)]
    calls = []

    def gemini(content: list[dict[str, object]]) -> tuple[object, dict[str, object]]:
        return complete_flash("fixture", content, request=provider)

    def review(request: dict[str, Any]) -> object:
        calls.append(request)
        assert set(request) == {"case_token", "photo_ids", "photos", "facts", "review_fields"}
        assert request["photo_ids"] == [1, 3]
        captured = provider.posts[0]["messages"][0]["content"]
        assert [item["image_url"] for item in request["photos"]] == [
            item["image_url"] for item in captured if item["type"] == "image_url"
        ]
        assert request["facts"] == json.loads(captured[0]["text"].split("Fatos técnicos oficiais:\n")[1])
        for field in request["review_fields"].values():
            fact = request["facts"][field["source_fact_id"]]
            assert set(fact) == {"reference_id", "angle", "field", "expected", "url", "citation"}
            assert fact["reference_id"] == "rolex-submariner-126610ln"
        encoded = json.dumps(request)
        assert "seller-fixture" not in encoded and '"probable_original"' not in encoded
        assert '"probable_fake"' not in encoded and '"hypothesis"' not in encoded
        assert "Mostrador preto" not in encoded and "Mostrador azul" not in encoded
        findings = [
            {
                "claim_id": key, "photo_id": item["photo_id"],
                "source_fact_id": item["source_fact_id"], "reading": "Leitura simulada.",
                "status": "contradicted" if (
                    scenario == "independently_confirmed_divergence" and item["field"] == "dial_color"
                ) else "supported",
            }
            for key, item in request["review_fields"].items()
        ]
        if scenario == "different_reading":
            findings[0]["status"] = "not_visible"
        if scenario == "wrong_photo":
            findings[0]["photo_id"] = 2
        if scenario == "wrong_fact":
            findings[0]["source_fact_id"] = "different-fact"
        token = "different-case" if scenario == "wrong_case" else request["case_token"]
        return None if scenario == "unavailable" else {"case_token": token, "findings": findings}

    if expected is None:
        with pytest.raises(ValueError, match="different case|requested evidence"):
            replay_with_review(builder, "seller-fixture", photos, gemini, review)
    else:
        result = replay_with_review(builder, "seller-fixture", photos, gemini, review)
        assert result["previous_policy"] == builder.build_report(answer, {1, 3})["conclusion"]
        assert [item["photo_id"] for item in result["prepared_photos"]] == [1, 3]
        assert result["gemini_hypothesis"] == answer["hypothesis"]
        assert result["gemini_only"] == ("inconclusive" if scenario == "mixed" else answer["hypothesis"])
        assert result["with_review"] == expected
        if scenario == "different_reading":
            disputed, = result["disputed_claim_ids"]
            assert result["review_findings"][disputed]["status"] == "not_visible"
    assert len(calls) == (0 if scenario == "mixed" else 1)
    assert len(provider.posts) == 1
