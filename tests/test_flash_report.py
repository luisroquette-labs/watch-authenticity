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
from pilot.flash_research import MODEL, complete_flash
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


def test_watchless_view_does_not_block_comparing_the_remaining_watch() -> None:
    answer = _answer()
    answer.update(linkage="single_photo", non_watch_photos=[3])
    answer["candidates"][0]["photo"] = 1
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["linkage"] == "single_photo"
    assert report["examined_photos"] == [1]
    assert report["conclusion"] == "inconclusive"
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert candidates[0]["reference_id"] == "rolex-submariner-126610ln"
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    assert [row["photo"] for row in checks] == [1]
    assert "Sem relógio ou componente visível nas fotos de número 3." in cast(list[str], report["limits"])


@pytest.mark.parametrize("candidate_provenance", [{"photo": 3}, {}])
def test_watchless_view_cannot_supply_reference_for_conflicting_checks(candidate_provenance: dict[str, int]) -> None:
    answer = _answer()
    answer.update(
        linkage="single_photo", non_watch_photos=[3], hypothesis="probable_fake",
        candidates=[{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                     "reference_id": "cartier-santos-wssa0061", **candidate_provenance}],
        observations=[
            {"photo": 1, "angle": "dial", "location": "ponteiros", "detail": "Ponteiros em bastão",
             "visibility": "visible", "field": "hand_shape", "value": "baton"},
            {"photo": 1, "angle": "crown", "location": "coroa", "detail": "Coroa redonda",
             "visibility": "visible", "field": "crown_shape", "value": "round"},
        ],
    )
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["examined_photos"] == [1]
    assert report["conclusion"] == "inconclusive"
    assert report["candidates"] == report["technical_checks"] == []


def test_watchless_batch_discards_candidates_and_asks_for_a_watch_photo() -> None:
    answer = _answer()
    answer.update(non_watch_photos=[1, 3], hypothesis="probable_fake")
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["decision_reason"] == "no_watch_visible"
    assert report["conclusion"] == report["model_hypothesis"] == "inconclusive"
    assert report["candidates"] == report["examined_photos"] == report["technical_checks"] == []
    assert "foto nítida do relógio inteiro" in str(report["next_verification"])


def test_one_photo_with_multiple_watches_is_still_inconclusive() -> None:
    answer = _answer()
    answer.update(linkage="mixed", non_watch_photos=[])
    answer["observations"] = answer["observations"][:1]
    report = FlashReport("fixture").build_report(answer, {1})
    assert report["linkage"] == "mixed"
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "unlinked_photos"
    assert report["technical_checks"] == []
    assert "fotografe cada peça separadamente" in str(report["next_verification"])


def test_uncertain_single_watch_view_requests_an_overlapping_second_view() -> None:
    answer = _answer()
    answer.update(linkage="uncertain", non_watch_photos=[3])
    answer["candidates"][0]["photo"] = 1
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["decision_reason"] == "unlinked_photos"
    assert report["conclusion"] == "inconclusive"
    assert report["examined_photos"] == [1]
    assert "uma parte que já aparece na primeira" in str(report["next_verification"])


@pytest.mark.parametrize(("value", "visibility", "second_reading", "reject"), [
    ("no date", "visible", None, True),
    ("no date", "ambiguous", None, False),
    ("6 o'clock", "visible", None, False),
    ("no date", "visible", "6 o'clock", False),
    ("no date", "visible", "3 o'clock", False),
    ("3 o'clock", "visible", "12 o'clock", False),
])
def test_visible_date_layout_conflict_rejects_the_exact_candidate(
    value: str, visibility: str, second_reading: str | None, reject: bool,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Santos de Cartier", "reference": "WSSA0018",
                     "reference_id": "cartier-santos-wssa0018", "photo": 1}],
        observations=[{"photo": 1, "angle": "dial", "location": "6 horas", "detail": "Janela de data.",
                       "visibility": visibility, "field": "date_position", "value": value}],
        next_verification="Procure a referência 4072 no fundo.",
    )
    if second_reading is not None:
        answer["observations"].append({**answer["observations"][0], "photo": 3, "value": second_reading})
    report = FlashReport("fixture").build_report(answer, {1, 3})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert (candidate["reference"] is None) is reject
    assert report["conclusion"] == "inconclusive"
    if second_reading is not None:
        assert report["decision_reason"] == "date_readings_disagree"
        assert "mostrador de frente" in str(report["next_verification"])
    if reject:
        assert candidate["reference_id"] is None and candidate["family"] == "Santos de Cartier"
        assert report["decision_reason"] == "reference_conflict"
        assert "WSSA0018" in str(report["decision_reason_text"])
        assert "4072" not in str(report["next_verification"])
        assert "fundo" in str(report["next_verification"])
        check, = cast(list[dict[str, Any]], report["technical_checks"])
        assert check["reference_id"] == "cartier-santos-wssa0018"
        assert check["status"] == "conflicting" and check["photo"] == 1
        assert check["url"].startswith("https://www.cartier.com/")


@pytest.mark.parametrize("multiple", [False, True])
def test_date_rejects_exact_candidate_without_promoting_another_identity(multiple: bool) -> None:
    answer = _answer()
    santos = {"brand": "Cartier", "family": "Santos de Cartier", "photo": 1}
    answer.update(
        candidates=[{**santos, "reference": "WSSA0018", "reference_id": None}],
        observations=[{"photo": 1, "angle": "dial", "location": "mostrador", "detail": "Sem janela de data.",
                       "visibility": "visible", "field": "date_position", "value": "no date"}],
        next_verification="Procure a referência 4072 no fundo.",
    )
    if multiple:
        answer["candidates"].append({**santos, "reference": "WSSA0029", "reference_id": "cartier-santos-wssa0029"})
    result = FlashReport("fixture").build_report(answer, {1, 3})
    candidates = cast(list[dict[str, Any]], result["candidates"])
    assert candidates[0]["reference"] is None and candidates[0]["reference_id"] is None
    assert result["decision_reason"] == "reference_conflict"
    assert result["conclusion"] == "inconclusive"
    assert "4072" not in str(result["next_verification"])
    checks = cast(list[dict[str, Any]], result["technical_checks"])
    statuses = {"cartier-santos-wssa0018": "conflicting"}
    if multiple:
        assert candidates[1]["reference"] == "WSSA0029"
        statuses["cartier-santos-wssa0029"] = "matched"
    assert {check["reference_id"]: check["status"] for check in checks} == statuses


@pytest.mark.parametrize(("rear_visibility", "with_visible_screws"), [
    (None, False),
    ("ambiguous", False),
    ("visible", False),
    (None, True),
    ("ambiguous", True),
])
def test_catalog_gap_does_not_require_resending_an_examined_rear_marking(
    rear_visibility: str | None, with_visible_screws: bool,
) -> None:
    answer = _answer()
    with_rear = rear_visibility is not None or with_visible_screws
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Santos de Cartier", "reference": "WSSA0018",
                     "reference_id": "cartier-santos-wssa0018", "photo": 1}],
        hypothesis="inconclusive",
        linkage="consistent" if with_rear else "single_photo",
        observations=[
            {"photo": 1, "angle": "dial", "location": "6 horas", "detail": "Janela de data.",
             "visibility": "visible", "field": "date_position", "value": "6 o'clock"},
        ],
        next_verification="Examinar o movimento com um relojoeiro.",
    )
    if rear_visibility:
        answer["observations"].append({
            "photo": 2, "angle": "caseback", "location": "caseback text", "field": None, "value": None,
            "visibility": rear_visibility,
            "detail": "Gravação na tampa com STAINLESS STEEL e código 4072."
            if rear_visibility == "visible" else "Gravação da tampa ilegível.",
        })
    if with_visible_screws:
        answer["observations"].append({
            "photo": 2, "angle": "caseback", "location": "parafusos da tampa",
            "detail": "Parafusos visíveis.", "visibility": "visible", "field": None, "value": None,
        })

    report = FlashReport("fixture").build_report(answer, {1, 2} if with_rear else {1})

    assert report["decision_reason"] == "catalog_gap"
    assert report["conclusion"] == "inconclusive"
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    assert check["field"] == "date_position" and check["status"] == "matched"
    guidance = str(report["next_verification"])
    if rear_visibility == "visible":
        observations = cast(list[dict[str, Any]], report["observations"])
        assert "4072" in observations[1]["detail"]
        assert guidance.startswith("Se o código da caixa já estiver legível")
        assert "relojoeiro" in guidance
        assert "Se estiver ilegível, envie uma macro em foco" in guidance
    else:
        assert guidance.startswith("Fotografe o fundo da caixa de frente")
    assert "número de série" in guidance


def test_santos_medium_date_layout_is_sent_and_matches_without_guessing_a_case_code() -> None:
    builder = FlashReport("fixture")
    content, _ = builder.prepare([])
    prompt = str(content[0]["text"])
    assert "cartier-santos-wssa0029: no date" in prompt
    assert "cartier-santos-wssa0018: 6 o'clock" in prompt
    assert "field date_position" in prompt
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Santos de Cartier", "reference": "WSSA0029",
                     "reference_id": "cartier-santos-wssa0029", "photo": 1}],
        observations=[{"photo": 1, "angle": "dial", "location": "mostrador", "detail": "Sem janela de data.",
                       "visibility": "visible", "field": "date_position", "value": "no date"}],
        next_verification="Procure a referência 4072 no fundo.",
    )
    report = builder.build_report(answer, {1, 3})
    assert report["conclusion"] == "inconclusive"
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference"] == "WSSA0029"
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    assert check["reference_id"] == "cartier-santos-wssa0029" and check["status"] == "matched"
    assert "4072" not in str(report["next_verification"])


@pytest.mark.parametrize(("field", "shape", "visibility", "expected_value", "expected_detail"), [
    ("crown_shape", "faceted", "visible", "faceted", "Coroa de contorno facetado."),
    (" CROWN_SHAPE ", "heptagonal", "visible", "7 sided", "Coroa com sete lados."),
    ("crown_shape", "round", "visible", "round", "Coroa de contorno arredondado."),
    ("crown_shape", "espinélio sintético azul", "visible", None, "Formato da coroa indeterminado nesta foto."),
    ("crown_shape", "seven-sided", "ambiguous", None, "Formato da coroa indeterminado nesta foto."),
])
def test_crown_shape_observation_displays_geometry_only(
    field: str, shape: str, visibility: str, expected_value: str | None, expected_detail: str,
) -> None:
    answer = _answer()
    answer["candidates"] = [{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                             "reference_id": "cartier-santos-wssa0061", "photo": 3}]
    answer["observations"] = [
        {"photo": 3, "angle": "crown", "location": "espinélio azul",
         "detail": "Coroa heptagonal facetada engastada com espinélio sintético azul",
         "visibility": visibility, "field": field, "value": shape},
    ]
    result = FlashReport("fixture").build_report(answer, {1, 3})
    observation, = cast(list[dict[str, Any]], result["observations"])
    assert observation == {
        "photo": 3, "angle": "crown", "location": "coroa", "detail": expected_detail,
        "visibility": "visible" if expected_value else "ambiguous",
        "field": "crown_shape", "value": expected_value,
    }
    checks = cast(list[dict[str, Any]], result["technical_checks"])
    if expected_value in ("7 sided", "round"):
        check, = checks
        assert check["photo"] == 3 and check["field"] == "crown_shape"
        assert check["status"] == ("matched" if expected_value == "7 sided" else "conflicting")
    else:
        assert not checks
    assert result["conclusion"] == "inconclusive"


@pytest.mark.parametrize("non_watch", [None, "3", [2], [True], [1, 1]])
def test_watchless_photo_indices_must_be_distinct_prepared_inputs(non_watch: object) -> None:
    answer = {**_answer(), "non_watch_photos": non_watch}
    with pytest.raises(ValueError, match="foto sem relógio"):
        FlashReport("fixture").build_report(answer, {1, 3})


@pytest.mark.parametrize(("family", "reference", "reference_id", "expected_id", "reason"), [
    ("Collection", "F-91W-1", None, "casio-f91w-1", "catalog_gap"),
    ("Classic", "F-91W-1", "casio-f91w-1", "casio-f91w-1", "catalog_gap"),
    ("F-91W", "F-91W-1", "casio-f91w-1", "casio-f91w-1", "catalog_gap"),
    ("F-91W", "1", "casio-f91w-1", None, "reference_unresolved"),
    ("G-Shock", "F-91W-1", None, None, "reference_unresolved"),
    ("Collection", "F-91W", None, None, "reference_unresolved"),
    ("Collection", "F-91W-3", None, None, "catalog_gap"),
    ("Collection", "F-91W-1/F-91W-3", None, None, "reference_unresolved"),
    ("Collection", "F-91W-1", "rolex-submariner-126610ln", None, "reference_unresolved"),
])
def test_flash_resolves_only_exact_official_commercial_designation(
    family: str, reference: str, reference_id: str | None, expected_id: str | None, reason: str,
) -> None:
    answer = {
        "candidates": [{"brand": "Casio", "family": family, "reference": reference,
                        "reference_id": reference_id}],
        "linkage": "consistent",
        "hypothesis": "probable_original",
        "observations": [
            {"photo": 1, "angle": "dial", "location": "visor", "detail": "Visor digital",
             "visibility": "visible", "field": "display_type", "value": "digital"},
        ],
        "next_verification": "Fotografe a inscrição do fundo.",
    }
    report = FlashReport("fixture").build_report(answer, {1, 2})
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert candidates[0]["reference_id"] == expected_id
    assert candidates[0]["reference"] == reference
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    if expected_id:
        check, = checks
        assert check["photo"] == 1
        assert check["field"] == "display_type"
        assert check["status"] == "matched"
        assert check["url"].startswith("https://www.casio.com/")
    else:
        assert not checks
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == reason


@pytest.mark.parametrize("rear_detail", [
    None,
    "Fundo metálico com a inscrição 593",
    "Tampa metálica visível, gravação ilegível",
])
def test_casio_f91w_guidance_requires_evidence_beyond_the_display_test(rear_detail: str | None) -> None:
    rear_visible = rear_detail is not None
    answer: dict[str, Any] = {
        "candidates": [{"brand": "Casio", "family": "Classic", "reference": "F-91W-1",
                        "reference_id": "casio-f91w-1", "photo": 1}],
        "linkage": "consistent" if rear_visible else "single_photo",
        "hypothesis": "probable_original",
        "observations": [
            {"photo": 1, "angle": "dial", "location": "visor", "detail": "Visor digital",
             "visibility": "visible", "field": "display_type", "value": "digital"},
        ],
        "next_verification": "Acione o botão para ver CASIo e confirmar a autenticidade.",
    }
    if rear_visible:
        answer["observations"].append(
            {"photo": 2, "angle": "caseback", "location": "fundo",
             "detail": rear_detail, "visibility": "visible",
             "field": None, "value": None},
        )
    report = FlashReport("fixture").build_report(answer, {1, 2} if rear_visible else {1})
    next_verification = str(report["next_verification"])
    assert "confirmar a autenticidade" not in next_verification
    if rear_visible:
        assert "Se a inscrição da tampa estiver ilegível" in next_verification
        assert "Se já estiver legível" in next_verification
        assert "relojoeiro" in next_verification
        assert "placa" in next_verification
    else:
        assert "Fotografe o fundo da caixa" in next_verification
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "catalog_gap"


def test_flash_compares_one_exact_reference_repeated_across_linked_views() -> None:
    answer = _answer()
    candidate = {"brand": "Casio", "family": "Classic", "reference": "F-91W-1",
                 "reference_id": "casio-f91w-1"}
    answer["candidates"] = [{**candidate, "photo": 1}, {**candidate, "photo": 2}]
    answer["observations"] = [
        {"photo": 1, "angle": "dial", "location": "visor", "detail": "Mostrador digital LCD",
         "visibility": "visible", "field": "display_type", "value": "digital"},
        {"photo": 2, "angle": "caseback", "location": "fundo", "detail": "Inscrição 593 no fundo",
         "visibility": "visible", "field": None, "value": None},
    ]
    report = FlashReport("fixture").build_report(answer, {1, 2})
    assert report["candidates"] == [candidate]
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    assert check["reference_id"] == "casio-f91w-1"
    assert check["photo"] == 1 and check["field"] == "display_type" and check["status"] == "matched"
    assert check["url"].startswith("https://www.casio.com/")
    assert report["examined_photos"] == [1, 2]
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "catalog_gap"


@pytest.mark.parametrize("case", [
    "wrong_brand", "mixed", "uncertain", "multiple_candidates", "conflicting_ids", "catalogue_collision",
])
def test_commercial_designation_cannot_override_uncertain_identity(case: str) -> None:
    answer = _answer()
    candidate = {"brand": "Casio", "family": "Collection", "reference": "F-91W-1", "reference_id": None}
    answer["candidates"] = [candidate]
    reporter = FlashReport("fixture")
    if case == "wrong_brand":
        candidate["brand"] = "Seiko"
    elif case in ("mixed", "uncertain"):
        answer["linkage"] = case
        candidate["reference_id"] = "casio-f91w-1"
        answer["candidates"] = [{**candidate, "photo": 1}, {**candidate, "photo": 3}]
    elif case == "multiple_candidates":
        answer["candidates"].append({**candidate, "reference": "F-91W-3"})
    elif case == "conflicting_ids":
        answer["candidates"].append({**candidate, "reference_id": "rolex-submariner-126610ln"})
    else:
        reporter.reference_code_owners[("casio", "f91w1")].add("casio-other-registered-id")
    report = reporter.build_report(answer, {1, 3})
    assert not report["technical_checks"]
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == ("unlinked_photos" if case in ("mixed", "uncertain") else "reference_unresolved")
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert len(candidates) == len(answer["candidates"])
    assert all(item["reference_id"] is None for item in candidates)


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


@pytest.mark.parametrize("color_view", ["white", "blue"])
def test_crown_color_requires_chromatic_evidence_in_its_own_photo(color_view: str) -> None:
    answer = _answer()
    answer["candidates"] = [{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                             "reference_id": "cartier-santos-wssa0061", "photo": 1}]
    answer["observations"] = [
        {"photo": 1, "angle": "crown", "location": "coroa", "detail": "Coroa com sete lados.",
         "visibility": "visible", "field": "crown_shape", "value": "7 sided"},
        {"photo": 3, "angle": "crown", "location": "pedra azul", "detail": "Pedra azul.",
         "visibility": "visible", "field": "crown_stone_color", "value": "blue"},
    ]
    reporter = FlashReport("fixture", request=StubOpenRouter(answer))
    first = base64.b64decode(_photo("red"))
    third = base64.b64decode(_photo(color_view))
    photos = [("crown", "image/png", data) for data in (first, first, third)]
    report = reporter(photos)
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    color = cast(list[dict[str, Any]], report["observations"])[1]
    assert color["photo"] == 3
    if color_view == "blue":
        assert [check["field"] for check in checks] == ["crown_shape", "crown_stone_color"]
        assert report["conclusion"] == "visually_compatible"
    else:
        assert [check["field"] for check in checks] == ["crown_shape"]
        assert color["visibility"] == "ambiguous" and color["value"] is None
        assert "azul" not in color["detail"] and "azul" not in color["location"]
        assert report["conclusion"] == "inconclusive"

        def agree(request: dict[str, Any]) -> dict[str, Any]:
            fields = request["review_fields"]
            assert [field["field"] for field in fields.values()] == ["crown_shape"]
            return {"case_token": request["case_token"], "findings": [
                {"claim_id": key, "photo_id": field["photo_id"], "source_fact_id": field["source_fact_id"],
                 "status": "supported", "reading": "Sete lados visíveis."} for key, field in fields.items()
            ]}

        review = replay_with_review(reporter, "gray-color", photos,
                                    lambda _: (answer, {"model": MODEL, "model_response_id": "fixture"}), agree)
        assert review["with_review"] == "inconclusive"


@pytest.mark.parametrize(("bezel", "hypothesis", "comparison", "conclusion"), [
    ("blue and red", "probable_original", "matched", "visually_compatible"),
    ("blue", "probable_fake", "conflicting", "inconclusive"),
])
def test_gmt_pepsi_compares_exterior_without_authenticating(
    bezel: str, hypothesis: str, comparison: str, conclusion: str,
) -> None:
    reference_id = "rolex-gmtmasterii-126710blro"
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Rolex", "family": "GMT-Master II",
                     "reference": "126710BLRO", "reference_id": reference_id}],
        linkage="single_photo",
        hypothesis=hypothesis,
    )
    answer["observations"][1]["photo"] = 1
    answer["observations"].extend([
        {"photo": 1, "angle": "dial", "location": "aro", "detail": bezel,
         "visibility": "visible", "field": "bezel_color", "value": bezel},
        {"photo": 1, "angle": "dial", "location": "caixa", "detail": "3285",
         "visibility": "visible", "field": "calibre", "value": "3285"},
    ])
    provider = StubOpenRouter(answer)
    report = FlashReport("fixture", request=provider)(
        [("dial", "image/png", base64.b64decode(_photo("#100912")))],
    )
    assert len(provider.posts) == 1
    assert reference_id in provider.posts[0]["messages"][0]["content"][0]["text"]
    assert report["conclusion"] == conclusion
    observations = cast(list[dict[str, Any]], report["observations"])
    assert {item["field"] for item in observations} == {"dial_color", "date_magnifier", "bezel_color"}
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    assert {item["field"]: item["status"] for item in checks} == {
        "dial_color": "matched", "date_magnifier": "matched", "bezel_color": comparison,
    }
    brochure = "https://assets.rolex.com/api/brochure/en-us/gmt-master-ii/m126710blro-0001.pdf"
    assert all(item["photo"] == 1 and item["url"] == brochure for item in checks)


@pytest.mark.parametrize("angle,detail", [
    ("caseback", "Inscrição com código de módulo 593 gravado na tampa traseira metálica aparafusada"),
    ("movement", "Inscrição com código de módulo 593 gravado na tampa traseira metálica aparafusada"),
    ("movement", "Código 593 gravado na face externa da tampa traseira."),
    ("movement", "593 stamped on the outer caseback."),
    ("movement", "O código 593 está na tampa traseira fechada."),
    ("movement", "The code 593 is on the exterior of the caseback."),
    ("movement", "Este é um calibre genuíno garantido"),
    ("movement", "plate: 593"),
])
def test_external_caseback_code_does_not_become_a_movement_reading(angle: str, detail: str) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Casio", "family": "Classic", "reference": "F-91W-1",
                     "reference_id": "casio-f91w-1", "photo": 1}],
        observations=[
            {"photo": 1, "angle": "dial", "location": "dial", "detail": "Mostrador digital LCD",
             "visibility": "visible", "field": "display_type", "value": "digital"},
            {"photo": 2, "angle": angle, "location": "module", "detail": detail,
             "visibility": "visible", "field": "movement_marking", "value": "593"},
            {"photo": 2, "angle": "caseback", "location": "caseback",
             "detail": "Tampa traseira fixada por parafusos", "visibility": "visible",
             "field": "caseback_type", "value": "screwed"},
        ],
    )
    report = FlashReport("fixture").build_report(answer, {1, 2})
    observations = cast(list[dict[str, Any]], report["observations"])
    reading, = [item for item in observations if "593" in item["detail"]]
    assert reading["photo"] == 2
    assert reading["location"] == "fundo"
    assert reading["field"] is None
    assert reading["detail"] == "Inscrição legível na vista do fundo · 593"
    assert all(item["field"] != "movement_marking" for item in observations)
    assert report["conclusion"] != "probable_original"


def test_visible_internal_marking_through_transparent_caseback_keeps_its_origin() -> None:
    answer = _answer()
    answer.update(linkage="single_photo", observations=[
        {"photo": 1, "angle": "caseback", "location": "fundo",
         "detail": "Rotor visível através do fundo transparente.",
         "visibility": "visible", "field": None, "value": None},
        {"photo": 1, "angle": "movement", "location": "rotor",
         "detail": "rotor: 26-330", "visibility": "visible", "field": "movement_marking", "value": "26-330"},
    ])
    report = FlashReport("fixture").build_report(answer, {1})
    reading, = [item for item in cast(list[dict[str, Any]], report["observations"])
                if item["field"] == "movement_marking"]
    assert reading["field"] == "movement_marking"
    assert reading["location"] == "rotor"


@pytest.mark.parametrize("marking", ["324 S C", "26-330 S C"])
def test_aquanaut_catalogue_keeps_movement_marking_separate_from_reference_checks(marking: str) -> None:
    reference_id = "patek-aquanaut-5168g010"
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Patek Philippe", "family": "Aquanaut", "reference": "5168G-010",
                     "reference_id": reference_id, "photo": 1}],
        hypothesis="probable_fake",
        observations=[
            {"photo": 1, "angle": "dial", "location": "janela da data", "detail": "Data visível às 3h.",
             "visibility": "visible", "field": "date_position", "value": "3 o'clock"},
            {"photo": 2, "angle": "movement", "location": "bridge", "detail": f"bridge: {marking}",
             "visibility": "visible", "field": "movement_marking", "value": marking},
        ],
    )
    reporter = FlashReport("fixture")
    lot_url = "https://www.artcurial.com/en/sales/MC-6026/lots/166-a"
    lot_claims = [claim for claim in reporter.claims[reference_id]
                  if claim.field == "calibre" and claim.source_url == lot_url]
    assert {claim.value for claim in lot_claims} == {"324 S C", "26-330 S C"}
    assert all(not claim.applicable_angles for claim in lot_claims)
    report = reporter.build_report(answer, {1, 2})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference_id"] == reference_id
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    assert (check["photo"], check["field"], check["status"]) == (1, "date_position", "matched")
    assert check["url"] == "https://www.patek.com/en/collection/aquanaut/5168g-010"
    readings = cast(list[dict[str, Any]], report["observations"])
    assert any(row["field"] == "movement_marking" and row["value"] == marking for row in readings)
    assert report["conclusion"] == "inconclusive"
    assert report["quality_score_0_1000"] is None


@pytest.mark.parametrize("angle,location,visibility,value,retained", [
    ("movement", "rotor", "visible", "26-330 S C", True),
    ("movement", "plate", "visible", "1847 MC", True),
    ("movement", "module", "visible", "593", True),
    ("movement", "bridge", "visible", "899/1", True),
    ("movement", "plate", "visible", "C.79350", True),
    ("movement", "bridge", "visible", "C. 79350", True),
    ("movement", "plate", "visible", "No.12345", False),
    ("movement", "plate", "visible", "C.123456", False),
    ("movement", "bridge", "ambiguous", "26-330", False),
    ("caseback", "exterior da tampa", "visible", "593", False),
    ("dial", "rotor", "visible", "26-330", False),
    ("caseback", "bridge", "visible", "1234567890", False),
    ("caseback", "bridge", "visible", "A123-4567", False),
    ("caseback", "bridge", "visible", "", False),
    ("caseback", "bridge", "visible", 3235, False),
    ("caseback", None, "visible", "26-330", False),
    ("movement", "caseback", "visible", "26-330", False),
    ("caseback", "bridge", "unknown", "26-330", False),
])
def test_movement_marking_is_literal_photo_evidence(
    angle: str, location: str | None, visibility: str, value: str | int, retained: bool,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Patek Philippe", "family": "Nautilus", "reference": "5711/1A-014",
                     "reference_id": None}],
        linkage="single_photo",
        observations=[{"photo": 1, "angle": angle, "location": location, "visibility": visibility,
                       "field": "movement_marking", "value": value,
                       "detail": f"{location}: {value}"}],
    )
    provider = StubOpenRouter(answer)
    report = FlashReport("fixture", request=provider)(
        [("caseback", "image/png", base64.b64decode(_photo("black")))],
    )
    observations = cast(list[dict[str, Any]], report["observations"])
    if retained:
        observation, = observations
        assert observation["detail"] == f"Inscrição legível no movimento · {value}"
        assert observation["field"] == "movement_marking"
        assert observation["photo"] == 1
        assert any("arquitetura do movimento" in limit for limit in cast(list[str], report["limits"]))
    else:
        assert not observations
    assert report["examined_photos"] == ([1] if retained else [])
    assert not report["technical_checks"]
    assert report["conclusion"] == "inconclusive"
    assert report["quality_score_0_1000"] is None


@pytest.mark.parametrize("scenario, status, conclusion", [
    ("compatible", 200, "visually_compatible"),
    ("mixed", 200, "inconclusive"),
    ("uncertain", 200, "inconclusive"),
    ("color-only", 200, "inconclusive"),
    ("weak-compatibility", 200, "inconclusive"),
    ("calibre-unseen", 200, "inconclusive"),
    ("invalid-movement-marking", 200, "visually_compatible"),
    ("movement-description", 200, "visually_compatible"),
    ("wrong-reference-code", 200, "inconclusive"),
    ("wrong-reference-brand", 200, "inconclusive"),
    ("outside-catalog", 200, "inconclusive"),
    ("equivalent-spec-words", 200, "inconclusive"),
    ("cartier-crown-ambiguous", 200, "inconclusive"),
    ("structural-divergence", 200, "probable_fake"),
    ("wrong-photo", 503, None),
])
def test_flash_batch_http_route(scenario: str, status: int, conclusion: str | None) -> None:
    answer = _answer()
    if scenario in {"mixed", "uncertain"}:
        answer["linkage"] = scenario
    elif scenario == "color-only":
        answer["hypothesis"] = "probable_fake"
        answer["observations"][0]["value"] = "blue"
    elif scenario == "weak-compatibility":
        answer["observations"][1].update(visibility="ambiguous", value=None)
        answer["next_verification"] = "Peça a abertura da caixa."
    elif scenario == "calibre-unseen":
        answer["observations"][0].update(field="calibre", value="3235")
    elif scenario == "invalid-movement-marking":
        answer["observations"].append({"photo": 3, "angle": "movement", "location": None, "detail": "",
                                       "visibility": "unknown", "field": "movement_marking", "value": 3235})
    elif scenario == "movement-description":
        answer["observations"].append({"photo": 3, "angle": "movement", "location": "rotor",
                                       "detail": "Rotor com gravação da coroa", "visibility": "visible",
                                       "field": None, "value": None})
    elif scenario == "wrong-reference-code":
        answer["candidates"][0]["reference"] = "126610LV"
    elif scenario == "wrong-reference-brand":
        answer["candidates"][0].update(brand="Cartier", family="Santos")
    elif scenario == "outside-catalog":
        answer["candidates"] = [{"brand": "Rolex", "family": "Cosmograph Daytona",
                                 "reference": "116588TBR", "reference_id": None}]
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
    dial = _photo("#100912")
    body = {"images": [{"angle": angle, "content_type": "image/png", "data": image}
                      for angle, image in [("dial", dial), ("rehaut", dial), ("cyclops", _photo("white"))]]}
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
        assert data["examined_photos"] == ([3] if scenario == "calibre-unseen" else [1, 3])
        checks = cast(list[dict[str, Any]], data["technical_checks"])
        expected_photos = {
            "compatible": [1, 3], "mixed": [], "uncertain": [], "color-only": [1, 3],
            "weak-compatibility": [1], "calibre-unseen": [3], "invalid-movement-marking": [1, 3],
            "movement-description": [1, 3],
            "wrong-reference-code": [], "wrong-reference-brand": [],
            "outside-catalog": [],
            "equivalent-spec-words": [1, 3], "structural-divergence": [1, 3], "cartier-crown-ambiguous": [1],
        }
        assert [check["photo"] for check in checks] == expected_photos[scenario]
        brand = answer["candidates"][0]["brand"].lower()
        if scenario == "wrong-reference-brand":
            assert not checks
        assert all(check["url"].startswith(f"https://www.{brand}.com/") for check in checks)
        if scenario == "equivalent-spec-words":
            assert {check["status"] for check in checks} == {"matched"}
        if scenario == "movement-description":
            assert {
                "photo": 3, "angle": "caseback", "location": "rotor", "detail": "Rotor com gravação da coroa",
                "visibility": "visible", "field": None, "value": None,
            } in data["observations"]
        assert data["model_hypothesis"] == answer["hypothesis"]
        expected_reason = {
            "mixed": "unlinked_photos", "uncertain": "unlinked_photos", "wrong-reference-code": "reference_unresolved",
            "wrong-reference-brand": "reference_unresolved", "structural-divergence": None,
            "equivalent-spec-words": "compatible_or_insufficient", "cartier-crown-ambiguous": "detail_not_verified",
        }.get(scenario, "catalog_gap")
        assert data["decision_reason"] == expected_reason
        if scenario == "outside-catalog":
            assert "não tem ficha técnica" in data["decision_reason_text"]
            assert all("A comparação usa a ficha" not in limit for limit in data["limits"])
        if scenario == "cartier-crown-ambiguous":
            assert "coroa de perfil" in data["next_verification"]
        if scenario == "weak-compatibility":
            assert "janela de data perpendicular" in data["next_verification"]
            assert "autenticidade segue em aberto" in data["next_verification"]
        if scenario == "mixed":
            assert "um relógio por vez" in data["next_verification"]
            assert "lotes" in data["next_verification"]
        if scenario == "uncertain":
            assert "ângulo oblíquo" in data["next_verification"]
            assert "a mesma marca" in data["next_verification"]
            assert "uma das fotos anteriores" in data["next_verification"]


@pytest.mark.parametrize(("family", "dial_color", "magnifier", "expected_id"), [
    ("Day-Date", "green", "present", "rolex-daydate-228235"),
    ("Day-Date 40", "white", "cyclops lens", "rolex-daydate-228235"),
    ("Day-Date 36", "green", "cyclops", None),
])
def test_daydate_228235_compares_shared_features_without_dial_or_calibre_inference(
    family: str, dial_color: str, magnifier: str, expected_id: str | None,
) -> None:
    answer = {
        "candidates": [{"brand": "Rolex", "family": family, "reference": "228235",
                        "reference_id": None, "photo": 1}],
        "linkage": "single_photo",
        "hypothesis": "probable_fake",
        "observations": [
            {"photo": 1, "angle": "dial", "location": "data", "detail": "Janela às 3 horas",
             "visibility": "visible", "field": "date_position", "value": "3 o'clock"},
            {"photo": 1, "angle": "dial", "location": "vidro", "detail": "Lente acima da data",
             "visibility": "visible", "field": "date_magnifier", "value": magnifier},
            {"photo": 1, "angle": "dial", "location": "mostrador", "detail": "Cor do mostrador",
             "visibility": "visible", "field": "dial_color", "value": dial_color},
            {"photo": 1, "angle": "dial", "location": "movimento", "detail": "Calibre 3255",
             "visibility": "visible", "field": "calibre", "value": "3255"},
        ],
        "next_verification": "Fotografe o mostrador perpendicularmente.",
    }
    report = FlashReport("fixture").build_report(answer, {1})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference_id"] == expected_id
    assert report["conclusion"] == "inconclusive"
    assert all(row["field"] != "calibre" for row in cast(list[dict[str, Any]], report["observations"]))
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    if expected_id:
        assert report["decision_reason"] == "compatible_or_insufficient"
        assert {row["field"] for row in checks} == {"date_position", "date_magnifier"}
        assert all(row["photo"] == 1 and row["status"] == "matched" for row in checks)
        assert all(row["url"] == "https://www.rolex.com/watches/day-date/m228235-0025" for row in checks)
    else:
        assert report["decision_reason"] == "reference_unresolved"
        assert not checks


@pytest.mark.parametrize(("date_value", "hypothesis", "expected_id", "expected_status"), [
    ("6 o'clock", "probable_original", "cartier-tankmust-wsta0053", "matched"),
    ("6 o'clock", "probable_fake", "cartier-tankmust-wsta0053", "matched"),
    ("no date", "probable_fake", None, "conflicting"),
])
def test_tank_must_wsta0053_checks_date_without_authenticating_a_copied_marking(
    date_value: str, hypothesis: str, expected_id: str | None, expected_status: str,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Tank Must XL", "reference": "WSTA0053",
                     "reference_id": None, "photo": 1}],
        hypothesis=hypothesis,
        observations=[
            {"photo": 1, "angle": "dial", "location": "data", "detail": "Análise da janela de data.",
             "visibility": "visible", "field": "date_position", "value": date_value},
            {"photo": 2, "angle": "movement", "location": "rotor", "detail": "rotor: 1847 MC",
             "visibility": "visible", "field": "movement_marking", "value": "1847 MC"},
        ],
    )
    report = FlashReport("fixture").build_report(answer, {1, 2})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    assert candidate["reference_id"] == expected_id
    assert report["conclusion"] == "inconclusive"
    assert report["quality_score_0_1000"] is None
    assert check["field"] == "date_position" and check["photo"] == 1
    assert check["status"] == expected_status
    assert check["url"] == "https://www.cartier.com/en-us/watches/collections/tank/tank-must-watch-CRWSTA0053.html"
    assert any(row["field"] == "movement_marking" and row["photo"] == 2
               for row in cast(list[dict[str, Any]], report["observations"]))


@pytest.mark.parametrize(("reference", "date_value", "expected_status"), [
    ("W6920046", "no date", "matched"),
    ("W6920046", "3 o'clock", "conflicting"),
    ("W69012Z4", "no date", "conflicting"),
    ("WSBB0049", "no date", "conflicting"),
])
def test_ballon_bleu_date_distinguishes_commercial_variants(
    reference: str, date_value: str, expected_status: str,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Ballon Bleu de Cartier", "reference": reference,
                     "reference_id": None, "photo": 1}],
        linkage="single_photo",
        hypothesis="probable_fake",
        observations=[
            {"photo": 1, "angle": "dial", "location": "mostrador", "detail": "Inspeção da posição da data.",
             "visibility": "visible", "field": "date_position", "value": date_value},
        ],
    )
    report = FlashReport("fixture").build_report(answer, {1})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    check, = cast(list[dict[str, Any]], report["technical_checks"])
    if expected_status == "matched":
        assert candidate["reference_id"] == f"cartier-ballonbleu-{reference.lower()}"
    else:
        assert candidate["reference_id"] is None
        assert report["decision_reason"] == "reference_conflict"
    assert (check["field"], check["photo"], check["status"]) == ("date_position", 1, expected_status)
    assert check["url"].startswith(f"https://www.cartier.mx/{reference.lower()}-")
    assert report["conclusion"] == "inconclusive"
    assert report["quality_score_0_1000"] is None


@pytest.mark.parametrize("suggested", ["W69012Z4", "WSBB0049", None])
@pytest.mark.parametrize("extra", ["none", "crown", "caseback", "movement-description", "1847 MC", "049"])
def test_ballon_bleu_42_shared_details_keep_both_generations(suggested: str | None, extra: str) -> None:
    answer = _answer()
    models = [suggested] if suggested else ["W69012Z4", "WSBB0049"]
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Ballon Bleu de Cartier", "reference": reference,
                     "reference_id": None, "photo": 1} for reference in models],
        linkage="single_photo",
        observations=[
            {"photo": 1, "angle": "dial", "location": "janela da data", "detail": "Data visível às 3h.",
             "visibility": "visible", "field": "date_position", "value": "3 o'clock"},
            {"photo": 1, "angle": "dial", "location": "ponteiros", "detail": "Ponteiros em forma de espada.",
             "visibility": "visible", "field": "hand_shape", "value": "sword"},
        ],
    )
    indices = {1}
    if extra == "crown":
        answer["observations"].append(
            {"photo": 1, "angle": "crown", "location": "coroa", "detail": "Coroa visível na frente.",
             "visibility": "visible", "field": None, "value": None},
        )
    elif extra != "none":
        indices.add(2)
        answer["linkage"] = "consistent"
        observation = {"photo": 2, "angle": "caseback", "location": "fundo", "detail": "Tampa metálica visível.",
                       "visibility": "visible", "field": None, "value": None}
        if extra == "movement-description":
            observation.update(angle="movement", location="rotor", detail="Rotor com gravação.")
        elif extra != "caseback":
            observation.update(angle="movement", location="rotor", detail=f"rotor: {extra}",
                               field="movement_marking", value=extra)
        answer["observations"].append(observation)

    report = FlashReport("fixture").build_report(answer, indices)
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert {candidate["reference"] for candidate in candidates} == {"W69012Z4", "WSBB0049"}
    assert candidates[0]["reference"] == (suggested or "W69012Z4")
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    assert {(check["reference_id"], check["field"], check["photo"], check["status"]) for check in checks} == {
        (f"cartier-ballonbleu-{reference.lower()}", field, 1, "matched")
        for reference in ("W69012Z4", "WSBB0049") for field in ("date_position", "hand_shape")
    }
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "reference_unresolved"
    requested = "cartão de garantia" if extra == "caseback" else "encaixe da pulseira"
    assert requested in str(report["next_verification"])
    assert report["quality_score_0_1000"] is None
    if extra in {"1847 MC", "049"}:
        assert any(item["photo"] == 2 and item["field"] == "movement_marking" and item["value"] == extra
                   for item in cast(list[dict[str, Any]], report["observations"]))


@pytest.mark.parametrize("case_code", ("3284", "3001", "3765"))
def test_ballon_bleu_case_code_does_not_select_one_commercial_variant(case_code: str) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Ballon Bleu", "reference": case_code,
                     "reference_id": None, "photo": 1}],
        linkage="single_photo",
        hypothesis="probable_original",
        observations=[],
    )
    report = FlashReport("fixture").build_report(answer, {1})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference_id"] is None
    assert not report["technical_checks"]
    assert report["conclusion"] == "inconclusive"


def test_exact_reference_missing_from_catalogue_requests_an_independent_examination() -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Rolex", "family": "Datejust", "reference": "16233",
                     "reference_id": None, "photo": 1}],
        observations=[
            {"photo": 1, "angle": "dial", "location": "janela da data",
             "detail": "Janela de data às 3h.", "visibility": "visible",
             "field": "date_position", "value": "3 o'clock"},
            {"photo": 2, "angle": "movement", "location": "bridge",
             "detail": "bridge: 3135", "visibility": "visible",
             "field": "movement_marking", "value": "3135"},
        ],
        next_verification="Fotografe novamente o mostrador de frente.",
    )
    report = FlashReport("fixture").build_report(answer, {1, 2})
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "catalog_gap"
    assert not report["technical_checks"]
    assert report["next_verification"] != answer["next_verification"]
    guidance = cast(str, report["next_verification"])
    assert "relojoeiro" in guidance and "exemplar documentado" in guidance


@pytest.mark.parametrize("scenario, reason", [
    ("partial-reference", "reference_unresolved"),
    ("multiple-candidates", "reference_unresolved"),
    ("catalogued-reference-without-id", "reference_unresolved"),
    ("catalogued-reference-with-alternate-family", "reference_unresolved"),
    ("catalogued-code-with-wrong-brand", "reference_unresolved"),
    ("unknown-reference-text", "reference_unresolved"),
    ("alternatives-in-reference-text", "reference_unresolved"),
    ("slash-separated-references", "reference_unresolved"),
    ("unknown-brand-text", "reference_unresolved"),
    ("unknown-family-text", "reference_unresolved"),
    ("unlinked-photos", "unlinked_photos"),
])
def test_catalogue_feedback_preserves_uncertainty(scenario: str, reason: str) -> None:
    answer = _answer()
    candidate = {"brand": "Rolex", "family": "Cosmograph Daytona",
                 "reference": "116588TBR", "reference_id": None}
    answer["candidates"] = [candidate]
    if scenario == "partial-reference":
        candidate["reference"] = None
    elif scenario == "multiple-candidates":
        answer["candidates"].append({**candidate, "reference": "116508"})
    elif scenario == "catalogued-reference-without-id":
        candidate.update(family="Submariner", reference="126610LN")
    elif scenario == "catalogued-reference-with-alternate-family":
        candidate.update(family="Submariner Date", reference="126610LN")
    elif scenario == "catalogued-code-with-wrong-brand":
        candidate.update(brand="Cartier", family="Santos", reference="126610LN")
    elif scenario == "unknown-reference-text":
        candidate["reference"] = "desconhecida"
    elif scenario == "alternatives-in-reference-text":
        candidate["reference"] = "116588TBR ou 116508"
    elif scenario == "slash-separated-references":
        candidate["reference"] = "116588TBR/116508"
    elif scenario == "unknown-brand-text":
        candidate["brand"] = "desconhecida"
    elif scenario == "unknown-family-text":
        candidate["family"] = "não identificada"
    else:
        answer["linkage"] = "uncertain"
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["decision_reason"] == reason
    assert report["conclusion"] == "inconclusive"


@pytest.mark.parametrize(("scenario", "expected"), [
    ("agree", "visually_compatible"),
    ("different_reading", "inconclusive"),
    ("independently_confirmed_divergence", "inconclusive"),
    ("supported_structural_divergence", "probable_fake"),
    ("tank_date_only_review", "inconclusive"),
    ("unavailable", "unavailable"),
    ("mixed", "inconclusive"),
    ("date_reference_conflict", "inconclusive"),
    ("date_readings_disagree", "inconclusive"),
    ("date_reference_unresolved", "inconclusive"),
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
    date_scenario = scenario in ("date_reference_conflict", "date_readings_disagree", "date_reference_unresolved")
    if date_scenario:
        answer["candidates"] = [{"brand": "Cartier", "family": "Santos de Cartier", "reference": "WSSA0018",
                                 "reference_id": "cartier-santos-wssa0018", "photo": 1}]
        answer["observations"] = [{"photo": 1, "angle": "dial", "location": "6 horas",
                                   "detail": "Sem janela de data.", "visibility": "visible",
                                   "field": "date_position", "value": "no date"}]
    if scenario == "date_readings_disagree":
        answer["observations"][0].update(value="3 o'clock", detail="Data às 3.")
        answer["observations"].append({**answer["observations"][0], "photo": 3,
                                       "field": "DATE_POSITION", "value": "12 o'clock", "detail": "Data às 12."})
    if scenario == "date_reference_unresolved":
        answer["candidates"][0]["reference_id"] = None
        answer["observations"][0].update(value="6 o'clock", detail="Data às 6.")
    if scenario == "independently_confirmed_divergence":
        answer["hypothesis"] = "probable_fake"
        answer["observations"][0].update(value="blue", detail="Mostrador azul")
    if scenario == "supported_structural_divergence":
        answer["hypothesis"] = "probable_fake"
        answer["candidates"] = [{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                                 "reference_id": "cartier-santos-wssa0061", "photo": 1}]
        answer["observations"] = [
            {"photo": 1, "angle": "dial", "location": "ponteiros", "detail": "Ponteiros em bastão.",
             "visibility": "visible", "field": "hand_shape", "value": "baton"},
            {"photo": 3, "angle": "crown", "location": "coroa", "detail": "Coroa arredondada.",
             "visibility": "visible", "field": "crown_shape", "value": "round"},
        ]
    if scenario == "tank_date_only_review":
        answer["candidates"] = [{"brand": "Cartier", "family": "Tank Must XL", "reference": "WSTA0053",
                                 "reference_id": "cartier-tankmust-wsta0053", "photo": 1}]
        answer["observations"] = [
            {"photo": 1, "angle": "dial", "location": "data", "detail": "Data às 6.",
             "visibility": "visible", "field": "date_position", "value": "6 o'clock"},
            {"photo": 3, "angle": "movement", "location": "rotor", "detail": "rotor: 1847 MC",
             "visibility": "visible", "field": "movement_marking", "value": "1847 MC"},
        ]
    provider = StubOpenRouter(answer)
    builder = FlashReport("fixture", request=provider)
    dial, white = (base64.b64decode(_photo(color)) for color in ("#100912", "white"))
    photos = [("dial", "image/png", dial), ("rehaut", "image/png", dial),
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
            expected_reference = {
                "tank_date_only_review": "cartier-tankmust-wsta0053",
                "supported_structural_divergence": "cartier-santos-wssa0061",
            }.get(scenario, "cartier-santos-wssa0018" if date_scenario else "rolex-submariner-126610ln")
            assert fact["reference_id"] == expected_reference
            if scenario == "tank_date_only_review":
                assert fact["field"] == "date_position"
        encoded = json.dumps(request)
        assert "seller-fixture" not in encoded and '"probable_original"' not in encoded
        assert '"probable_fake"' not in encoded and '"hypothesis"' not in encoded
        assert "Mostrador preto" not in encoded and "Mostrador azul" not in encoded
        findings = [
            {
                "claim_id": key, "photo_id": item["photo_id"],
                "source_fact_id": item["source_fact_id"], "reading": "Leitura simulada.",
                "status": "contradicted" if (
                    scenario in ("date_reference_conflict", "date_readings_disagree", "supported_structural_divergence")
                    or scenario == "independently_confirmed_divergence" and item["field"] == "dial_color"
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
        if date_scenario:
            reason = scenario if scenario == "date_readings_disagree" else scenario.removeprefix("date_")
            assert result["previous_policy_reason"] == reason
            assert result["review_status"] == "needs_verification"
            assert not result["disputed_claim_ids"]
        if scenario == "different_reading":
            disputed, = result["disputed_claim_ids"]
            assert result["review_findings"][disputed]["status"] == "not_visible"
        if scenario in ("independently_confirmed_divergence", "tank_date_only_review"):
            assert result["review_status"] == "needs_verification"
            assert not result["disputed_claim_ids"]
        if scenario == "supported_structural_divergence":
            assert result["previous_policy"] == "probable_fake"
            assert result["review_status"] == "supported"
    assert len(calls) == (0 if scenario == "mixed" else 1)
    assert len(provider.posts) == 1


@pytest.mark.parametrize(("linkage", "marking", "detail", "date_value", "shared"), [
    ("single_photo", "4324", "Código legível 4324 na tampa.", None, True),
    ("uncertain", "4324", "Código legível 4324 na tampa.", "6 o'clock", True),
    ("consistent", "4324", "Código legível 4324 na tampa.", "6 o'clock", True),
    ("consistent", "4324", "Código legível 4324 na tampa.", "no date", False),
    ("single_photo", "4390", "Marcação legível 4390. Não há inscrição 4324.", None, False),
    ("single_photo", "4324-1", "Gravação externa 4324-1 na tampa traseira.", None, False),
    ("single_photo", None, "Leitura incerta entre 4324 e 4390.", None, False),
    ("mixed", "4324", "Código legível 4324 na tampa.", "6 o'clock", False),
])
def test_tank_shared_case_marking_preserves_both_commercial_hypotheses(
    linkage: str, marking: str | None, detail: str, date_value: str | None, shared: bool,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Cartier", "family": "Tank Must", "reference": "WSTA0053",
                     "reference_id": "cartier-tankmust-wsta0053", "photo": 1}],
        linkage=linkage,
        hypothesis="inconclusive",
        observations=[
            {"photo": 1, "angle": "caseback", "location": "tampa", "detail": detail,
             "visibility": "visible", "field": None, "value": marking},
        ],
    )
    indices = {1}
    if date_value is not None:
        answer["candidates"][0]["photo"] = 2
        answer["observations"].append(
            {"photo": 2, "angle": "dial", "location": "data", "detail": "Leitura da janela de data.",
             "visibility": "visible", "field": "date_position", "value": date_value},
        )
        indices.add(2)
    report = FlashReport("fixture").build_report(answer, indices)
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert report["conclusion"] == "inconclusive"
    assert report["linkage"] == linkage
    if not shared:
        assert all(candidate["reference"] != "WSTA0040" for candidate in candidates)
        if date_value == "no date":
            assert report["decision_reason"] == "reference_conflict"
        return
    assert {candidate["reference"] for candidate in candidates} == {"WSTA0040", "WSTA0053"}
    assert all(candidate["reference_id"] is None for candidate in candidates)
    assert any("4324" in limit and "WSTA0040" in limit for limit in cast(list[str], report["limits"]))
    guidance = cast(str, report["next_verification"])
    if linkage == "uncertain":
        assert report["decision_reason"] == "unlinked_photos"
        assert "oblíquo" in guidance
        assert any("se a gravação estiver no mesmo relógio" in limit for limit in cast(list[str], report["limits"]))
    else:
        assert report["decision_reason"] == "reference_unresolved"
        assert "garantia" in guidance and "sem compartilhá-lo" in guidance


@pytest.mark.parametrize(("rear_type", "linkage", "expected", "conflict"), [
    ("exhibition", "single_photo", {"310.30.42.50.01.002"}, "310.30.42.50.01.001"),
    ("solid", "single_photo", {"310.30.42.50.01.001"}, "310.30.42.50.01.002"),
    ("open", "single_photo", {"310.30.42.50.01.001", "310.30.42.50.01.002"}, None),
    ("open_and_window", "single_photo", {"310.30.42.50.01.001", "310.30.42.50.01.002"}, None),
    ("transparent", "uncertain", {"310.30.42.50.01.001", "310.30.42.50.01.002"}, None),
])
def test_moonwatch_fitted_caseback_checks_factory_variant_only(
    rear_type: str, linkage: str, expected: set[str], conflict: str | None,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[
            {"brand": "Omega", "family": "Speedmaster", "reference": "310.30.42.50.01.002",
             "reference_id": None, "photo": 1},
            {"brand": "Omega", "family": "Speedmaster", "reference": "310.30.42.50.01.001",
             "reference_id": "omega-speedmaster-31030425001001", "photo": 1},
        ],
        observations=[
            {"photo": 1, "angle": "caseback", "location": "fundo", "detail": "Leitura do fundo da caixa.",
             "visibility": "visible", "field": "caseback_type",
             "value": "open" if rear_type == "open_and_window" else rear_type},
            {"photo": 1, "angle": "movement", "location": "bridge", "detail": "bridge: 3861",
             "visibility": "visible", "field": "movement_marking", "value": "3861"},
        ],
        hypothesis="probable_original",
        linkage=linkage,
    )
    if rear_type == "open_and_window":
        answer["observations"].append({**answer["observations"][0], "value": "transparent"})
    report = FlashReport("fixture").build_report(answer, {1})
    candidates = cast(list[dict[str, Any]], report["candidates"])
    assert {row["reference"] for row in candidates} == expected
    assert all(row["reference_id"] is None for row in candidates)
    assert report["conclusion"] == "inconclusive"
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    if conflict:
        assert report["decision_reason"] == "factory_caseback_difference"
        reason = cast(str, report["decision_reason_text"])
        assert conflict in reason and "substituída" in reason
        assert {check["status"] for check in checks} == {"matched", "conflicting"}
        assert all(check["photo"] == 1 and check["field"] == "caseback_type" for check in checks)
        assert {check["citation"].split(" > ")[0] for check in checks} == {
            "Omega 310.30.42.50.01.001", "Omega 310.30.42.50.01.002",
        }
    else:
        assert not checks
    if rear_type.startswith("open"):
        assert "tampa" in cast(str, report["next_verification"])
        assert report["decision_reason"] == ("caseback_removed" if rear_type == "open" else "caseback_state_uncertain")


def test_moonwatch_replacement_caseback_keeps_identified_family_and_conditions_factory_reference() -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Omega", "family": "Speedmaster", "reference": "310.30.42.50.01.001",
                     "reference_id": "omega-speedmaster-31030425001001", "photo": 1}],
        observations=[{"photo": 1, "angle": "caseback", "location": "fundo", "detail": "Janela instalada.",
                       "visibility": "visible", "field": "caseback_type", "value": "transparent"}],
        hypothesis="probable_fake", linkage="single_photo",
    )
    report = FlashReport("fixture").build_report(answer, {1})
    assert report["candidates"] == [
        {"brand": "Omega", "family": "Speedmaster", "reference": None, "reference_id": None},
    ]
    assert report["conclusion"] == "inconclusive"
    assert "310.30.42.50.01.001" in cast(str, report["decision_reason_text"])
    assert "troca da tampa" in cast(str, report["next_verification"])


@pytest.mark.parametrize(("reference", "proposed", "compares"), [
    ("Moonwatch Professional Co-Axial Master Chronometer Chronograph 42 mm", "omega-speedmaster-31030425001002", True),
    ("310.30.42.50.01.002", "omega-speedmaster-31030425001002", True),
    ("310.30.42.50.01.001", "omega-speedmaster-31030425001002", False),
    ("310.30.42.50.01.001", "rolex-daytona-126506", False),
    ("Moonwatch 310.30.42.50.01.003", "omega-speedmaster-31030425001002", False),
])
def test_moonwatch_caseback_comparison_preserves_reference_scope(
    reference: str, proposed: str, compares: bool,
) -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Omega", "family": "Speedmaster", "reference": reference,
                     "reference_id": proposed, "photo": 1}],
        observations=[{"photo": 1, "angle": "caseback", "location": "fundo", "detail": "Visor na tampa.",
                       "visibility": "visible", "field": "caseback_type", "value": "transparent"}],
        linkage="single_photo", hypothesis="probable_original",
    )
    report = FlashReport("fixture").build_report(answer, {1})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference"] == reference and candidate["reference_id"] is None
    assert report["conclusion"] == "inconclusive"
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    if compares:
        assert len(checks) == 2 and report["decision_reason"] == "factory_caseback_difference"
        assert "310.30.42.50.01.001" in cast(str, report["decision_reason_text"])
    else:
        assert not checks


@pytest.mark.parametrize(("code", "family"), [
    ("SPB383", "Prospex"), ("SPB383J1", "1968 Diver’s Modern Re-interpretation GMT"),
])
@pytest.mark.parametrize(("visibility", "observed", "rejected"), [
    ("visible", "300 m", True),
    ("visible", "200m", False),
    ("ambiguous", "300m", False),
    ("visible", "300ft", False),
    ("visible", "3 00m", False),
])
def test_seiko_dial_depth_marking_checks_identity_only(
    code: str, family: str, visibility: str, observed: str, rejected: bool,
) -> None:
    answer = _answer()
    answer.update(
        linkage="single_photo", hypothesis="probable_fake",
        candidates=[{"brand": "Seiko", "family": family, "reference": code,
                     "reference_id": "seiko-prospex-spb383j1", "photo": 1}],
        observations=[{"photo": 1, "angle": "dial", "location": "text near 6",
                       "detail": "Watch certified for diving at 300 metres.",
                       "visibility": visibility, "field": "dial_depth_marking", "value": observed}],
    )
    report = FlashReport("fixture").build_report(answer, {1})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert (candidate["reference"] is None) is rejected
    assert report["conclusion"] == "inconclusive"
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    if visibility == "visible" and observed in {"200m", "300 m"}:
        check, = checks
        assert (check["field"], check["photo"]) == ("dial_depth_marking", 1)
        assert check["status"] == ("conflicting" if rejected else "matched")
        assert check["url"].startswith("https://www.seikowatches.com/")
        observation, = cast(list[dict[str, Any]], report["observations"])
        assert observation["detail"] == f"Inscrição no mostrador · {observed.replace(' ', '')}."
    else:
        assert not checks
    if rejected:
        assert code in str(report["decision_reason_text"])
        assert "cartão de garantia" in str(report["next_verification"])


def test_seiko_300m_marking_does_not_choose_between_two_compatible_variants() -> None:
    answer = _answer()
    answer.update(
        linkage="single_photo", hypothesis="probable_original",
        candidates=[{"brand": "Seiko", "family": "Prospex", "reference": code, "reference_id": None, "photo": 1}
                    for code in ("HBC002J1", "SPB534J1")],
        observations=[{"photo": 1, "angle": "dial", "location": "mostrador",
                       "detail": "DIVER'S 300m", "visibility": "visible",
                       "field": "dial_depth_marking", "value": "300m"}],
    )
    report = FlashReport("fixture").build_report(answer, {1})
    assert [item["reference"] for item in cast(list[dict[str, Any]], report["candidates"])] == ["HBC002J1", "SPB534J1"]
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    assert {(item["reference_id"], item["status"]) for item in checks} == {
        ("seiko-prospex-hbc002j1", "matched"), ("seiko-prospex-spb534j1", "matched"),
    }
    assert report["conclusion"] == "inconclusive"


def test_prompt_supplies_official_seiko_variant_comparators() -> None:
    content, _ = FlashReport("fixture").prepare([("dial", "image/png", base64.b64decode(_photo("red")))])
    prompt = str(content[0]["text"])
    assert "seiko-prospex-hbc002j1: 300m, borda silver-tone" in prompt
    assert "seiko-prospex-spb383j1: 200m, borda silver-tone" in prompt
    assert "seiko-prospex-spb534j1: 300m, borda gold-tone" in prompt


def test_disagreeing_depth_readings_do_not_reject_the_candidate() -> None:
    answer = _answer()
    answer.update(
        candidates=[{"brand": "Seiko", "family": "Prospex", "reference": "HBC002",
                     "reference_id": None, "photo": 1}],
        observations=[{"photo": photo, "angle": "dial", "location": "mostrador", "detail": value,
                       "visibility": "visible", "field": "dial_depth_marking", "value": value}
                      for photo, value in ((1, "300m"), (3, "200m"))],
    )
    report = FlashReport("fixture").build_report(answer, {1, 3})
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert candidate["reference"] == "HBC002" and candidate["reference_id"] is None
    assert report["conclusion"] == "inconclusive"
    assert report["decision_reason"] == "dial_marking_readings_disagree"
    assert report["technical_checks"] == []


@pytest.mark.parametrize(("angle", "tone", "monochrome", "resolved"), [
    ("dial", "silver-tone", frozenset(), True),
    ("dial", "gold-tone", frozenset(), False),
    ("dial", "silver-tone", frozenset({1}), False),
    ("dial", "unverified solid gold", frozenset(), False),
    ("caseback", "silver-tone", frozenset(), False),
    (None, None, frozenset(), False),
])
def test_same_marking_needs_visible_bezel_tone_to_resolve_diver_variant(
    angle: str | None, tone: str | None, monochrome: frozenset[int], resolved: bool,
) -> None:
    answer = _answer()
    answer.update(
        linkage="single_photo",
        candidates=[{"brand": "Seiko", "family": "1968 Heritage Diver's GMT Watch",
                     "reference": "HBC002J1", "reference_id": None, "photo": 1}],
        observations=[{"photo": 1, "angle": "dial", "location": "mostrador",
                       "detail": "DIVER'S 300m", "visibility": "visible",
                       "field": "dial_depth_marking", "value": "300m"}],
    )
    if angle:
        answer["observations"].append(
            {"photo": 1, "angle": angle, "location": "bisel",
             "detail": "Solid gold proves genuine.", "visibility": "visible", "field": "bezel_edge_color", "value": tone},
        )
    report = FlashReport("fixture").build_report(answer, {1}, monochrome=monochrome)
    candidate, = cast(list[dict[str, Any]], report["candidates"])
    assert report["conclusion"] == "inconclusive"
    assert (candidate["reference_id"] == "seiko-prospex-hbc002j1") is resolved
    assert "proves genuine" not in str(report["observations"])
    if resolved:
        checks = cast(list[dict[str, Any]], report["technical_checks"])
        assert {(check["field"], check["status"]) for check in checks} == {
            ("dial_depth_marking", "matched"), ("bezel_edge_color", "matched"),
        }
    else:
        assert candidate["reference_id"] is None
        assert report["decision_reason"] == "dial_marking_variant_unresolved"
        assert all(code in str(report["decision_reason_text"]) for code in ("HBC002J1", "SPB534J1"))
        assert "em cores" in str(report["next_verification"])


def test_depth_readings_do_not_change_unrelated_structural_discrepancies() -> None:
    answer = _answer()
    answer.update(
        hypothesis="probable_fake",
        candidates=[{"brand": "Cartier", "family": "Santos", "reference": "WSSA0061",
                     "reference_id": "cartier-santos-wssa0061", "photo": 1}],
        observations=[
            {"photo": 1, "angle": "dial", "location": "ponteiros", "detail": "Ponteiros em bastão.",
             "visibility": "visible", "field": "hand_shape", "value": "baton"},
            {"photo": 3, "angle": "crown", "location": "coroa", "detail": "Coroa arredondada.",
             "visibility": "visible", "field": "crown_shape", "value": "round"},
            *({"photo": photo, "angle": "dial", "location": "mostrador", "detail": value,
               "visibility": "visible", "field": "dial_depth_marking", "value": value}
              for photo, value in ((1, "300m"), (3, "200m"))),
        ],
    )
    report = FlashReport("fixture").build_report(answer, {1, 3})
    assert report["conclusion"] == "probable_fake"
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    assert {check["field"] for check in checks} == {"crown_shape", "hand_shape"}


def test_identification_checks_are_excluded_from_authenticity_review() -> None:
    answer = _answer()
    answer.update(
        linkage="single_photo", hypothesis="probable_original",
        candidates=[{"brand": "Seiko", "family": "Prospex", "reference": "HBC002J1", "reference_id": None, "photo": 1}],
        observations=[{"photo": 1, "angle": "dial", "location": "mostrador", "detail": "DIVER'S 300m",
                       "visibility": "visible", "field": "dial_depth_marking", "value": "300m"}],
    )

    def review(body: object) -> Any:
        pytest.fail("Variant evidence must not request an authenticity review.")

    result = replay_with_review(
        FlashReport("fixture"), "seiko-identification",
        [("dial", "image/png", base64.b64decode(_photo("red")))],
        gemini=lambda _: (answer, {"model": MODEL, "model_response_id": "fixture-review-scope"}),
        review=review,
    )
    assert result["previous_policy"] == "inconclusive"
    assert result["review_status"] == "no_comparable_fields"
