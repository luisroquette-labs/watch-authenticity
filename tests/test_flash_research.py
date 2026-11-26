from __future__ import annotations

import io
import json
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from pilot.flash_research import _validate, research


def _photo(path: Path) -> None:
    output = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(output, format="JPEG")
    path.write_bytes(output.getvalue())


def _answer() -> dict[str, object]:
    return {
        "candidates": [{"brand": "Cartier", "family": "Santos", "reference": None}],
        "observations": [{"location": "mostrador", "detail": "números romanos", "visibility": "visible"}],
        "authenticity_hypothesis": "probable_original",
        "alternatives": ["peça modificada"],
        "next_verification": "Foto nítida do fundo.",
    }


def test_research_preserves_model_hypothesis_without_promoting_to_verdict(tmp_path: Path) -> None:
    photo = tmp_path / "watch.jpg"
    _photo(photo)

    class Response:
        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, object]:
            return self.payload

    class Client:
        def get(self, url: str, **_kwargs: object) -> Response:
            if url.endswith("/auth/key"):
                return Response({"data": {"limit_remaining": 247.0, "usage": 2.0}})
            return Response({"data": [{"id": "google/gemini-3.8-flash", "context_length": 1_048_576, "pricing": {"prompt": "0.000000825", "image": "0.000000825", "completion": "0.000004125", "internal_reasoning": "0.000004125"}}]})

        def post(self, url: str, **kwargs: object) -> Response:
            assert url == "https://openrouter.ai/api/v1/chat/completions"
            request_body = cast(dict[str, object], kwargs["json"])
            assert request_body["model"] == "google/gemini-3.8-flash"
            assert request_body["provider"] == {
                "only": ["google-vertex/global"],
                "order": ["google-vertex/global"],
                "allow_fallbacks": False,
                "data_collection": "deny",
                "zdr": True,
            }
            assert kwargs["allow_redirects"] is False
            return Response({"model": "google/gemini-3.8-flash", "id": "test", "usage": {"cost": 0.01}, "choices": [{"message": {"content": json.dumps(_answer())}}]})

    result = research(photo, "test-key", request=Client())
    assert cast(dict[str, object], result["research"])["authenticity_hypothesis"] == "probable_original"
    assert result["authenticity_verdict"] == "inconclusivo"
    assert result["visual_claims_status"] == "unverified_model_observations"
    assert result["quality_score_0_1000"] is None


def test_invalid_visibility_and_unavailable_budget_fail_closed(tmp_path: Path) -> None:
    answer = _answer()
    answer["observations"] = [{"location": "movimento", "detail": "calibre original", "visibility": []}]
    with pytest.raises(ValueError, match="observações inválidas"):
        _validate(answer)

    photo = tmp_path / "watch.jpg"
    _photo(photo)

    class Client:
        def get(self, url: str, **_kwargs: object) -> object:
            class Response:
                def raise_for_status(self) -> None:
                    pass

                def json(self) -> dict[str, object]:
                    if url.endswith("/auth/key"):
                        return {"data": {"limit_remaining": 247.0, "usage": 4.9}}
                    return {"data": [{"id": "google/gemini-3.8-flash", "context_length": 1_048_576, "pricing": {"prompt": "0.000000825", "image": "0.000000825", "completion": "0.000004125", "internal_reasoning": "0.000004125"}}]}
            return Response()

        def post(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("paid call must not happen")

    with pytest.raises(RuntimeError, match="envelope"):
        research(photo, "test-key", request=Client())


@pytest.mark.parametrize("bad_field", ["prompt", "image", "completion", "internal_reasoning"])
def test_nonfinite_catalog_price_blocks_paid_call(tmp_path: Path, bad_field: str) -> None:
    photo = tmp_path / "watch.jpg"
    _photo(photo)

    class Client:
        def get(self, url: str, **_kwargs: object) -> object:
            class Response:
                def raise_for_status(self) -> None:
                    pass

                def json(self) -> dict[str, object]:
                    if url.endswith("/auth/key"):
                        return {"data": {"limit_remaining": 247.0, "usage": 2.0}}
                    prices = {"prompt": "0.000000825", "image": "0.000000825", "completion": "0.000004125", "internal_reasoning": "0.000004125"}
                    prices[bad_field] = "NaN"
                    return {"data": [{"id": "google/gemini-3.8-flash", "context_length": 1_048_576, "pricing": prices}]}
            return Response()

        def post(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("paid call must not happen")

    with pytest.raises(RuntimeError, match="preço"):
        research(photo, "test-key", request=Client())
