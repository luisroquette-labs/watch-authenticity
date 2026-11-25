"""Bounded, local Gemini Flash research pass; output is never an authenticity verdict."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

MODEL = "google/gemini-3.8-flash"
KEYCHAIN_SERVICE = "codex.openrouter.watch-authenticity"
MAX_PROJECT_USAGE_USD = 5.0
MAX_COMPLETION_TOKENS = 1800
MAX_MODEL_CONTEXT_TOKENS = 1_048_576
MAX_INPUT_PRICE_USD = 0.000000825
MAX_OUTPUT_PRICE_USD = 0.000004125
PROMPT = """Examine esta foto de relógio como pesquisa, não como laudo. Não há referência nem procedência confirmada. Responda SOMENTE com um objeto JSON com estas chaves exatas: candidates (lista de até 3 objetos com brand, family, reference, cada valor texto ou null); observations (lista de até 8 objetos com location, detail, visibility; visibility deve ser visible ou ambiguous); authenticity_hypothesis (probable_fake, probable_original ou inconclusive); alternatives (lista de até 3 textos); next_verification (um texto). Descreva somente detalhes efetivamente discerníveis nesta foto. Não infira calibre, metal, autenticidade de componente, fábrica ou gravação invisível. Uma cor, plástico, fundo ou etiqueta não comprova falsificação. Se a foto não distinguir original de réplica, use inconclusive. Não invente fonte ou URL. O parecer final será feito separadamente."""


class ReportUnavailable(RuntimeError):
    pass


def _image_part(path: Path) -> dict[str, object]:
    from PIL import Image, ImageOps

    from pilot.server import MAX_UPLOAD_BYTES, decode_image

    data = path.read_bytes()
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("imagem maior que 15 MB")
    suffix_mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = suffix_mime.get(path.suffix.lower())
    if mime is None:
        raise ValueError("formato aceito: JPEG, PNG ou WebP")
    image = decode_image(data, mime)
    try:
        image = ImageOps.exif_transpose(image)
        image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88)
    finally:
        image.close()
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()}}


def _validate(answer: object) -> dict[str, object]:
    if not isinstance(answer, dict) or set(answer) != {"candidates", "observations", "authenticity_hypothesis", "alternatives", "next_verification"}:
        raise ValueError("resposta do modelo fora do contrato")
    candidates, observations = answer["candidates"], answer["observations"]
    alternatives, next_verification = answer["alternatives"], answer["next_verification"]
    if not isinstance(candidates, list) or len(candidates) > 3 or not all(
        isinstance(item, dict) and set(item) == {"brand", "family", "reference"}
        and all(value is None or isinstance(value, str) and len(value) <= 120 for value in item.values())
        for item in candidates
    ):
        raise ValueError("candidatos inválidos")
    if not isinstance(observations, list) or len(observations) > 8 or not all(
        isinstance(item, dict) and set(item) == {"location", "detail", "visibility"}
        and isinstance(item["visibility"], str)
        and item["visibility"] in {"visible", "ambiguous"}
        and all(isinstance(item[key], str) and 0 < len(item[key]) <= 300 for key in ("location", "detail"))
        for item in observations
    ):
        raise ValueError("observações inválidas")
    if not isinstance(answer["authenticity_hypothesis"], str) or answer["authenticity_hypothesis"] not in {"probable_fake", "probable_original", "inconclusive"}:
        raise ValueError("hipótese inválida")
    if not isinstance(alternatives, list) or len(alternatives) > 3 or not all(
        isinstance(item, str) and 0 < len(item) <= 300 for item in alternatives
    ) or not isinstance(next_verification, str) or not 0 < len(next_verification) <= 300:
        raise ValueError("alternativas ou próxima verificação inválidas")
    return answer


def research(path: Path, key: str, *, request: Any = requests) -> dict[str, object]:
    image = _image_part(path)
    answer, metadata = complete_flash(key, [{"type": "text", "text": PROMPT}, image], request=request)
    return {**metadata, "research": _validate(answer), "visual_claims_status": "unverified_model_observations", "authenticity_verdict": "inconclusivo", "reference_coverage": "pending", "technical_sources": [], "quality_score_0_1000": None}


def check_flash_budget(
    key: str, *, request: Any = requests,
    max_usage_usd: float = MAX_PROJECT_USAGE_USD,
) -> tuple[dict[str, str], float]:
    headers = {"Authorization": "Bearer " + key, "HTTP-Referer": "https://watch-authenticity.local/pilot", "X-Title": "Watch Authenticity Research Pilot"}
    auth = request.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=10)
    auth.raise_for_status()
    account = auth.json().get("data", {})
    remaining, spent = account.get("limit_remaining"), account.get("usage")
    if not isinstance(remaining, (int, float)) or not isinstance(spent, (int, float)) or not math.isfinite(remaining) or not math.isfinite(spent) or spent < 0:
        raise RuntimeError("saldo ou limite da chave do projeto insuficiente")
    catalog = request.get("https://openrouter.ai/api/v1/models", timeout=10)
    catalog.raise_for_status()
    model = next((item for item in catalog.json().get("data", []) if item.get("id") == MODEL), None)
    if model is None:
        raise RuntimeError("modelo não encontrado no catálogo OpenRouter")
    prices = model.get("pricing") or {}
    try:
        price_values = {field: float(prices[field]) for field in ("prompt", "image", "completion", "internal_reasoning")}
        if not all(math.isfinite(value) and value >= 0 for value in price_values.values()):
            raise ValueError("nonfinite price")
        input_price = max(price_values["prompt"], price_values["image"])
        output_price = max(price_values["completion"], price_values["internal_reasoning"])
        context = int(model["context_length"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("preço ou limite de contexto indisponível") from exc
    if context <= 0 or context > MAX_MODEL_CONTEXT_TOKENS or not math.isfinite(input_price) or not math.isfinite(output_price) or input_price < 0 or input_price > MAX_INPUT_PRICE_USD or output_price < 0 or output_price > MAX_OUTPUT_PRICE_USD:
        raise RuntimeError("preço ou limite do modelo excede o envelope aprovado")
    worst_case_cost = context * input_price + MAX_COMPLETION_TOKENS * output_price
    if not math.isfinite(max_usage_usd) or remaining < worst_case_cost or spent + worst_case_cost > max_usage_usd:
        raise RuntimeError("envelope local insuficiente para a chamada")
    return headers, worst_case_cost


def complete_flash(
    key: str, content: list[dict[str, object]], *, request: Any = requests,
    max_usage_usd: float = MAX_PROJECT_USAGE_USD,
) -> tuple[object, dict[str, object]]:
    headers, worst_case_cost = check_flash_budget(key, request=request, max_usage_usd=max_usage_usd)
    start = time.monotonic()
    response = request.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=headers,
        json={"model": MODEL, "messages": [{"role": "user", "content": content}],
              "max_tokens": MAX_COMPLETION_TOKENS, "temperature": 0,
              "reasoning": {"effort": "low"}, "provider": {"allow_fallbacks": False}},
        timeout=75,
    )
    response.raise_for_status()
    payload = response.json()
    usage = payload.get("usage") or {}
    cost = usage.get("cost")
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0:
        raise RuntimeError("custo real ausente na resposta da API")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("resposta vazia do modelo")
    message = choices[0].get("message")
    output = message.get("content") if isinstance(message, dict) else None
    if not isinstance(output, str) or choices[0].get("finish_reason") == "length":
        raise ValueError("resposta vazia do modelo")
    try:
        answer = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
    except json.JSONDecodeError as exc:
        raise ValueError("resposta do modelo não é JSON") from exc
    if payload.get("model") != MODEL or cost > worst_case_cost:
        raise RuntimeError("modelo retornado ou custo fora do envelope; resposta descartada")
    return answer, {"model": MODEL, "model_response_id": payload.get("id"), "latency_seconds": round(time.monotonic() - start, 2), "cost_usd": cost}


def _check_code_engine_secret() -> None:
    try:
        secret = json.load(sys.stdin)
        data = secret.get("data", {})
        key = data.get("WATCH_AUTH_OPENROUTER_PILOT_KEY")
        valid = (
            secret.get("format") == "generic" and set(data) == {"WATCH_AUTH_OPENROUTER_PILOT_KEY"}
            and isinstance(key, str) and key and key.isascii() and key.isprintable()
            and not any(char.isspace() for char in key)
        )
    except (ValueError, AttributeError, TypeError):
        valid = False
    if not valid:
        raise SystemExit("Gemini pilot requires the dedicated Code Engine provider secret")
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.max_redirects = 0
            check_flash_budget(key, request=session)
    except (requests.RequestException, RuntimeError, ValueError, TypeError, AttributeError, KeyError):
        raise SystemExit("Gemini pilot provider or spending envelope unavailable") from None


def main() -> None:
    if sys.argv[1:] == ["--check-code-engine-secret"]:
        _check_code_engine_secret()
        return
    parser = argparse.ArgumentParser(description="Pesquisa visual local, sem laudo de autenticidade")
    parser.add_argument("photo", type=Path)
    args = parser.parse_args()
    key = subprocess.run(["security", "find-generic-password", "-w", "-s", KEYCHAIN_SERVICE], capture_output=True, text=True, check=False).stdout.strip()
    if not key:
        raise RuntimeError("chave OpenRouter do projeto ausente no Keychain")
    print(json.dumps(research(args.photo, key), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
