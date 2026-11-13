"""Local upload page for the provisional online pilot."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

from pilot.prototype import extract_embeddings, load_artifact, predict_embedding
from quality_gate.blur_light_check import Pass, quality_gate
from reference_intelligence.knowledge import load_knowledge
from reference_intelligence.types import TechnicalClaim

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ANGLES = frozenset({"dial", "cyclops", "rehaut", "crown", "clasp", "caseback"})
_HTML = """<!doctype html><html lang=\"pt-BR\"><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width\"><title>Piloto Submariner 126610LN</title><style>body{font:16px system-ui;max-width:680px;margin:40px auto;padding:0 18px;background:#101214;color:#f5f5f5}main{background:#1a1e21;padding:24px;border-radius:16px}button,input,select{font:inherit;margin:8px 0;padding:10px}button{cursor:pointer}#r{white-space:pre-wrap;padding:16px;background:#0b0d0e;border-radius:10px}.warn{color:#ffcc66}</style><main><h1>Piloto Rolex 126610LN</h1><p class=\"warn\">Experimental: compara a foto com exemplos online. Nao autentica o relogio.</p><label>Angulo<br><select id=\"a\"><option value=\"dial\">Mostrador</option><option value=\"cyclops\">Data/lupa</option><option value=\"rehaut\">Aro interno</option><option value=\"crown\">Coroa/lateral</option><option value=\"clasp\">Fecho</option><option value=\"caseback\">Traseira</option></select></label><br><input id=\"f\" type=\"file\" accept=\"image/jpeg,image/png,image/webp\"><button id=\"b\">Analisar</button><div id=\"r\">Escolha uma foto nítida.</div></main><script>b.onclick=async()=>{if(!f.files[0])return;r.textContent='Analisando...';let x=await fetch('/predict?angle='+a.value,{method:'POST',headers:{'Content-Type':f.files[0].type},body:f.files[0]});let j=await x.json();r.textContent=x.ok?j.message+'\nProbabilidade provisoria de replica: '+Math.round(j.replica_probability*100)+'%\nProxima foto: '+j.next_photo+'\nPistas online: '+j.online_checks.map(v=>v.field+' - '+v.url).join(' | '):j.error}</script></html>""".encode("utf-8")


class PilotService:
    def __init__(self, artifact_path: Path, knowledge_path: Path) -> None:
        self.artifact = load_artifact(artifact_path)
        self.claims = load_knowledge(knowledge_path, "rolex-submariner-126610ln")

    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, object]:
        image = decode_image(body, content_type)
        if angle not in ANGLES:
            raise ValueError("angulo invalido")
        quality = quality_gate(image, angle=angle)
        if not isinstance(quality, Pass):
            return {
                "verdict": "foto_invalida",
                "replica_probability": 0.5,
                "message": "Foto sem nitidez suficiente. Tire outra mais perto e sem movimento.",
                "next_photo": angle,
                "online_checks": checks_for_angle(self.claims, angle),
            }
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temporary:
            image.save(temporary, format="JPEG", quality=95)
            temporary.flush()
            embedding = extract_embeddings((Path(temporary.name),), batch_size=1)[0]
        result = predict_embedding(self.artifact, embedding)
        messages = {
            "provavel_replica": "A imagem se parece mais com as replicas conhecidas do piloto.",
            "sem_inconsistencia_visivel": "Nenhuma inconsistencia visivel foi encontrada nesta foto.",
            "inconclusivo": "A evidencia desta foto nao e suficiente.",
        }
        if not result.in_distribution:
            messages["inconclusivo"] = "A foto esta fora dos exemplos conhecidos do piloto."
        next_photo = "rehaut" if angle != "rehaut" else "clasp"
        return {
            "verdict": result.verdict,
            "replica_probability": round(result.replica_probability, 6),
            "nearest_similarity": round(result.nearest_similarity, 6),
            "message": messages[result.verdict],
            "next_photo": next_photo,
            "notice": result.notice,
            "online_checks": checks_for_angle(self.claims, angle),
        }


def checks_for_angle(
    claims: tuple[TechnicalClaim, ...], angle: str
) -> list[dict[str, str]]:
    return [
        {
            "field": claim.field,
            "expected": claim.value,
            "role": "official_spec" if claim.source_class.authoritative else "community_lead",
            "url": claim.source_url,
        }
        for claim in claims
        if any(value.value == angle for value in claim.applicable_angles)
    ]


def decode_image(body: bytes, content_type: str) -> Image.Image:
    if not body or len(body) > MAX_UPLOAD_BYTES:
        raise ValueError("imagem vazia ou maior que 15 MB")
    if content_type.split(";", 1)[0].strip().lower() not in {
        "image/jpeg",
        "image/png",
        "image/webp",
    }:
        raise ValueError("formato aceito: JPEG, PNG ou WebP")
    try:
        with Image.open(io.BytesIO(body)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(body)) as opened:
            if opened.width < 224 or opened.height < 224 or opened.width * opened.height > 25_000_000:
                raise ValueError("imagem pequena demais ou com dimensoes excessivas")
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("arquivo de imagem invalido") from exc


def handler(service: PilotService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if urlsplit(self.path).path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlsplit(self.path)
            if parsed.path != "/predict":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_UPLOAD_BYTES:
                    raise ValueError("imagem vazia ou maior que 15 MB")
                body = self.rfile.read(length)
                angle = parse_qs(parsed.query).get("angle", [""])[0]
                payload = service.predict(body, self.headers.get("Content-Type", ""), angle)
                self._json(HTTPStatus.OK, payload)
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=Path(".runtime/pilot/model.json"))
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("config/references/rolex-submariner-126610ln.json"),
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("port must be between 1024 and 65535")
    service = PilotService(args.artifact, args.knowledge)
    server = HTTPServer(("127.0.0.1", args.port), handler(service))
    print(f"Piloto local: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
