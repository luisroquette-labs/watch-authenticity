"""Local upload page for the evidence-constrained online pilot."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

from reference_intelligence.knowledge import load_knowledge
from reference_intelligence.types import TechnicalClaim

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_BATCH_IMAGES = 6
MAX_BATCH_DECODED_BYTES = MAX_BATCH_IMAGES * MAX_UPLOAD_BYTES
MAX_BATCH_ENCODED_BYTES = ((MAX_BATCH_DECODED_BYTES + 2) // 3) * 4
MAX_BATCH_REQUEST_BYTES = MAX_BATCH_ENCODED_BYTES + 16_384
ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ANGLES = frozenset({"dial", "cyclops", "rehaut", "crown", "clasp", "caseback"})
_HTML = """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Piloto Submariner 126610LN</title><style>body{font:16px system-ui;max-width:680px;margin:40px auto;padding:0 18px;background:#101214;color:#f5f5f5}main{background:#1a1e21;padding:24px;border-radius:16px}button,input,select{font:inherit;margin:8px 0;padding:10px}button{cursor:pointer}.item{border:1px solid #444;padding:8px;margin:8px 0}#r{white-space:pre-wrap;padding:16px;background:#0b0d0e;border-radius:10px}.warn{color:#ffcc66}</style><main><h1>Piloto Rolex 126610LN</h1><p class="warn">Experimental: compara apenas detalhes visíveis com fontes online citadas. Não autentica o relógio.</p><label>Fotos (1 a 6, até 15 MB cada)<br><input id="f" type="file" multiple accept="image/jpeg,image/png,image/webp"></label><div id="angles"></div><button id="b">Analisar</button><div id="r">Escolha fotos nítidas.</div></main><script>
const labels={dial:'Mostrador',cyclops:'Data/lupa',rehaut:'Aro interno',crown:'Coroa/lateral',clasp:'Fecho',caseback:'Traseira'};
f.onchange=()=>{angles.replaceChildren();if(f.files.length>6){r.textContent='Máximo de 6 imagens.';return}[...f.files].forEach((file,i)=>{let row=document.createElement('div');row.className='item';let text=document.createElement('span');text.textContent=file.name+' — Selecione o ângulo: ';let select=document.createElement('select');select.dataset.index=i;Object.entries(labels).forEach(([value,label])=>select.add(new Option(label,value)));row.append(text,select);angles.append(row)})};
const encode=file=>new Promise((resolve,reject)=>{let reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('Não foi possível ler a imagem.'));reader.readAsDataURL(file)});
b.onclick=async()=>{if(!f.files.length){r.textContent='Selecione de 1 a 6 imagens.';return}if(f.files.length>6){r.textContent='Máximo de 6 imagens.';return}b.disabled=true;r.textContent='Analisando...';try{let selects=[...angles.querySelectorAll('select')];let images=[];for(let i=0;i<f.files.length;i++){images.push({angle:selects[i].value,content_type:f.files[i].type,data:await encode(f.files[i])})}let x=await fetch("/predict/batch",{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images})});let j=await x.json();if(!x.ok)throw new Error(j.error||'Falha ao analisar as imagens.');r.textContent=j.joint.message+'\nÂngulos ausentes: '+(j.joint.missing_angles.join(', ')||'nenhum')+'\n\nResultados e fontes:\n'+JSON.stringify(j.results,null,2)}catch(error){r.textContent='Erro: '+error.message}finally{b.disabled=false}};
</script></html>""".encode("utf-8")


class PredictService(Protocol):
    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, object]: ...


class PilotService:
    def __init__(self, checkpoint: Path, knowledge_path: Path) -> None:
        from pilot.online_assessor import LocalVisionAssessor

        self.claims = load_knowledge(knowledge_path, "rolex-submariner-126610ln")
        self.assessor = LocalVisionAssessor(checkpoint)

    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, object]:
        from quality_gate.blur_light_check import Pass, quality_gate

        image = decode_image(body, content_type)
        if angle not in ANGLES:
            raise ValueError("angulo invalido")
        checks = checks_for_angle(self.claims, angle)
        quality = quality_gate(image, angle=angle)
        if not isinstance(quality, Pass):
            return {
                "verdict": "foto_invalida",
                "replica_probability": None,
                "message": "Foto sem nitidez suficiente. Tire outra mais perto e sem movimento.",
                "next_photo": angle,
                "online_checks": checks,
            }
        assessment = self.assessor.assess(image, angle, self.claims)
        messages = {
            "inconsistencia_com_referencia": "Ha conflito visivel com a referencia 126610LN.",
            "sem_inconsistencia_visivel": "Nenhuma inconsistencia oficial ficou visivel nesta foto.",
            "inconclusivo": "A evidencia desta foto nao e suficiente.",
        }
        return {
            "verdict": assessment.verdict,
            "replica_probability": None,
            "message": messages[assessment.verdict],
            "next_photo": assessment.missing_detail,
            "official_conflicts": [
                {"field": field, "observed": observed}
                for field, observed in assessment.official_conflicts
            ],
            "online_checks": checks,
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
    mime_type = content_type.split(";", 1)[0].strip().lower()
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError("formato aceito: JPEG, PNG ou WebP")
    try:
        with Image.open(io.BytesIO(body)) as opened:
            if opened.format != {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}[mime_type]:
                raise ValueError("conteudo da imagem nao corresponde ao formato informado")
            opened.verify()
        with Image.open(io.BytesIO(body)) as opened:
            if opened.width < 224 or opened.height < 224 or opened.width * opened.height > 25_000_000:
                raise ValueError("imagem pequena demais ou com dimensoes excessivas")
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("arquivo de imagem invalido") from exc


def analyze_batch(
    service: PredictService, payload: object
) -> dict[str, Any]:
    if not isinstance(payload, dict) or not isinstance(payload.get("images"), list):
        raise ValueError("images deve ser uma lista")
    images = payload["images"]
    if not 1 <= len(images) <= MAX_BATCH_IMAGES:
        raise ValueError("envie de 1 a 6 imagens")
    if any(not isinstance(item, dict) for item in images):
        raise ValueError("cada imagem deve ser um objeto")
    encoded_total = sum(
        len(item.get("data", ""))
        for item in images
        if isinstance(item.get("data"), str)
    )
    if encoded_total > MAX_BATCH_ENCODED_BYTES:
        raise ValueError("lote codificado excede o limite")

    validated: list[tuple[str, str, str]] = []
    decoded_total = 0
    for item in images:
        angle = item.get("angle")
        content_type = item.get("content_type")
        data = item.get("data")
        if not isinstance(angle, str) or angle not in ANGLES:
            raise ValueError("angulo invalido")
        if not isinstance(content_type, str) or content_type.lower() not in ALLOWED_MIME_TYPES:
            raise ValueError("formato aceito: JPEG, PNG ou WebP")
        if not isinstance(data, str):
            raise ValueError("imagem base64 invalida")
        try:
            body = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("imagem base64 invalida") from exc
        if len(body) > MAX_UPLOAD_BYTES:
            raise ValueError("cada imagem deve ter no maximo 15 MB")
        decoded_total += len(body)
        if decoded_total > MAX_BATCH_DECODED_BYTES:
            raise ValueError("lote decodificado excede o limite")
        validated.append((angle, content_type, data))

    results: list[dict[str, Any]] = []
    submitted_angles: set[str] = set()
    for angle, content_type, data in validated:
        body = base64.b64decode(data, validate=True)
        result = service.predict(body, content_type, angle)
        results.append({"angle": angle, **result})
        submitted_angles.add(angle)

    verdicts = {str(result.get("verdict")) for result in results}
    if "inconsistencia_com_referencia" in verdicts:
        verdict = "inconsistencia_com_referencia"
        message = "Há conflito oficial visível em pelo menos uma foto."
    elif verdicts == {"foto_invalida"}:
        verdict = "foto_invalida"
        message = "Nenhuma foto pôde ser utilizada. Envie fotos mais nítidas."
    else:
        verdict = "inconclusivo"
        message = "O conjunto não autentica o relógio; a análise permanece inconclusiva."
    return {
        "results": results,
        "joint": {
            "verdict": verdict,
            "message": message,
            "missing_angles": sorted(ANGLES - submitted_angles),
        },
    }


def handler(service: PredictService) -> type[BaseHTTPRequestHandler]:
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
            if parsed.path not in {"/predict", "/predict/batch"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit = MAX_BATCH_REQUEST_BYTES if parsed.path == "/predict/batch" else MAX_UPLOAD_BYTES
                if length <= 0 or length > limit:
                    raise ValueError("imagem vazia ou maior que 15 MB")
                body = self.rfile.read(length)
                if parsed.path == "/predict/batch":
                    if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                        raise ValueError("o lote deve usar application/json")
                    payload = analyze_batch(service, json.loads(body))
                else:
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--knowledge", type=Path,
        default=Path("config/references/rolex-submariner-126610ln.json"),
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("port must be between 1024 and 65535")
    service = PilotService(args.checkpoint, args.knowledge)
    server = HTTPServer(("127.0.0.1", args.port), handler(service))
    print(f"Piloto local: http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
