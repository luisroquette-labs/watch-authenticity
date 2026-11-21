"""Local upload page for the evidence-constrained online pilot."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import io
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError

from pilot.visual_evidence import (
    INSPECTION_SOURCE,
    bezel_color_hint,
    consensus,
)
from reference_intelligence.knowledge import load_knowledge
from reference_intelligence.types import TechnicalClaim

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_BATCH_IMAGES = 6
MAX_BATCH_DECODED_BYTES = MAX_BATCH_IMAGES * MAX_UPLOAD_BYTES
MAX_BATCH_ENCODED_BYTES = ((MAX_BATCH_DECODED_BYTES + 2) // 3) * 4
MAX_BATCH_REQUEST_BYTES = MAX_BATCH_ENCODED_BYTES + 16_384
MAX_HEADER_SECONDS = 10
MAX_REQUEST_SECONDS = 120
ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ANGLES = frozenset({"dial", "cyclops", "rehaut", "crown", "clasp", "caseback"})
_HTML = r"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Prévia visual de relógios</title><style>body{font:16px system-ui;max-width:680px;margin:40px auto;padding:0 18px;background:#101214;color:#f5f5f5}main{background:#1a1e21;padding:24px;border-radius:16px}button,input,select{font:inherit;margin:8px 0;padding:10px}button{cursor:pointer}input[type=file]{width:100%;box-sizing:border-box;margin-left:0;margin-right:0}.item{border:1px solid #444;padding:8px;margin:8px 0}#r{white-space:pre-wrap;padding:16px;background:#0b0d0e;border-radius:10px}.item,#r{overflow-wrap:anywhere}.warn{color:#ffcc66}</style><main><h1>Prévia visual de relógios</h1><p class="warn">Ainda não detecta falsificações: não há classificador de autenticidade treinado e validado neste serviço. Esta prévia descreve cores do mostrador e aro somente quando a vista informada é mostrador; outras vistas são recebidas, mas não examinadas. Não identifica marca ou referência nem confirma autenticidade.</p><label>Fotos (1 a 6, até 15 MB cada)<br><input id="f" type="file" multiple accept="image/jpeg,image/png,image/webp"></label><div id="angles"></div><button id="b">Analisar</button><div id="r">Escolha fotos nítidas.</div></main><script>
const labels={dial:'Mostrador',cyclops:'Data/lupa',rehaut:'Aro interno',crown:'Coroa/lateral',clasp:'Fecho',caseback:'Traseira'};
f.onchange=()=>{angles.replaceChildren();if(f.files.length>6){r.textContent='Máximo de 6 imagens.';return}[...f.files].forEach((file,i)=>{let row=document.createElement('div');row.className='item';let text=document.createElement('span');text.textContent=file.name+' — Vista informada (esta prévia verifica mostrador e aro): ';let select=document.createElement('select');select.dataset.index=i;Object.entries(labels).forEach(([value,label])=>select.add(new Option(label,value)));row.append(text,select);angles.append(row)})};
const encode=file=>new Promise((resolve,reject)=>{let reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);reader.onerror=()=>reject(new Error('Não foi possível ler a imagem.'));reader.readAsDataURL(file)});
b.onclick=async()=>{if(!f.files.length){r.textContent='Selecione de 1 a 6 imagens.';return}if(f.files.length>6){r.textContent='Máximo de 6 imagens.';return}b.disabled=true;f.disabled=true;let files=[...f.files];let selectedAngles=[...angles.querySelectorAll('select')].map(x=>x.value);let start=Date.now();let timer=setInterval(()=>r.textContent='Analisando '+files.length+' foto(s)... '+Math.floor((Date.now()-start)/1000)+' s. O tempo aumenta com o número de fotos.',1000);r.textContent='Analisando '+files.length+' foto(s)...';try{let images=[];for(let i=0;i<files.length;i++){images.push({angle:selectedAngles[i],content_type:files[i].type,data:await encode(files[i])})}let x=await fetch("/predict/batch",{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images})});let j=await x.json();if(!x.ok)throw new Error(j.error||'Falha ao analisar as imagens.');let lines=j.results.map((item,i)=>(i+1)+'. '+files[i].name+': '+item.message);let ev=j.joint.visible_evidence;let colors={green:'verde',black:'preto',blue:'azul',uncertain:'não verificada'};let count=n=>n+' '+(n===1?'foto':'fotos');let findings='Mostrador: '+(colors[ev.dial_color.value]||ev.dial_color.value)+' ('+count(ev.dial_color.supporting_photos)+'). Aro: '+(colors[ev.bezel_color.value]||ev.bezel_color.value)+' ('+count(ev.bezel_color.supporting_photos)+').';r.textContent=j.joint.message+'\nFotos enviadas: '+j.joint.submitted_photos+'. Conteúdos visuais distintos: '+j.joint.unique_visual_contents+'. Análises executadas: '+j.joint.model_checked_images+'. Resultados reaproveitados: '+j.joint.reused_results+'.\nAs fotos são tratadas como um relógio; não são provas independentes.\n\nObservado: '+findings+'\n\nAinda não verificado: '+j.joint.unverified.join('; ')+'.\n\nPor foto:\n'+lines.join('\n')+'\n\nFonte da referência: '+(j.joint.reference_candidate?j.joint.reference_candidate.source_url:'não disponível')+'\nProcesso de inspeção: '+j.joint.inspection_source_url+'\nReferência comparada: nenhuma; marca não identificada.';let facts=j.joint.cited_reference_facts.map(x=>x.field+': '+x.url).join('; ');r.textContent+='\n\nFotos usadas: '+(j.joint.used_photo_indices.join(', ')||'nenhuma')+'\nFatos oficiais citados: '+(facts||'nenhum detalhe confirmado')+'\nPróxima verificação útil: '+j.joint.next_useful_photo+'\nModelo visual de cores: carregado; classificador de autenticidade: '+(j.joint.model_sha256||'não liberado')+'; confiança calibrada: '+(j.joint.calibrated_confidence??'indisponível')+'\nQualidade da falsificação (somente após confirmação): '+(j.joint.quality_score_0_1000??j.joint.quality_assessment)}catch(error){r.textContent='Erro: '+error.message}finally{clearInterval(timer);f.disabled=false;b.disabled=false}};
</script></html>""".encode("utf-8")


class PredictService(Protocol):
    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, object]: ...


@dataclass(frozen=True)
class _VisualInspection:
    status: str
    dial_color: str = "uncertain"
    bezel_color: str = "uncertain"


class PilotService:
    def __init__(self, checkpoint: Path, knowledge_path: Path | None = None) -> None:
        from pilot.online_assessor import LocalVisionAssessor
        from pilot.watch_locator import LocalWatchLocator

        self.claims = (
            load_knowledge(knowledge_path, "rolex-submariner-126610ln")
            if knowledge_path is not None else ()
        )
        self.assessor = LocalVisionAssessor(checkpoint)
        detector = os.environ.get("WATCH_AUTH_DETECTOR_CHECKPOINT")
        self.locator = LocalWatchLocator(detector) if detector else LocalWatchLocator()

    def predict(self, body: bytes, content_type: str, angle: str) -> dict[str, object]:
        if angle not in ANGLES:
            raise ValueError("angulo invalido")
        image = decode_image(body, content_type)
        if angle != "dial":
            image.close()
            return self.render(_VisualInspection("unsupported"), angle)
        try:
            return self.render(self.inspect(image), angle)
        finally:
            image.close()

    def inspect(self, image: Image.Image) -> _VisualInspection:
        region = self.locator.locate(image)
        if region is None or region.face is None:
            return _VisualInspection("no_face")
        from quality_gate.blur_light_check import Pass, quality_gate

        if not isinstance(quality_gate(region.face, angle="dial"), Pass):
            return _VisualInspection("blurry")
        return _VisualInspection(
            "inspected", self.assessor.dial_color(region.face), bezel_color_hint(region.face)
        )

    def render(self, visual: _VisualInspection, angle: str) -> dict[str, object]:
        if angle not in ANGLES:
            raise ValueError("angulo invalido")
        if angle != "dial":
            return {
                "verdict": "inconclusivo",
                "replica_probability": None,
                "message": "Vista recebida, mas esta prévia ainda não examina este ângulo.",
                "next_photo": "Envie uma foto do mostrador de frente e em foco.",
                "model_invoked": False,
                "online_checks": [],
            }
        checks = checks_for_angle(self.claims, angle)
        if visual.status == "no_face":
            return {
                "verdict": "inconclusivo",
                "replica_probability": None,
                "message": "Nao localizei um mostrador com confianca suficiente nesta foto.",
                "next_photo": (
                    "Se a foto está nítida, confirme marca e referência presencialmente; "
                    "senão, envie outra foto frontal nítida."
                ),
                "model_invoked": False,
                "online_checks": [],
            }
        if visual.status == "blurry":
            return {
                "verdict": "foto_invalida",
                "replica_probability": None,
                "message": "O mostrador localizado nao tem nitidez suficiente.",
                "next_photo": "Mostrador de frente e sem movimento.",
                "model_invoked": False,
                "online_checks": checks,
            }
        color = visual.dial_color
        bezel_color = visual.bezel_color
        labels = {"black": "preto", "green": "verde", "blue": "azul"}
        dial_label = labels.get(color, "cor indefinida")
        bezel_label = labels.get(bezel_color, "cor não verificável")
        message = f"Mostrador: {dial_label}. Aro: {bezel_label}. Esta foto não comprova autenticidade."
        return {
            "verdict": "inconclusivo",
            "replica_probability": None,
            "message": message,
            "next_photo": "Mostrador de frente e em foco, com a marca legível.",
            "model_invoked": True,
            "detected_angle": "dial",
            "angle_mismatch": False,
            "observed_dial_color": color,
            "observed_bezel_color": bezel_color,
            "possible_reference_mismatch": False,
            "online_checks": [],
        }


def pilot_photo_quality(image: Image.Image, angle: str) -> str:
    from quality_gate.blur_light_check import Pass, quality_gate

    if isinstance(quality_gate(image, angle=angle), Pass):
        return "pass"
    width, height = image.size
    subject = image.crop((width // 4, height // 4, width - width // 4, height - height // 4))
    return "uncertain" if isinstance(quality_gate(subject, angle=angle), Pass) else "reject"


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
    except Image.DecompressionBombError as exc:
        raise ValueError("imagem com dimensoes excessivas") from exc
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

    encoded_images: list[tuple[str, str, bytes]] = []
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
        encoded_images.append((angle, content_type, body))

    results: list[dict[str, Any]] = []
    observed_angles: set[str] = set()
    model_checked_images = 0
    reused_results = 0
    used_photo_indices: list[int] = []
    used_digests: set[str] = set()
    cache: dict[tuple[str, str, str], dict[str, object]] = {}
    visual_cache: dict[str, _VisualInspection] = {}
    unique_evidence: dict[str, dict[str, Any]] = {}
    for index, (angle, content_type, body) in enumerate(encoded_images, start=1):
        image = decode_image(body, content_type)
        digest = hashlib.sha256(
            f"{image.width}x{image.height}:".encode() + image.tobytes()
        ).hexdigest()
        if isinstance(service, PilotService):
            if angle != "dial":
                result = service.render(_VisualInspection("unsupported"), angle)
            else:
                if digest in visual_cache:
                    visual = visual_cache[digest]
                    reused_results += 1
                else:
                    visual = service.inspect(image)
                    visual_cache[digest] = visual
                    if visual.status == "inspected":
                        model_checked_images += 1
                result = service.render(visual, angle)
        else:
            key = (digest, content_type, angle)
            if key in cache:
                result = cache[key]
                reused_results += 1
            else:
                result = service.predict(body, content_type, angle)
                cache[key] = result
                if result.get("model_invoked") is True:
                    model_checked_images += 1
        results.append({"angle": angle, **result})
        if result.get("model_invoked") is True and digest not in used_digests:
            used_photo_indices.append(index)
            used_digests.add(digest)
        if digest not in unique_evidence or (
            not unique_evidence[digest].get("model_invoked") and result.get("model_invoked")
        ):
            unique_evidence[digest] = result
        if result.get("model_invoked") is True:
            detected = result.get("detected_angle")
            if isinstance(detected, str) and detected in ANGLES:
                observed_angles.add(detected)
        image.close()
        del image

    verdicts = {str(result.get("verdict")) for result in results}
    if "inconsistencia_com_referencia" in verdicts:
        verdict = "inconsistencia_com_referencia"
        message = "Há conflito oficial visível em pelo menos uma foto."
    elif verdicts == {"foto_invalida"}:
        verdict = "foto_invalida"
        message = "Nenhuma foto pôde ser utilizada. Envie fotos mais nítidas."
    else:
        verdict = "inconclusivo"
        message = (
            "Fotos recebidas, mas nenhuma foi enviada ao modelo."
            if model_checked_images == 0
            else "O conjunto não autentica o relógio; a análise permanece inconclusiva."
        )
    dial, dial_count = consensus([str(result.get("observed_dial_color", "uncertain")) for result in unique_evidence.values()])
    bezel, bezel_count = consensus([str(result.get("observed_bezel_color", "uncertain")) for result in unique_evidence.values()])
    if model_checked_images and verdict == "inconclusivo":
        message = "Cores observadas; marca, referência e autenticidade inconclusivas."
    missing_angles = sorted(ANGLES - observed_angles)
    next_useful_angle = (
        None
        if model_checked_images == 0
        and verdict == "inconclusivo"
        and any(angle == "dial" for angle, _, _ in encoded_images)
        else "dial"
    )
    examined_next_photo = next(
        (
            result["next_photo"]
            for result in results
            if result.get("model_invoked") is True
            and isinstance(result.get("next_photo"), str)
            and result["next_photo"]
        ),
        None,
    )
    next_useful_photo = (
        "Esta prévia não examinou o relógio; confirme marca e referência presencialmente."
        if next_useful_angle is None
        else examined_next_photo or "Mostrador de frente e em foco, com a marca legível."
    )
    cited_facts = {
        (check["field"], check["url"])
        for result in results
        if result.get("model_invoked") is True
        for check in result.get("online_checks", [])
        if isinstance(check, dict)
        and check.get("role") == "official_spec"
        and isinstance(check.get("field"), str)
        and isinstance(check.get("url"), str)
    }
    return {
        "results": results,
        "joint": {
            "verdict": verdict,
            "message": message,
            "missing_angles": missing_angles,
            "next_useful_angle": next_useful_angle,
            "next_useful_photo": next_useful_photo,
            "used_photo_indices": used_photo_indices,
            "cited_reference_facts": [
                {"field": field, "url": url} for field, url in sorted(cited_facts)
            ],
            "model_sha256": None,
            "calibrated_confidence": None,
            "quality_score_0_1000": None,
            "quality_assessment": "não avaliada",
            "submitted_photos": len(results),
            "unique_visual_contents": len(unique_evidence),
            "reused_results": reused_results,
            "model_checked_images": model_checked_images,
            "reference_candidate": None,
            "visible_evidence": {
                "dial_color": {"value": dial, "supporting_photos": dial_count},
                "bezel_color": {"value": bezel, "supporting_photos": bezel_count},
            },
            "unverified": (
                ["marca, referência e autenticidade; nenhum detalhe foi examinado pelo modelo"]
                if model_checked_images == 0
                else [
                    "impressão e índices do mostrador em macro",
                    "gravações internas, referência e número de série",
                    "lupa e data em macro",
                    "acabamento da caixa, pulseira e interior do fecho",
                    "movimento e peças internas",
                ]
            ),
            "source_urls": [],
            "inspection_source_url": INSPECTION_SOURCE,
        },
    }


def handler(service: PredictService, password: str | None = None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(MAX_HEADER_SECONDS)
            self._header_timer = threading.Timer(MAX_HEADER_SECONDS, self._abort_connection)
            self._header_timer.daemon = True
            self._header_timer.start()

        def parse_request(self) -> bool:
            try:
                return super().parse_request()
            finally:
                self._header_timer.cancel()

        def finish(self) -> None:
            self._header_timer.cancel()
            super().finish()

        def _abort_connection(self) -> None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        def _authorized(self) -> bool:
            if password is None:
                return True
            authorization = self.headers.get("Authorization", "")
            if authorization.startswith("Basic "):
                try:
                    supplied = base64.b64decode(authorization[6:], validate=True).decode()
                    if hmac.compare_digest(supplied, f"watch:{password}"):
                        return True
                except (binascii.Error, UnicodeDecodeError, ValueError):
                    pass
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="Watch Auth Pilot"')
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            if urlsplit(self.path).path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_HTML)))
            self.end_headers()
            self.wfile.write(_HTML)

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            parsed = urlsplit(self.path)
            if parsed.path not in {"/predict", "/predict/batch"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                limit = MAX_BATCH_REQUEST_BYTES if parsed.path == "/predict/batch" else MAX_UPLOAD_BYTES
                if length <= 0 or length > limit:
                    raise ValueError("imagem vazia ou maior que 15 MB")
                deadline = time.monotonic() + MAX_REQUEST_SECONDS
                chunks: list[bytes] = []
                remaining = length
                while remaining:
                    seconds = deadline - time.monotonic()
                    if seconds <= 0:
                        raise TimeoutError("tempo de envio excedido")
                    self.connection.settimeout(min(10, seconds))
                    chunk = self.rfile.read(min(65_536, remaining))
                    if not chunk:
                        raise ValueError("envio da imagem incompleto")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                body = b"".join(chunks)
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
            except (TimeoutError, socket.timeout):
                self._json(HTTPStatus.REQUEST_TIMEOUT, {"error": "tempo de envio excedido"})

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


class PilotHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: Callable[..., BaseHTTPRequestHandler],
        bind_and_activate: bool = True,
    ) -> None:
        self._slots = threading.BoundedSemaphore(2)
        super().__init__(server_address, handler_class, bind_and_activate)

    def process_request(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        if not self._slots.acquire(blocking=False):
            self.close_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: Any
    ) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--knowledge", type=Path,
        default=None,
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        raise SystemExit("port must be between 1024 and 65535")
    password = os.environ.get("WATCH_AUTH_PILOT_PASSWORD")
    if args.host != "127.0.0.1" and (not password or len(password) < 24):
        raise SystemExit("public bind requires WATCH_AUTH_PILOT_PASSWORD (24+ characters)")
    service = PilotService(args.checkpoint, args.knowledge)
    server = PilotHTTPServer((args.host, args.port), handler(service, password))
    print(f"Prévia visual: {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
