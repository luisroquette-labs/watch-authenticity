from __future__ import annotations

import base64
import gc
import http.client
import io
import json
import math
import subprocess
import sys
import threading
import weakref
from http.server import HTTPServer
from pathlib import Path
from typing import Any, cast

import cv2
import pytest
from PIL import Image, ImageDraw, ImageFilter

from pilot.server import (
    _HTML,
    ANGLES,
    MAX_BATCH_DECODED_BYTES,
    MAX_BATCH_IMAGES,
    MAX_UPLOAD_BYTES,
    PilotService,
    _VisualInspection,
    analyze_batch,
    checks_for_angle,
    decode_image,
    handler,
    pilot_photo_quality,
)
from pilot.visual_evidence import bezel_color_hint, consensus
from pilot.watch_locator import WatchRegion, choose_region
from reference_intelligence.knowledge import load_knowledge


def _jpeg(size: tuple[int, int] = (300, 300)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="JPEG")
    return output.getvalue()


def _jpeg_variant(index: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (300, 300), (index, index * 3, index * 5)).save(output, format="JPEG")
    return output.getvalue()


def test_decode_image_accepts_bounded_jpeg() -> None:
    assert decode_image(_jpeg(), "image/jpeg").size == (300, 300)


def test_pilot_marks_small_sharp_watch_as_uncertain_without_accepting_blurry_watch() -> None:
    image = Image.new("RGB", (1200, 1600), "#d2c8b8")
    draw = ImageDraw.Draw(image)
    draw.ellipse((450, 650, 750, 950), fill="#17442f", outline="white", width=4)
    for index in range(12):
        angle = index * 2 * math.pi / 12
        x, y = 600 + 112 * math.cos(angle), 800 + 112 * math.sin(angle)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="white")

    assert pilot_photo_quality(image, "dial") == "uncertain"
    assert pilot_photo_quality(image.filter(ImageFilter.GaussianBlur(radius=8)), "dial") == "reject"


def test_uncertain_subject_focus_never_calls_the_assessor() -> None:
    image = Image.new("RGB", (1200, 1600), "#d2c8b8")
    draw = ImageDraw.Draw(image)
    draw.ellipse((450, 650, 750, 950), fill="#17442f", outline="white", width=4)
    image = image.filter(ImageFilter.GaussianBlur(radius=8))
    draw = ImageDraw.Draw(image)
    for y in range(410, 460, 5):
        for x in range(310, 360, 5):
            draw.rectangle((x, y, x + 4, y + 4), fill="black" if (x + y) // 5 % 2 else "white")

    class FailAssessor:
        def assess(self, *_args: object) -> None:
            raise AssertionError("blurry watch must not reach model")

    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(
        Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln"
    )
    service.assessor = cast(Any, FailAssessor())
    service.locator = cast(Any, type("Locator", (), {"locate": lambda self, image: WatchRegion(image, image.crop((450, 650, 750, 950)), .9, .8)})())
    output = io.BytesIO()
    image.save(output, format="PNG")

    result = service.predict(output.getvalue(), "image/png", "dial")
    assert result["verdict"] == "foto_invalida"
    assert result["model_invoked"] is False


def test_locator_ignores_face_outside_watch() -> None:
    image = Image.new("RGB", (500, 500), "white")
    region = choose_region(image, [
        (.9, 0, (100, 100, 400, 400)),
        (.95, 1, (0, 0, 80, 80)),
        (.7, 1, (170, 170, 330, 330)),
    ])
    assert region is not None
    assert region.face is not None
    assert region.face.size == (160, 160)
    assert region.face_score == .7


def test_locator_keeps_face_with_small_detector_box_overhang() -> None:
    # Real 1440px vendor photo: detector face extends ~5px beyond its watch box.
    image = Image.new("RGB", (1440, 1440), "white")
    region = choose_region(image, [
        (.4671, 0, (417.1, 246.5, 888.2, 1162.4)),
        (.4002, 1, (418.8, 262.3, 893.3, 793.3)),
    ])

    assert region is not None
    assert region.face is not None
    assert region.face_score == .4002


@pytest.mark.parametrize(
    ("watch_score", "face_score", "watch_box", "face_box"),
    [
        (.336, .410, (20.8, 99.3, 1424.2, 2174.9), (6.8, 609.9, 1305.8, 1776.9)),
        (.162, .349, (1.9, -17.4, 1415.3, 1957.2), (7.0, 774.9, 1268.6, 1793.1)),
    ],
)
def test_locator_keeps_white_background_watch_with_correlated_face(
    watch_score: float, face_score: float,
    watch_box: tuple[float, float, float, float], face_box: tuple[float, float, float, float],
) -> None:
    # Detector outputs observed on two clear 126610LN listing photos.
    image = Image.new("RGB", (1412, 2160), "white")
    region = choose_region(image, [(watch_score, 0, watch_box), (face_score, 1, face_box)])

    assert region is not None
    assert region.face is not None
    assert region.watch_score == watch_score
    assert region.face_score == face_score


def test_locator_rejects_weak_watch_without_correlated_face() -> None:
    image = Image.new("RGB", (500, 500), "white")
    assert choose_region(image, [
        (.20, 0, (100, 100, 400, 400)),
        (.29, 1, (130, 130, 370, 370)),
    ]) is None
    assert choose_region(image, [
        (.20, 0, (100, 100, 400, 400)),
        (.40, 1, (0, 0, 80, 80)),
    ]) is None


def test_locator_rejects_face_with_large_detector_box_overhang() -> None:
    image = Image.new("RGB", (500, 500), "white")
    region = choose_region(image, [
        (.9, 0, (100, 100, 400, 400)),
        (.8, 1, (50, 150, 350, 350)),
    ])

    assert region is not None
    assert region.face is None


def test_locator_rejects_image_without_watch_or_face() -> None:
    assert choose_region(Image.new("RGB", (500, 500)), []) is None


@pytest.mark.parametrize(
    ("ring_color", "expected"),
    [("#137746", "green"), ("#111111", "black"), ("#153f9d", "blue")],
)
def test_bezel_color_reads_ring_and_ignores_dial(ring_color: str, expected: str) -> None:
    face = Image.new("RGB", (300, 300), "white")
    draw = ImageDraw.Draw(face)
    draw.ellipse((5, 5, 295, 295), fill=ring_color)
    draw.ellipse((48, 48, 252, 252), fill="white")
    assert bezel_color_hint(face) == expected
    assert bezel_color_hint(face.resize((140, 140))) == "uncertain"


def test_dial_without_visible_bezel_cannot_supply_bezel_color() -> None:
    face = Image.new("RGB", (300, 300), "white")
    ImageDraw.Draw(face).ellipse((5, 5, 295, 295), fill="#137746")
    assert bezel_color_hint(face) == "uncertain"


def test_partly_cropped_bezel_uses_dial_boundary_and_rejects_green_backdrop() -> None:
    photo = Image.new("RGB", (360, 360), "#dadbd5")
    draw = ImageDraw.Draw(photo)
    draw.ellipse((6, 6, 354, 354), fill="#137746")
    draw.ellipse((48, 48, 312, 312), fill="#263b31")
    assert bezel_color_hint(photo.crop((0, 12, 360, 348))) == "green"

    backdrop = Image.new("RGB", (360, 360), "#137746")
    ImageDraw.Draw(backdrop).ellipse((48, 48, 312, 312), fill="#263b31")
    assert bezel_color_hint(backdrop) == "uncertain"


def test_bezel_analysis_downsamples_large_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    original = cv2.cvtColor
    shapes = []

    def checked(image, conversion):
        shapes.append(image.shape)
        return original(image, conversion)

    monkeypatch.setattr(cv2, "cvtColor", checked)
    assert bezel_color_hint(Image.new("RGB", (5000, 5000), "green")) == "uncertain"
    assert shapes and all(max(shape[:2]) <= 512 for shape in shapes)


def test_color_consensus_requires_consistent_clues() -> None:
    assert consensus(["green", "green", "uncertain"]) == ("green", 2)
    assert consensus(["green"]) == ("green", 1)
    assert consensus(["green", "black", "green"]) == ("uncertain", 0)


@pytest.mark.parametrize("color", ["green", "black"])
def test_pilot_analyzes_detected_dial_without_claiming_authenticity(color: str) -> None:
    image = Image.new("RGB", (300, 300), "#164d30")
    draw = ImageDraw.Draw(image)
    for x in range(0, 300, 10):
        draw.line((x, 0, x, 300), fill="white", width=1)

    class Locator:
        def locate(self, photo: Image.Image) -> WatchRegion:
            return WatchRegion(photo, photo, .9, .8)

    class Assessor:
        def dial_color(self, photo: Image.Image) -> str:
            assert photo.size == (300, 300)
            return color

    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    service.locator = cast(Any, Locator())
    service.assessor = cast(Any, Assessor())
    output = io.BytesIO()
    image.save(output, format="PNG")
    result = service.predict(output.getvalue(), "image/png", "dial")
    assert result["verdict"] == "inconclusivo"
    assert result["replica_probability"] is None
    assert result["detected_angle"] == "dial"
    assert result["possible_reference_mismatch"] is False
    assert result["online_checks"] == []
    assert result["model_invoked"] is True


def test_decode_image_rejects_mime_mismatch() -> None:
    output = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(output, format="PNG")

    with pytest.raises(ValueError, match="corresponde"):
        decode_image(output.getvalue(), "image/jpeg")


def test_decode_image_rejects_decompression_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    with pytest.raises(ValueError, match="dimensoes excessivas"):
        decode_image(_jpeg(), "image/jpeg")


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
    preflight = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert not Path(".github/workflows/pilot-ci.yml").exists()
    for path in (
        "pilot/**", "reference_intelligence/**", "config/**", "training/**",
        "evaluation/**", "dataset/**", "infra/**", "labeling/**",
        "Containerfile.*", "requirements*.lock", "tests/**",
    ):
        assert f'"{path}"' in preflight
    assert "timeout-minutes: 10" in preflight
    assert "bash scripts/preflight-ci.sh" in preflight
    assert "next build" not in preflight


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
            "model_invoked": verdict != "foto_invalida",
            "detected_angle": angle if verdict != "foto_invalida" else None,
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
    first, second = _jpeg_variant(1), _jpeg_variant(2)

    result = analyze_batch(
        service,
        {"images": [_batch_item("dial", first), _batch_item("crown", second)]},
    )

    assert [call[0] for call in service.calls] == [first, second]
    assert [item["angle"] for item in result["results"]] == ["dial", "crown"]
    assert result["results"][0]["online_checks"][0]["url"].endswith("/dial")
    assert result["joint"]["verdict"] == "inconclusivo"
    assert result["joint"]["missing_angles"] == sorted(ANGLES - {"dial", "crown"})
    assert result["joint"]["model_checked_images"] == 2


def test_batch_reuses_identical_photo_and_counts_evidence_once() -> None:
    class CountingService:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *_args: object) -> dict[str, object]:
            self.calls += 1
            return {
                "verdict": "inconclusivo",
                "model_invoked": True,
                "detected_angle": "dial",
                "observed_dial_color": "green",
                "observed_bezel_color": "green",
            }

    service = CountingService()
    batch = analyze_batch(service, {"images": [_batch_item() for _ in range(6)]})
    assert service.calls == 1
    assert batch["joint"]["submitted_photos"] == 6
    assert batch["joint"]["unique_visual_contents"] == 1
    assert batch["joint"]["reused_results"] == 5
    assert batch["joint"]["model_checked_images"] == 1
    assert batch["joint"]["visible_evidence"]["dial_color"]["supporting_photos"] == 1


def test_batch_does_not_claim_other_angle_from_reused_dial_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    calls = []

    def inspect(_image: Image.Image) -> _VisualInspection:
        calls.append(True)
        return _VisualInspection("inspected", "green", "green")

    monkeypatch.setattr(service, "inspect", inspect)
    png = io.BytesIO()
    Image.new("RGB", (300, 300), "white").save(png, format="PNG")
    result = analyze_batch(service, {"images": [
        _batch_item("dial", _jpeg()),
        {**_batch_item("clasp", png.getvalue()), "content_type": "image/png"},
    ]})
    assert len(calls) == 1
    assert result["joint"]["unique_visual_contents"] == 1
    assert result["joint"]["reused_results"] == 0
    assert result["joint"]["model_checked_images"] == 1
    assert result["results"][1]["model_invoked"] is False
    assert "detected_angle" not in result["results"][1]
    assert result["results"][1]["online_checks"] == []


def test_batch_caseback_before_dial_only_uses_dial(monkeypatch: pytest.MonkeyPatch) -> None:
    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    calls = []

    def inspect(_image: Image.Image) -> _VisualInspection:
        calls.append(True)
        return _VisualInspection("inspected", "black", "black")

    monkeypatch.setattr(service, "inspect", inspect)
    result = analyze_batch(service, {"images": [
        _batch_item("caseback", _jpeg_variant(1)),
        _batch_item("dial", _jpeg_variant(2)),
    ]})

    assert len(calls) == 1
    assert result["joint"]["used_photo_indices"] == [2]
    assert result["joint"]["model_checked_images"] == 1
    assert result["results"][0]["model_invoked"] is False
    assert "detected_angle" not in result["results"][0]


def test_batch_releases_each_decoded_image_before_inspecting_next(monkeypatch: pytest.MonkeyPatch) -> None:
    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    prior_images: list[weakref.ReferenceType[Image.Image]] = []

    def inspect(image: Image.Image) -> _VisualInspection:
        gc.collect()
        assert all(prior() is None for prior in prior_images)
        prior_images.append(weakref.ref(image))
        return _VisualInspection("inspected", "green", "green")

    monkeypatch.setattr(service, "inspect", inspect)
    result = analyze_batch(service, {"images": [
        _batch_item(body=_jpeg_variant(1)),
        _batch_item(body=_jpeg_variant(2)),
    ]})
    assert result["joint"]["unique_visual_contents"] == 2
    assert len(prior_images) == 2


def test_selected_clasp_does_not_claim_clasp_when_only_dial_is_located() -> None:
    image = Image.new("RGB", (300, 300), "#164d30")
    for x in range(0, 300, 10):
        ImageDraw.Draw(image).line((x, 0, x, 300), fill="white", width=1)

    class Locator:
        def locate(self, photo: Image.Image) -> WatchRegion:
            return WatchRegion(photo, photo, .9, .8)

    class Assessor:
        def dial_color(self, _photo: Image.Image) -> str:
            return "green"

    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    service.locator = cast(Any, Locator())
    service.assessor = cast(Any, Assessor())
    output = io.BytesIO()
    image.save(output, format="PNG")
    result = service.predict(output.getvalue(), "image/png", "clasp")
    assert result["model_invoked"] is False
    assert "detected_angle" not in result
    checks = cast(list[dict[str, str]], result["online_checks"])
    assert all(item["field"] != "clasp" for item in checks)


def test_caseback_selection_never_counts_a_false_face_as_examined() -> None:
    class Locator:
        def locate(self, photo: Image.Image) -> WatchRegion:
            return WatchRegion(photo, photo, .9, .8)

    class Assessor:
        def dial_color(self, _photo: Image.Image) -> str:
            raise AssertionError("dial-only model must not run on a selected caseback")

    service = PilotService.__new__(PilotService)
    service.claims = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), "rolex-submariner-126610ln")
    service.locator = cast(Any, Locator())
    service.assessor = cast(Any, Assessor())
    result = service.predict(_jpeg(), "image/jpeg", "caseback")

    assert result["verdict"] == "inconclusivo"
    assert result["model_invoked"] is False
    assert "detected_angle" not in result
    assert result["online_checks"] == []


def test_batch_does_not_infer_reference_from_colors_without_brand_identification() -> None:
    class GreenWatch:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, *_args: object) -> dict[str, object]:
            self.calls += 1
            return {
                "verdict": "inconclusivo",
                "model_invoked": True,
                "observed_dial_color": "green",
                "observed_bezel_color": "green" if self.calls < 3 else "uncertain",
            }

    result = analyze_batch(GreenWatch(), {"images": [_batch_item(body=_jpeg_variant(i + 1)) for i in range(3)]})
    joint = result["joint"]
    assert joint["verdict"] == "inconclusivo"
    assert joint["reference_candidate"] is None
    assert joint["next_useful_angle"] == "dial"
    assert joint["visible_evidence"]["dial_color"]["supporting_photos"] == 3
    assert joint["visible_evidence"]["bezel_color"]["supporting_photos"] == 2
    assert "marca, referência e autenticidade inconclusivas" in joint["message"]


def test_black_dial_and_bezel_on_non_target_watch_never_suggest_rolex() -> None:
    class NonTargetWatch:
        def predict(self, *_args: object) -> dict[str, object]:
            return {
                "verdict": "inconclusivo",
                "model_invoked": True,
                "observed_dial_color": "black",
                "observed_bezel_color": "black",
            }

    joint = analyze_batch(NonTargetWatch(), {"images": [_batch_item(body=_jpeg_variant(1))]})["joint"]
    assert joint["reference_candidate"] is None
    assert joint["source_urls"] == []
    assert "Rolex" not in joint["message"]


def test_conflicting_dial_colors_block_reference_candidate() -> None:
    colors = iter(("green", "black"))

    class ConflictingPhotos:
        def predict(self, *_args: object) -> dict[str, object]:
            return {
                "verdict": "inconclusivo",
                "model_invoked": True,
                "detected_angle": "dial",
                "observed_dial_color": next(colors),
                "observed_bezel_color": "green",
            }

    joint = analyze_batch(ConflictingPhotos(), {"images": [_batch_item("crown", _jpeg_variant(1)), _batch_item("crown", _jpeg_variant(2))]})["joint"]
    assert joint["reference_candidate"] is None
    assert joint["visible_evidence"]["dial_color"]["value"] == "uncertain"
    assert "crown" in joint["missing_angles"]


def test_submitted_but_unchecked_photos_are_reported_separately() -> None:
    class Unassessed:
        def predict(self, *_args: object) -> dict[str, object]:
            return {"verdict": "inconclusivo", "model_invoked": False}

    result = analyze_batch(Unassessed(), {"images": [_batch_item(angle) for angle in ANGLES]})

    assert result["joint"]["missing_angles"] == sorted(ANGLES)
    assert result["joint"]["model_checked_images"] == 0
    assert "nenhuma foi enviada ao modelo" in result["joint"]["message"]


def test_untrained_pilot_separates_seen_photos_uncertainty_and_fake_quality() -> None:
    result = analyze_batch(
        _StubService(["inconclusivo"]),
        {"images": [_batch_item("dial", _jpeg_variant(1))]},
    )
    joint = result["joint"]
    assert joint["used_photo_indices"] == [1]
    assert joint["next_useful_angle"] == "dial"
    assert joint["model_sha256"] is None
    assert joint["calibrated_confidence"] is None
    assert joint["quality_score_0_1000"] is None
    assert joint["quality_assessment"] == "não avaliada"
    assert joint["cited_reference_facts"] == []
    page = _HTML.decode("utf-8")
    assert "Fotos usadas:" in page
    assert "confiança calibrada:" in page
    assert "Qualidade da falsificação (somente após confirmação):" in page
    assert "nenhuma foto adicional desta prévia comprova autenticidade" in page


def test_rejected_photo_cannot_supply_cited_facts_or_fake_quality() -> None:
    class UnsupportedClaim:
        def predict(self, *_args: object) -> dict[str, object]:
            return {
                "verdict": "foto_invalida",
                "model_invoked": False,
                "online_checks": [{"field": "dial_color", "url": "https://www.rolex.com/x", "role": "official_spec"}],
                "quality_score_0_1000": 1000,
                "replica_probability": 0.99,
            }

    joint = analyze_batch(UnsupportedClaim(), {"images": [_batch_item()]})["joint"]

    assert joint["used_photo_indices"] == []
    assert joint["cited_reference_facts"] == []
    assert joint["quality_score_0_1000"] is None
    assert joint["calibrated_confidence"] is None
    assert joint["next_useful_angle"] == "dial"


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
        {"images": [_batch_item("dial", _jpeg_variant(1)), _batch_item("rehaut", _jpeg_variant(2))]},
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
    assert "Prévia visual de relógios" in html
    assert "Referência comparada: nenhuma; marca não identificada." in html
    assert "Ainda não detecta falsificações" in html
    assert "não há classificador de autenticidade treinado e validado" in html

    assert "j.joint.message+'\\nFotos enviadas: '" in html
    assert "j.joint.message+'\nFotos enviadas: '" not in html
    assert "Conteúdos visuais distintos:" in html
    assert "Resultados reaproveitados:" in html
    assert 'multiple' in html
    assert 'Máximo de 6 imagens' in html
    assert 'fetch("/predict/batch"' in html
    assert 'finally' in html
    assert 'b.disabled=false' in html
    assert 'let files=[...f.files]' in html
    assert "files[i].name+': '+item.message" in html
    assert 'f.disabled=true' in html and 'f.disabled=false' in html
    assert 'Vista informada' in html
