from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from pilot.server import MAX_UPLOAD_BYTES, checks_for_angle, decode_image
from reference_intelligence.knowledge import load_knowledge


def _jpeg(size: tuple[int, int] = (300, 300)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, "white").save(output, format="JPEG")
    return output.getvalue()


def test_decode_image_accepts_bounded_jpeg() -> None:
    assert decode_image(_jpeg(), "image/jpeg").size == (300, 300)


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
    assert "tests/test_online_assessor.py" in workflow
    assert "tests/test_predict.py" in workflow
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
)
def test_decode_image_rejects_unsafe_input(body: bytes, content_type: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode_image(body, content_type)
