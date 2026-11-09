from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from acquisition.collect import (
    AcquisitionRefused,
    IntegrityError,
    UnsafeImage,
    collect_batch,
    sanitize_image,
)
from acquisition.partner_import import import_partner_export
from tests.conftest import NOW


def test_blocked_policy_makes_no_http_or_cos_call(blocked_policy, rights, http_client, cos):
    with pytest.raises(AcquisitionRefused):
        collect_batch(
            [],
            blocked_policy,
            rights,
            http_client=http_client,
            cos_client=cos,
            bucket="private",
            batch_id="b1",
            now=NOW,
        )

    http_client.get.assert_not_called()
    assert cos.write_count == 0


@pytest.mark.parametrize(
    "payload_case",
    [
        "partial",
        "hash_mismatch",
        "oversized",
        "non_image",
        "mime_magic_mismatch",
        "truncated",
        "polyglot",
        "multi_frame",
        "decompression_bomb",
    ],
)
def test_unsafe_or_incomplete_payload_never_reaches_cos(payload_case, collector_fixture, cos):
    with pytest.raises((IntegrityError, UnsafeImage)):
        collector_fixture.collect(payload_case)

    assert cos.write_count == 0


def test_repeated_batch_reuses_verified_raw_derivative_and_receipt(collector_fixture, cos):
    first = collector_fixture.collect("valid")
    writes = cos.write_count
    second = collector_fixture.collect("valid")

    assert second == first
    assert writes == 3
    assert cos.write_count == writes


def test_derivative_is_rgb_readable_and_has_no_source_metadata(collector_fixture, cos):
    collector_fixture.collect("metadata")
    derivative = next(body for key, body in cos.bodies.items() if key.startswith("datasets/derivatives/"))

    with Image.open(io.BytesIO(derivative)) as image:
        assert image.mode == "RGB"
        assert image.getexif() == {}
        image.load()


@pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP"])
def test_supported_single_frame_formats_are_sanitized(image_format: str):
    source = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(source, format=image_format)

    derivative, phash = sanitize_image(source.getvalue())

    with Image.open(io.BytesIO(derivative)) as image:
        assert image.mode == "RGB"
        image.load()
    assert len(phash) == 16


def test_429_retries_are_bounded_and_do_not_write(collector_fixture, cos):
    collector_fixture.set_statuses(429, 429, 429)

    with pytest.raises(IntegrityError, match="after 3 attempts"):
        collector_fixture.collect("valid")

    assert collector_fixture.http_client.get.call_count == 3
    assert cos.write_count == 0


def test_missing_rights_refuses_before_network(allowed_policy, http_client, cos):
    with pytest.raises(AcquisitionRefused, match="rights missing"):
        collect_batch(
            [],
            allowed_policy,
            None,
            http_client=http_client,
            cos_client=cos,
            bucket="private",
            batch_id="b1",
            now=NOW,
        )

    http_client.get.assert_not_called()
    assert cos.write_count == 0


def test_receipt_collision_refuses_without_overwriting(collector_fixture, cos):
    first = collector_fixture.collect("valid")
    candidate_id = first.accepted_candidate_ids[0]
    key = f"datasets/receipts/b1/{candidate_id}.json"
    cos.bodies[key] = b'{"different":true}'

    with pytest.raises(IntegrityError, match="receipt"):
        collector_fixture.collect("valid")

    assert cos.bodies[key] == b'{"different":true}'


@pytest.mark.parametrize("unsafe_path", ["/tmp/image.png", "../image.png", "nested/../../image.png"])
def test_partner_import_rejects_unsafe_paths(tmp_path: Path, rights, unsafe_path: str):
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row(unsafe_path)) + "\n")

    with pytest.raises(ValueError, match="safe relative path"):
        import_partner_export(manifest, rights)


def test_partner_import_rejects_symlinks(tmp_path: Path, rights):
    outside = tmp_path.parent / "outside-partner-image.png"
    Image.new("RGB", (4, 4), "red").save(outside)
    linked = tmp_path / "image.png"
    linked.symlink_to(outside)
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row("image.png")) + "\n")

    with pytest.raises(ValueError, match="symlink"):
        import_partner_export(manifest, rights)


def test_partner_import_rejects_undeclared_files(tmp_path: Path, rights):
    Image.new("RGB", (4, 4), "red").save(tmp_path / "image.png")
    (tmp_path / "extra.txt").write_text("not declared")
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row("image.png")) + "\n")

    with pytest.raises(ValueError, match="undeclared file"):
        import_partner_export(manifest, rights)


def test_partner_import_hashes_declared_image(tmp_path: Path, rights):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row("image.png")) + "\n")

    [candidate] = import_partner_export(manifest, rights)

    assert candidate.raw_sha256 == hashlib.sha256(image_path.read_bytes()).hexdigest()
    assert candidate.rights_record_id == rights.rights_record_id


def _partner_row(file: str) -> dict[str, object]:
    return {
        "file": file,
        "candidate_id": "partner-candidate-1",
        "acquisition_batch_id": "partner-batch-1",
        "source_policy_id": "partner-upload-v1",
        "source_record_id": "partner-record-1",
        "source_url": "https://partner.example/record-1",
        "collected_at": NOW.isoformat(),
        "declared_reference": "rolex-submariner-126610ln",
    }
