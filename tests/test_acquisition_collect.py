from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import ibm_botocore.session
import pytest
from PIL import Image

import acquisition.collect as collect_module
import acquisition.partner_import as partner_import
from acquisition.collect import (
    AcquisitionRefused,
    IntegrityError,
    UnsafeImage,
    collect_batch,
    sanitize_image,
)
from acquisition.partner_import import collect_partner_batch, import_partner_export
from acquisition.types import SourceAccessRecord, SourceAccessStatus
from tests.conftest import NOW, _Response


def test_blocked_policy_makes_no_http_or_cos_call(blocked_policy, rights, http_client, cos):
    with pytest.raises(AcquisitionRefused):
        collect_batch(
            [],
            blocked_policy,
            rights,
            access=None,
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


def test_concurrent_identical_receipt_create_is_idempotent(
    allowed_policy, rights, cos
):
    body = _png("red")
    cos.receipt_write_barrier = Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: _collect_body(body, allowed_policy, rights, cos),
                range(2),
            )
        )

    assert results[0] == results[1]
    candidate_id = results[0].accepted_candidate_ids[0]
    assert len(_receipt_keys(cos, candidate_id)) == 1


def test_concurrent_different_receipts_are_preserved_and_both_detect_collision(
    allowed_policy, rights, cos
):
    cos.receipt_write_barrier = Barrier(2)

    def collect_or_error(body: bytes):
        try:
            return _collect_body(body, allowed_policy, rights, cos)
        except IntegrityError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(collect_or_error, [_png("red"), _png("blue")]))

    assert all(isinstance(result, IntegrityError) for result in results)
    candidate_id = "candidate-" + hashlib.sha256(
        b"b1\0partner-upload-v1\0record-1"
    ).hexdigest()[:24]
    receipt_keys = _receipt_keys(cos, candidate_id)
    assert len(receipt_keys) == 2
    assert len({cos.bodies[key] for key in receipt_keys}) == 2


def test_receipt_put_uses_only_ibm_sdk_supported_parameters():
    members = (
        ibm_botocore.session.get_session()
        .get_service_model("s3")
        .operation_model("PutObject")
        .input_shape.members
    )

    assert "IfNoneMatch" not in members
    assert "IfNoneMatch" not in inspect.getsource(collect_module._write_receipt)


def test_mixed_source_batch_refuses_before_any_io(
    allowed_policy, rights, http_client, cos
):
    valid = {
        "source_record_id": "record-1",
        "source_url": "https://partner.example/record-1",
        "image_url": "https://partner.example/image-1.png",
        "declared_reference": "rolex-submariner-126610ln",
    }
    wrong_source = {
        "source": "wrong-source",
        "source_record_id": "record-2",
        "source_url": "https://wrong.example/record-2",
        "image_url": "https://wrong.example/image-2.png",
        "declared_reference": "rolex-submariner-126610ln",
    }

    with pytest.raises(AcquisitionRefused, match="record source"):
        collect_batch(
            [valid, wrong_source],
            allowed_policy,
            rights,
            access=_access(rights),
            http_client=http_client,
            cos_client=cos,
            bucket="private",
            batch_id="b1",
            now=NOW,
        )

    http_client.get.assert_not_called()
    assert cos.write_count == 0


def test_derivative_is_rgb_readable_and_has_no_source_metadata(collector_fixture, cos):
    collector_fixture.collect("metadata")
    derivative = next(body for key, body in cos.bodies.items() if key.startswith("datasets/derivatives/"))

    with Image.open(io.BytesIO(derivative)) as image:
        assert image.mode == "RGB"
        assert image.getexif() == {}
        image.load()


def test_jpeg_gps_exif_is_removed_and_pixels_remain_readable(jpeg_with_gps):
    with Image.open(io.BytesIO(jpeg_with_gps)) as source:
        assert source.getexif().get_ifd(0x8825)
        source.load()
        expected_pixel = source.getpixel((0, 0))

    derivative, _ = sanitize_image(jpeg_with_gps)

    with Image.open(io.BytesIO(derivative)) as image:
        assert image.getexif() == {}
        assert image.getpixel((0, 0)) == expected_pixel


def test_sanitizer_never_assigns_global_pillow_pixel_limit():
    assert "Image.MAX_IMAGE_PIXELS =" not in inspect.getsource(sanitize_image)


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
            access=None,
            http_client=http_client,
            cos_client=cos,
            bucket="private",
            batch_id="b1",
            now=NOW,
        )

    http_client.get.assert_not_called()
    assert cos.write_count == 0


@pytest.mark.parametrize(
    "image_url",
    [
        "http://partner.example/image.png",
        "https://user@partner.example/image.png",
        "https://partner.example:8443/image.png",
        "https://partner.example.evil.test/image.png",
    ],
)
def test_external_download_rejects_url_tricks_before_http(
    image_url, allowed_policy, rights, cos
):
    http = MagicMock()

    with pytest.raises(AcquisitionRefused):
        _collect_url(image_url, http, allowed_policy, rights, cos, _public_dns)

    http.get.assert_not_called()
    assert cos.write_count == 0


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.169.254", "224.0.0.1", "0.0.0.0"],
)
def test_external_download_rejects_non_public_dns(
    address, allowed_policy, rights, cos
):
    http = MagicMock()

    def resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", (address, 443))]

    with pytest.raises(AcquisitionRefused, match="non-public"):
        _collect_url(
            "https://partner.example/image.png",
            http,
            allowed_policy,
            rights,
            cos,
            resolver,
        )

    http.get.assert_not_called()


def test_redirect_is_revalidated_before_second_connection(allowed_policy, rights, cos):
    redirect = _Response(b"", "image/png", status_code=302, length=0)
    redirect.headers["Location"] = "https://evil.example/internal.png"
    http = MagicMock()
    http.get.return_value = redirect

    with pytest.raises(AcquisitionRefused, match="host is not approved"):
        _collect_url(
            "https://partner.example/image.png",
            http,
            allowed_policy,
            rights,
            cos,
            _public_dns,
        )

    assert http.get.call_count == 1
    assert http.get.call_args.kwargs["allow_redirects"] is False
    assert cos.write_count == 0


def test_receipt_collision_refuses_without_overwriting(collector_fixture, cos):
    first = collector_fixture.collect("valid")
    candidate_id = first.accepted_candidate_ids[0]
    [key] = _receipt_keys(cos, candidate_id)
    cos.bodies[key] = b'{"different":true}'

    with pytest.raises(IntegrityError, match="receipt"):
        collector_fixture.collect("valid")

    assert cos.bodies[key] == b'{"different":true}'


def test_existing_different_receipt_refuses_before_new_cos_uploads(
    allowed_policy, rights, cos
):
    first = _collect_body(_png("red"), allowed_policy, rights, cos)
    candidate_id = first.accepted_candidate_ids[0]
    writes = cos.write_count

    with pytest.raises(IntegrityError, match="receipt collision"):
        _collect_body(_png("blue"), allowed_policy, rights, cos)

    assert cos.write_count == writes
    assert len(_receipt_keys(cos, candidate_id)) == 1


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


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="platform lacks O_NOFOLLOW")
def test_partner_import_refuses_file_swapped_to_symlink(
    tmp_path: Path, rights, monkeypatch
):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    outside = tmp_path.parent / "outside-swapped-partner-image.png"
    Image.new("RGB", (4, 4), "blue").save(outside)
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row("image.png")) + "\n")
    original_open = os.open
    swapped = False

    def swap_then_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if Path(path) == image_path and not swapped:
            swapped = True
            image_path.unlink()
            image_path.symlink_to(outside)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(partner_import.os, "open", swap_then_open)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        import_partner_export(manifest, rights)

    assert swapped


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


def test_partner_batch_persists_twenty_images_and_resumes_without_duplicates(
    tmp_path: Path, allowed_policy, rights, cos
):
    evidence = b'{"permission":"owner-authorized","purpose":["download","train"]}'
    evidence_path = tmp_path / "rights-evidence.json"
    evidence_path.write_bytes(evidence)
    rows = []
    for index in range(20):
        filename = f"image-{index}.png"
        Image.new("RGB", (4, 4), (index, index * 2, index * 3)).save(tmp_path / filename)
        row = _partner_row(filename)
        row["candidate_id"] = f"partner-candidate-{index}"
        row["source_record_id"] = "partner-listing-1"
        rows.append(row)
    manifest = tmp_path / "export.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    declared_rights = replace(
        rights,
        evidence_sha256=hashlib.sha256(evidence).hexdigest(),
    )

    first = collect_partner_batch(
        manifest,
        evidence_path,
        allowed_policy,
        declared_rights,
        cos_client=cos,
        bucket="private",
        batch_id="partner-batch-1",
        now=NOW,
    )
    writes = cos.write_count
    second = collect_partner_batch(
        manifest,
        evidence_path,
        allowed_policy,
        declared_rights,
        cos_client=cos,
        bucket="private",
        batch_id="partner-batch-1",
        now=NOW,
    )

    assert len(first.accepted_candidate_ids) == 20
    assert second == first
    assert cos.write_count == writes
    assert f"datasets/evidence/{declared_rights.evidence_sha256}" in cos.bodies


def test_partner_batch_refuses_missing_or_mismatched_rights_evidence_before_cos(
    tmp_path: Path, allowed_policy, rights, cos
):
    image = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "red").save(image)
    manifest = tmp_path / "export.jsonl"
    manifest.write_text(json.dumps(_partner_row(image.name)) + "\n")
    evidence_path = tmp_path / "rights-evidence.json"
    evidence_path.write_text("{}")

    with pytest.raises(ValueError, match="rights evidence hash"):
        collect_partner_batch(
            manifest,
            evidence_path,
            allowed_policy,
            rights,
            cos_client=cos,
            bucket="private",
            batch_id="partner-batch-1",
            now=NOW,
        )

    assert cos.write_count == 0


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


def _png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def _collect_body(body, policy, rights, cos):
    http = MagicMock()
    http.get.return_value = _Response(body, "image/png")
    return collect_batch(
        [
            {
                "source_record_id": "record-1",
                "source_url": "https://partner.example/record-1",
                "image_url": "https://partner.example/image-1.png",
                "declared_reference": "rolex-submariner-126610ln",
            }
        ],
        policy,
        rights,
        access=_access(rights),
        http_client=http,
        cos_client=cos,
        bucket="private",
        batch_id="b1",
        now=NOW,
        dns_resolver=_public_dns,
    )


def _collect_url(image_url, http, policy, rights, cos, resolver):
    return collect_batch(
        [
            {
                "source_record_id": "record-1",
                "source_url": "https://partner.example/record-1",
                "image_url": image_url,
                "declared_reference": "rolex-submariner-126610ln",
            }
        ],
        policy,
        rights,
        access=_access(rights),
        http_client=http,
        cos_client=cos,
        bucket="private",
        batch_id="b1",
        now=NOW,
        dns_resolver=resolver,
    )


def _access(rights) -> SourceAccessRecord:
    return SourceAccessRecord(
        access_record_id="access-partner-test",
        source="partner-upload",
        acquisition_method="partner-export",
        status=SourceAccessStatus.ACTIVE,
        evidence_url="https://partner.example/access-evidence",
        evidence_sha256="9" * 64,
        verified_at=rights.effective_at,
        expires_at=rights.expires_at,
        allowed_image_hosts=("partner.example",),
        requests_per_minute=60,
        reason="test fixture",
    )


def _public_dns(*_args, **_kwargs):
    return [(2, 1, 6, "", ("93.184.216.34", 443))]


def _receipt_keys(cos, candidate_id: str) -> list[str]:
    prefix = f"datasets/receipts/b1/{candidate_id}-"
    return sorted(key for key in cos.bodies if key.startswith(prefix))
