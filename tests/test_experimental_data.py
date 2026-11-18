from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.experimental_data import (
    load_cpo_claim_samples,
    load_owner_fake_samples,
    load_vendor_replica_samples,
)


def _image(root: Path, url: str, body: bytes) -> str:
    digest = hashlib.sha256(body).hexdigest()
    path = root / "images" / digest[:2] / f"{digest}.webp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    receipt = root / "receipts" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({
        "image_url": url, "image_path": str(path.relative_to(root)),
        "sha256": digest, "bytes": len(body),
    }))
    return digest


def test_vendor_images_keep_weak_label_and_shared_product_group(tmp_path: Path) -> None:
    one, two = "https://example.com/one.webp", "https://example.com/two.webp"
    _image(tmp_path, one, b"one")
    _image(tmp_path, two, b"two")
    (tmp_path / "status.json").write_text('{"state":"complete"}')
    (tmp_path / "catalog.json").write_text(json.dumps([
        {"product_id": 1, "source_label": "vendor_declared_replica", "image_urls": [one]},
        {"product_id": 2, "source_label": "vendor_declared_replica", "image_urls": [one, two]},
    ]))

    samples = load_vendor_replica_samples(tmp_path)

    assert len(samples) == 2
    assert {sample.label_basis for sample in samples} == {"vendor_declared_replica"}
    assert {sample.provisional_group for sample in samples} == {"supercloneluxury:1"}
    (tmp_path / "images" / samples[0].sha256[:2] / f"{samples[0].sha256}.webp").write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_vendor_replica_samples(tmp_path)


def test_reencoded_repost_is_not_mistaken_for_confirmed_watch_identity(tmp_path: Path) -> None:
    first, repost = "https://example.com/first.webp", "https://example.com/repost.webp"
    _image(tmp_path, first, b"watch photo")
    _image(tmp_path, repost, b"same watch, recoded pixels")
    (tmp_path / "status.json").write_text('{"state":"complete"}')
    (tmp_path / "catalog.json").write_text(json.dumps([
        {"product_id": 1, "source_label": "vendor_declared_replica", "image_urls": [first]},
        {"product_id": 2, "source_label": "vendor_declared_replica", "image_urls": [repost]},
    ]))
    samples = load_vendor_replica_samples(tmp_path)
    assert {sample.provisional_group for sample in samples} == {
        "supercloneluxury:1", "supercloneluxury:2",
    }


def test_owner_label_keeps_one_provisional_watch_and_refuses_changed_bytes(tmp_path: Path) -> None:
    image = tmp_path / "watch.jpg"
    image.write_bytes(b"user-photo")
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({
        "label": "fake", "label_source": "user-attested", "watch_entity_id": "one-watch",
        "photos": [{"file_name": image.name, "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}],
    }))
    samples = load_owner_fake_samples(attestation, tmp_path)
    assert len(samples) == 1
    assert (samples[0].label, samples[0].label_basis, samples[0].provisional_group) == (
        "fake", "user-attested", "one-watch"
    )
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_owner_fake_samples(attestation, tmp_path)


def test_owner_label_rejects_symlink_outside_downloads(tmp_path: Path) -> None:
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    (downloads / "watch.png").symlink_to(outside)
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps({
        "label": "fake", "label_source": "user-attested", "watch_entity_id": "one-watch",
        "photos": [{"file_name": "watch.png", "sha256": hashlib.sha256(outside.read_bytes()).hexdigest()}],
    }))
    with pytest.raises(ValueError, match="escapes downloads"):
        load_owner_fake_samples(attestation, downloads)


def test_cpo_claim_keeps_weak_label_and_listing_group(tmp_path: Path) -> None:
    image = tmp_path / "images" / "watch.png"
    image.parent.mkdir()
    image.write_bytes(b"retailer-photo")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    receipt = tmp_path / "receipts" / "one.json"
    receipt.parent.mkdir()
    receipt.write_text(json.dumps({
        "label": "genuine", "label_basis": "cpo_seller_claim",
        "source": "retailer", "source_record_id": "listing-1",
        "image_path": "images/watch.png", "sha256": digest,
    }))

    samples = load_cpo_claim_samples(tmp_path)
    assert len(samples) == 1
    assert (samples[0].label, samples[0].label_basis, samples[0].provisional_group) == (
        "genuine", "cpo_seller_claim", "cpo:retailer:listing-1",
    )
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        load_cpo_claim_samples(tmp_path)
