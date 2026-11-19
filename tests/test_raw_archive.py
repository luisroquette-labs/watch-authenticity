import hashlib
import io
import json

import pytest
from PIL import Image

from acquisition import raw_archive


def _fixture(tmp_path):
    root = tmp_path / "aurum"
    triage = root / "triage"
    triage.mkdir(parents=True)
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(image, format="JPEG")
    photo = root / "watch.jpg"
    photo.write_bytes(image.getvalue())
    inventory = root / "inventory.csv"
    inventory.write_text("filename,sha256\n" + (f"watch.jpg,{hashlib.sha256(image.getvalue()).hexdigest()}\n" * 796))
    evidence_name = "owner_use_authorization_20261118.json"
    (root / evidence_name).write_text(json.dumps({
        "owner_authorized_restricted_project_use": True,
        "source": "Aurum Time VIP 4 BLACK FRIDAY 23/11",
        "scope": "796 existing media records archived from the group; restricted project use only",
        "applies_to_rows": 796,
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "training_admitted_now": False,
    }))
    manifest = triage / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "staged_not_uploaded",
        "source": "Aurum Time VIP 4 BLACK FRIDAY 23/11",
        "photo_count": 1,
        "raw_bytes": len(image.getvalue()),
        "photos": [{
            "filename": photo.name,
            "sha256": hashlib.sha256(image.getvalue()).hexdigest(),
            "bytes": len(image.getvalue()),
            "owner_use_evidence": evidence_name,
            "training_admitted": False,
        }],
    }))
    return manifest, photo


def test_raw_archive_uploads_original_and_evidence_last_manifest(tmp_path, monkeypatch):
    manifest, _ = _fixture(tmp_path)
    monkeypatch.setattr(raw_archive, "EXPECTED_AURUM_PHOTOS", 1)
    keys = []
    monkeypatch.setattr(raw_archive, "upload_verified", lambda client, path, bucket, key, digest: keys.append(key))
    result = raw_archive.archive_aurum_raw(manifest, cos_client=object(), bucket="test")
    assert result["photos"] == 1
    assert result["training_admitted"] is False
    assert keys[0].startswith("datasets/evidence/")
    assert keys[1].startswith("datasets/evidence/")
    assert keys[2].startswith("datasets/raw/")
    assert keys[3].startswith("datasets/raw-staging/")
    assert keys[4].startswith("datasets/raw-staging/receipts/")


def test_raw_archive_rejects_changed_photo_before_upload(tmp_path, monkeypatch):
    manifest, photo = _fixture(tmp_path)
    monkeypatch.setattr(raw_archive, "EXPECTED_AURUM_PHOTOS", 1)
    photo.write_bytes(b"altered")
    keys = []
    monkeypatch.setattr(raw_archive, "upload_verified", lambda *args: keys.append(args))
    with pytest.raises(ValueError, match="photo bytes differ"):
        raw_archive.archive_aurum_raw(manifest, cos_client=object(), bucket="test")
    assert keys == []


def test_raw_archive_dry_run_validates_without_upload(tmp_path, monkeypatch):
    manifest, _ = _fixture(tmp_path)
    monkeypatch.setattr(raw_archive, "EXPECTED_AURUM_PHOTOS", 1)
    monkeypatch.setattr(raw_archive, "upload_verified", lambda *args: pytest.fail("uploaded during dry run"))
    result = raw_archive.archive_aurum_raw(manifest, cos_client=None, bucket="test", dry_run=True)
    assert result["status"] == "validated_raw_only"
    assert result["photos"] == 1


def test_raw_archive_rejects_corrupt_jpeg_with_matching_hash(tmp_path, monkeypatch):
    manifest, photo = _fixture(tmp_path)
    monkeypatch.setattr(raw_archive, "EXPECTED_AURUM_PHOTOS", 1)
    damaged = photo.read_bytes()[:-20]
    photo.write_bytes(damaged)
    inventory = photo.parent / "inventory.csv"
    inventory.write_text("filename,sha256\n" + (f"watch.jpg,{hashlib.sha256(damaged).hexdigest()}\n" * 796))
    evidence_path = photo.parent / "owner_use_authorization_20261118.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["inventory_sha256"] = hashlib.sha256(inventory.read_bytes()).hexdigest()
    evidence_path.write_text(json.dumps(evidence))
    document = json.loads(manifest.read_text())
    document["raw_bytes"] = document["photos"][0]["bytes"] = len(damaged)
    document["photos"][0]["sha256"] = hashlib.sha256(damaged).hexdigest()
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(raw_archive, "upload_verified", lambda *args: pytest.fail("uploaded corrupt JPEG"))
    with pytest.raises((OSError, ValueError)):
        raw_archive.archive_aurum_raw(manifest, cos_client=object(), bucket="test")


def test_raw_archive_rejects_truncated_self_consistent_manifest(tmp_path):
    manifest, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="photo count"):
        raw_archive.archive_aurum_raw(manifest, cos_client=object(), bucket="test", dry_run=True)
