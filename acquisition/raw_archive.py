"""Archive owner-provided original photos before dataset admission."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any

from PIL import Image

from acquisition.collect import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from infra.ibm_cos import upload_verified

EXPECTED_AURUM_PHOTOS = 654


def archive_aurum_raw(manifest_path: Path, *, cos_client: Any, bucket: str, dry_run: bool = False) -> dict[str, object]:
    """Verify a complete Aurum staging set, then upload originals and evidence.

    This archive has no derivative or training receipt. Admission stays a later,
    separately reviewed operation.
    """
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("staging manifest must be a regular file")
    root = manifest_path.parent.parent.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "staged_not_uploaded" or manifest.get("source") != "Aurum Time VIP 4 BLACK FRIDAY 23/11":
        raise ValueError("unexpected Aurum staging manifest")
    photos = manifest.get("photos")
    if not isinstance(photos, list) or len(photos) != EXPECTED_AURUM_PHOTOS or len(photos) != manifest.get("photo_count"):
        raise ValueError("photo count differs from manifest")
    evidence_name = "owner_use_authorization_20261118.json"
    evidence_path = _regular_child(root, evidence_name)
    evidence = json.loads(evidence_path.read_bytes())
    if evidence.get("owner_authorized_restricted_project_use") is not True:
        raise ValueError("restricted project use is not authorized")
    if evidence.get("source") != manifest["source"]:
        raise ValueError("owner evidence belongs to another source")
    if (evidence.get("training_admitted_now") is not False
        or evidence.get("applies_to_rows") != 796
        or "restricted project use only" not in str(evidence.get("scope", ""))):
        raise ValueError("owner evidence does not cover restricted raw staging")
    inventory_path = _regular_child(root, "inventory.csv")
    inventory_bytes = inventory_path.read_bytes()
    inventory_hash = hashlib.sha256(inventory_bytes).hexdigest()
    if inventory_hash != evidence.get("inventory_sha256"):
        raise ValueError("owner evidence is not bound to this inventory")
    inventory_rows = list(csv.DictReader(io.StringIO(inventory_bytes.decode("utf-8"))))
    if len(inventory_rows) != evidence["applies_to_rows"]:
        raise ValueError("inventory row count differs from owner evidence")
    inventoried = {(row["filename"], row["sha256"]) for row in inventory_rows}
    manifest_names = {row.get("filename") for row in photos if isinstance(row, dict)}
    if {path.name for path in root.glob("*.jpg")} != manifest_names:
        raise ValueError("Aurum JPG inventory differs from staging manifest")

    checked: list[tuple[Path, str]] = []
    seen: set[str] = set()
    total_bytes = 0
    for row in photos:
        if not isinstance(row, dict):
            raise ValueError("photo row must be an object")
        name = row.get("filename")
        digest = row.get("sha256")
        if not isinstance(name, str) or name in seen:
            raise ValueError("duplicate or invalid photo filename")
        seen.add(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid photo SHA-256")
        if (name, digest) not in inventoried:
            raise ValueError("photo is missing from owner-authorized inventory")
        if row.get("owner_use_evidence") != evidence_name or row.get("training_admitted") is not False:
            raise ValueError("photo lacks restricted-use evidence or is marked admitted")
        path = _regular_child(root, name)
        data = path.read_bytes()
        if len(data) > MAX_IMAGE_BYTES or len(data) != row.get("bytes") or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"photo bytes differ from manifest: {name}")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "JPEG" or getattr(image, "n_frames", 1) != 1 or image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(f"photo is not a safe single JPEG: {name}")
                image.verify()
            with Image.open(io.BytesIO(data)) as decoded:
                decoded.load()
        total_bytes += len(data)
        checked.append((path, digest))
    if total_bytes != manifest.get("raw_bytes"):
        raise ValueError("raw byte total differs from manifest")

    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if dry_run:
        return {"status": "validated_raw_only", "photos": len(checked), "raw_bytes": total_bytes, "manifest_sha256": manifest_hash, "training_admitted": False}

    upload_verified(cos_client, evidence_path, bucket, f"datasets/evidence/{evidence_hash}", evidence_hash)
    upload_verified(cos_client, inventory_path, bucket, f"datasets/evidence/{inventory_hash}", inventory_hash)
    for path, digest in checked:
        upload_verified(cos_client, path, bucket, f"datasets/raw/{digest}", digest)
    upload_verified(cos_client, manifest_path, bucket, f"datasets/raw-staging/{manifest_hash}", manifest_hash)
    receipt = {
        "status": "archived_raw_only",
        "source": manifest["source"],
        "photos": len(checked),
        "raw_bytes": total_bytes,
        "manifest_sha256": manifest_hash,
        "owner_evidence_sha256": evidence_hash,
        "inventory_sha256": inventory_hash,
        "training_admitted": False,
    }
    receipt_bytes = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    with tempfile.TemporaryDirectory() as temporary:
        receipt_path = Path(temporary) / "receipt.json"
        receipt_path.write_bytes(receipt_bytes)
        upload_verified(cos_client, receipt_path, bucket, f"datasets/raw-staging/receipts/{receipt_hash}", receipt_hash)
    return receipt


def _regular_child(root: Path, name: str) -> Path:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("archive filename must be a basename")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"archive file must be regular and non-symlink: {name}")
    return path
