"""Local weak-label inventory; provisional groups are not safe train/test splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentalSample:
    path: Path
    sha256: str
    label: str
    label_basis: str
    provisional_group: str
    source: str


def _verified_image(path: Path, digest: str) -> None:
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"image missing or changed: {path}")


def load_vendor_replica_samples(root: Path) -> tuple[ExperimentalSample, ...]:
    """Bind each distinct archived image to its seller claim and product group."""
    if json.loads((root / "status.json").read_text())["state"] != "complete":
        raise ValueError("replica archive is incomplete")
    catalog = json.loads((root / "catalog.json").read_text())
    if not isinstance(catalog, list):
        raise ValueError("replica catalog is invalid")
    by_hash: dict[str, tuple[Path, set[int]]] = {}
    parent: dict[int, int] = {}

    def find(product_id: int) -> int:
        while parent[product_id] != product_id:
            parent[product_id] = parent[parent[product_id]]
            product_id = parent[product_id]
        return product_id

    for product in catalog:
        if product.get("source_label") != "vendor_declared_replica":
            continue
        product_id = product["product_id"]
        if isinstance(product_id, bool) or not isinstance(product_id, int):
            raise ValueError("product ID is invalid")
        parent[product_id] = product_id
        for url in product["image_urls"]:
            receipt_path = root / "receipts" / f"{hashlib.sha256(url.encode()).hexdigest()}.json"
            receipt = json.loads(receipt_path.read_text())
            relative = Path(receipt["image_path"])
            if (
                receipt["image_url"] != url
                or relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != "images"
            ):
                raise ValueError("replica receipt does not match its catalog image")
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve() / "images"):
                raise ValueError("replica image path escapes archive")
            digest = receipt["sha256"]
            _verified_image(path, digest)
            if path.stat().st_size != receipt["bytes"]:
                raise ValueError("replica receipt byte count differs")
            if digest not in by_hash:
                by_hash[digest] = (path, set())
            by_hash[digest][1].add(product_id)

    # Shared bytes across listings keep those product groups in one future split.
    for _, products in by_hash.values():
        first = min(products)
        for product_id in products:
            parent[find(product_id)] = find(first)
    return tuple(
        ExperimentalSample(
            path=path,
            sha256=digest,
            label="fake",
            label_basis="vendor_declared_replica",
            provisional_group=f"supercloneluxury:{find(min(products))}",
            source="supercloneluxury.com",
        )
        for digest, (path, products) in sorted(by_hash.items())
    )


def load_owner_fake_samples(attestation: Path, downloads: Path) -> tuple[ExperimentalSample, ...]:
    """Use the owner's explicit fake label without inventing a reference or grade."""
    record = json.loads(attestation.read_text())
    if record.get("label") != "fake" or record.get("label_source") != "user-attested":
        raise ValueError("owner fake attestation is missing")
    group = record["watch_entity_id"]
    samples = []
    for photo in record["photos"]:
        name = photo["file_name"]
        if Path(name).name != name:
            raise ValueError("owner photo name is unsafe")
        path = downloads / name
        if not path.resolve().is_relative_to(downloads.resolve()):
            raise ValueError("owner photo path escapes downloads")
        digest = photo["sha256"]
        _verified_image(path, digest)
        samples.append(ExperimentalSample(path, digest, "fake", "user-attested", group, "owner"))
    if len({sample.sha256 for sample in samples}) != len(samples):
        raise ValueError("owner attestation repeats an image")
    return tuple(samples)


def load_cpo_claim_samples(root: Path) -> tuple[ExperimentalSample, ...]:
    """Load retailer claims as weak genuine labels, preserving each listing group."""
    samples = []
    for receipt_path in sorted((root / "receipts").glob("*.json")):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("label") != "genuine" or receipt.get("label_basis") != "cpo_seller_claim":
            raise ValueError("CPO receipt has an unsupported label")
        relative = Path(receipt["image_path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or relative.parts[0] != "images"
        ):
            raise ValueError("CPO receipt has an unsafe image path")
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve() / "images"):
            raise ValueError("CPO image path escapes archive")
        digest = receipt["sha256"]
        _verified_image(path, digest)
        samples.append(ExperimentalSample(
            path, digest, "genuine", "cpo_seller_claim",
            f"cpo:{receipt['source']}:{receipt['source_record_id']}", receipt["source"],
        ))
    if len({sample.sha256 for sample in samples}) != len(samples):
        raise ValueError("CPO archive repeats image bytes across listings")
    return tuple(samples)
