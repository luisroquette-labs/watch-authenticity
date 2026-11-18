"""Conservative listing census for the private Super Clone Luxury archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from acquisition.supercloneluxury_archive import CatalogError, _atomic_json

BRAND_TERMS = (
    "audemars piguet", "breitling", "cartier", "iwc", "omega", "panerai",
    "patek philippe", "richard mille", "rolex", "tag heuer", "tudor",
)


def census_archive(root: Path) -> dict[str, object]:
    """Count listings and exact-image links; never infer physical-watch identity."""
    status = json.loads((root / "status.json").read_text())
    if status.get("state") != "complete":
        raise CatalogError("archive is not complete")
    products = json.loads((root / "catalog.json").read_text())
    if not isinstance(products, list) or not products:
        raise CatalogError("catalog is empty or invalid")
    receipt_by_url: dict[str, str] = {}
    for path in (root / "receipts").glob("*.json"):
        receipt = json.loads(path.read_text())
        url, digest = receipt.get("image_url"), receipt.get("sha256")
        if not isinstance(url, str) or not isinstance(digest, str):
            raise CatalogError("receipt URL or hash is invalid")
        if hashlib.sha256(url.encode()).hexdigest() != path.stem:
            raise CatalogError("receipt filename does not match its URL")
        receipt_by_url[url] = digest

    ids: set[int] = set()
    parent: dict[int, int] = {}
    hash_owners: dict[str, set[int]] = defaultdict(set)
    brand_mentions: Counter[str] = Counter()
    urls: set[str] = set()
    for product in products:
        product_id = product.get("product_id")
        name, image_urls = product.get("name"), product.get("image_urls")
        if (not isinstance(product_id, int) or isinstance(product_id, bool)
                or product_id in ids or not isinstance(name, str)
                or not isinstance(image_urls, list)
                or product.get("training_eligible") is not False
                or product.get("confirmed_fake") is not False):
            raise CatalogError("catalog product identity or images are invalid")
        ids.add(product_id)
        parent[product_id] = product_id
        matched = [term for term in BRAND_TERMS if term in name.lower()]
        brand_mentions[matched[0] if len(matched) == 1 else "unknown_or_multiple"] += 1
        for url in image_urls:
            if not isinstance(url, str) or url not in receipt_by_url:
                raise CatalogError("catalog image has no receipt")
            urls.add(url)
            hash_owners[receipt_by_url[url]].add(product_id)

    def find(product_id: int) -> int:
        while parent[product_id] != product_id:
            parent[product_id] = parent[parent[product_id]]
            product_id = parent[product_id]
        return product_id

    for owners in hash_owners.values():
        first, *rest = sorted(owners)
        for other in rest:
            parent[find(other)] = find(first)
    sizes = Counter(find(product_id) for product_id in ids)
    linked = sum(size for size in sizes.values() if size > 1)
    return {
        "source": "supercloneluxury.com",
        "catalog_products": len(ids),
        "image_urls": len(urls),
        "unique_image_hashes": len(hash_owners),
        "hashes_shared_between_products": sum(len(owners) > 1 for owners in hash_owners.values()),
        "exact_hash_linked_listing_groups": len(sizes),
        "listings_in_shared_hash_groups": linked,
        "largest_shared_hash_group": max(sizes.values()),
        "brand_name_mentions": dict(sorted(brand_mentions.items())),
        "physical_watches_confirmed": 0,
        "training_eligible_images": 0,
        "census_limit": "Exact shared image hashes link listings; groups are not verified physical watches or authenticity labels.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    report = census_archive(args.root)
    _atomic_json(args.root / "census.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
