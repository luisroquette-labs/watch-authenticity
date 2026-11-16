"""Resumable private archive of a vendor-declared replica catalog."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import requests
from PIL import Image, UnidentifiedImageError

API = "https://supercloneluxury.com/wp-json/wc/store/v1/products"
HOST = "supercloneluxury.com"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 100_000_000
IMAGE_TYPES = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
SELLER_TERMS = ("super clone", "replica", "fake", "clone")


class CatalogError(RuntimeError):
    pass


def _get(session: requests.Session, url: str, *, max_bytes: int) -> bytes:
    response = session.get(
        url,
        headers={"User-Agent": "watch-authenticity/0.1 (private research archive)"},
        timeout=(10, 30),
        allow_redirects=False,
        stream=True,
    )
    try:
        if response.status_code != 200:
            raise CatalogError(f"HTTP {response.status_code}: {url}")
        data = bytearray()
        for chunk in response.iter_content(64 * 1024):
            data.extend(chunk)
            if len(data) > max_bytes:
                raise CatalogError(f"response exceeds {max_bytes} bytes: {url}")
        return bytes(data)
    finally:
        response.close()


def _image_url(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.startswith("/wp-content/uploads/")
        or parsed.fragment
    ):
        raise CatalogError("product image URL is outside the public uploads host")
    return url


def _product(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or isinstance(raw.get("id"), bool) or not isinstance(raw.get("id"), int):
        raise CatalogError("product ID is missing")
    name, permalink, images = raw.get("name"), raw.get("permalink"), raw.get("images")
    if not isinstance(name, str) or not name or not isinstance(permalink, str):
        raise CatalogError("product name or permalink is missing")
    parsed = urlsplit(permalink)
    if parsed.scheme != "https" or parsed.hostname != HOST or not parsed.path.startswith("/product/"):
        raise CatalogError("product permalink is outside the catalog")
    if not isinstance(images, list):
        raise CatalogError("product image list is missing")
    urls = []
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("src"), str):
            raise CatalogError("product image URL is missing")
        urls.append(_image_url(image["src"]))
    urls = list(dict.fromkeys(urls))
    description = raw.get("description", "")
    if not isinstance(description, str):
        raise CatalogError("product description is invalid")
    declaration = name + " " + re.sub(r"<[^>]+>", " ", description)
    terms = [term for term in SELLER_TERMS if term in declaration.lower()]
    return {
        "source": HOST,
        "product_id": raw["id"],
        "name": name,
        "permalink": permalink,
        "sku": raw.get("sku") if isinstance(raw.get("sku"), str) else None,
        "seller_terms": terms,
        "search_tags": list(SELLER_TERMS) if terms else [],
        "source_label": "vendor_declared_replica" if terms else "unlabeled",
        "confirmed_fake": False,
        "quality_score_0_1000": None,
        "quality_score_status": "not_assessed",
        "image_urls": urls,
        "training_eligible": False,
        "source_product": raw,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        temporary.chmod(0o600)
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def discover_catalog(
    session: requests.Session, root: Path, *, refresh: bool = False, deadline: float | None = None
) -> list[dict[str, object]]:
    snapshot = root / "catalog.json"
    if snapshot.exists() and not refresh:
        saved_products = json.loads(snapshot.read_text())
        if not isinstance(saved_products, list):
            raise CatalogError("stored catalog is invalid")
        return saved_products
    products: dict[int, dict[str, object]] = {}
    staging = root / "discovery-pages"
    expected_total = None
    pages = None
    page = 1
    while pages is None or page <= pages:
        if deadline is not None and time.monotonic() >= deadline:
            raise CatalogError("discovery time budget exceeded")
        checkpoint = staging / f"{page:03d}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            total, total_pages, raw = saved["total"], saved["pages"], saved["products"]
        else:
            response = session.get(
                API,
                params={"per_page": 100, "page": page},
                headers={"User-Agent": "watch-authenticity/0.1 (private research archive)"},
                timeout=(10, 30),
                allow_redirects=False,
                stream=True,
            )
            try:
                if response.status_code != 200:
                    raise CatalogError(f"catalog page {page} returned HTTP {response.status_code}")
                total = int(response.headers["X-WP-Total"])
                total_pages = int(response.headers["X-WP-TotalPages"])
                data = bytearray()
                for chunk in response.iter_content(64 * 1024):
                    data.extend(chunk)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise CatalogError("catalog API page is too large")
                raw = json.loads(data)
            finally:
                response.close()
            _atomic_json(checkpoint, {"total": total, "pages": total_pages, "products": raw})
        if not isinstance(total, int) or not isinstance(total_pages, int) or total < 1 or not 1 <= total_pages <= 100:
            raise CatalogError("catalog page count is invalid")
        if expected_total is None:
            expected_total, pages = total, total_pages
        elif expected_total != total or pages != total_pages:
            shutil.rmtree(staging)
            raise CatalogError("catalog changed during discovery; restart the snapshot")
        if not isinstance(raw, list) or not raw:
            raise CatalogError("catalog page is empty or invalid")
        for item in raw:
            product = _product(item)
            product_id = cast(int, product["product_id"])
            if product_id in products:
                shutil.rmtree(staging)
                raise CatalogError("catalog pagination repeated a product")
            products[product_id] = product
        page += 1
    if len(products) != expected_total:
        shutil.rmtree(staging)
        raise CatalogError(f"catalog incomplete: {len(products)} of {expected_total}")
    ordered = [products[key] for key in sorted(products)]
    _atomic_json(snapshot, ordered)
    shutil.rmtree(staging)
    return ordered


def _archive_image(session: requests.Session, root: Path, url: str) -> bool:
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    receipt_path = root / "receipts" / f"{url_hash}.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        relative_path = Path(str(receipt["image_path"]))
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or ".." in relative_path.parts
            or relative_path.parts[0] != "images"
        ):
            raise CatalogError("stored receipt has an unsafe image path")
        image_path = root / relative_path
        if (
            receipt.get("image_url") == url
            and image_path.is_file()
            and image_path.stat().st_size == receipt.get("bytes")
            and hashlib.sha256(image_path.read_bytes()).hexdigest() == receipt.get("sha256")
        ):
            return False
    body = _get(session, _image_url(url), max_bytes=MAX_IMAGE_BYTES)
    try:
        with Image.open(io.BytesIO(body)) as image:
            width, height = image.size
            image_format = image.format
            if width * height > MAX_PIXELS or image_format not in IMAGE_TYPES:
                raise CatalogError("image dimensions or format are unsupported")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise CatalogError("downloaded image is invalid") from exc
    digest = hashlib.sha256(body).hexdigest()
    relative = Path("images") / digest[:2] / (digest + IMAGE_TYPES[image_format])
    image_path = root / relative
    if not image_path.exists() or hashlib.sha256(image_path.read_bytes()).hexdigest() != digest:
        image_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        image_path.parent.chmod(0o700)
        temporary = image_path.with_name(image_path.name + f".{os.getpid()}.tmp")
        with temporary.open("wb") as file:
            temporary.chmod(0o600)
            file.write(body)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, image_path)
    receipt = {
        "image_url": url,
        "sha256": digest,
        "bytes": len(body),
        "width": width,
        "height": height,
        "image_path": str(relative),
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(receipt_path, receipt)
    return True


def archive_catalog(
    root: Path, *, max_seconds: int = 3600, interval: float = 0.25, refresh: bool = False
) -> dict[str, object]:
    if max_seconds < 1 or interval < 0:
        raise ValueError("max_seconds and interval must be non-negative")
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root.chmod(0o700)
    result: dict[str, object] = {
        "source": HOST,
        "products": 0,
        "images_total": 0,
        "images_archived": 0,
        "state": "discovering",
        "quality_score_0_1000": None,
        "quality_score_status": "not_assessed",
        "training_eligible": False,
    }
    _atomic_json(root / "status.json", result)
    completed = 0
    try:
        with requests.Session() as session:
            products = discover_catalog(session, root, refresh=refresh, deadline=started + max_seconds)
            urls = tuple(dict.fromkeys(url for product in products for url in cast(list[str], product["image_urls"])))
            result.update(products=len(products), images_total=len(urls), state="downloading")
            _atomic_json(root / "status.json", result)
            for url in urls:
                if time.monotonic() - started >= max_seconds:
                    break
                downloaded = _archive_image(session, root, url)
                completed += 1
                if completed % 100 == 0:
                    result["images_archived"] = completed
                    _atomic_json(root / "status.json", result)
                if downloaded and interval:
                    time.sleep(interval)
    except Exception as exc:
        result.update(
            state="partial" if str(exc) == "discovery time budget exceeded" else "failed",
            images_archived=completed,
            error_type=type(exc).__name__,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_json(root / "status.json", result)
        raise
    result.update(
        images_archived=completed,
        state="complete" if completed == len(urls) else "partial",
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(root / "status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(".runtime/supercloneluxury"))
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(archive_catalog(args.root, max_seconds=args.max_seconds, refresh=args.refresh), ensure_ascii=False))


if __name__ == "__main__":
    main()
