from __future__ import annotations

import hashlib
import io
import json
import stat
from typing import cast

import pytest
import requests
from PIL import Image

from acquisition import supercloneluxury_archive as archive
from acquisition.supercloneluxury_census import census_archive


class Response:
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.content = body
        self.headers = headers or {}
        self.status_code = 200

    def iter_content(self, _size: int):
        yield self.content

    def close(self) -> None:
        pass


class Session:
    def __init__(self, image: bytes) -> None:
        self.image = image
        self.calls: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def get(self, url: str, **kwargs):
        self.calls.append(url)
        assert kwargs["allow_redirects"] is False
        if url == archive.API:
            assert kwargs["params"] == {"per_page": 100, "page": 1}
            products = [{
                "id": 1,
                "name": "Rolex Submariner Super Clone",
                "permalink": "https://supercloneluxury.com/product/rolex-submariner-super-clone/",
                "sku": "SCL-1",
                "description": "<p>Replica advertised by seller</p>",
                "images": [
                    {"src": "https://supercloneluxury.com/wp-content/uploads/watch.webp"},
                    {"src": "https://supercloneluxury.com/wp-content/uploads/watch.webp"},
                ],
            }]
            return Response(json.dumps(products).encode(), {"X-WP-Total": "1", "X-WP-TotalPages": "1"})
        assert url == "https://supercloneluxury.com/wp-content/uploads/watch.webp"
        return Response(self.image)


def test_archive_is_resumable_and_never_assigns_quality_from_sales_claim(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "green").save(output, format="WEBP")
    session = Session(output.getvalue())
    monkeypatch.setattr(archive.requests, "Session", lambda: session)
    first = archive.archive_catalog(tmp_path, interval=0)
    assert first["state"] == "complete"
    assert (first["products"], first["images_total"]) == (1, 1)
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog[0]["source_label"] == "vendor_declared_replica"
    assert catalog[0]["search_tags"] == ["super clone", "replica", "fake", "clone"]
    assert catalog[0]["confirmed_fake"] is False
    assert catalog[0]["quality_score_0_1000"] is None
    assert catalog[0]["training_eligible"] is False
    assert catalog[0]["source_product"]["sku"] == "SCL-1"
    assert json.loads((tmp_path / "census.json").read_text())["catalog_products"] == 1
    receipt = next((tmp_path / "receipts").glob("*.json"))
    stored_image = tmp_path / json.loads(receipt.read_text())["image_path"]
    assert stored_image.read_bytes() == output.getvalue()
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(stored_image.stat().st_mode) == 0o600
    assert archive.archive_catalog(tmp_path, interval=0)["state"] == "complete"
    assert len(session.calls) == 2  # no duplicate page or image request
    assert archive.archive_catalog(tmp_path, interval=0, refresh=True)["state"] == "complete"
    assert len(session.calls) == 3  # refreshes metadata, not unchanged images
    stored_image.write_bytes(b"x" * len(output.getvalue()))
    assert archive.archive_catalog(tmp_path, interval=0)["state"] == "complete"
    assert stored_image.read_bytes() == output.getvalue()
    assert len(session.calls) == 4


def test_external_image_host_is_rejected() -> None:
    with pytest.raises(archive.CatalogError, match="outside"):
        archive._image_url("https://example.com/wp-content/uploads/watch.webp")


def test_census_links_exact_reposts_without_claiming_physical_watches(tmp_path) -> None:
    urls = [f"https://supercloneluxury.com/wp-content/uploads/{index}.webp" for index in range(3)]
    archive._atomic_json(tmp_path / "status.json", {"state": "complete"})
    archive._atomic_json(tmp_path / "catalog.json", [
        {"product_id": 1, "name": "Rolex watch", "image_urls": urls[:2], "training_eligible": False, "confirmed_fake": False},
        {"product_id": 2, "name": "Rolex repost", "image_urls": [urls[2]], "training_eligible": False, "confirmed_fake": False},
        {"product_id": 3, "name": "Omega watch", "image_urls": [], "training_eligible": False, "confirmed_fake": False},
    ])
    for url, digest in zip(urls, ("a", "b", "b")):
        archive._atomic_json(tmp_path / "receipts" / f"{hashlib.sha256(url.encode()).hexdigest()}.json", {"image_url": url, "sha256": digest})

    report = census_archive(tmp_path)

    assert report["catalog_products"] == 3
    assert report["image_urls"] == 3
    assert report["unique_image_hashes"] == 2
    assert report["exact_hash_linked_listing_groups"] == 2
    assert report["listings_in_shared_hash_groups"] == 2
    assert report["physical_watches_confirmed"] == 0
    assert report["training_eligible_images"] == 0


def test_census_refuses_incomplete_archive(tmp_path) -> None:
    archive._atomic_json(tmp_path / "status.json", {"state": "partial"})
    with pytest.raises(archive.CatalogError, match="not complete"):
        census_archive(tmp_path)


def test_census_accepts_valid_catalog_without_images(tmp_path) -> None:
    archive._atomic_json(tmp_path / "status.json", {"state": "complete"})
    archive._atomic_json(tmp_path / "catalog.json", [
        {"product_id": 1, "name": "Watch", "image_urls": [], "training_eligible": False, "confirmed_fake": False}
    ])

    report = census_archive(tmp_path)

    assert report["image_urls"] == 0
    assert report["unique_image_hashes"] == 0
    assert report["exact_hash_linked_listing_groups"] == 1
    assert report["largest_shared_hash_group"] == 1


def test_archive_records_census_failure_instead_of_claiming_complete(tmp_path, monkeypatch) -> None:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "green").save(output, format="WEBP")
    monkeypatch.setattr(archive.requests, "Session", lambda: Session(output.getvalue()))
    from acquisition import supercloneluxury_census

    def fail(_root):
        raise archive.CatalogError("census unavailable")

    monkeypatch.setattr(supercloneluxury_census, "census_archive", fail)
    with pytest.raises(archive.CatalogError, match="census unavailable"):
        archive.archive_catalog(tmp_path, interval=0)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "census_failed"
    assert not (tmp_path / "census.json").exists()


def test_product_without_replica_declaration_is_not_labeled_fake() -> None:
    product = archive._product({
        "id": 1,
        "name": "Watch",
        "permalink": "https://supercloneluxury.com/product/watch/",
        "description": "Watch photo",
        "images": [],
    })
    assert product["source_label"] == "unlabeled"
    assert product["quality_score_0_1000"] is None


def test_refresh_failure_never_leaves_old_complete_status(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "green").save(output, format="WEBP")
    session = Session(output.getvalue())
    monkeypatch.setattr(archive.requests, "Session", lambda: session)
    assert archive.archive_catalog(tmp_path, interval=0)["state"] == "complete"
    original_get = session.get

    def failing_get(url: str, **kwargs):
        if url == archive.API:
            response = original_get(url, **kwargs)
            products = json.loads(response.content)
            products[0]["images"] = [{"src": "https://supercloneluxury.com/wp-content/uploads/new.webp"}]
            return Response(json.dumps(products).encode(), response.headers)
        response = Response(b"")
        response.status_code = 404
        return response

    session.get = failing_get  # type: ignore[method-assign]
    with pytest.raises(archive.CatalogError, match="HTTP 404"):
        archive.archive_catalog(tmp_path, interval=0, refresh=True)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed"
    assert not (tmp_path / "census.json").exists()
    assert status["images_archived"] == 0
    assert json.loads((tmp_path / "catalog.json").read_text())[0]["image_urls"] == [
        "https://supercloneluxury.com/wp-content/uploads/new.webp"
    ]


def test_discovery_resumes_at_failed_page(tmp_path) -> None:
    class PageSession(Session):
        def __init__(self) -> None:
            super().__init__(b"")
            self.fail_second = True

        def get(self, url: str, **kwargs):
            page = kwargs["params"]["page"]
            self.calls.append(str(page))
            if page == 2 and self.fail_second:
                response = Response(b"")
                response.status_code = 503
                return response
            product = {
                "id": page,
                "name": "Super Clone",
                "permalink": f"https://supercloneluxury.com/product/watch-{page}/",
                "images": [],
            }
            return Response(
                json.dumps([product]).encode(),
                {"X-WP-Total": "2", "X-WP-TotalPages": "2"},
            )

    session = PageSession()
    with pytest.raises(archive.CatalogError, match="page 2"):
        archive.discover_catalog(cast(requests.Session, session), tmp_path, refresh=True)
    assert (tmp_path / "discovery-pages/001.json").exists()
    session.fail_second = False
    assert len(archive.discover_catalog(cast(requests.Session, session), tmp_path, refresh=True)) == 2
    assert session.calls == ["1", "2", "2"]


def test_invalid_200_page_is_not_checkpointed(tmp_path) -> None:
    class FlakySession(Session):
        def __init__(self) -> None:
            super().__init__(b"")

        def get(self, url: str, **kwargs):
            page = kwargs["params"]["page"]
            self.calls.append(str(page))
            product = {
                "id": page,
                "name": "Super Clone",
                "permalink": f"https://supercloneluxury.com/product/watch-{page}/",
                "images": [],
            }
            if page == 2 and self.calls.count("2") == 1:
                product.pop("name")
            return Response(
                json.dumps([product]).encode(),
                {"X-WP-Total": "2", "X-WP-TotalPages": "2"},
            )

    session = FlakySession()
    with pytest.raises(archive.CatalogError, match="name"):
        archive.discover_catalog(cast(requests.Session, session), tmp_path, refresh=True)
    assert (tmp_path / "discovery-pages/001.json").exists()
    assert not (tmp_path / "discovery-pages/002.json").exists()
    assert len(archive.discover_catalog(cast(requests.Session, session), tmp_path, refresh=True)) == 2
    assert session.calls == ["1", "2", "2"]
