from __future__ import annotations

import io
import json
import stat

import pytest
from PIL import Image

from acquisition import supercloneluxury_archive as archive


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
