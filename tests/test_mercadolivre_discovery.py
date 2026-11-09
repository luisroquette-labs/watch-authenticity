from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.source_policy import get_source_policy, load_source_policies
from acquisition.types import SourceAccessRecord, SourceAccessStatus
from scrapers.mercadolivre import MercadoLivreAPIError, discover_mercadolivre

NOW = datetime(2026, 11, 9, 7, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, payload: object, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def _access() -> SourceAccessRecord:
    return SourceAccessRecord(
        access_record_id="access-ml-1",
        source="mercadolivre",
        acquisition_method="official-api",
        status=SourceAccessStatus.ACTIVE,
        evidence_url="https://developers.mercadolivre.com.br/itens-e-buscas",
        evidence_sha256="a" * 64,
        verified_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(days=1),
        allowed_image_hosts=("http2.mlstatic.com",),
        requests_per_minute=60,
        reason="test fixture",
    )


def _policy():
    return get_source_policy(
        load_source_policies(Path("config/source_policies.json")), "mercadolivre"
    )


def _search(offset: int, total: int, *ids: str) -> dict[str, object]:
    return {
        "results": [{"id": item_id} for item_id in ids],
        "paging": {"offset": offset, "limit": 20, "total": total},
    }


def _item(item_id: str, *, host: str = "http2.mlstatic.com") -> dict[str, object]:
    return {
        "id": item_id,
        "seller_id": 42,
        "permalink": f"https://produto.mercadolivre.com.br/{item_id}",
        "last_updated": "2026-11-09T06:00:00.000Z",
        "pictures": [
            {"id": f"{item_id}-1", "secure_url": f"https://{host}/{item_id}-1.jpg"},
            {"id": f"{item_id}-2", "secure_url": f"https://{host}/{item_id}-2.jpg"},
        ],
    }


def _discover(http, cos, **overrides):
    values = {
        "http_client": http,
        "policy": _policy(),
        "access": _access(),
        "cos_client": cos,
        "bucket": "private",
        "prefix": "datasets/discovery",
        "token": "test-token",
        "now": NOW,
    }
    values.update(overrides)
    return discover_mercadolivre(**values)


def test_official_api_pages_commit_in_order_and_preserve_picture_ids(cos):
    http = MagicMock()
    http.get.side_effect = [
        _Response(_search(0, 2, "MLB1")),
        _Response(_item("MLB1")),
        _Response(_search(1, 2, "MLB2")),
        _Response(_item("MLB2")),
    ]

    first = _discover(http, cos)
    second = _discover(http, cos, previous=first, now=NOW + timedelta(minutes=1))

    assert first.next_cursor == "1"
    assert second.previous_sha256 == first.sha256
    assert second.next_cursor is None
    page = json.loads(cos.bodies[f"datasets/discovery/mercadolivre/pages/{first.page_sha256}.json"])
    assert page["records"][0]["image_ids"] == ["MLB1-1", "MLB1-2"]
    assert page["records"][0]["source_record_id"] == "MLB1"
    assert page["records"][0]["seller_group_id"] == "42"
    assert page["records"][0]["image_urls"] == [
        "https://http2.mlstatic.com/MLB1-1.jpg",
        "https://http2.mlstatic.com/MLB1-2.jpg",
    ]
    assert all(call.args[0].startswith("https://api.mercadolibre.com/") for call in http.get.call_args_list)
    assert http.get.call_args_list[0].kwargs["params"]["q"] == "Rolex Submariner 126610LN"


@pytest.mark.parametrize(
    "access", [None, replace(_access(), status=SourceAccessStatus.UNAVAILABLE)]
)
def test_missing_or_inactive_access_refuses_before_network(cos, access):
    http = MagicMock()
    with pytest.raises(PermissionError):
        _discover(http, cos, access=access)
    http.get.assert_not_called()
    assert cos.write_count == 0


def test_missing_token_refuses_before_network(cos):
    http = MagicMock()
    with pytest.raises(PermissionError, match="token"):
        _discover(http, cos, token=" ")
    http.get.assert_not_called()
    assert cos.write_count == 0


def test_rate_limit_does_not_commit_and_can_resume_later(cos):
    http = MagicMock()
    http.get.return_value = _Response({}, status_code=429)
    with pytest.raises(MercadoLivreAPIError, match="resume later"):
        _discover(http, cos)
    assert cos.write_count == 0


@pytest.mark.parametrize(
    "responses, message",
    [
        ([_Response({"results": "bad", "paging": {}})], "results"),
        ([_Response(_search(0, 1, "../token"))], "item ID"),
        ([_Response(_search(0, 1, "MLB1")), _Response(_item("MLB1", host="evil.example"))], "host"),
    ],
)
def test_invalid_api_page_never_commits(cos, responses, message):
    http = MagicMock()
    http.get.side_effect = responses
    with pytest.raises(MercadoLivreAPIError, match=message):
        _discover(http, cos)
    assert cos.write_count == 0
