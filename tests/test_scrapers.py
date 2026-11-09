"""Source adapters normalize only pre-fetched, policy-approved metadata."""

from unittest.mock import MagicMock

import pytest

from acquisition.collect import SourceUnavailable
from scrapers.chinese_replica import chinese_replica_ingest
from scrapers.chrono24 import chrono24_ingest
from scrapers.mercadolivre import mercadolivre_ingest
from scrapers.olx import olx_ingest
from scrapers.reddit import reddit_ingest
from scrapers.reptime import reptime_ingest
from scrapers.watchuseek import watchuseek_ingest


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "source_record_id": "listing-1",
        "source_url": "https://source.example/listing-1",
        "image_url": "https://source.example/image-1.jpg",
        "declared_reference": "rolex-submariner-126610ln",
        "listing_text": "Rolex Submariner replica",
        "price": "120.00",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "adapter",
    [
        olx_ingest,
        chrono24_ingest,
        reptime_ingest,
        watchuseek_ingest,
        reddit_ingest,
        chinese_replica_ingest,
    ],
)
def test_adapter_without_approved_input_returns_typed_unavailable(adapter):
    assert isinstance(adapter(), SourceUnavailable)


@pytest.mark.parametrize(
    "adapter",
    [
        olx_ingest,
        chrono24_ingest,
        reptime_ingest,
        watchuseek_ingest,
        reddit_ingest,
        chinese_replica_ingest,
    ],
)
def test_unapproved_adapter_does_not_accept_supplied_records(adapter):
    records = MagicMock()

    assert isinstance(adapter(records), SourceUnavailable)
    records.__iter__.assert_not_called()


def test_mercadolivre_scrape_refuses_before_reading_records():
    records = MagicMock()

    with pytest.raises(PermissionError):
        mercadolivre_ingest("scrape", records)

    records.__iter__.assert_not_called()


def test_mercadolivre_requires_prefetched_official_api_records():
    assert isinstance(mercadolivre_ingest("official_api"), SourceUnavailable)
    [candidate] = mercadolivre_ingest("official_api", [_record()])
    assert candidate.source == "mercadolivre"
