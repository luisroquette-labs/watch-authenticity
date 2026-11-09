from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from acquisition.discovery import (
    DiscoveryConflict,
    DiscoveryPage,
    DiscoveryRecord,
    append_discovery_page,
)

NOW = datetime(2026, 11, 9, 5, 0, tzinfo=timezone.utc)


def _record(record_id: str, *, image: str | None = None) -> DiscoveryRecord:
    return DiscoveryRecord(
        source="mercadolivre",
        source_record_id=record_id,
        source_version="v1",
        source_url=f"https://produto.mercadolivre.com.br/{record_id}",
        seller_group_id="seller-1",
        declared_reference="rolex-submariner-126610ln",
        image_ids=(f"picture-{record_id}",),
        image_urls=(image or f"https://http2.mlstatic.com/{record_id}.jpg",),
    )


def _page(now, records, *, cursor=None, next_cursor="page-2") -> DiscoveryPage:
    return DiscoveryPage(
        source="mercadolivre",
        access_record_id="access-ml-1",
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=now,
        records=tuple(records),
    )


def test_discovery_page_is_idempotent_and_input_order_independent(cos):
    first = _page(NOW, [_record("MLB-2"), _record("MLB-1")])
    reordered = _page(NOW, [_record("MLB-1"), _record("MLB-2")])

    commit = append_discovery_page(cos, "private", "datasets/discovery", first)
    writes = cos.write_count
    repeated = append_discovery_page(cos, "private", "datasets/discovery", reordered)

    assert repeated == commit
    assert cos.write_count == writes
    assert commit.next_cursor == "page-2"


def test_cursor_advances_only_after_commit_is_durable(cos):
    page = _page(NOW, [_record("MLB-1")])
    original_put = cos.put_object

    def fail_commit(**kwargs):
        if "/commits/" in kwargs["Key"]:
            raise RuntimeError("simulated commit outage")
        return original_put(**kwargs)

    cos.put_object = fail_commit
    with pytest.raises(RuntimeError, match="commit outage"):
        append_discovery_page(cos, "private", "datasets/discovery", page)
    cos.put_object = original_put

    commit = append_discovery_page(cos, "private", "datasets/discovery", page)
    assert commit.next_cursor == "page-2"


def test_discovery_refuses_cursor_fork_and_divergent_record(cos):
    first = append_discovery_page(
        cos,
        "private",
        "datasets/discovery",
        _page(NOW, [_record("MLB-1")]),
    )

    with pytest.raises(DiscoveryConflict, match="cursor"):
        append_discovery_page(
            cos,
            "private",
            "datasets/discovery",
            _page(NOW + timedelta(minutes=1), [_record("MLB-2")], cursor="wrong"),
            previous=first,
        )

    changed = _record("MLB-1", image="https://http2.mlstatic.com/changed.jpg")
    with pytest.raises(DiscoveryConflict, match="divergent"):
        append_discovery_page(
            cos,
            "private",
            "datasets/discovery",
            _page(NOW + timedelta(minutes=1), [changed], cursor="page-2"),
            previous=first,
        )


def test_discovery_allows_only_one_child_for_each_committed_cursor(cos):
    root = append_discovery_page(
        cos,
        "private",
        "datasets/discovery",
        _page(NOW, [_record("MLB-1")]),
    )
    append_discovery_page(
        cos,
        "private",
        "datasets/discovery",
        _page(NOW + timedelta(minutes=1), [_record("MLB-2")], cursor="page-2"),
        previous=root,
    )

    with pytest.raises(DiscoveryConflict, match="immutable discovery object diverged"):
        append_discovery_page(
            cos,
            "private",
            "datasets/discovery",
            _page(NOW + timedelta(minutes=1), [_record("MLB-3")], cursor="page-2"),
            previous=root,
        )


def test_discovery_rejects_forged_parent_hash(cos):
    root = append_discovery_page(
        cos,
        "private",
        "datasets/discovery",
        _page(NOW, [_record("MLB-1")]),
    )

    with pytest.raises(DiscoveryConflict, match="commit hash is invalid"):
        append_discovery_page(
            cos,
            "private",
            "datasets/discovery",
            _page(NOW + timedelta(minutes=1), [_record("MLB-2")], cursor="page-2"),
            previous=replace(root, sha256="f" * 64),
        )


def test_concurrent_children_leave_exactly_one_authoritative_commit(cos):
    root = append_discovery_page(
        cos,
        "private",
        "datasets/discovery",
        _page(NOW, [_record("MLB-1")]),
    )

    def append(record_id):
        try:
            return append_discovery_page(
                cos,
                "private",
                "datasets/discovery",
                _page(NOW + timedelta(minutes=1), [_record(record_id)], cursor="page-2"),
                previous=root,
            )
        except DiscoveryConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(append, ("MLB-2", "MLB-3")))

    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, DiscoveryConflict) for result in results) == 1
