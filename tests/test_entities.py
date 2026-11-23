from __future__ import annotations

import random
from dataclasses import replace
from itertools import combinations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import dataset.entities as entities_module
from acquisition.types import CandidateRecord
from core.types import Label
from dataset.entities import (
    EntityEvidence,
    assign_watch_entities,
    build_content_clusters,
    build_grouping_decision,
    find_near_duplicate_pairs,
    hamming_distance_hex,
)


def test_syndicated_listing_with_identical_bytes_shares_content_cluster(grouped_records):
    records = [grouped_records[0], replace(grouped_records[1], raw_sha256=grouped_records[0].raw_sha256)]
    clusters = build_content_clusters(records)

    assert clusters["front"] == clusters["repost"]


def test_identical_photo_in_different_listings_is_quarantined_even_with_same_label(
    grouped_records,
):
    records = [
        grouped_records[0],
        replace(
            grouped_records[1],
            raw_sha256=grouped_records[0].raw_sha256,
            source_url="https://partner.example/another-listing",
        ),
    ]
    clusters = build_content_clusters(records)
    entities = assign_watch_entities(records, clusters)

    assert clusters["front"] == clusters["repost"]
    assert entities["front"] != entities["repost"]
    assert all(build_grouping_decision(record, clusters, entities).quarantined for record in records)
    assert entities.conflicts[clusters["front"]].candidate_ids == ("front", "repost")


def test_distinct_listings_with_similar_phash_do_not_become_one_image(grouped_records):
    # Distinct catalog watches can have near-identical composition and pHash.
    records = [
        replace(grouped_records[0], candidate_id="sku-188633", acquisition_batch_id="batch-a", phash="bed0d503522f2f54"),
        replace(grouped_records[1], candidate_id="sku-188165", acquisition_batch_id="batch-b", phash="bed0d503522f2fd0"),
    ]
    clusters = build_content_clusters(records)
    entities = assign_watch_entities(records, clusters)

    assert clusters["sku-188633"] != clusters["sku-188165"]
    assert entities.split_families["sku-188633"] == entities.split_families["sku-188165"]


def test_near_phash_with_conflicting_labels_is_quarantined(grouped_records):
    records = [grouped_records[0], replace(grouped_records[1], proposed_label=Label.FAKE)]
    clusters = build_content_clusters(records)
    assignments = assign_watch_entities(records, clusters)

    assert clusters["front"] != clusters["repost"]
    assert build_grouping_decision(records[0], clusters, assignments).quarantined
    assert build_grouping_decision(records[1], clusters, assignments).quarantined


def test_uncertain_same_watch_is_grouped_not_split(grouped_records):
    clusters = build_content_clusters(grouped_records)
    evidence = {
        "front": EntityEvidence(listing_id="listing-1", seller_account_id="seller-1"),
        "clasp": EntityEvidence(listing_id="listing-1", seller_account_id="seller-1"),
    }

    entity_ids = assign_watch_entities(grouped_records, clusters, evidence)

    assert entity_ids["front"] == entity_ids["clasp"]
    front = next(record for record in grouped_records if record.candidate_id == "front")
    clasp = next(record for record in grouped_records if record.candidate_id == "clasp")
    assert build_grouping_decision(front, clusters, entity_ids).split_family_id == (
        build_grouping_decision(clasp, clusters, entity_ids).split_family_id
    )


def test_visual_similarity_alone_does_not_assert_physical_watch_identity(grouped_records):
    clusters = build_content_clusters(grouped_records)

    entity_ids = assign_watch_entities(grouped_records, clusters)

    assert clusters["front"] != clusters["repost"]
    assert entity_ids["front"] != entity_ids["repost"]
    assert entity_ids.split_families["front"] == entity_ids.split_families["repost"]


def test_reviewer_link_and_hashed_serial_can_join_angles(grouped_records):
    clusters = build_content_clusters(grouped_records)
    serial_hash = "9" * 64
    evidence = {
        "front": EntityEvidence(seller_account_id="seller-1", declared_serial_sha256=serial_hash),
        "clasp": EntityEvidence(
            seller_account_id="seller-1",
            declared_serial_sha256=serial_hash,
            reviewer_linked_candidate_ids=("front",),
        ),
    }

    entity_ids = assign_watch_entities(grouped_records, clusters, evidence)

    assert entity_ids["front"] == entity_ids["clasp"]


def test_reused_serial_across_sellers_stays_separate_watch_but_one_split(grouped_records):
    records = [
        replace(grouped_records[0], acquisition_batch_id="batch-a"),
        replace(grouped_records[2], acquisition_batch_id="batch-b"),
    ]
    clusters = build_content_clusters(records)
    serial_hash = "9" * 64
    evidence = {
        "front": EntityEvidence(seller_account_id="seller-a", declared_serial_sha256=serial_hash),
        "clasp": EntityEvidence(seller_account_id="seller-b", declared_serial_sha256=serial_hash),
    }

    assignments = assign_watch_entities(records, clusters, evidence)

    assert assignments["front"] != assignments["clasp"]
    assert assignments.split_families["front"] == assignments.split_families["clasp"]


def test_conflicting_labels_quarantine_entire_content_cluster(grouped_records):
    grouped_records[1] = replace(grouped_records[1], raw_sha256=grouped_records[0].raw_sha256)
    clusters = build_content_clusters(grouped_records)
    evidence = {
        "front": EntityEvidence(accepted_label=Label.GENUINE, evidence_record_id="evidence-genuine"),
        "repost": EntityEvidence(
            accepted_label=Label.FAKE,
            accepted_grade=4,
            evidence_record_id="evidence-fake",
        ),
    }

    entity_ids = assign_watch_entities(grouped_records, clusters, evidence)
    decision = build_grouping_decision(grouped_records[0], clusters, entity_ids)

    assert decision.quarantined
    assert decision.conflict_candidate_ids == ("front", "repost")
    assert decision.conflict_evidence_ids == ("evidence-fake", "evidence-genuine")


def test_grouping_decision_fails_closed_when_conflict_metadata_is_dropped(grouped_records):
    clusters = build_content_clusters(grouped_records)
    entity_ids = assign_watch_entities(grouped_records, clusters)

    with pytest.raises(ValueError, match="complete entity assignments are required"):
        build_grouping_decision(grouped_records[0], clusters, dict(entity_ids))


@settings(deadline=None)
@given(st.lists(st.integers(min_value=0, max_value=2**64 - 1), min_size=1, max_size=30, unique=True))
def test_indexed_near_duplicate_search_matches_brute_force(values):
    hashes = [f"{value:016x}" for value in values]
    expected = {
        pair
        for pair in combinations(range(len(hashes)), 2)
        if hamming_distance_hex(hashes[pair[0]], hashes[pair[1]])
        <= entities_module.HAMMING_DEDUP_THRESHOLD
    }

    assert find_near_duplicate_pairs(hashes) == expected


def test_ten_thousand_hash_index_avoids_quadratic_distance_calls(monkeypatch):
    rng = random.Random(20261109)
    hashes = [f"{rng.getrandbits(64):016x}" for _ in range(10_000)]
    calls = 0
    real_distance = entities_module.hamming_distance_hex

    def count_distance(left: str, right: str) -> int:
        nonlocal calls
        calls += 1
        return real_distance(left, right)

    monkeypatch.setattr(entities_module, "hamming_distance_hex", count_distance)

    find_near_duplicate_pairs(hashes)

    all_pairs = len(hashes) * (len(hashes) - 1) // 2
    assert calls < all_pairs * 0.05


def test_dense_shared_block_does_not_expand_to_all_pairs(monkeypatch):
    rng = random.Random(20261110)
    hashes = [f"abcd{rng.getrandbits(48):012x}" for _ in range(1_000)]
    calls = 0
    real_distance = entities_module.hamming_distance_hex

    def count_distance(left: str, right: str) -> int:
        nonlocal calls
        calls += 1
        return real_distance(left, right)

    monkeypatch.setattr(entities_module, "hamming_distance_hex", count_distance)

    find_near_duplicate_pairs(hashes)

    all_pairs = len(hashes) * (len(hashes) - 1) // 2
    assert calls < all_pairs * 0.05


def test_grouping_is_deterministic_when_input_order_changes(grouped_records: list[CandidateRecord]):
    forward = build_content_clusters(grouped_records)
    reverse = build_content_clusters(list(reversed(grouped_records)))

    assert forward == reverse
