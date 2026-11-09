"""Conservative content, physical-watch and split-family grouping."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from acquisition.types import CandidateRecord
from core.types import Label
from dataset.dedup import HAMMING_DEDUP_THRESHOLD

_PHASH_64 = re.compile(r"^[0-9a-f]{16}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PHASH_BITS = 64


@dataclass(frozen=True)
class EntityEvidence:
    listing_id: str | None = None
    seller_account_id: str | None = None
    declared_serial_sha256: str | None = None
    reviewer_linked_candidate_ids: tuple[str, ...] = ()
    accepted_label: Label | None = None
    accepted_grade: int | None = None
    evidence_record_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("listing_id", self.listing_id),
            ("seller_account_id", self.seller_account_id),
            ("evidence_record_id", self.evidence_record_id),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when present")
        if self.declared_serial_sha256 is not None and not _SHA256.fullmatch(
            self.declared_serial_sha256
        ):
            raise ValueError("declared_serial_sha256 must be lowercase SHA-256")
        if len(set(self.reviewer_linked_candidate_ids)) != len(
            self.reviewer_linked_candidate_ids
        ) or any(not item.strip() for item in self.reviewer_linked_candidate_ids):
            raise ValueError("reviewer links must be unique non-empty candidate IDs")
        if self.accepted_label is Label.GENUINE and self.accepted_grade is not None:
            raise ValueError("genuine evidence cannot have a grade")
        if self.accepted_grade is not None and self.accepted_label is not Label.FAKE:
            raise ValueError("accepted grade requires a fake label")


@dataclass(frozen=True)
class ClusterConflict:
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroupingDecision:
    candidate_id: str
    content_cluster_id: str
    watch_entity_id: str
    split_family_id: str
    quarantined: bool
    conflict_candidate_ids: tuple[str, ...] = ()
    conflict_evidence_ids: tuple[str, ...] = ()


class EntityAssignments(dict[str, str]):
    """Watch-entity dictionary carrying the separately computed split family."""

    def __init__(
        self,
        entities: Mapping[str, str],
        split_families: Mapping[str, str],
        conflicts: Mapping[str, ClusterConflict],
    ):
        super().__init__(entities)
        self.split_families = dict(split_families)
        self.conflicts = dict(conflicts)


class _UnionFind:
    def __init__(self, ids: Iterable[str]):
        self.parent = {item: item for item in ids}

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def hamming_distance_hex(left: str, right: str) -> int:
    _validate_phash(left)
    _validate_phash(right)
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_near_duplicate_pairs(phashes: Sequence[str]) -> set[tuple[int, int]]:
    """Find every pHash pair within the threshold without an all-pairs scan."""
    values = [int(_validate_phash(value), 16) for value in phashes]
    block_count = HAMMING_DEDUP_THRESHOLD + 1
    base, remainder = divmod(_PHASH_BITS, block_count)
    block_lengths = [base + (index < remainder) for index in range(block_count)]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()

    for index, value in enumerate(values):
        remaining = _PHASH_BITS
        for block_index, length in enumerate(block_lengths):
            remaining -= length
            block_value = (value >> remaining) & ((1 << length) - 1)
            key = (block_index, block_value)
            candidates.update((prior, index) for prior in buckets[key])
            buckets[key].append(index)

    # The threshold+1 index is complete by the pigeonhole rule, but its
    # 5-6 bit blocks collide often. A second complete filter splits into four
    # 16-bit blocks: distance <= 10 guarantees at least one block differs by
    # at most two bits. This keeps full-distance calls well below all-pairs.
    refined_buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    refined: set[tuple[int, int]] = set()
    for index, value in enumerate(values):
        blocks = tuple((value >> shift) & 0xFFFF for shift in (48, 32, 16, 0))
        for block_index, block_value in enumerate(blocks):
            for nearby in _within_two_bits(block_value):
                refined.update(
                    (prior, index) for prior in refined_buckets[(block_index, nearby)]
                )
            refined_buckets[(block_index, block_value)].append(index)

    candidates.intersection_update(refined)

    return {
        pair
        for pair in candidates
        if hamming_distance_hex(phashes[pair[0]], phashes[pair[1]])
        <= HAMMING_DEDUP_THRESHOLD
    }


def _within_two_bits(value: int) -> Iterable[int]:
    yield value
    for first in range(16):
        one_bit = value ^ (1 << first)
        yield one_bit
        for second in range(first + 1, 16):
            yield one_bit ^ (1 << second)


def build_content_clusters(records: Sequence[CandidateRecord]) -> dict[str, str]:
    _validate_records(records)
    union = _UnionFind(record.candidate_id for record in records)
    by_raw: dict[str, str] = {}
    by_phash: dict[str, str] = {}
    for record in records:
        prior = by_raw.setdefault(record.raw_sha256, record.candidate_id)
        union.union(prior, record.candidate_id)
        prior_phash = by_phash.setdefault(record.phash, record.candidate_id)
        union.union(prior_phash, record.candidate_id)

    unique_phashes = sorted(by_phash)
    for left, right in find_near_duplicate_pairs(unique_phashes):
        union.union(by_phash[unique_phashes[left]], by_phash[unique_phashes[right]])

    groups = _groups(union, (record.candidate_id for record in records))
    ids = {
        member: _stable_id("content", groups[root])
        for root, members in groups.items()
        for member in members
    }
    return ids


def assign_watch_entities(
    records: Sequence[CandidateRecord],
    content_clusters: Mapping[str, str],
    evidence: Mapping[str, EntityEvidence] | None = None,
) -> EntityAssignments:
    _validate_records(records)
    evidence = evidence or {}
    _validate_mapping_keys(records, content_clusters, "content clusters")
    _validate_evidence(records, evidence)
    candidate_ids = [record.candidate_id for record in records]
    entities = _UnionFind(candidate_ids)

    _union_shared_value(entities, records, lambda record: content_clusters[record.candidate_id])
    _union_shared_value(
        entities,
        records,
        lambda record: (
            record.source_policy_id,
            evidence.get(record.candidate_id, EntityEvidence()).listing_id
            or record.source_record_id,
        ),
    )
    _union_shared_value(
        entities,
        records,
        lambda record: _seller_serial_key(record.candidate_id, evidence),
        ignore_none=True,
    )
    for candidate_id, item in evidence.items():
        for linked_id in item.reviewer_linked_candidate_ids:
            entities.union(candidate_id, linked_id)

    entity_groups = _groups(entities, candidate_ids)
    entity_ids = {
        member: _stable_id("watch", members)
        for members in entity_groups.values()
        for member in members
    }

    split_union = _UnionFind(candidate_ids)
    _union_shared_value(split_union, records, lambda record: entity_ids[record.candidate_id])
    _union_shared_value(split_union, records, lambda record: record.acquisition_batch_id)
    _union_shared_value(
        split_union,
        records,
        lambda record: evidence.get(record.candidate_id, EntityEvidence()).seller_account_id,
        ignore_none=True,
    )
    split_groups = _groups(split_union, candidate_ids)
    split_ids = {
        member: _stable_id("split", members)
        for members in split_groups.values()
        for member in members
    }
    conflicts = detect_cluster_conflicts(records, content_clusters, evidence)
    return EntityAssignments(entity_ids, split_ids, conflicts)


def detect_cluster_conflicts(
    records: Sequence[CandidateRecord],
    content_clusters: Mapping[str, str],
    evidence: Mapping[str, EntityEvidence] | None = None,
) -> dict[str, ClusterConflict]:
    evidence = evidence or {}
    _validate_records(records)
    _validate_mapping_keys(records, content_clusters, "content clusters")
    _validate_evidence(records, evidence)
    by_cluster: dict[str, list[CandidateRecord]] = defaultdict(list)
    for record in records:
        by_cluster[content_clusters[record.candidate_id]].append(record)

    conflicts: dict[str, ClusterConflict] = {}
    for cluster_id, members in by_cluster.items():
        labels: set[tuple[Label, int | None]] = set()
        conflict_candidates: list[str] = []
        conflict_evidence: list[str] = []
        for record in members:
            item = evidence.get(record.candidate_id)
            label = item.accepted_label if item and item.accepted_label is not None else record.proposed_label
            grade = item.accepted_grade if item and item.accepted_label is not None else record.proposed_grade
            if label is not None:
                labels.add((label, grade))
                conflict_candidates.append(record.candidate_id)
                if item and item.evidence_record_id:
                    conflict_evidence.append(item.evidence_record_id)
        if len(labels) > 1:
            conflicts[cluster_id] = ClusterConflict(
                candidate_ids=tuple(sorted(conflict_candidates)),
                evidence_ids=tuple(sorted(set(conflict_evidence))),
            )
    return conflicts


def build_grouping_decision(
    candidate: CandidateRecord,
    clusters: Mapping[str, str],
    entities: Mapping[str, str],
    evidence: Mapping[str, EntityEvidence] | None = None,
    conflicts: Mapping[str, ClusterConflict] | None = None,
) -> GroupingDecision:
    del evidence  # evidence is consumed while assigning entities and detecting conflicts.
    cluster_id = clusters[candidate.candidate_id]
    conflict_map = conflicts if conflicts is not None else getattr(entities, "conflicts", {})
    conflict = conflict_map.get(cluster_id)
    split_families = getattr(entities, "split_families", {})
    split_family_id = split_families.get(
        candidate.candidate_id,
        _stable_id("split", (candidate.acquisition_batch_id, entities[candidate.candidate_id])),
    )
    return GroupingDecision(
        candidate_id=candidate.candidate_id,
        content_cluster_id=cluster_id,
        watch_entity_id=entities[candidate.candidate_id],
        split_family_id=split_family_id,
        quarantined=conflict is not None,
        conflict_candidate_ids=() if conflict is None else conflict.candidate_ids,
        conflict_evidence_ids=() if conflict is None else conflict.evidence_ids,
    )


def _validate_phash(value: str) -> str:
    if not isinstance(value, str) or _PHASH_64.fullmatch(value) is None:
        raise ValueError("pHash must be exactly 64-bit lowercase hexadecimal")
    return value


def _validate_records(records: Sequence[CandidateRecord]) -> None:
    ids = [record.candidate_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate IDs must be unique")
    for record in records:
        _validate_phash(record.phash)


def _validate_mapping_keys(
    records: Sequence[CandidateRecord], mapping: Mapping[str, object], name: str
) -> None:
    expected = {record.candidate_id for record in records}
    if set(mapping) != expected:
        raise ValueError(f"{name} must contain exactly the input candidate IDs")


def _validate_evidence(
    records: Sequence[CandidateRecord], evidence: Mapping[str, EntityEvidence]
) -> None:
    ids = {record.candidate_id for record in records}
    if not set(evidence).issubset(ids):
        raise ValueError("entity evidence references an unknown candidate")
    for candidate_id, item in evidence.items():
        if not isinstance(item, EntityEvidence):
            raise ValueError("entity evidence must contain EntityEvidence values")
        if not set(item.reviewer_linked_candidate_ids).issubset(ids):
            raise ValueError(f"reviewer link for {candidate_id} references an unknown candidate")


def _seller_serial_key(
    candidate_id: str, evidence: Mapping[str, EntityEvidence]
) -> tuple[str, str] | None:
    item = evidence.get(candidate_id)
    if item is None or item.seller_account_id is None or item.declared_serial_sha256 is None:
        return None
    return item.seller_account_id, item.declared_serial_sha256


def _union_shared_value(
    union: _UnionFind,
    records: Sequence[CandidateRecord],
    key,
    *,
    ignore_none: bool = False,
) -> None:
    seen: dict[object, str] = {}
    for record in records:
        value = key(record)
        if ignore_none and value is None:
            continue
        prior = seen.setdefault(value, record.candidate_id)
        union.union(prior, record.candidate_id)


def _groups(union: _UnionFind, candidate_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate_id in candidate_ids:
        grouped[union.find(candidate_id)].append(candidate_id)
    return {root: tuple(sorted(members)) for root, members in grouped.items()}


def _stable_id(prefix: str, members: Iterable[str]) -> str:
    digest = hashlib.sha256("\0".join(sorted(members)).encode()).hexdigest()[:24]
    return f"{prefix}-{digest}"
