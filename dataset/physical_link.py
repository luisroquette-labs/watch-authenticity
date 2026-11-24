"""Hash-bound human reviews of two views of one physical watch."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

from acquisition.types import CandidateRecord
from infra.ibm_cos import download_verified, upload_verified
from infra.ibm_types import ObjectRef
from labeling.review import ReviewerRecord, reviewer_digest


@dataclass(frozen=True)
class PhysicalLinkReview:
    candidate_ids: tuple[str, str]
    raw_sha256s: tuple[str, str]
    artifact_sha256: str
    visual_basis: str
    reviewer_id: str
    reviewer_registry_sha256: str
    reviewed_at: datetime
    review_sha256: str


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _payload(review: PhysicalLinkReview) -> dict[str, object]:
    return {
        "candidate_ids": list(review.candidate_ids),
        "raw_sha256s": list(review.raw_sha256s),
        "artifact_sha256": review.artifact_sha256,
        "visual_basis": review.visual_basis,
        "reviewer_id": review.reviewer_id,
        "reviewer_registry_sha256": review.reviewer_registry_sha256,
        "reviewed_at": review.reviewed_at.isoformat(),
    }


def physical_link_artifact(
    left: CandidateRecord,
    right: CandidateRecord,
    left_raw: bytes,
    right_raw: bytes,
) -> bytes:
    if left.candidate_id == right.candidate_id:
        raise ValueError("physical link requires two candidates")
    pair = sorted(((left.candidate_id, left.raw_sha256, left_raw),
                   (right.candidate_id, right.raw_sha256, right_raw)))
    if any(not isinstance(body, bytes) or not body or len(body) > 25 * 1024 * 1024
           or hashlib.sha256(body).hexdigest() != digest for _, digest, body in pair):
        raise ValueError("physical link images differ from archived raw hashes")
    return _canonical({
        "schema": "watch-auth-physical-link-images-v1",
        "images": [
            {"candidate_id": candidate_id, "raw_sha256": digest,
             "raw_base64": base64.b64encode(body).decode("ascii")}
            for candidate_id, digest, body in pair
        ],
    })


def _verify_artifact(artifact: bytes, pair: Sequence[tuple[str, str]]) -> None:
    if len(pair) != 2:
        raise ValueError("physical link artifact requires exactly two images")
    if not isinstance(artifact, bytes) or len(artifact) > 70 * 1024 * 1024:
        raise ValueError("physical link artifact is invalid or too large")
    try:
        document = json.loads(artifact)
        if (not isinstance(document, dict)
            or set(document) != {"schema", "images"}
            or document["schema"] != "watch-auth-physical-link-images-v1"
            or not isinstance(document["images"], list)
            or len(document["images"]) != 2):
            raise ValueError("physical link artifact schema is invalid")
        for item, (candidate_id, digest) in zip(document["images"], pair, strict=True):
            if (not isinstance(item, dict)
                or set(item) != {"candidate_id", "raw_sha256", "raw_base64"}
                or item["candidate_id"] != candidate_id
                or item["raw_sha256"] != digest
                or not isinstance(item["raw_base64"], str)):
                raise ValueError("physical link artifact image binding is invalid")
            body = base64.b64decode(item["raw_base64"], validate=True)
            if not body or len(body) > 25 * 1024 * 1024 or hashlib.sha256(body).hexdigest() != digest:
                raise ValueError("physical link artifact image hash differs")
    except (UnicodeDecodeError, json.JSONDecodeError, binascii.Error, TypeError) as exc:
        raise ValueError("physical link artifact cannot be decoded") from exc
    if _canonical(document) != artifact:
        raise ValueError("physical link artifact must be canonical")


def record_physical_link_review(
    left: CandidateRecord,
    right: CandidateRecord,
    artifact: bytes,
    visual_basis: str,
    reviewer: ReviewerRecord,
    reviewed_at: datetime,
) -> PhysicalLinkReview:
    if left.candidate_id == right.candidate_id:
        raise ValueError("physical link requires two candidates")
    if not isinstance(artifact, bytes) or not artifact:
        raise ValueError("physical link requires the examined artifact bytes")
    pair = tuple(sorted(((left.candidate_id, left.raw_sha256),
                         (right.candidate_id, right.raw_sha256))))
    _verify_artifact(artifact, pair)
    if not isinstance(visual_basis, str) or len(visual_basis.strip()) < 12:
        raise ValueError("physical link requires a specific visual basis")
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError("review time must be timezone-aware")
    if not reviewer.effective_at <= reviewed_at < reviewer.expires_at:
        raise ValueError("physical link reviewer is not active")
    if reviewed_at < max(left.collected_at, right.collected_at):
        raise ValueError("physical link review predates one of its photos")
    if "physical_identity_review" not in reviewer.allowed_evidence_types:
        raise ValueError("reviewer is not qualified for physical identity")
    if {left.acquisition_batch_id, right.acquisition_batch_id,
        left.source_policy_id, right.source_policy_id} & set(reviewer.prohibited_contributor_groups):
        raise ValueError("reviewer cannot approve own contributor group")
    review = PhysicalLinkReview(
        candidate_ids=(pair[0][0], pair[1][0]),
        raw_sha256s=(pair[0][1], pair[1][1]),
        artifact_sha256=hashlib.sha256(artifact).hexdigest(),
        visual_basis=visual_basis.strip(),
        reviewer_id=reviewer.reviewer_id,
        reviewer_registry_sha256=reviewer_digest(reviewer),
        reviewed_at=reviewed_at,
        review_sha256="",
    )
    return replace(review, review_sha256=hashlib.sha256(_canonical(_payload(review))).hexdigest())


def verified_physical_links(
    records: Sequence[CandidateRecord],
    reviews: Sequence[PhysicalLinkReview],
    artifacts: Mapping[str, bytes],
    reviewers: Mapping[str, ReviewerRecord],
) -> set[tuple[str, str]]:
    by_id = {record.candidate_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("physical link candidate IDs must be unique")
    qualifications: dict[tuple[str, str], set[str]] = {}
    artifact_by_pair: dict[tuple[str, str], str] = {}
    for review in reviews:
        if not isinstance(review, PhysicalLinkReview):
            raise ValueError("physical link review type is invalid")
        if review.candidate_ids[0] >= review.candidate_ids[1]:
            raise ValueError("physical link candidate pair is not canonical")
        try:
            left, right = (by_id[item] for item in review.candidate_ids)
            reviewer = reviewers[review.reviewer_id]
            artifact = artifacts[review.artifact_sha256]
        except KeyError as exc:
            raise ValueError("physical link candidate, reviewer or artifact is missing") from exc
        if (review.raw_sha256s != (left.raw_sha256, right.raw_sha256)
            or hashlib.sha256(artifact).hexdigest() != review.artifact_sha256
            or review.reviewer_registry_sha256 != reviewer_digest(reviewer)
            or not reviewer.effective_at <= review.reviewed_at < reviewer.expires_at
            or review.reviewed_at < max(left.collected_at, right.collected_at)
            or "physical_identity_review" not in reviewer.allowed_evidence_types
            or {left.acquisition_batch_id, right.acquisition_batch_id,
                left.source_policy_id, right.source_policy_id} & set(reviewer.prohibited_contributor_groups)
            or len(review.visual_basis.strip()) < 12
            or hashlib.sha256(_canonical(_payload(review))).hexdigest() != review.review_sha256):
            raise ValueError("physical link review differs from examined evidence")
        _verify_artifact(artifact, tuple(zip(review.candidate_ids, review.raw_sha256s, strict=True)))
        pair = review.candidate_ids
        if pair in artifact_by_pair and artifact_by_pair[pair] != review.artifact_sha256:
            raise ValueError("physical link reviewers disagree on evidence")
        artifact_by_pair[pair] = review.artifact_sha256
        qualifications.setdefault(pair, set()).add(reviewer.qualification_artifact_sha256)
    return {pair for pair, agreeing in qualifications.items() if len(agreeing) >= 2}


def persist_physical_link_review(
    cos_client: object,
    bucket: str,
    review: PhysicalLinkReview,
    artifact: bytes,
) -> tuple[ObjectRef, ObjectRef]:
    if hashlib.sha256(artifact).hexdigest() != review.artifact_sha256:
        raise ValueError("physical link artifact bytes differ from review")
    if hashlib.sha256(_canonical(_payload(review))).hexdigest() != review.review_sha256:
        raise ValueError("physical link review digest differs from content")
    with tempfile.TemporaryDirectory(prefix="watch-physical-link-") as directory:
        artifact_path = Path(directory) / "artifact"
        review_path = Path(directory) / "review.json"
        artifact_path.write_bytes(artifact)
        review_path.write_bytes(_canonical({**_payload(review), "review_sha256": review.review_sha256}))
        artifact_ref = upload_verified(
            cos_client, artifact_path, bucket,
            f"datasets/physical-links/artifacts/{review.artifact_sha256}",
            review.artifact_sha256,
        )
        review_bytes_sha256 = hashlib.sha256(review_path.read_bytes()).hexdigest()
        review_ref = upload_verified(
            cos_client, review_path, bucket,
            f"datasets/physical-links/reviews/{review.review_sha256}",
            review_bytes_sha256,
        )
    return artifact_ref, review_ref


def verify_persisted_physical_link_reviews(
    cos_client: object,
    bucket: str,
    reviews: Sequence[PhysicalLinkReview],
    artifacts: Mapping[str, bytes],
) -> None:
    if any(not isinstance(review, PhysicalLinkReview) for review in reviews):
        raise ValueError("persisted physical link review type is invalid")
    if set(artifacts) != {review.artifact_sha256 for review in reviews}:
        raise ValueError("persisted physical link artifact set differs from reviews")
    with tempfile.TemporaryDirectory(prefix="watch-physical-link-readback-") as directory:
        destination = Path(directory)
        for artifact_sha256, artifact in artifacts.items():
            if not isinstance(artifact, bytes) or hashlib.sha256(artifact).hexdigest() != artifact_sha256:
                raise ValueError("persisted physical link artifact hash differs")
            path = destination / artifact_sha256
            download_verified(
                cos_client, bucket,
                f"datasets/physical-links/artifacts/{artifact_sha256}",
                path, artifact_sha256,
            )
            if path.read_bytes() != artifact:
                raise ValueError("persisted physical link artifact differs from examined bytes")
        for review in reviews:
            if hashlib.sha256(_canonical(_payload(review))).hexdigest() != review.review_sha256:
                raise ValueError("persisted physical link review digest differs")
            body = _canonical({**_payload(review), "review_sha256": review.review_sha256})
            path = destination / review.review_sha256
            download_verified(
                cos_client, bucket,
                f"datasets/physical-links/reviews/{review.review_sha256}/"
                f"{hashlib.sha256(body).hexdigest()}",
                path, hashlib.sha256(body).hexdigest(),
            )
            if path.read_bytes() != body:
                raise ValueError("persisted physical link review differs from supplied decision")
