import hashlib
import io
import json
from datetime import datetime, timezone

import pytest

import acquisition.research_census as research_census
from acquisition.research_census import run_research_census_once

NOW = datetime(2026, 11, 20, 22, 0, tzinfo=timezone.utc)


class COS:
    def __init__(self):
        self.bodies = {}
        self.page_size = 1

    def list_objects_v2(self, **kwargs):
        keys = sorted(key for key in self.bodies if key.startswith(kwargs["Prefix"]))
        offset = int(kwargs.get("ContinuationToken", "0"))
        end = offset + self.page_size
        return {
            "Contents": [{"Key": key} for key in keys[offset:end]],
            "IsTruncated": end < len(keys),
            **({"NextContinuationToken": str(end)} if end < len(keys) else {}),
        }

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.bodies[kwargs["Key"]])}

    def put_object_if_absent(self, **kwargs):
        if kwargs["Key"] in self.bodies:
            return False
        self.bodies[kwargs["Key"]] = kwargs["Body"]
        return True


def _seed(cos, *, offset=0, source_offset=None, query=6):
    files = []
    for index in range(offset, offset + 2):
        receipt = {
            "candidate_id": f"candidate-{index}",
            "source_record_id": f"q{query}:{index if source_offset is None else source_offset + index - offset}",
            "source_url": f"https://commons.wikimedia.org/wiki/File:Example_{index}.jpg",
            "policy_id": "wikimedia-commons-licensed-v1",
            "raw_sha256": f"{index + 1:064x}",
            "derivative_sha256": f"{index + 3:064x}",
            "declared_reference": "unverified",
            "proposed_label": None,
            "proposed_grade": None,
        }
        body = json.dumps(receipt, sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        cos.bodies[f"datasets/receipts/wikimedia-{index}/candidate-{index}-{digest}.json"] = body
        files.append({**{key: receipt[key] for key in (
            "candidate_id", "source_record_id", "raw_sha256", "derivative_sha256"
        )}, "receipt_sha256": digest, "role": "front" if index == 0 else "caseback"})
    manifest = {
        "schema": "watch-auth-research-entity-v1",
        "source": "wikimedia",
        "reference": "unverified",
        "physical_watch_group": "probable-one-watch",
        "visual_basis": "Front and rear appear to depict one piece; provisional.",
        "physical_watch_count_probable": 1,
        "physical_watch_count_confirmed": 0,
        "content_file_count": 2,
        "training_admission": False,
        "test_admission": False,
        "files": files,
    }
    body = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    cos.bodies[f"datasets/research-entities/wikimedia/2026-11-20/{digest}.json"] = body


def _seed_multi(cos):
    _seed(cos, offset=10)
    key = next(key for key, body in cos.bodies.items()
               if "/research-entities/" in key and '"candidate-10"' in body.decode())
    manifest = json.loads(cos.bodies.pop(key))
    manifest.update(
        schema="watch-auth-research-entity-v2",
        physical_watch_group="probable-multiple-watches",
        physical_watch_count_probable=2,
        content_file_count=1,
        files=manifest["files"][:1],
        instances=[
            {"ordinal": 1, "box_permille": [10, 20, 480, 980],
             "visual_basis": "Left watch visible."},
            {"ordinal": 2, "box_permille": [520, 20, 990, 980],
             "visual_basis": "Right watch visible."},
        ],
    )
    body = json.dumps(manifest, sort_keys=True).encode()
    digest = hashlib.sha256(body).hexdigest()
    cos.bodies[f"datasets/research-entities/wikimedia/2026-11-20/{digest}.json"] = body


def test_census_reads_paginated_receipts_and_persists_research_only_counts():
    cos = COS()
    _seed(cos)

    report = run_research_census_once(cos, "private", "datasets", now=NOW)
    repeated = run_research_census_once(cos, "private", "datasets", now=NOW)

    assert report == repeated
    assert report["file_count"] == 2
    assert report["probable_watch_count"] == 1
    assert report["confirmed_watch_count"] == 0
    assert report["training_admission_count"] == 0
    assert len([key for key in cos.bodies if "/research-census/" in key]) == 1


def test_census_counts_multiple_watches_in_one_photo_without_admission():
    cos = COS()
    _seed(cos)
    _seed_multi(cos)

    report = run_research_census_once(cos, "private", "datasets", now=NOW)

    assert report["manifest_count"] == 2
    assert report["file_count"] == 3
    assert report["probable_watch_count"] == 3
    assert report["confirmed_watch_count"] == 0
    assert report["training_admission_count"] == 0


def test_census_accepts_legacy_slash_hash_receipts_in_inventory():
    cos = COS()
    _seed(cos)
    receipt_key = next(key for key in cos.bodies if "/receipts/" in key)
    body = cos.bodies.pop(receipt_key)
    digest = hashlib.sha256(body).hexdigest()
    cos.bodies[f"datasets/receipts/wikimedia-legacy/{digest}.json"] = body
    orphan = b"legacy unrelated receipt"
    orphan_digest = hashlib.sha256(orphan).hexdigest()
    cos.bodies[f"datasets/receipts/wikimedia-old/{orphan_digest}.json"] = orphan

    report = run_research_census_once(cos, "private", "datasets", now=NOW)

    assert report["file_count"] == 2
    assert report["probable_watch_count"] == 1


def test_census_rejects_same_receipt_hash_at_two_keys():
    cos = COS()
    _seed(cos)
    receipt_key = next(key for key in cos.bodies if "/receipts/" in key)
    digest = hashlib.sha256(cos.bodies[receipt_key]).hexdigest()
    cos.bodies[f"datasets/receipts/wikimedia-duplicate/{digest}.json"] = cos.bodies[receipt_key]

    with pytest.raises(ValueError, match="inventory is ambiguous"):
        run_research_census_once(cos, "private", "datasets", now=NOW)
    assert not any("/research-census/" in key for key in cos.bodies)


def test_census_rejects_reacquired_source_records_before_persisting():
    cos = COS()
    _seed(cos)
    _seed(cos, offset=2, source_offset=0, query=9)

    with pytest.raises(ValueError, match="reuse a source record"):
        run_research_census_once(cos, "private", "datasets", now=NOW)
    assert not any("/research-census/" in key for key in cos.bodies)


def test_census_refuses_a_second_terminal_for_the_same_date():
    cos = COS()
    _seed(cos)
    run_research_census_once(cos, "private", "datasets", now=NOW)
    key = next(key for key in cos.bodies if "/research-census/" in key)
    cos.bodies[key] = b'{}'

    with pytest.raises(RuntimeError, match="already closed"):
        run_research_census_once(cos, "private", "datasets", now=NOW)


def test_late_manifest_creates_immutable_revision_without_rewriting_terminal():
    cos = COS()
    _seed(cos)
    first = run_research_census_once(cos, "private", "datasets", now=NOW)
    terminal_key = next(key for key in cos.bodies if key.endswith("/terminal.json"))
    terminal = cos.bodies[terminal_key]
    _seed(cos, offset=2)

    revised = run_research_census_once(cos, "private", "datasets", now=NOW)
    repeated = run_research_census_once(cos, "private", "datasets", now=NOW)

    assert revised == repeated
    assert revised["file_count"] == 4
    assert revised["probable_watch_count"] == 2
    assert revised["training_admission_count"] == 0
    assert cos.bodies[terminal_key] == terminal
    assert json.loads(terminal)["file_count"] == first["file_count"] == 2
    revisions = [key for key in cos.bodies if "/research-census/" in key and "/revisions/" in key]
    assert len(revisions) == 1
    assert json.loads(cos.bodies[revisions[0]]) == revised


def test_census_fails_before_persisting_corrupt_receipt():
    cos = COS()
    _seed(cos)
    receipt_key = next(key for key in cos.bodies if "/receipts/" in key)
    cos.bodies[receipt_key] += b" "

    with pytest.raises(ValueError, match="missing or changed"):
        run_research_census_once(cos, "private", "datasets", now=NOW)
    assert not any("/research-census/" in key for key in cos.bodies)


def test_census_rejects_oversized_manifest_before_persisting():
    cos = COS()
    _seed(cos)
    manifest_key = next(key for key in cos.bodies if "/research-entities/" in key)
    cos.bodies[manifest_key] += b" " * 1_000_000

    with pytest.raises(ValueError, match="body is unavailable"):
        run_research_census_once(cos, "private", "datasets", now=NOW)
    assert not any("/research-census/" in key for key in cos.bodies)


def test_census_rejects_oversized_output_before_persisting(monkeypatch):
    cos = COS()
    _seed(cos)
    monkeypatch.setattr(research_census, "_MAX_OUTPUT_BYTES", 100)

    with pytest.raises(ValueError, match="output exceeds"):
        run_research_census_once(cos, "private", "datasets", now=NOW)
    assert not any("/research-census/" in key for key in cos.bodies)
