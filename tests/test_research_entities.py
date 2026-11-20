import hashlib
import json

import pytest

from dataset.research_entities import read_research_entity_manifest


def _fixture():
    receipts = []
    files = []
    for index in range(2):
        receipt = {
            "candidate_id": f"candidate-{index}",
            "source_record_id": f"q6:{index}",
            "raw_sha256": f"{index + 1:064x}",
            "derivative_sha256": f"{index + 3:064x}",
            "declared_reference": "unverified",
            "policy_id": "wikimedia-commons-licensed-v1",
            "source_url": f"https://commons.wikimedia.org/wiki/File:Example_{index}.jpg",
            "proposed_label": None,
            "proposed_grade": None,
        }
        body = json.dumps(receipt, sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        receipts.append((digest, body))
        files.append({
            "candidate_id": receipt["candidate_id"],
            "source_record_id": receipt["source_record_id"],
            "raw_sha256": receipt["raw_sha256"],
            "derivative_sha256": receipt["derivative_sha256"],
            "receipt_sha256": digest,
            "role": "front" if index == 0 else "caseback",
        })
    manifest = {
        "schema": "watch-auth-research-entity-v1",
        "source": "wikimedia",
        "physical_watch_group": "probable-one-watch",
        "visual_basis": "Front and rear appear to depict one piece; provisional.",
        "reference": "unverified",
        "training_admission": False,
        "test_admission": False,
        "physical_watch_count_probable": 1,
        "physical_watch_count_confirmed": 0,
        "content_file_count": 2,
        "files": files,
    }
    return manifest, dict(receipts)


def _read(manifest, receipts):
    return read_research_entity_manifest(json.dumps(manifest, sort_keys=True).encode(), receipts)


def test_research_group_counts_probable_watch_without_admitting_files():
    manifest, receipts = _fixture()

    report = _read(manifest, receipts)

    assert report.file_count == 2
    assert report.probable_watch_count == 1
    assert report.confirmed_watch_count == 0
    assert report.candidate_ids == ("candidate-0", "candidate-1")


@pytest.mark.parametrize("change", [
    lambda manifest: manifest.update(training_admission=True),
    lambda manifest: manifest.update(physical_watch_count_confirmed=1),
    lambda manifest: manifest.update(physical_watch_count_probable=3),
    lambda manifest: manifest.update(physical_watch_count_probable=2),
    lambda manifest: manifest["files"][1].update(candidate_id="candidate-0"),
    lambda manifest: manifest["files"][1].update(raw_sha256="0" * 64),
    lambda manifest: manifest["files"][1].update(candidate_id=None),
    lambda manifest: manifest.update(reference=None),
    lambda manifest: manifest.update(content_file_count=True),
    lambda manifest: manifest.update(source="other-source"),
])
def test_research_group_rejects_admission_or_lineage_inflation(change):
    manifest, receipts = _fixture()
    change(manifest)

    with pytest.raises(ValueError):
        _read(manifest, receipts)


def test_research_group_rejects_changed_receipt_bytes():
    manifest, receipts = _fixture()
    digest = manifest["files"][0]["receipt_sha256"]
    receipts[digest] += b" "

    with pytest.raises(ValueError, match="missing or changed"):
        _read(manifest, receipts)


def test_research_group_rejects_receipt_from_another_source():
    manifest, receipts = _fixture()
    old_digest = manifest["files"][0]["receipt_sha256"]
    receipt = json.loads(receipts.pop(old_digest))
    receipt["source_url"] = "https://example.com/watch.jpg"
    body = json.dumps(receipt, sort_keys=True).encode()
    new_digest = hashlib.sha256(body).hexdigest()
    receipts[new_digest] = body
    manifest["files"][0]["receipt_sha256"] = new_digest

    with pytest.raises(ValueError, match="source differs"):
        _read(manifest, receipts)
