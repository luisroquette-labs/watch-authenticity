from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pilot.prototype import (
    HOLDOUT_REPLICA_POLICY,
    PilotArtifact,
    PilotExample,
    fit_artifact,
    load_artifact,
    load_examples,
    predict_embedding,
    write_artifact,
)


def _example(index: int, label: int, policy: str, vector_group: str) -> PilotExample:
    return PilotExample(
        candidate_id=f"candidate-{index:024x}",
        derivative_sha256=f"{index + 1:064x}",
        image_path=Path(f"unused-{index}.jpg"),
        label=label,
        phash=f"{index + 1:016x}",
        policy_id=policy,
        source_domain=vector_group,
    )


def test_fit_keeps_one_replica_source_and_genuine_quarter_as_holdout() -> None:
    examples = [
        *[_example(i, 0, "wikimedia-commons-licensed-v1", "commons") for i in range(8)],
        *[_example(20 + i, 1, "vsfactoryusa-public-private-download-v1", "vsf") for i in range(8)],
        *[_example(40 + i, 1, HOLDOUT_REPLICA_POLICY, "hub") for i in range(4)],
    ]
    embeddings = np.zeros((len(examples), 768), dtype=np.float64)
    embeddings[:8, 0] = -1.0
    embeddings[8:, 0] = 1.0

    artifact = fit_artifact(examples, embeddings)

    assert artifact.training_count == 14
    assert artifact.holdout_count == 6
    assert artifact.holdout_decided <= artifact.holdout_count
    assert artifact.holdout_correct <= artifact.holdout_decided


def test_artifact_round_trip_and_abstention(tmp_path: Path) -> None:
    positive = (1.0,) + (0.0,) * 767
    negative = (-1.0,) + (0.0,) * 767
    orthogonal = (0.0, 1.0) + (0.0,) * 766
    artifact = PilotArtifact(
        schema_version=1,
        backbone_revision="f9e44c814b77203eaa57a6bdbbd535f21ede1415",
        dataset_sha256="a" * 64,
        weights=(3.0,) + (0.0,) * 767,
        reference_embeddings=(positive, negative, orthogonal),
        intercept=0.0,
        genuine_max=0.2,
        replica_min=0.8,
        similarity_min=0.4,
        training_count=10,
        holdout_count=4,
        holdout_decided=2,
        holdout_correct=2,
        holdout_inconclusive=2,
        provisional_notice="provisional",
    )
    path = tmp_path / "model.json"
    write_artifact(path, artifact)

    loaded = load_artifact(path)

    ambiguous = np.zeros(768)
    ambiguous[1] = 1
    assert predict_embedding(loaded, ambiguous).verdict == "inconclusivo"
    replica = np.zeros(768)
    replica[0] = 3
    assert predict_embedding(loaded, replica).verdict == "provavel_replica"
    genuine = np.zeros(768)
    genuine[0] = -3
    assert predict_embedding(loaded, genuine).verdict == "sem_inconsistencia_visivel"

    unknown = np.zeros(768)
    unknown[2] = 1
    rejected = predict_embedding(loaded, unknown)
    assert rejected.verdict == "inconclusivo"
    assert not rejected.in_distribution


def test_manifest_rejects_unknown_source_policy(tmp_path: Path) -> None:
    image = Image.new("RGB", (224, 224), "black")
    image_path = tmp_path / "source.jpg"
    image.save(image_path)
    digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
    image_path.rename(tmp_path / f"provisional_replica-{digest}.jpg")
    receipts = tmp_path / "receipts.json"
    receipts.write_text(json.dumps([{
        "candidate_id": "candidate-" + "1" * 24,
        "derivative_sha256": digest,
        "phash": "1" * 16,
        "policy_id": "unknown-policy",
        "source_url": "https://example.com/watch",
    }]))

    with pytest.raises(ValueError, match="not explicitly classified"):
        load_examples(receipts, tmp_path)
