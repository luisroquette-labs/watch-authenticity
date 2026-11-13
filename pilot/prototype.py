"""Conservative DINOv2 linear probe over provisional online examples."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor

from adapters.backbone import BACKBONE_REVISION, load_frozen_backbone

SCHEMA_VERSION = 1
GENUINE_MAX = 0.20
REPLICA_MIN = 0.80
SIMILARITY_MIN = 0.40
HOLDOUT_REPLICA_POLICY = "vsfwatcheshub-public-private-download-v1"
POLICY_LABELS = {
    "wikimedia-commons-licensed-v1": 0,
    "richfactoryus-public-private-download-v1": 1,
    "superclonerolex-public-private-download-v1": 1,
    "vsfactoryusa-public-private-download-v1": 1,
    HOLDOUT_REPLICA_POLICY: 1,
}


@dataclass(frozen=True)
class PilotExample:
    candidate_id: str
    derivative_sha256: str
    image_path: Path
    label: int
    phash: str
    policy_id: str
    source_domain: str


@dataclass(frozen=True)
class PilotArtifact:
    schema_version: int
    backbone_revision: str
    dataset_sha256: str
    weights: tuple[float, ...]
    reference_embeddings: tuple[tuple[float, ...], ...]
    intercept: float
    genuine_max: float
    replica_min: float
    similarity_min: float
    training_count: int
    holdout_count: int
    holdout_decided: int
    holdout_correct: int
    holdout_inconclusive: int
    provisional_notice: str


@dataclass(frozen=True)
class PilotPrediction:
    verdict: str
    replica_probability: float
    nearest_similarity: float
    in_distribution: bool
    notice: str


def load_examples(receipts_path: Path, images_dir: Path) -> tuple[PilotExample, ...]:
    raw = json.loads(receipts_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("pilot receipts must be a non-empty list")
    unique: dict[str, dict[str, Any]] = {}
    for receipt in raw:
        if not isinstance(receipt, dict):
            raise ValueError("pilot receipt must be an object")
        policy_id = receipt.get("policy_id")
        if policy_id not in POLICY_LABELS:
            raise ValueError(f"pilot policy is not explicitly classified: {policy_id}")
        phash = _safe_hex(receipt.get("phash"), 16, "phash")
        candidate_id = receipt.get("candidate_id")
        digest = _safe_hex(receipt.get("derivative_sha256"), 64, "derivative_sha256")
        source_url = receipt.get("source_url")
        if not isinstance(candidate_id, str) or not candidate_id.startswith("candidate-"):
            raise ValueError("pilot candidate_id is invalid")
        parsed = urlsplit(source_url if isinstance(source_url, str) else "")
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("pilot source_url is invalid")
        previous = unique.get(phash)
        if previous is None or candidate_id < previous["candidate_id"]:
            unique[phash] = receipt

    examples: list[PilotExample] = []
    for receipt in sorted(unique.values(), key=lambda item: item["candidate_id"]):
        digest = receipt["derivative_sha256"]
        label_name = "provisional_genuine" if POLICY_LABELS[receipt["policy_id"]] == 0 else "provisional_replica"
        path = images_dir / f"{label_name}-{digest}.jpg"
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"pilot image is missing or corrupt: {digest}")
        examples.append(
            PilotExample(
                candidate_id=receipt["candidate_id"],
                derivative_sha256=digest,
                image_path=path,
                label=POLICY_LABELS[receipt["policy_id"]],
                phash=receipt["phash"],
                policy_id=receipt["policy_id"],
                source_domain=urlsplit(receipt["source_url"]).hostname or "",
            )
        )
    if {item.label for item in examples} != {0, 1}:
        raise ValueError("pilot requires both provisional classes")
    return tuple(examples)


def extract_embeddings(
    image_paths: Sequence[Path], *, batch_size: int = 8
) -> np.ndarray:
    if not image_paths or batch_size <= 0:
        raise ValueError("pilot embedding input is empty or invalid")
    processor = AutoImageProcessor.from_pretrained(
        "facebook/dinov2-base",
        revision=BACKBONE_REVISION,
        local_files_only=True,
    )
    model = load_frozen_backbone()
    model.eval()
    rows: list[np.ndarray] = []
    for offset in range(0, len(image_paths), batch_size):
        images = []
        for path in image_paths[offset : offset + batch_size]:
            with Image.open(path) as opened:
                images.append(opened.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            output = model(pixel_values=inputs["pixel_values"]).pooler_output
        normalized = torch.nn.functional.normalize(output, dim=1)
        rows.append(normalized.cpu().numpy().astype(np.float64))
    return np.concatenate(rows, axis=0)


def fit_artifact(
    examples: Sequence[PilotExample], embeddings: np.ndarray
) -> PilotArtifact:
    if embeddings.ndim != 2 or embeddings.shape[0] != len(examples):
        raise ValueError("pilot embeddings do not align with examples")
    labels = np.asarray([item.label for item in examples], dtype=np.int64)
    replica_holdout = np.asarray(
        [item.policy_id == HOLDOUT_REPLICA_POLICY for item in examples], dtype=bool
    )
    genuine_indices = sorted(
        (index for index, item in enumerate(examples) if item.label == 0),
        key=lambda index: examples[index].derivative_sha256,
    )
    genuine_holdout = set(genuine_indices[::4])
    holdout = replica_holdout | np.asarray(
        [index in genuine_holdout for index in range(len(examples))], dtype=bool
    )
    train = ~holdout
    if set(labels[train]) != {0, 1} or set(labels[holdout]) != {0, 1}:
        raise ValueError("pilot train and holdout must contain both classes")
    classifier = LogisticRegression(
        C=100.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="liblinear",
    ).fit(embeddings[train], labels[train])
    probabilities = classifier.predict_proba(embeddings[holdout])[:, 1]
    holdout_labels = labels[holdout]
    decided = (probabilities <= GENUINE_MAX) | (probabilities >= REPLICA_MIN)
    decisions = probabilities >= REPLICA_MIN
    correct = int(np.sum(decided & (decisions == holdout_labels)))
    dataset_sha256 = hashlib.sha256(
        "\n".join(sorted(item.derivative_sha256 for item in examples)).encode()
    ).hexdigest()
    return PilotArtifact(
        schema_version=SCHEMA_VERSION,
        backbone_revision=BACKBONE_REVISION,
        dataset_sha256=dataset_sha256,
        weights=tuple(float(value) for value in classifier.coef_[0]),
        reference_embeddings=tuple(
            tuple(float(value) for value in row) for row in embeddings[train]
        ),
        intercept=float(classifier.intercept_[0]),
        genuine_max=GENUINE_MAX,
        replica_min=REPLICA_MIN,
        similarity_min=SIMILARITY_MIN,
        training_count=int(np.sum(train)),
        holdout_count=int(np.sum(holdout)),
        holdout_decided=int(np.sum(decided)),
        holdout_correct=correct,
        holdout_inconclusive=int(np.sum(~decided)),
        provisional_notice=(
            "Experimental similarity pilot trained from provisional online labels; "
            "it is not an authentication or proof of genuineness."
        ),
    )


def write_artifact(path: Path, artifact: PilotArtifact) -> None:
    _validate_artifact(artifact)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(asdict(artifact), sort_keys=True, separators=(",", ":"))
    path.write_text(body, encoding="utf-8")


def load_artifact(path: Path) -> PilotArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("pilot artifact must be an object")
    raw["weights"] = tuple(raw.get("weights", ()))
    raw["reference_embeddings"] = tuple(
        tuple(row) for row in raw.get("reference_embeddings", ())
    )
    artifact = PilotArtifact(**raw)
    _validate_artifact(artifact)
    return artifact


def predict_embedding(artifact: PilotArtifact, embedding: np.ndarray) -> PilotPrediction:
    _validate_artifact(artifact)
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.shape[0] != len(artifact.weights) or not np.isfinite(vector).all():
        raise ValueError("pilot embedding is invalid")
    norm = float(np.linalg.norm(vector))
    if norm <= 0:
        raise ValueError("pilot embedding has zero norm")
    vector = vector / norm
    references = np.asarray(artifact.reference_embeddings, dtype=np.float64)
    nearest_similarity = float(np.max(references @ vector))
    logit = float(vector @ np.asarray(artifact.weights) + artifact.intercept)
    probability = 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, logit))))
    in_distribution = nearest_similarity >= artifact.similarity_min
    if not in_distribution:
        verdict = "inconclusivo"
    elif probability >= artifact.replica_min:
        verdict = "provavel_replica"
    elif probability <= artifact.genuine_max:
        verdict = "sem_inconsistencia_visivel"
    else:
        verdict = "inconclusivo"
    return PilotPrediction(
        verdict,
        probability,
        nearest_similarity,
        in_distribution,
        artifact.provisional_notice,
    )


def _validate_artifact(artifact: PilotArtifact) -> None:
    if artifact.schema_version != SCHEMA_VERSION or artifact.backbone_revision != BACKBONE_REVISION:
        raise ValueError("pilot artifact version or backbone differs")
    if len(artifact.weights) != 768 or not all(math.isfinite(value) for value in artifact.weights):
        raise ValueError("pilot artifact weights are invalid")
    if (
        not artifact.reference_embeddings
        or any(
            len(row) != 768 or not all(math.isfinite(value) for value in row)
            for row in artifact.reference_embeddings
        )
    ):
        raise ValueError("pilot reference embeddings are invalid")
    if not math.isfinite(artifact.intercept) or not 0 < artifact.genuine_max < artifact.replica_min < 1:
        raise ValueError("pilot artifact thresholds are invalid")
    if not -1 < artifact.similarity_min < 1:
        raise ValueError("pilot similarity threshold is invalid")
    if (
        artifact.holdout_correct > artifact.holdout_decided
        or artifact.holdout_decided > artifact.holdout_count
        or artifact.holdout_inconclusive != artifact.holdout_count - artifact.holdout_decided
    ):
        raise ValueError("pilot artifact holdout metrics are invalid")


def _safe_hex(value: object, length: int, field: str) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"pilot {field} is invalid")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"pilot {field} is invalid") from exc
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
