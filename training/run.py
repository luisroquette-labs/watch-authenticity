"""Deterministic linear-probe training with verified, resumable artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from adapters.backbone import BACKBONE_CHECKPOINT, BACKBONE_EMBED_DIM, BACKBONE_REVISION
from core.types import Label, ModelClass
from dataset.augment import augment_pipeline
from dataset.crop import ANGLE_CROP_FRACTIONS, crop_angle
from dataset.splits import TRAIN, VAL
from infra.ibm_cos import IntegrityError, download_verified
from infra.ibm_types import RunSpec
from model.heads import AngleHeadModel, build_trainable_model, trainable_head_parameters
from training.data import (
    TrainingBundle,
    TrainingRecord,
    entity_wave_asset_ids,
    training_wave_bundle,
)

SEED = 42
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
IMAGE_SIZE = 224
_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)


class TrainingRefused(RuntimeError):
    """Training or resume lineage is incomplete, mismatched or unsafe."""


@dataclass(frozen=True)
class RunResult:
    epochs_completed: int
    best_epoch: int
    best_validation_nll: float
    artifacts: dict[Path, str]


@dataclass(frozen=True)
class WaveRunResult:
    fraction: float
    entity_count: int
    result: RunResult


@dataclass(frozen=True)
class _TrainingState:
    epochs_completed: int
    best_epoch: int
    best_validation_nll: float
    epochs_without_improvement: int
    history: list[dict[str, float | int]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_state_dict(path: Path, state: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _class_map() -> dict[str, int]:
    return {model_class.name.lower(): model_class.value for model_class in ModelClass}


def _lineage(bundle: TrainingBundle, spec: RunSpec, class_map_sha256: str) -> dict[str, str]:
    if bundle.manifest_sha256 != spec.manifest_sha256 or bundle.splits_sha256 != spec.splits_sha256:
        raise TrainingRefused("training bundle lineage does not match RunSpec")
    return {
        "code_sha": spec.code_sha,
        "image_digest": spec.image_digest,
        "backbone_checkpoint": BACKBONE_CHECKPOINT,
        "backbone_revision": BACKBONE_REVISION,
        "manifest_sha256": spec.manifest_sha256,
        "splits_sha256": spec.splits_sha256,
        "class_map_sha256": class_map_sha256,
        "train_assets_sha256": hashlib.sha256(
            _canonical(sorted(bundle.splits[TRAIN]))
        ).hexdigest(),
    }


def _seed_everything() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_head_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as exc:
        raise TrainingRefused("checkpoint is not a safe tensor state dictionary") from exc
    expected_keys = {
        f"{angle}.{parameter}"
        for angle in ANGLE_CROP_FRACTIONS
        for parameter in ("weight", "bias")
    }
    if (
        not isinstance(state, dict)
        or not state
        or set(state) != expected_keys
        or any(not isinstance(key, str) or not isinstance(value, torch.Tensor) for key, value in state.items())
        or any("backbone" in key for key in state)
    ):
        raise TrainingRefused("checkpoint is not a head-only tensor state dictionary")
    for key, tensor in state.items():
        expected_shape = (
            (len(ModelClass), BACKBONE_EMBED_DIM) if key.endswith(".weight") else (len(ModelClass),)
        )
        if tuple(tensor.shape) != expected_shape:
            raise TrainingRefused("checkpoint head tensor shape does not match the model contract")
    return state


def _completed_result(
    output_dir: Path, lineage: dict[str, str], class_map_bytes: bytes
) -> RunResult | None:
    result_path = output_dir / "run-result.json"
    if not result_path.exists():
        return None
    try:
        raw = result_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingRefused("run-result.json is unreadable") from exc
    if raw != _canonical(payload) or payload.get("lineage") != lineage:
        raise TrainingRefused("completed run lineage does not match requested lineage")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "checkpoint.pt",
        "metrics.json",
        "class-map.json",
    }:
        raise TrainingRefused("completed run artifact inventory is invalid")
    resolved: dict[Path, str] = {}
    for name, digest in artifacts.items():
        path = output_dir / name
        if not isinstance(digest, str) or not path.is_file() or _sha256(path) != digest:
            raise TrainingRefused("completed run artifact hash does not match")
        resolved[path] = digest
    resumable = _resumable_state(output_dir, lineage, class_map_bytes)
    if resumable is None:
        raise TrainingRefused("completed run training state is missing")
    try:
        epochs_completed = int(payload["epochs_completed"])
        best_epoch = int(payload["best_epoch"])
        best_validation_nll = float(payload["best_validation_nll"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingRefused("completed run metrics are invalid") from exc
    if (
        epochs_completed != resumable.epochs_completed
        or best_epoch != resumable.best_epoch
        or best_validation_nll != resumable.best_validation_nll
    ):
        raise TrainingRefused("completed run and metrics disagree")
    if epochs_completed <= 0 or not 1 <= best_epoch <= epochs_completed or not math.isfinite(best_validation_nll):
        raise TrainingRefused("completed run metrics are invalid")
    return RunResult(epochs_completed, best_epoch, best_validation_nll, resolved)


def _resumable_state(
    output_dir: Path, lineage: dict[str, str], class_map_bytes: bytes
) -> _TrainingState | None:
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    class_map_path = output_dir / "class-map.json"
    present = tuple(path.exists() for path in (checkpoint_path, metrics_path, class_map_path))
    if not any(present):
        return None
    try:
        class_map_matches = class_map_path.read_bytes() == class_map_bytes
    except OSError as exc:
        raise TrainingRefused("class map is unreadable") from exc
    if not all(present) or not class_map_matches:
        raise TrainingRefused("partial prior training artifacts cannot be resumed safely")
    try:
        raw = metrics_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingRefused("training metrics are unreadable") from exc
    if raw != _canonical(payload) or payload.get("lineage") != lineage:
        raise TrainingRefused("training resume lineage does not match requested lineage")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or _sha256(checkpoint_path) != checkpoint_sha256:
        raise TrainingRefused("training resume checkpoint hash does not match")
    _load_head_state(checkpoint_path)
    try:
        epochs_completed_raw = payload["epochs_completed"]
        best_epoch_raw = payload["best_epoch"]
        epochs_without_improvement_raw = payload["epochs_without_improvement"]
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (
                epochs_completed_raw,
                best_epoch_raw,
                epochs_without_improvement_raw,
            )
        ):
            raise TypeError("integer metric has invalid type")
        epochs_completed = epochs_completed_raw
        best_epoch = best_epoch_raw
        best_validation_nll = float(payload["best_validation_nll"])
        epochs_without_improvement = epochs_without_improvement_raw
        history = payload["history"]
    except (KeyError, TypeError, ValueError) as exc:
        raise TrainingRefused("training resume metrics are invalid") from exc
    if (
        not isinstance(history, list)
        or epochs_completed != len(history)
        or epochs_completed <= 0
        or not 1 <= best_epoch <= epochs_completed
        or not math.isfinite(best_validation_nll)
        or not 0 <= epochs_without_improvement <= epochs_completed
    ):
        raise TrainingRefused("training resume metrics are invalid")
    for expected_epoch, entry in enumerate(history, start=1):
        if (
            not isinstance(entry, dict)
            or set(entry) != {"epoch", "train_nll", "validation_nll"}
            or entry["epoch"] != expected_epoch
            or not all(
                isinstance(entry[name], (int, float))
                and not isinstance(entry[name], bool)
                and math.isfinite(float(entry[name]))
                for name in ("train_nll", "validation_nll")
            )
        ):
            raise TrainingRefused("training resume history is invalid")
    return _TrainingState(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        epochs_without_improvement=epochs_without_improvement,
        history=history,
    )


def _write_metrics(
    path: Path,
    *,
    lineage: dict[str, str],
    checkpoint_path: Path,
    epochs_completed: int,
    best_epoch: int,
    best_validation_nll: float,
    epochs_without_improvement: int,
    history: list[dict[str, float | int]],
) -> None:
    _atomic_bytes(
        path,
        _canonical(
            {
                "best_epoch": best_epoch,
                "best_validation_nll": best_validation_nll,
                "checkpoint_sha256": _sha256(checkpoint_path),
                "epochs_completed": epochs_completed,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
                "lineage": lineage,
            }
        ),
    )


def _publish_result(
    output_dir: Path,
    *,
    lineage: dict[str, str],
    epochs_completed: int,
    best_epoch: int,
    best_validation_nll: float,
) -> RunResult:
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    class_map_path = output_dir / "class-map.json"
    artifacts = {
        checkpoint_path: _sha256(checkpoint_path),
        metrics_path: _sha256(metrics_path),
        class_map_path: _sha256(class_map_path),
    }
    _atomic_bytes(
        output_dir / "run-result.json",
        _canonical(
            {
                "artifacts": {path.name: digest for path, digest in artifacts.items()},
                "best_epoch": best_epoch,
                "best_validation_nll": best_validation_nll,
                "epochs_completed": epochs_completed,
                "lineage": lineage,
            }
        ),
    )
    return RunResult(epochs_completed, best_epoch, best_validation_nll, artifacts)


def _materialize(record: TrainingRecord, bundle: TrainingBundle, directory: Path) -> Path:
    target = directory / record.image_sha256
    if target.exists() and _sha256(target) == record.image_sha256:
        return target
    target.unlink(missing_ok=True)
    try:
        return download_verified(
            bundle.cos,
            bundle.bucket,
            record.object_key,
            target,
            record.image_sha256,
        )
    except (FileNotFoundError, IntegrityError) as exc:
        raise TrainingRefused(f"image bytes are missing or corrupt: {record.asset_id}") from exc


def _image_tensor(
    record: TrainingRecord,
    angle: str,
    bundle: TrainingBundle,
    cache_dir: Path,
    *,
    epoch: int,
    augment: bool,
) -> torch.Tensor:
    path = _materialize(record, bundle, cache_dir)
    try:
        with Image.open(path) as source:
            if getattr(source, "is_animated", False) or source.width * source.height > 100_000_000:
                raise TrainingRefused("training image is animated or too large")
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise TrainingRefused(f"training image cannot be decoded: {record.asset_id}") from exc
    image = crop_angle(image, angle)
    if augment:
        pixels = augment_pipeline(image, rng=random.Random(f"{SEED}:{epoch}:{record.asset_id}:{angle}"))
        image = Image.fromarray(pixels)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1) / 255.0
    return (tensor - _MEAN) / _STD


def _expanded_records(
    bundle: TrainingBundle, split: str
) -> list[tuple[TrainingRecord, str]]:
    selected = {
        asset_id for asset_id, assigned_split in bundle.split_by_asset.items() if assigned_split == split
    }
    expanded: list[tuple[TrainingRecord, str]] = []
    for record in bundle.records:
        if record.asset_id not in selected:
            continue
        angles = ANGLE_CROP_FRACTIONS if record.label is Label.REJECTION else (record.angle,)
        for angle in angles:
            assert isinstance(angle, str)
            expanded.append((record, angle))
    return expanded


def _batches(
    records: list[tuple[TrainingRecord, str]], *, epoch: int, shuffle: bool
) -> list[list[tuple[TrainingRecord, str]]]:
    by_angle: dict[str, list[tuple[TrainingRecord, str]]] = {
        angle: [] for angle in ANGLE_CROP_FRACTIONS
    }
    for item in records:
        by_angle[item[1]].append(item)
    batches: list[list[tuple[TrainingRecord, str]]] = []
    for angle, items in by_angle.items():
        if shuffle:
            random.Random(f"{SEED}:{epoch}:{angle}").shuffle(items)
        batches.extend(items[index : index + BATCH_SIZE] for index in range(0, len(items), BATCH_SIZE))
    if shuffle:
        random.Random(SEED + epoch).shuffle(batches)
    return batches


def _criteria(bundle: TrainingBundle, device: torch.device) -> dict[str, nn.CrossEntropyLoss]:
    criteria: dict[str, nn.CrossEntropyLoss] = {}
    for angle, counts in bundle.head_class_counts.items():
        total = sum(counts.values())
        weights = torch.tensor(
            [total / (len(ModelClass) * counts[model_class]) for model_class in ModelClass],
            dtype=torch.float32,
            device=device,
        )
        criteria[angle] = nn.CrossEntropyLoss(weight=weights)
    return criteria


def _epoch_loss(
    model: AngleHeadModel,
    bundle: TrainingBundle,
    split: str,
    cache_dir: Path,
    criteria: dict[str, nn.CrossEntropyLoss],
    device: torch.device,
    *,
    epoch: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.heads.train(training)
    model.backbone.eval()
    records = _expanded_records(bundle, split)
    if not records:
        raise TrainingRefused(f"{split} split has no trainable records")
    total_loss = 0.0
    total_examples = 0
    for batch in _batches(records, epoch=epoch, shuffle=training):
        angle = batch[0][1]
        pixels = torch.stack(
            [
                _image_tensor(record, angle, bundle, cache_dir, epoch=epoch, augment=training)
                for record, _ in batch
            ]
        ).to(device)
        labels = torch.tensor(
            [record.model_class.value for record, _ in batch], dtype=torch.long, device=device
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model({angle: pixels})[angle]
            loss = (
                criteria[angle](logits, labels)
                if training
                else nn.functional.cross_entropy(logits, labels)
            )
        if not torch.isfinite(loss):
            raise TrainingRefused("training produced a non-finite loss")
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(batch)
        total_examples += len(batch)
    return total_loss / total_examples


def run_training(bundle: TrainingBundle, spec: RunSpec, output_dir: Path) -> RunResult:
    """Train six-class per-angle heads and atomically publish verified artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_map_path = output_dir / "class-map.json"
    class_map_bytes = _canonical(_class_map())
    class_map_sha256 = hashlib.sha256(class_map_bytes).hexdigest()
    lineage = _lineage(bundle, spec, class_map_sha256)
    completed = _completed_result(output_dir, lineage, class_map_bytes)
    if completed is not None:
        return completed
    resumable = _resumable_state(output_dir, lineage, class_map_bytes)
    if resumable is None:
        _atomic_bytes(class_map_path, class_map_bytes)
    checkpoint_path = output_dir / "checkpoint.pt"
    metrics_path = output_dir / "metrics.json"
    history = [] if resumable is None else list(resumable.history)
    best_validation_nll = math.inf if resumable is None else resumable.best_validation_nll
    best_epoch = 0 if resumable is None else resumable.best_epoch
    epochs_without_improvement = (
        0 if resumable is None else resumable.epochs_without_improvement
    )
    start_epoch = 1 if resumable is None else resumable.epochs_completed + 1

    if start_epoch > MAX_EPOCHS or epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        return _publish_result(
            output_dir,
            lineage=lineage,
            epochs_completed=len(history),
            best_epoch=best_epoch,
            best_validation_nll=best_validation_nll,
        )

    _seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_trainable_model(bundle.head_class_counts).to(device)
    optimizer = torch.optim.AdamW(trainable_head_parameters(model), lr=LEARNING_RATE)
    criteria = _criteria(bundle, device)
    cache_dir = Path(tempfile.mkdtemp(prefix="watch-auth-training-inputs-"))
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            train_nll = _epoch_loss(
                model, bundle, TRAIN, cache_dir, criteria, device, epoch=epoch, optimizer=optimizer
            )
            validation_nll = _epoch_loss(
                model, bundle, VAL, cache_dir, criteria, device, epoch=epoch, optimizer=None
            )
            epoch_metrics = {
                "epoch": epoch,
                "train_nll": train_nll,
                "validation_nll": validation_nll,
            }
            if epoch < start_epoch:
                if epoch_metrics != history[epoch - 1]:
                    raise TrainingRefused("replayed training history does not match")
                continue
            history.append(epoch_metrics)
            if validation_nll < best_validation_nll:
                best_validation_nll = validation_nll
                best_epoch = epoch
                epochs_without_improvement = 0
                state = {
                    key: value.detach().cpu() for key, value in model.heads.state_dict().items()
                }
                _atomic_state_dict(checkpoint_path, state)
            else:
                epochs_without_improvement += 1
            _write_metrics(
                metrics_path,
                lineage=lineage,
                checkpoint_path=checkpoint_path,
                epochs_completed=len(history),
                best_epoch=best_epoch,
                best_validation_nll=best_validation_nll,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
            )
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    epochs_completed = len(history)
    if best_epoch == 0 or not checkpoint_path.exists() or not math.isfinite(best_validation_nll):
        raise TrainingRefused("training did not produce a finite best checkpoint")
    _load_head_state(checkpoint_path)
    return _publish_result(
        output_dir,
        lineage=lineage,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
    )


def run_training_waves(
    bundle: TrainingBundle, spec: RunSpec, output_dir: Path
) -> tuple[WaveRunResult, ...]:
    """Train the same seeded probe on nested 25/50/75/100% entity waves."""
    results: list[WaveRunResult] = []
    records_by_id = {record.asset_id: record for record in bundle.records}
    for fraction, asset_ids in entity_wave_asset_ids(bundle).items():
        wave_bundle = training_wave_bundle(bundle, asset_ids)
        result = run_training(
            wave_bundle,
            spec,
            output_dir / f"wave-{round(fraction * 100):03d}",
        )
        entity_count = len({records_by_id[asset_id].subject_id for asset_id in asset_ids})
        results.append(WaveRunResult(fraction, entity_count, result))
    return tuple(results)
