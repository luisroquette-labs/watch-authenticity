"""End-to-end contracts for the immutable six-class training producer."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import training.run as training_run
from adapters.backbone import BACKBONE_EMBED_DIM, BACKBONE_REVISION
from core.types import ModelClass
from dataset.crop import ANGLE_CROP_FRACTIONS
from infra.ibm_types import RunSpec
from model.heads import build_trainable_model
from tests.conftest import valid_run_spec
from training.data import TrainingDataRefused, load_training_data
from training.run import RunResult, TrainingRefused, run_training, run_training_waves


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes | Path) -> str:
    body = value.read_bytes() if isinstance(value, Path) else value
    return hashlib.sha256(body).hexdigest()


def _png(index: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), (index % 255, (index * 7) % 255, (index * 13) % 255)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


class _NotFound(Exception):
    def __init__(self) -> None:
        self.response = {"Error": {"Code": "NoSuchKey"}}


class FakeCOS:
    def __init__(self, spec: RunSpec) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self._watch_auth_bucket = spec.bucket
        self._watch_auth_prefix = spec.prefix
        self._watch_auth_configuration_verified = True

    def put(self, key: str, body: bytes, *, metadata_sha256: str | None = None) -> None:
        self.objects[key] = (
            body,
            {
                "sha256": metadata_sha256 or _sha256(body),
                "length": str(len(body)),
            },
        )

    def head_object(self, *, Bucket: str, Key: str) -> dict[str, object]:
        assert Bucket == self._watch_auth_bucket
        try:
            body, metadata = self.objects[Key]
        except KeyError as exc:
            raise _NotFound() from exc
        return {"Metadata": metadata, "ContentLength": len(body), "ETag": '"fixture"'}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        assert bucket == self._watch_auth_bucket
        Path(filename).write_bytes(self.objects[key][0])


def _record(
    spec: RunSpec,
    asset_id: str,
    image: bytes,
    *,
    label: str,
    grade: int | None,
    angle: str | None,
    source_domain: str,
    subject_id: str,
    provenance_id: str,
    rights: str = "licensed",
    factory: str | None = None,
    rejection_subtype: str | None = None,
) -> dict[str, object]:
    digest = _sha256(image)
    return {
        "asset_id": asset_id,
        "object_key": f"{spec.prefix}/inputs/images/{digest}",
        "image_sha256": digest,
        "label": label,
        "grade": grade,
        "angle": angle,
        "source_domain": source_domain,
        "subject_id": subject_id,
        "provenance_id": provenance_id,
        "rights": rights,
        "rights_record_id": f"rights-{asset_id}",
        "evidence_tier": "b",
        "factory": factory,
        "rejection_subtype": rejection_subtype,
    }


def _valid_payloads(spec: RunSpec) -> tuple[dict[str, object], dict[str, object], dict[str, bytes]]:
    records: list[dict[str, object]] = []
    images: dict[str, bytes] = {}
    index = 1
    for angle in ANGLE_CROP_FRACTIONS:
        genuine = _png(index)
        index += 1
        genuine_record = _record(
            spec,
            f"genuine-{angle}",
            genuine,
            label="genuine",
            grade=None,
            angle=angle,
            source_domain="dealer-one.com" if index % 2 else "dealer-two.com",
            subject_id=f"genuine-watch-{angle}",
            provenance_id=f"genuine-proof-{angle}",
        )
        records.append(genuine_record)
        images[str(genuine_record["object_key"])] = genuine
        for grade in range(1, 5):
            fake = _png(index)
            index += 1
            fake_record = _record(
                spec,
                f"grade-{grade}-{angle}",
                fake,
                label="fake",
                grade=grade,
                angle=angle,
                source_domain="replica-one.com" if grade % 2 else "replica-two.com",
                subject_id=f"fake-watch-{grade}-{angle}",
                provenance_id=f"fake-proof-{grade}-{angle}",
                factory="vsf" if grade == 4 else None,
            )
            records.append(fake_record)
            images[str(fake_record["object_key"])] = fake

    for subtype in ("no-watch", "non-submariner"):
        rejection = _png(index)
        index += 1
        rejection_record = _record(
            spec,
            f"rejection-{subtype}",
            rejection,
            label="rejection",
            grade=None,
            angle=None,
            source_domain="owner-upload.com",
            subject_id=f"negative-{subtype}",
            provenance_id=f"owner-proof-{subtype}",
            rights="owner",
            rejection_subtype=subtype,
        )
        records.append(rejection_record)
        images[str(rejection_record["object_key"])] = rejection

    train_records = list(records)
    validation_records: list[dict[str, object]] = []
    for source_record in train_records:
        validation = _png(index)
        index += 1
        validation_record = _record(
            spec,
            f"validation-{source_record['asset_id']}",
            validation,
            label=str(source_record["label"]),
            grade=source_record["grade"] if isinstance(source_record["grade"], int) else None,
            angle=source_record["angle"] if isinstance(source_record["angle"], str) else None,
            source_domain=str(source_record["source_domain"]),
            subject_id=f"validation-{source_record['subject_id']}",
            provenance_id=f"validation-{source_record['provenance_id']}",
            rights=str(source_record["rights"]),
            rejection_subtype=(
                str(source_record["rejection_subtype"])
                if source_record["rejection_subtype"] is not None
                else None
            ),
        )
        validation_records.append(validation_record)
        images[str(validation_record["object_key"])] = validation
    records.extend(validation_records)

    splits = {
        "schema_version": 1,
        "splits": {
            "train": [str(record["asset_id"]) for record in train_records],
            "val": [str(record["asset_id"]) for record in validation_records],
            "test": [],
            "cross_source": [],
            "factory_holdout": [],
            "temporal_holdout": [],
            "gold": [],
        },
    }
    split_sha256 = _sha256(_canonical(splits))
    freeze = {
        "manifest_sha256": _sha256(_canonical(records)),
        "split_sha256": split_sha256,
        "policy_set_sha256": "a" * 64,
        "eligible_asset_ids": sorted(str(record["asset_id"]) for record in records),
        "excluded_asset_ids": [],
        "exclusion_reason_counts": [],
        "readiness_sha256": "b" * 64,
        "rights_sha256": "c" * 64,
        "exclusions": [],
        "blocked": False,
        "asset_rights": sorted(
            [str(record["asset_id"]), str(record["rights_record_id"])] for record in records
        ),
    }
    manifest = {"schema_version": 2, "freeze": freeze, "records": records}
    return manifest, splits, images


def _seed(
    manifest: dict[str, object],
    splits: dict[str, object],
    images: dict[str, bytes],
    *,
    missing_key: str | None = None,
    bad_image_metadata_key: str | None = None,
    refresh_freeze: bool = True,
) -> tuple[FakeCOS, RunSpec]:
    if refresh_freeze and isinstance(manifest.get("freeze"), dict):
        records = manifest["records"]
        assert isinstance(records, list)
        freeze = manifest["freeze"]
        assert isinstance(freeze, dict)
        freeze["manifest_sha256"] = _sha256(_canonical(records))
        freeze["split_sha256"] = _sha256(_canonical(splits))
        freeze["eligible_asset_ids"] = sorted(str(record["asset_id"]) for record in records)
        freeze["asset_rights"] = sorted(
            [str(record["asset_id"]), str(record["rights_record_id"])] for record in records
        )
    manifest_bytes = _canonical(manifest)
    splits_bytes = _canonical(splits)
    spec = replace(
        valid_run_spec(),
        manifest_sha256=_sha256(manifest_bytes),
        splits_sha256=_sha256(splits_bytes),
    )
    cos = FakeCOS(spec)
    cos.put(f"{spec.prefix}/inputs/manifest/{spec.manifest_sha256}.json", manifest_bytes)
    cos.put(f"{spec.prefix}/inputs/splits/{spec.splits_sha256}.json", splits_bytes)
    for key, body in images.items():
        if key != missing_key:
            cos.put(key, body, metadata_sha256="0" * 64 if key == bad_image_metadata_key else None)
    return cos, spec


@pytest.fixture
def training_fixture() -> tuple[FakeCOS, RunSpec, dict[str, object], dict[str, object], dict[str, bytes]]:
    base = valid_run_spec()
    manifest, splits, images = _valid_payloads(base)
    cos, spec = _seed(manifest, splits, images)
    return cos, spec, manifest, splits, images


def test_loader_accepts_complete_content_addressed_freeze(training_fixture):
    cos, spec, _manifest, _splits, _images = training_fixture

    bundle = load_training_data(cos, spec)

    assert set(bundle.class_counts) == set(ModelClass)
    assert all(count > 0 for count in bundle.class_counts.values())
    assert set(bundle.head_class_counts) == set(ANGLE_CROP_FRACTIONS)
    assert all(
        set(counts) == set(ModelClass) and all(count > 0 for count in counts.values())
        for counts in bundle.head_class_counts.values()
    )
    assert bundle.manifest_sha256 == spec.manifest_sha256
    assert bundle.splits_sha256 == spec.splits_sha256
    assert bundle.freeze.split_sha256 == spec.splits_sha256


def test_loader_refuses_missing_or_blocked_dataset_freeze(training_fixture):
    _cos, spec, manifest, splits, images = training_fixture
    manifest["freeze"]["blocked"] = True
    cos, spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused, match="freeze is blocked"):
        load_training_data(cos, spec)


def test_wave_runner_uses_four_nested_entity_waves(training_fixture, monkeypatch, tmp_path):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    called: list[Path] = []

    def fake_run(_bundle, _spec, output_dir):
        called.append(output_dir)
        return RunResult(1, 1, 0.5, {})

    monkeypatch.setattr(training_run, "run_training", fake_run)

    results = run_training_waves(bundle, spec, tmp_path)

    assert tuple(item.fraction for item in results) == (0.25, 0.5, 0.75, 1.0)
    assert [path.name for path in called] == ["wave-025", "wave-050", "wave-075", "wave-100"]


@pytest.mark.parametrize("failure", ["placeholder", "single-source", "missing", "bad-metadata"])
def test_loader_refuses_untrusted_or_missing_inputs(training_fixture, failure: str):
    _cos, _spec, manifest, splits, images = training_fixture
    records = manifest["records"]
    assert isinstance(records, list)
    missing_key = None
    bad_key = None
    if failure == "placeholder":
        records[0]["source_domain"] = "photos.example.com"
    elif failure == "single-source":
        for record in records:
            if record["label"] == "genuine":
                record["source_domain"] = "only-genuine-source.com"
    elif failure == "missing":
        missing_key = str(records[0]["object_key"])
    else:
        bad_key = str(records[0]["object_key"])
    cos, spec = _seed(
        manifest,
        splits,
        images,
        missing_key=missing_key,
        bad_image_metadata_key=bad_key,
    )

    with pytest.raises(TrainingDataRefused):
        load_training_data(cos, spec)


@pytest.mark.parametrize("grade", [0, 5])
def test_loader_refuses_fake_grades_outside_market_taxonomy(training_fixture, grade: int):
    _cos, _spec, manifest, splits, images = training_fixture
    records = manifest["records"]
    assert isinstance(records, list)
    next(record for record in records if record["label"] == "fake")["grade"] = grade
    cos, spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused, match="between 1 and 4"):
        load_training_data(cos, spec)


@pytest.mark.parametrize("failure", ["overlap", "empty-class", "no-rejection", "bad-rights"])
def test_loader_refuses_incomplete_or_leaking_training_set(training_fixture, failure: str):
    _cos, _spec, manifest, splits, images = training_fixture
    records = manifest["records"]
    split_map = splits["splits"]
    assert isinstance(records, list) and isinstance(split_map, dict)
    if failure == "overlap":
        split_map["val"].append(split_map["train"][0])
    elif failure == "empty-class":
        removed = {record["asset_id"] for record in records if record["grade"] == 4}
        split_map["train"] = [asset for asset in split_map["train"] if asset not in removed]
        records[:] = [record for record in records if record["asset_id"] not in removed]
    elif failure == "no-rejection":
        removed = {record["asset_id"] for record in records if record["label"] == "rejection"}
        split_map["train"] = [asset for asset in split_map["train"] if asset not in removed]
        records[:] = [record for record in records if record["asset_id"] not in removed]
    else:
        next(record for record in records if record["label"] == "rejection")["rights"] = "unknown"
    cos, spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused):
        load_training_data(cos, spec)


def test_rejection_provenance_and_subject_groups_cannot_cross_splits(training_fixture):
    _cos, spec, manifest, splits, images = training_fixture
    records = manifest["records"]
    split_map = splits["splits"]
    assert isinstance(records, list) and isinstance(split_map, dict)
    original = next(record for record in records if record["label"] == "rejection")
    duplicate_image = _png(250)
    duplicate = _record(
        spec,
        "rejection-leak",
        duplicate_image,
        label="rejection",
        grade=None,
        angle=None,
        source_domain="owner-upload.com",
        subject_id=str(original["subject_id"]),
        provenance_id=str(original["provenance_id"]),
        rights="owner",
        rejection_subtype="no-watch",
    )
    records.append(duplicate)
    split_map["val"].append("rejection-leak")
    images[str(duplicate["object_key"])] = duplicate_image
    cos, final_spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused, match="rejection group"):
        load_training_data(cos, final_spec)


def test_same_image_bytes_cannot_cross_splits_under_different_asset_ids(training_fixture):
    _cos, _spec, manifest, splits, images = training_fixture
    records = manifest["records"]
    assert isinstance(records, list)
    train_record = records[0]
    validation_record = next(
        record for record in records if record["asset_id"] == f"validation-{train_record['asset_id']}"
    )
    validation_record["image_sha256"] = train_record["image_sha256"]
    validation_record["object_key"] = train_record["object_key"]
    cos, final_spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused, match="image content crosses"):
        load_training_data(cos, final_spec)


class _StubBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(BACKBONE_EMBED_DIM), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> object:
        pooled = pixel_values.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return type(
            "BackboneOutput",
            (),
            {"pooler_output": pooled.expand(pixel_values.shape[0], BACKBONE_EMBED_DIM)},
        )()


class _UnsafePayload:
    pass


def test_two_epoch_cpu_training_writes_verified_head_only_artifacts(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 2)
    monkeypatch.setattr(training_run, "EARLY_STOPPING_PATIENCE", 2)
    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda counts: build_trainable_model(counts, backbone=_StubBackbone()),
    )

    result = run_training(bundle, spec, tmp_path)

    assert result.epochs_completed == 2
    assert math.isfinite(result.best_validation_nll)
    assert {path.name for path in result.artifacts} == {
        "checkpoint.pt",
        "metrics.json",
        "class-map.json",
    }
    for path, digest in result.artifacts.items():
        assert path.exists()
        assert _sha256(path) == digest
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=True, map_location="cpu")
    assert checkpoint
    assert all(key.split(".", 1)[0] in ANGLE_CROP_FRACTIONS for key in checkpoint)
    assert not any("backbone" in key for key in checkpoint)
    assert json.loads((tmp_path / "run-result.json").read_text())["lineage"]["code_sha"] == spec.code_sha
    assert not (tmp_path / ".inputs").exists()

    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda _counts: pytest.fail("valid completed run must resume without retraining"),
    )
    assert run_training(bundle, spec, tmp_path) == result


def test_resume_refuses_mismatched_lineage(training_fixture, tmp_path: Path):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)

    with pytest.raises(TrainingRefused, match="lineage"):
        run_training(bundle, replace(spec, manifest_sha256="f" * 64), tmp_path)


def test_interrupted_run_resumes_only_matching_head_state_and_metadata(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 1)
    monkeypatch.setattr(training_run, "EARLY_STOPPING_PATIENCE", 2)
    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda counts: build_trainable_model(counts, backbone=_StubBackbone()),
    )
    first = run_training(bundle, spec, tmp_path)
    assert first.epochs_completed == 1
    (tmp_path / "run-result.json").unlink()
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 2)

    resumed = run_training(bundle, spec, tmp_path)

    assert resumed.epochs_completed == 2
    assert len(json.loads((tmp_path / "metrics.json").read_text())["history"]) == 2


def test_interrupted_run_replays_optimizer_trajectory_after_non_improving_epoch(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    original_epoch_loss = training_run._epoch_loss
    validation_losses = {1: 1.0, 2: 2.0, 3: 0.5}

    def controlled_epoch_loss(*args, **kwargs):
        if args[2] == "val":
            return validation_losses[kwargs["epoch"]]
        return original_epoch_loss(*args, **kwargs)

    def fixture_tensor(record, _angle, _bundle, _cache_dir, *, epoch, augment):
        value = (record.model_class.value + epoch + int(augment)) / 10
        return torch.full((3, 8, 8), value)

    monkeypatch.setattr(training_run, "EARLY_STOPPING_PATIENCE", 5)
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 3)
    monkeypatch.setattr(training_run, "_epoch_loss", controlled_epoch_loss)
    monkeypatch.setattr(training_run, "_image_tensor", fixture_tensor)
    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda counts: build_trainable_model(counts, backbone=_StubBackbone()),
    )

    continuous_dir = tmp_path / "continuous"
    resumed_dir = tmp_path / "resumed"
    run_training(bundle, spec, continuous_dir)
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 2)
    run_training(bundle, spec, resumed_dir)
    (resumed_dir / "run-result.json").unlink()
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 3)
    run_training(bundle, spec, resumed_dir)

    continuous_state = torch.load(
        continuous_dir / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    resumed_state = torch.load(
        resumed_dir / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    assert continuous_state.keys() == resumed_state.keys()
    assert all(
        torch.equal(continuous_state[key], resumed_state[key]) for key in continuous_state
    )
    assert (continuous_dir / "metrics.json").read_bytes() == (
        resumed_dir / "metrics.json"
    ).read_bytes()


def test_validation_nll_is_invariant_to_batch_size(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    training_run._seed_everything()
    device = torch.device("cpu")
    model = build_trainable_model(bundle.head_class_counts, backbone=_StubBackbone()).to(device)
    criteria = training_run._criteria(bundle, device)

    monkeypatch.setattr(training_run, "BATCH_SIZE", 1)
    single = training_run._epoch_loss(
        model,
        bundle,
        "val",
        tmp_path / "single",
        criteria,
        device,
        epoch=1,
        optimizer=None,
    )
    monkeypatch.setattr(training_run, "BATCH_SIZE", 32)
    grouped = training_run._epoch_loss(
        model,
        bundle,
        "val",
        tmp_path / "grouped",
        criteria,
        device,
        epoch=1,
        optimizer=None,
    )

    assert grouped == pytest.approx(single, abs=1e-6)


def test_resume_refuses_serialized_python_object(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    monkeypatch.setattr(training_run, "MAX_EPOCHS", 1)
    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda counts: build_trainable_model(counts, backbone=_StubBackbone()),
    )
    run_training(bundle, spec, tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"unsafe": _UnsafePayload()}, checkpoint_path)
    checkpoint_digest = _sha256(checkpoint_path)
    metrics_path = tmp_path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["checkpoint_sha256"] = checkpoint_digest
    metrics_path.write_bytes(_canonical(metrics))
    result_path = tmp_path / "run-result.json"
    result = json.loads(result_path.read_text())
    result["artifacts"]["checkpoint.pt"] = checkpoint_digest
    result["artifacts"]["metrics.json"] = _sha256(metrics_path)
    result_path.write_bytes(_canonical(result))

    with pytest.raises(TrainingRefused, match="safe tensor"):
        run_training(bundle, spec, tmp_path)


def test_training_verifies_image_bytes_before_use(
    training_fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cos, spec, manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    records = manifest["records"]
    assert isinstance(records, list)
    key = str(records[0]["object_key"])
    _body, metadata = cos.objects[key]
    cos.objects[key] = (b"corrupt after preflight", metadata)
    monkeypatch.setattr(
        training_run,
        "build_trainable_model",
        lambda counts: build_trainable_model(counts, backbone=_StubBackbone()),
    )

    with pytest.raises(TrainingRefused, match="missing or corrupt"):
        run_training(bundle, spec, tmp_path)


def test_container_bakes_pinned_offline_backbone_and_training_package():
    containerfile = Path("Containerfile.train").read_text()
    backbone = Path("adapters/backbone.py").read_text()

    assert BACKBONE_REVISION == "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
    assert "COPY --chown=10001:10001 infra infra" in containerfile
    assert "COPY --chown=10001:10001 training training" in containerfile
    assert "AutoModel.from_pretrained" in containerfile
    assert BACKBONE_REVISION in containerfile
    assert "HF_HUB_OFFLINE=1" in containerfile
    assert "local_files_only=True" in backbone
