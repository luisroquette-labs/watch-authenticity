from __future__ import annotations

import importlib
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from PIL import Image

if TYPE_CHECKING:
    import torch
else:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

from acquisition.types import Angle  # noqa: E402
from core.rules import CONFIDENCE_CEILING, CONFIDENCE_FLOOR, GRADE4_GENUINE_CEILING  # noqa: E402
from core.types import ModelClass, Rejection  # noqa: E402
from inference import InferenceConclusion, InferenceResult, predict  # noqa: E402
from model.heads import NUM_CLASSES, AngleHeadModel  # noqa: E402
from reference_intelligence import VisualObservation, load_knowledge  # noqa: E402

REFERENCE = "rolex-submariner-126610ln"
CLAIMS = load_knowledge(Path("config/references/rolex-submariner-126610ln.json"), REFERENCE)
FRESH_AFTER = date(2026, 1, 1)


class StubModel:
    def __init__(self, class_index: int) -> None:
        self.class_index = class_index
        self.calls = 0

    def __call__(
        self, pixel_values_per_angle: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {angle: _logits(self.class_index) for angle in pixel_values_per_angle}


def _logits(class_index: int) -> torch.Tensor:
    logits = torch.full((1, NUM_CLASSES), -8.0)
    logits[0, class_index] = 8.0
    return logits


def _sharp_image() -> Image.Image:
    pixels = np.indices((64, 64)).sum(axis=0) % 2 * 255
    return Image.fromarray(pixels.astype(np.uint8), mode="L").convert("RGB")


def _blurry_image() -> Image.Image:
    return Image.new("RGB", (64, 64), color=(128, 128, 128))


def _tensor() -> torch.Tensor:
    return torch.rand(1, 3, 8, 8)


def _heatmap(angle: str, tensor: torch.Tensor) -> np.ndarray:
    del angle
    return np.eye(tensor.shape[-1], tensor.shape[-2], dtype=np.float32)


def _run(
    model: StubModel,
    *,
    image: Image.Image | None = None,
    observations: tuple[VisualObservation, ...] | None = None,
) -> InferenceResult | Rejection:
    return predict(
        {"dial": image or _sharp_image()},
        {"dial": _tensor()},
        model=model,
        temperature=1.0,
        reference_id=REFERENCE,
        claims=CLAIMS,
        observations=observations
        if observations is not None
        else (VisualObservation("dial_color", "black", Angle.DIAL, True),),
        minimum_source_date=FRESH_AFTER,
        heatmap_fn=_heatmap,
    )


def test_blurry_photo_short_circuits_before_model_and_heatmap() -> None:
    model = StubModel(ModelClass.GENUINE.value)
    heatmap_calls = 0

    def counted_heatmap(angle: str, tensor: torch.Tensor) -> Any:
        nonlocal heatmap_calls
        heatmap_calls += 1
        return _heatmap(angle, tensor)

    result = predict(
        {"dial": _blurry_image()},
        {"dial": _tensor()},
        model=model,
        temperature=1.0,
        reference_id=REFERENCE,
        claims=CLAIMS,
        observations=(),
        minimum_source_date=FRESH_AFTER,
        heatmap_fn=counted_heatmap,
    )

    assert isinstance(result, Rejection)
    assert result.reason == "blurry"
    assert model.calls == 0
    assert heatmap_calls == 0


def test_ood_short_circuits_after_one_model_call() -> None:
    model = StubModel(ModelClass.REJECTION.value)
    result = _run(model)

    assert isinstance(result, Rejection)
    assert result.reason == "not-evaluable"
    assert model.calls == 1


def test_genuine_visual_result_is_cited_but_never_authenticated() -> None:
    model = StubModel(ModelClass.GENUINE.value)
    result = _run(model)

    assert isinstance(result, InferenceResult)
    assert CONFIDENCE_FLOOR <= result.prediction.confidence <= CONFIDENCE_CEILING
    assert result.prediction.genuine
    assert result.prediction.grade is None
    assert result.conclusion is InferenceConclusion.NO_VISIBLE_INCONSISTENCY
    assert not result.reference.authenticates_watch
    assert result.reference.checks
    assert all(check.source_url.startswith("https://www.rolex.com/") for check in result.reference.checks)
    assert set(result.prediction.heatmap) == {"dial"}


def test_grade4_result_keeps_the_genuine_confidence_ceiling() -> None:
    model = StubModel(ModelClass.GRADE_4.value)
    result = _run(model)

    assert isinstance(result, InferenceResult)
    assert result.prediction.grade == 4
    assert not result.prediction.genuine
    assert result.prediction.ceiling_applied
    assert result.prediction.confidence <= GRADE4_GENUINE_CEILING
    assert result.conclusion is InferenceConclusion.FAKE_LIKELY


def test_technical_conflict_forces_inconclusive_without_changing_visual_score() -> None:
    model = StubModel(ModelClass.GENUINE.value)
    baseline = _run(model)
    conflict = _run(
        StubModel(ModelClass.GENUINE.value),
        observations=(VisualObservation("dial_color", "blue", Angle.DIAL, True),),
    )

    assert isinstance(baseline, InferenceResult)
    assert isinstance(conflict, InferenceResult)
    assert conflict.prediction.confidence == baseline.prediction.confidence
    assert conflict.conclusion is InferenceConclusion.INCONCLUSIVE
    assert conflict.next_angle is Angle.DIAL


def test_missing_technical_view_requests_one_smallest_angle() -> None:
    result = _run(StubModel(ModelClass.GENUINE.value), observations=())

    assert isinstance(result, InferenceResult)
    assert result.conclusion is InferenceConclusion.INCONCLUSIVE
    assert result.next_angle is Angle.CYCLOPS


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperature_refuses_before_model(temperature: float) -> None:
    model = StubModel(ModelClass.GENUINE.value)

    with pytest.raises(ValueError, match="temperature must be positive"):
        predict(
            {"dial": _sharp_image()},
            {"dial": _tensor()},
            model=model,
            temperature=temperature,
            reference_id=REFERENCE,
            claims=CLAIMS,
            observations=(),
            minimum_source_date=FRESH_AFTER,
            heatmap_fn=_heatmap,
        )

    assert model.calls == 0


def test_corrupt_model_logits_fail_before_calibration() -> None:
    class CorruptModel:
        def __call__(
            self, pixel_values_per_angle: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {angle: torch.full((1, NUM_CLASSES), float("nan")) for angle in pixel_values_per_angle}

    with pytest.raises(ValueError, match="logits must be finite"):
        predict(
            {"dial": _sharp_image()},
            {"dial": _tensor()},
            model=CorruptModel(),
            temperature=1.0,
            reference_id=REFERENCE,
            claims=CLAIMS,
            observations=(),
            minimum_source_date=FRESH_AFTER,
            heatmap_fn=_heatmap,
        )


def test_default_heatmap_path_wraps_one_real_angle_head(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyBackbone(torch.nn.Module):
        def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                pooler_output=torch.ones(pixel_values.shape[0], 768)
            )

    model = AngleHeadModel(TinyBackbone(), angles=("dial",))
    head = cast(torch.nn.Linear, model.heads["dial"])
    with torch.no_grad():
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.constant_(head.bias, -8.0)
        head.bias[ModelClass.GENUINE.value] = 8.0

    predict_module = importlib.import_module("inference.predict")

    def fake_grad_cam(classifier: Any, image: torch.Tensor) -> np.ndarray:
        assert classifier(image).shape == (1, NUM_CLASSES)
        return _heatmap("dial", image)

    monkeypatch.setattr(predict_module, "grad_cam", fake_grad_cam)
    result = predict(
        {"dial": _sharp_image()},
        {"dial": _tensor()},
        model=model,
        temperature=1.0,
        reference_id=REFERENCE,
        claims=CLAIMS,
        observations=(VisualObservation("dial_color", "black", Angle.DIAL, True),),
        minimum_source_date=FRESH_AFTER,
    )

    assert isinstance(result, InferenceResult)
    assert set(result.prediction.heatmap) == {"dial"}
