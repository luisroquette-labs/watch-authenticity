"""Ordered, reference-aware inference orchestration."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any, Protocol

import torch
from torch import nn

from acquisition.types import Angle
from calibration.temperature_scaling import calibrate
from core.rules import aggregate_angles
from core.types import Grade, ModelClass, Prediction, Rejection
from explainability.gradcam import grad_cam
from model.heads import GUIDED_ANGLES, AngleHeadModel
from model.rejection import classify_logits
from quality_gate.blur_light_check import quality_gate
from reference_intelligence.compare import assess_reference
from reference_intelligence.types import (
    AssessmentStatus,
    CheckStatus,
    ReferenceAssessment,
    TechnicalClaim,
    VisualObservation,
)


class InferenceModel(Protocol):
    def __call__(self, pixel_values_per_angle: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


class InferenceConclusion(Enum):
    FAKE_LIKELY = "fake_likely"
    NO_VISIBLE_INCONSISTENCY = "no_visible_inconsistency"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class InferenceResult:
    prediction: Prediction
    reference: ReferenceAssessment
    conclusion: InferenceConclusion
    next_angle: Angle | None


HeatmapFn = Callable[[str, torch.Tensor], Any]


def predict(
    images_per_angle: Mapping[str, Any],
    pixel_values_per_angle: dict[str, torch.Tensor],
    *,
    model: InferenceModel,
    temperature: float,
    reference_id: str,
    claims: tuple[TechnicalClaim, ...],
    observations: tuple[VisualObservation, ...],
    minimum_source_date: date,
    heatmap_fn: HeatmapFn | None = None,
) -> InferenceResult | Rejection:
    """Return a cited result or short-circuit before authenticity scoring."""
    _validate_inputs(images_per_angle, pixel_values_per_angle, observations, temperature)
    for angle, image in images_per_angle.items():
        quality = quality_gate(image, angle=angle)
        if isinstance(quality, Rejection):
            return quality

    logits_per_angle = model(pixel_values_per_angle)
    if set(logits_per_angle) != set(pixel_values_per_angle):
        raise ValueError("model logits must align with submitted angles")
    _validate_logits(logits_per_angle)
    rejection = classify_logits(logits_per_angle)
    if rejection is not None:
        return rejection

    predicted_class = _aggregate_class(logits_per_angle)
    grade = None if predicted_class is ModelClass.GENUINE else predicted_class.value
    calibrated_per_angle = {
        angle: calibrate(logits.squeeze(0), grade=grade or 0, temperature=temperature)
        for angle, logits in logits_per_angle.items()
    }
    confidence = aggregate_angles(calibrated_per_angle)["raw_genuine"]
    heatmaps = {
        angle: (
            heatmap_fn(angle, pixel_values)
            if heatmap_fn is not None
            else _default_heatmap(model, angle, pixel_values)
        )
        for angle, pixel_values in pixel_values_per_angle.items()
    }
    prediction = Prediction(
        confidence=confidence,
        genuine=predicted_class is ModelClass.GENUINE,
        grade=grade,
        ceiling_applied=grade == Grade.SUPERCLONE,
        heatmap=heatmaps,
    )
    reference = assess_reference(
        reference_id,
        claims,
        observations,
        minimum_source_date=minimum_source_date,
    )
    return InferenceResult(
        prediction=prediction,
        reference=reference,
        conclusion=_conclusion(prediction, reference),
        next_angle=_next_angle(reference),
    )


def _validate_inputs(
    images_per_angle: Mapping[str, Any],
    pixel_values_per_angle: Mapping[str, torch.Tensor],
    observations: tuple[VisualObservation, ...],
    temperature: float,
) -> None:
    if not images_per_angle or set(images_per_angle) != set(pixel_values_per_angle):
        raise ValueError("images and tensors must contain the same non-empty angles")
    if not set(images_per_angle).issubset(GUIDED_ANGLES):
        raise ValueError("input contains an unsupported guided angle")
    submitted_angles = {Angle(angle) for angle in images_per_angle}
    if any(observation.angle not in submitted_angles for observation in observations):
        raise ValueError("visual observations must belong to a submitted image angle")
    if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    for tensor in pixel_values_per_angle.values():
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError("each angle tensor must be a batch of one image")


def _aggregate_class(logits_per_angle: Mapping[str, torch.Tensor]) -> ModelClass:
    probabilities = torch.stack(
        [torch.softmax(logits.squeeze(0), dim=-1) for logits in logits_per_angle.values()]
    )
    scoreable = probabilities.mean(dim=0)[: ModelClass.REJECTION.value]
    return ModelClass(int(scoreable.argmax().item()))


def _validate_logits(logits_per_angle: Mapping[str, torch.Tensor]) -> None:
    expected_shape = (1, len(ModelClass))
    if any(tuple(logits.shape) != expected_shape for logits in logits_per_angle.values()):
        raise ValueError("each angle must return one complete class-logit row")
    if any(not bool(torch.isfinite(logits).all()) for logits in logits_per_angle.values()):
        raise ValueError("model logits must be finite")


def _conclusion(
    prediction: Prediction, reference: ReferenceAssessment
) -> InferenceConclusion:
    if not prediction.genuine:
        return InferenceConclusion.FAKE_LIKELY
    if reference.status is not AssessmentStatus.CONSISTENT:
        return InferenceConclusion.INCONCLUSIVE
    return InferenceConclusion.NO_VISIBLE_INCONSISTENCY


def _next_angle(reference: ReferenceAssessment) -> Angle | None:
    conflicting = next(
        (
            check.angle
            for check in reference.checks
            if check.status is CheckStatus.CONFLICTING and check.angle is not None
        ),
        None,
    )
    if conflicting is not None:
        return conflicting
    missing = {
        check.angle
        for check in reference.checks
        if check.status is CheckStatus.UNOBSERVABLE and check.angle is not None
    }
    return next((Angle(angle) for angle in GUIDED_ANGLES if Angle(angle) in missing), None)


def _default_heatmap(model: InferenceModel, angle: str, image: torch.Tensor) -> Any:
    if not isinstance(model, AngleHeadModel):
        raise TypeError("a non-AngleHeadModel requires heatmap_fn")
    return grad_cam(_AngleClassifier(model, angle), image)


class _AngleClassifier(nn.Module):
    """Expose one angle head as the plain classifier Grad-CAM expects."""

    def __init__(self, model: AngleHeadModel, angle: str) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.heads[angle]

    @property
    def blocks(self) -> Any:
        return getattr(self.backbone, "blocks", None)

    @property
    def encoder(self) -> Any:
        return getattr(self.backbone, "encoder", None)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(pixel_values).pooler_output)
