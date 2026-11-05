"""Temperature-scaling calibration + grade-4 ceiling/clip wrap (CK-15, CK-3, CK-1).

Reimplements Guo et al. 2017 (SKILL.md:157-168) rather than depending on the
abandoned `gpleiss/temperature_scaling` package (unmaintained per SKILL.md's
Sources & Verification table). `fit_temperature` MUST be called with logits
and labels drawn from the held-out validation split only — never the test
split — or the reported confidence stops meaning anything (Requirements
item 7). `fit_temperature_on_validation_split` is the one call site that
enforces that boundary: it reads `splits[VAL]` and nothing else, so a test
can hand it `logits_by_path`/`labels_by_path` mappings that simply contain
no TEST-split entries and prove the boundary by construction.

`calibrate` is the full reported-confidence path a single prediction goes
through: temperature -> `core.apply_grade4_ceiling` -> `core.clip_
confidence`. Both of those are the SAME functions `core/rules.py` already
defines and tests, imported here rather than re-implemented (CK-22) — this
module adds the temperature step and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from core.rules import apply_grade4_ceiling, clip_confidence
from core.types import ManifestRecord
from dataset.splits import VAL
from model.heads import GENUINE_CLASS_INDEX


def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Returns `softmax(logits / temperature)` (CK-15).

    Dividing logits by a temperature greater than 1 shrinks every gap
    between class scores before the softmax runs, which is exactly why a
    higher temperature can only ever flatten — never sharpen — the
    resulting distribution: `max probability` is monotonically
    non-increasing in `temperature` for any fixed logit vector.
    """
    return torch.softmax(logits / temperature, dim=-1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fits a single scalar temperature by minimizing NLL (SKILL.md:157-168).

    `logits`/`labels` cover many validation samples (shapes `[N,
    NUM_CLASSES]` / `[N]`), unlike `calibrate`'s single-sample inference
    use. Callers MUST source both from the held-out validation split —
    see `fit_temperature_on_validation_split` for the call site that
    enforces that.
    """
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def fit_temperature_on_validation_split(
    splits: Mapping[str, Sequence[ManifestRecord]],
    logits_by_path: Mapping[str, torch.Tensor],
    labels_by_path: Mapping[str, int],
) -> float:
    """Fits temperature from `splits[VAL]` rows only, never `splits[TEST]` (CK-15).

    Taking the whole `splits` mapping — rather than a single list the
    caller pre-sliced by hand — makes "the test split is never read" a
    property of this function's own indexing rather than something a
    caller has to get right: it looks up `splits[VAL]` and nothing else,
    so a caller can prove the boundary by handing it `logits_by_path`/
    `labels_by_path` mappings that contain no entries for TEST-split
    paths at all.
    """
    validation_records = splits[VAL]
    logits = torch.stack([logits_by_path[record.path] for record in validation_records])
    labels = torch.tensor(
        [labels_by_path[record.path] for record in validation_records], dtype=torch.long
    )
    return fit_temperature(logits, labels)


def calibrate(logits: torch.Tensor, grade: int, temperature: float) -> float:
    """Reported-confidence path for one prediction: temperature -> ceiling -> clip.

    `temperature` is the scalar `fit_temperature` already fit on
    validation — it has no default here on purpose, so a caller cannot
    silently skip calibration by omitting it. `grade` is forwarded as-is
    to `core.apply_grade4_ceiling`, which only caps the result when
    `grade` is grade-4 (superclone); any other value passes through
    unceilinged exactly as that function already defines (CK-3, CK-22).
    """
    probabilities = temperature_scale(logits, temperature)
    raw_genuine = probabilities[GENUINE_CLASS_INDEX].item()
    ceilinged = apply_grade4_ceiling(raw_genuine, grade)
    return clip_confidence(ceilinged)
