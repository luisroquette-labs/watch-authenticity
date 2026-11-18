# Classifier Training and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a real 126610LN classifier only on a ready frozen dataset and release it only after independent evaluation passes.

**Architecture:** Reuse `training/run.py` entity waves, frozen DINOv2 backbone, IBM cost guard and `evaluation/release_gate.py`. The local color pilot remains separate until an exact checkpoint, calibration, freeze and release decision can be traced together.

**Tech Stack:** Python 3.12, PyTorch/DINOv2, IBM VPC GPU and COS, pytest.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`, sub-tasks `06-backbone-heads.md`, `10-evaluate.md`, `11-release-gate.md`; parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md` Tasks 4–7.

## Global Constraints

- `dataset/readiness.py` floors and rights/evidence checks must pass on the exact freeze before GPU starts. Current archive candidates do not satisfy them.
- Split by watch/repost/seller/batch; validation and final holdouts never become train or wave-selection data.
- No GPU run without IBM credit, eligible SKU, cost ceiling, TTL and teardown guard. No paid AI fallback.
- One GPU attempt is capped at 90 minutes, with 30 minutes reserved for artifact verification and teardown.

## Review Focus

1. Same watch appears on two sites: Task 1 keeps all photos in one split.
2. A wave adds only duplicate photos: Task 2 refuses wasted training.
3. Checkpoint has mismatched dataset hash: Task 2 refuses resume or evaluation.
4. High average accuracy hides unseen-source failures: Task 3 blocks release.
5. Model abstains on every watch: Task 3 blocks release despite low false reassurance.

## File Structure and Ownership

- `dataset/entities.py`, `dataset/splits.py`, `dataset/freeze.py` own independent identity and split hashes.
- `training/data.py`, `training/run.py`, `training/worker.py` own input attestation, waves, checkpoint and GPU worker.
- `evaluation/evaluate.py`, `evaluation/release_gate.py` own class/slice metrics and release refusal.
- Tests: `tests/test_entities.py`, `tests/test_splits.py`, `tests/test_training_data.py`, `tests/test_training_run.py`, `tests/test_evaluate.py`, `tests/test_release_gate.py`.

### Task 1: Freeze an independent evaluation set

**Files:** Reuse dataset owners; modify only when an observed cross-site fixture fails.

**Interfaces:** `assign_watch_entities(...)`, `build_splits(...)`, `assert_no_connected_component_crosses_splits(...)`, freeze hash.

- [ ] Add one sanitized, real cross-site watch/re-encoded-crop fixture and a distinct lookalike; prove grouping neither leaks one watch nor merges two watches.
- [ ] Run `pytest tests/test_entities.py tests/test_splits.py tests/test_training_data.py -q`.
- [ ] Freeze train, validation, standard, cross-source, factory, temporal and gold cohorts before training; save class/angle/source/entity census and exact hash.

**Acceptance:** No known physical watch or connected repost component crosses splits; holdout rows are not trainable or used to tune thresholds.

### Task 2: Train only a useful measured wave

**Files:** Reuse `training/run.py`, `training/worker.py`; focused test `tests/test_training_run.py`.

**Interfaces:** `run_training_waves(...) -> tuple[WaveRunResult, ...]`; persisted checkpoint, freeze/model hashes and IBM cost/teardown receipts.

- [ ] Confirm `evaluate_readiness(...).ready` and rights checks on the exact freeze; if false, record deficit and stop this task before GPU allocation.
- [ ] Verify duplicate-only wave refusal and exact-lineage resume with `pytest tests/test_training_run.py -q`.
- [ ] Execute one bounded IBM GPU wave; persist checkpoint, complete metrics, input freeze hash, runtime/cost and teardown state.
- [ ] Compare the next independent-entity wave only if the first identifies an eligible underperforming cell and a predeclared measurable gain; keep final holdouts sealed.

**Acceptance:** A candidate checkpoint is reproducible from its freeze and saved receipts. Training does not imply release.

### Task 3: Decide release from independent errors

**Files:** Reuse `evaluation/evaluate.py`, `evaluation/release_gate.py`; tests `tests/test_evaluate.py`, `tests/test_release_gate.py`.

**Interfaces:** `EvaluationReport`, `release_gate(...) -> ReleaseDecision`; calibrated model artifact and exact freeze hash.

- [ ] Run focused tests for high-quality-fake false reassurance, insufficient holdout, all-inconclusive output and source regression.
- [ ] Evaluate standard, cross-source, unseen-factory, temporal and Tier-A gold cohorts on the saved checkpoint; report watch-level numerator, denominator and confidence bounds.
- [ ] Refuse release unless every existing gate passes on the same hash, including one-sided 95% false-reassurance upper bound ≤5% and ≥20% answered coverage separately for genuine/fake.
- [ ] Only after a pass, connect the released artifact to the real final inference route and perform one real upload; compare persisted result with checkpoint and model card.

**Acceptance:** Release has a dated exact-SHA decision and real inference proof; otherwise status remains candidate or blocked, with the failed cohort named.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: existing wave and release gates are implemented; train readiness and real cohorts are the gap. | Plan proof first. |
| 2 | Yes: keep final holdout sealed during wave choice and bind real inference to checkpoint hash. | Tighten Tasks 2–3. |
| 3 | No: split, train and release boundaries are explicit. | Keep. |
| 4 | No: further model design depends on confirmed data and measured errors. | Stop after two consecutive unchanged passes. |
