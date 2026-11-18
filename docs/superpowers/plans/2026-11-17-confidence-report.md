# Calibrated Confidence and User Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show calibrated uncertainty only for a released trained model, and explain observable evidence without false assurance.

**Architecture:** Keep calibration in `calibration/temperature_scaling.py`, final inference in `inference/predict.py`, and the local pilot report in `pilot/server.py`. The pilot's current null model/confidence fields are a correct representation of its state.

**Tech Stack:** Python 3.12, PyTorch, pytest, local HTTP service.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`, sub-tasks `07b-calibration.md`, `09-inference-predict.md`, `11-release-gate.md`; parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md` Tasks 7 and 10.

## Global Constraints

- Temperature is fitted on held-out validation, never final test; all reported confidence remains within [0.1%, 99.9%].
- A grade-4-consistent result respects the specified genuine-confidence ceiling. Confidence is never the 0–1,000 fake-quality score.
- No model probability, hash or calibration is shown as available in the local visual pilot before a released artifact is actually loaded.
- A visible reference match does not authenticate a watch. One unit is at most 120 minutes.

## Review Focus

1. Untrained pilot upload: Task 1 keeps confidence null.
2. A confident model and conflicting official visible fact: Task 2 remains cautious.
3. An unseen-source fake receives `no_visible_inconsistency`: Task 2 counts false reassurance.
4. Candidate-only “superclone” text: Task 1 never becomes a grade or score.
5. A rejected or unseen detail: Task 1 never becomes cited observation.

## File Structure and Ownership

- `calibration/temperature_scaling.py`, `inference/predict.py`, `core/rules.py` own probability, grade ceiling and conclusion.
- `evaluation/evaluate.py`, `evaluation/release_gate.py` own calibration/error measurement and release refusal.
- `pilot/server.py` owns current user-visible evidence inspection and null model fields.
- Tests: `tests/test_calibration.py`, `tests/test_predict.py`, `tests/test_release_gate.py`, `tests/test_pilot_server.py`.

### Task 1: Preserve honest untrained reporting

**Files:** Reuse `pilot/server.py`; tests above. Edit only after a failing actual response.

**Interfaces:** `/predict/batch` `joint` fields `model_sha256`, `calibrated_confidence`, `visible_evidence`, `unverified`, `next_useful_angle`.

- [ ] Send one clear and one insufficient local image to the actual endpoint; retain response and source URLs.
- [ ] Check `pytest tests/test_pilot_server.py tests/test_predict.py -q`, including rejected-photo citation, duplicate count and null confidence.
- [ ] If real output states an unobserved detail, reproduce it as one failing test and correct the existing report owner.

**Acceptance:** The report names what was seen, what remains unknown and one useful next photo; untrained authenticity probability remains absent.

### Task 2: Release calibrated confidence only after model proof

**Files:** Reuse calibration, inference and evaluation owners; no new probability calculator.

**Interfaces:** `predict(...) -> InferenceResult | Rejection`; `calibrate(...)`; `release_gate(...) -> ReleaseDecision`.

- [ ] Verify `pytest tests/test_calibration.py tests/test_predict.py tests/test_release_gate.py -q` for clipping, grade-4 ceiling, conflicting facts and abstention.
- [ ] On the exact released checkpoint, fit temperature using validation only; persist temperature, validation hash, model hash and calibration metrics.
- [ ] Run final independent cohorts once and check both calibration and watch-level false reassurance/coverage against the predeclared release thresholds.
- [ ] Expose confidence in the real route only after the exact model/calibration/release records are loaded; perform an actual upload and preserve response plus SHA.

**Acceptance:** Every displayed probability identifies its released model and calibration artifact. Failure or missing artifact leaves confidence unavailable and result inconclusive.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: the pilot already has truthful null confidence; avoid adding a fabricated score. | Prove current report first. |
| 2 | Yes: bind every later probability to exact released model and validation hash. | Tighten Task 2. |
| 3 | No: calibration, release and display owners are distinct and connected. | Keep. |
| 4 | No: remaining proof needs real labeled cohorts, not another output field. | Stop after two consecutive unchanged passes. |
