# Confirmed Watch Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a small, independently verifiable 126610LN dataset before any claim of model training or accuracy.

**Architecture:** Reuse the existing collection runner and immutable candidate/evidence/manifest pipeline. A seller's description creates a candidate, never a training label; only reviewed, rights-eligible, correctly grouped watches enter a frozen dataset.

**Tech Stack:** Python 3.12, pytest, IBM Cloud Object Storage; existing `acquisition`, `labeling` and `dataset` modules.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`, `04c-data-acquisition.md`, `04d-source-acquisition-connectors.md` in its sub-task directory; parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md`.

## Global Constraints

- 126610LN only for the first trained release; other references stay separate.
- Seller-declared replicas and original-claim ads remain candidates until admissible evidence review; the 1,057 archived products currently have no accepted labels or fake-quality scores.
- Keep source rights, immutable receipts, revocation, physical-watch grouping, source/seller caps, six view cells and readiness floors from `dataset/readiness.py`.
- Each delivery unit, including review and proof, must fit within 120 minutes. No paid AI API fallback.

## Review Focus

1. Reposted watch from two sellers counts as one entity: Task 2 checks grouping.
2. Seller says “superclone” without independent evidence: Task 2 retains candidate state.
3. Original listing provides only the word “original”: Task 2 refuses admission.
4. A source yields many files but zero eligible entities: Task 1 stops repeat waves.
5. Revoked rights after admission: Task 2 excludes the affected entity from a new freeze.

## File Structure and Ownership

- `acquisition/planner.py` selects a bounded source/cell wave; `acquisition/run.py` writes collection receipts.
- `labeling/adjudicate.py` decides accepted label; `labeling/manifest.py` binds evidence and view review.
- `dataset/entities.py`, `dataset/readiness.py`, `dataset/freeze.py` own identity, census and immutable freeze.
- Existing focused tests: `tests/test_acquisition_planner.py`, `tests/test_acquisition_run.py`, `tests/test_label_review.py`, `tests/test_entities.py`, `tests/test_dataset_readiness.py`.

### Task 1: Measure eligible yield from a bounded real wave

**Files:** Inspect `acquisition/planner.py`, `acquisition/run.py`, `dataset/readiness.py`; modify their focused tests and owner only if an observed contract fails. Save non-sensitive receipt hashes in the task handoff, not a new verification document.

**Interfaces:** `plan_acquisition(...) -> tuple[AcquisitionTarget, ...]`; `run_planned_public_collection_once(...)`; `evaluate_readiness(...) -> ReadinessReport`.

- [ ] Record current eligible image/entity counts by class and angle, candidate-only count, source snapshot hash and date. Zero is a valid measured result.
- [ ] Run one authorized 20-candidate exploratory wave from an available source; record attempted, downloaded, reviewed and admitted **distinct watch entities**.
- [ ] Check `test_measured_zero_yield_is_not_retried` and `test_twenty_reposts_of_one_watch_lose_to_twenty_distinct_watches` using `pytest tests/test_acquisition_planner.py -q`.
- [ ] If actual receipts contradict those tests, write a failing focused fixture, repair the responsible planner/runner function, rerun focused tests, commit that correction in the same delivery branch.

**Acceptance:** Every next wave cites a measurable deficit and observed eligible yield, or is explicitly a first bounded exploration. No downloaded-file count is presented as training volume.

### Task 2: Admit only confirmed, independent watches

**Files:** Reuse `labeling/adjudicate.py`, `labeling/manifest.py`, `dataset/entities.py`, `dataset/freeze.py`; focused tests named above. Do not create a second labeling path.

**Interfaces:** `adjudicate(...) -> EvidenceRecord`; `build_training_record(...)`; `assign_watch_entities(...)`; `evaluate_readiness(...)`.

- [ ] For each proposed watch, retain the source URL/receipt, exact image hash, rights basis, reference evidence, class evidence, reviewer decision and entity grouping.
- [ ] Put unsupported originals, seller-only replicas, disputed reference and revoked-rights cases in the candidate or excluded state; check `pytest tests/test_label_review.py tests/test_entities.py tests/test_dataset_readiness.py -q`.
- [ ] Freeze only accepted rows and record census by class, angle, watch, source and seller; preserve a dated hash of the freeze.
- [ ] If any admissibility or revocation invariant fails, add the smallest failing fixture, fix the existing owner and repeat the focused tests before a commit.

**Acceptance:** A real freeze can be traced from receipt through adjudication to the stored row. If no class meets the existing readiness floor, report the exact deficit and leave training blocked.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: separate real eligible-entity yield from raw image volume. | Make Task 1 receipt-driven. |
| 2 | Yes: make rights revocation and cross-seller identity part of admission proof. | Tighten Task 2. |
| 3 | No: collection, adjudication and freeze each have an existing owner and acceptance. | Keep. |
| 4 | No: no additional safe route converts the present catalog into confirmed labels. | Stop after two consecutive unchanged passes. |
