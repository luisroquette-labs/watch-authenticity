# Visual Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each visual finding traceable to a usable photo, visible watch detail and exact cited reference, while retaining an inconclusive outcome for insufficient evidence.

**Architecture:** Keep angle and usability review at the existing dataset boundary, reference claims in `reference_intelligence`, and the local pilot as a separate visual inspection path. Do not infer authenticity from a matching color or a selected angle.

**Tech Stack:** Python 3.12, pytest, local pilot vision model and versioned reference facts.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`, sub-task `04e-reference-intelligence.md`, parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md`.

## Global Constraints

- Initial trained reference is 126610LN. A cited 116610LV resemblance is a hypothesis, not a new trained class or an authenticity verdict.
- Preserve original image, derivative hash, angle usability review, source class and exact URL/citation.
- Avoid model or human claims about a detail that is outside the image. Each unit is at most 120 minutes.

## Review Focus

1. User selects clasp but photo shows only dial: Task 1 cannot count a clasp view.
2. Green surface comes from background rather than bezel: Task 2 refuses bezel evidence.
3. Two photos of the same pixels use different encodings: Task 2 counts one visual content.
4. Official reference facts conflict or are stale: Task 2 stays inconclusive.
5. Green dial conflicts with 126610LN: Task 2 proposes another reference only with separate dial and bezel clues.

## File Structure and Ownership

- `dataset/angle_review.py`, `labeling/manifest.py`, `dataset/freeze.py`, `training/data.py`: immutable usable-view binding.
- `pilot/watch_locator.py`, `pilot/visual_evidence.py`, `pilot/server.py`: what the local pilot actually saw.
- `reference_intelligence/compare.py` and versioned reference facts: source-supported reference checks.
- Tests: `tests/test_angle_review.py`, `tests/test_training_data.py`, `tests/test_pilot_server.py`, `tests/test_reference_intelligence.py`.

### Task 1: Prove view admission and training lineage

**Files:** Reuse the existing owners above; edit only the owner where a failing real fixture shows a gap.

**Interfaces:** `record_usability_review(...) -> AngleUsabilityReview`; `verify_usability_review(...)`; `build_training_record(...)`; freeze loader.

- [ ] Measure a held-out sample of correctly selected angle, wrong angle, blurred detail and detail outside frame; retain reviewed counts and content hashes.
- [ ] Run `pytest tests/test_angle_review.py tests/test_training_data.py tests/test_label_review.py -q` and confirm wrong/hidden views cannot fill readiness.
- [ ] If a held-out case passes wrongly, add that exact sanitized case as a failing fixture, repair the responsible shared function and rerun focused tests.
- [ ] Record view precision and usable-detail rate from the reviewed sample, including denominator; commit only if code or fixture changed.

**Acceptance:** Every admitted image's reviewed angle and usable detail survive manifest, freeze and training load with the same hash.

### Task 2: Prove cited visual conclusions on the six-photo pilot

**Files:** Reuse `pilot/visual_evidence.py`, `pilot/server.py`, `reference_intelligence/compare.py`; tests above.

**Interfaces:** `reference_candidate(dial, bezel)`; `assess_reference(...)`; `/predict/batch` `joint.visible_evidence`, `joint.reference_candidate`, `joint.unverified`.

- [ ] Run the exact six local photos against the same versioned reference facts; retain per-photo observations, source URLs, response JSON and elapsed time.
- [ ] Check background contamination, cropped bezel, duplicate encoding, conflicting photo colors and stale/conflicting citations with `pytest tests/test_pilot_server.py tests/test_reference_intelligence.py -q`.
- [ ] For any observed false claim, add one failing fixture and repair the locator, evidence aggregation or reference checker that owns it; retest.
- [ ] Compare the new output with the saved earlier response: note what improved, stayed unresolved or regressed, without presenting color as authentication.

**Acceptance:** A user can identify each actually inspected photo and the source behind every stated reference fact. The six-photo case stays inconclusive about authenticity.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: existing usability binding and reference candidate code make a rewrite unnecessary. | Plan proof and gap repair. |
| 2 | Yes: require held-out view precision and distinct pixel counts, not only fixture pass. | Add measured acceptance. |
| 3 | No: input, owners and evidence outputs are covered. | Keep. |
| 4 | No: additional visual signals require new confirmed observations, not a planning change. | Stop after two consecutive unchanged passes. |
