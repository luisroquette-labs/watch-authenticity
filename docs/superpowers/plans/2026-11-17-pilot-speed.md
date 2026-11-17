# Pilot Speed and Local Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the six-photo pilot responsive and make the reported inspection count accurate.

**Architecture:** Reuse the current single loaded vision model and request-local cache in `pilot/server.py`. Benchmark before optimization; change only a measured bottleneck and preserve uploaded-image limits, error recovery and all verdict fields.

**Tech Stack:** Python 3.12, Pillow, pytest, local HTTP service.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`; parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md` Task 8.

## Global Constraints

- Local pilot is visual inspection, not a released authenticity classifier.
- Six images maximum; 15 MB each; encoded/decoded batch limits remain. No paid API.
- One delivery unit takes at most 120 minutes, including benchmark and proof.

## Review Focus

1. Six identical photos: count six submissions and one unique visual content.
2. Same pixels with different encoding or selected angle: reuse inspection but render each requested view honestly.
3. Oversized, malformed or decompression-bomb input: reject before model invocation.
4. Failed request: restore the form and give a readable error.
5. Six distinct high-resolution photos: bounded memory and a result for each.

## File Structure and Ownership

- `pilot/server.py`: request validation, sequential decode, reuse and response.
- `pilot/watch_locator.py`: subject crop and visual work; `pilot/online_assessor.py`: model call.
- `tests/test_pilot_server.py`, `tests/test_online_assessor.py`: focused contract tests.

### Task 1: Establish comparable latency and correct accounting

**Files:** Inspect `pilot/server.py` and focused tests; modify only after a failing result.

**Interfaces:** `analyze_batch(service, payload) -> dict`; `/predict/batch` fields `submitted_photos`, `unique_visual_contents`, `reused_results`, `model_checked_images`.

- [ ] Benchmark three warm runs of the same six files on the same Mac, server SHA and model; record median, per-image dimensions, peak resident memory and full JSON response.
- [ ] Run `pytest tests/test_pilot_server.py tests/test_online_assessor.py -q`, including duplicate, selected-angle and upload-limit cases.
- [ ] If the measured median exceeds 60 seconds or counts are false, profile decode, locator and model separately; fix the highest verified cost in the responsible function and add a focused regression test.
- [ ] Re-run three warm batches under the same conditions and compare median and conclusions; commit only if a measured improvement survives focused tests.

**Acceptance:** The six-photo case completes within 60 seconds median on the same Mac, or the measured bottleneck and exact remaining cost are reported. Counts never present duplicates as independent inspections.

### Task 2: Verify the actual browser flow

**Files:** `pilot/server.py`, `tests/test_pilot_server.py` only if browser behavior fails.

**Interfaces:** HTML form at `/`, POST `/predict/batch`, rendered per-photo and joint report.

- [ ] Upload six photos in the local browser and observe progress, response, source links and ability to submit a second batch.
- [ ] Repeat with one invalid image; confirm the error is visible and the form unlocks.
- [ ] Repair any reproduced browser failure in the existing form/handler; add its focused regression and rerun `pytest tests/test_pilot_server.py -q`.

**Acceptance:** A real browser session can complete, fail safely and retry. Record response and server SHA; do not call a unit test an end-to-end result.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: current code already reuses exact visual contents, so benchmark before changes. | Remove proposed cache rewrite. |
| 2 | Yes: include real browser retry and memory alongside latency. | Add Task 2 and memory proof. |
| 3 | No: one profiler target and one browser contract cover the known risk. | Keep. |
| 4 | No: further optimization depends on measured bottleneck. | Stop after two consecutive unchanged passes. |
