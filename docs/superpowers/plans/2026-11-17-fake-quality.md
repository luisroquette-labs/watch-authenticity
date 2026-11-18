# Confirmed Fake Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assign a 0–1,000 fidelity score only to an independently confirmed fake with a complete, reviewable inspection.

**Architecture:** Reuse `labeling/quality.py` and the existing separate rubric. The score describes how closely the confirmed fake matches the reference across exterior, interior and mechanics; it is neither authenticity probability nor the training grade.

**Tech Stack:** Python 3.12, immutable evidence records, pytest.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`; parent plan `docs/superpowers/plans/2026-11-17-watch-pipeline-optimization.md` Tasks 2 and 10.

## Global Constraints

- The archived seller-declared replicas have null quality scores. Exterior catalog photos alone cannot support a full score.
- Complete inspection requires exterior, opened-case inside and movement/function artifacts for the same confirmed fake.
- Two qualified independent reviewers score the same rubric; disagreement over 100 points requires qualified independent adjudication.
- Preserve rubric/version/hash and watch/evidence identity; each work unit ≤120 minutes.

## Review Focus

1. Seller says “superclone” with no accepted fake label: Task 1 leaves score null.
2. Two exterior reviewers without opened-case evidence: Task 1 leaves score null.
3. Reviews refer to changed inspection hash: Task 1 rejects replay.
4. Reviewers share conflict group: Task 1 refuses publication.
5. Score is 1,000: Task 2 still does not claim authenticity or absence of hidden defects.

## File Structure and Ownership

- `labeling/quality.py`: rubric, reviewer qualification, independent decisions and assessment hash.
- `labeling/adjudicate.py`: prerequisite accepted fake label; `dataset/entities.py`: watch identity.
- `pilot/server.py`: user-visible nullable `quality_score_0_1000` and `quality_assessment`.
- `tests/test_fake_quality.py`, `tests/test_label_review.py`, `tests/test_pilot_server.py`: focused contracts.

### Task 1: Prove the complete fake-quality gate

**Files:** Reuse `labeling/quality.py`; edit only after a failing real inspection case.

**Interfaces:** `submit_quality_review(...) -> QualityReviewDecision`; `assess_fake_quality(...) -> FakeQualityAssessment | None`; `verify_fake_quality(...)`.

- [ ] Run `pytest tests/test_fake_quality.py tests/test_label_review.py -q` for confirmed label, all three inspection attributes, reviewer independence, conflict, changed hash and adjudication.
- [ ] Obtain one actual confirmed fake with exterior, inside and mechanical examination tied to one physical-watch entity; if unavailable, keep the operational result `unassessed` and record the exact missing artifact.
- [ ] If that complete case exposes a contract gap, add a sanitized failing fixture, fix `labeling/quality.py`, rerun the focused tests and commit.

**Acceptance:** A score has a stable accepted-fake record, complete inspection, qualified votes, versioned rubric and verifiable digest. No partial case receives a score.

### Task 2: Display quality distinctly from uncertainty

**Files:** Reuse `pilot/server.py` and `tests/test_pilot_server.py`; change only when a real complete assessment exists and the final route can load it safely.

**Interfaces:** `joint.quality_score_0_1000: int | None`, `joint.quality_assessment`; model `calibrated_confidence` remains independent.

- [ ] Test one unassessed catalog candidate and one confirmed fake without full inspection through the real local endpoint; expect null quality in both.
- [ ] When Task 1 produces a complete assessment, verify its hash and watch identity at the output boundary; add focused tests for wrong watch, altered hash and a valid score.
- [ ] Upload that watch's photos in the actual final route and verify the displayed label says “quality of a confirmed fake”, cites the inspection and keeps confidence separate.

**Acceptance:** A user cannot mistake the quality score for probability that an unknown watch is fake. Until full inspection exists, “não avaliada” is the complete and correct output.

## Optimization Loop

| Pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: `labeling/quality.py` already implements the rubric and gates; eliminate duplicate implementation. | Focus on real inspected fake. |
| 2 | Yes: distinguish safe null display from future assessment lookup tied to watch identity. | Split Tasks 1–2. |
| 3 | No: existing gate covers reviewer, evidence and rubric constraints. | Keep. |
| 4 | No: the missing evidence requires a physical inspection, not another score formula. | Stop after two consecutive unchanged passes. |
