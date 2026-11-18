# Watch Authenticity Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is a separately reviewable result and must finish within two hours; split or stop the task before that limit.

**Goal:** Improve the ten stages from image acquisition to user-facing evidence without claiming authenticity or training on unverified catalog photos.

**Architecture:** Extend the existing acquisition, dataset, training, evaluation, reference-intelligence and pilot owners. Work in dependency order: measure eligible data, establish evidence and leakage safety, train only after readiness, then validate inference and the user-facing report. The current pilot and the trained DINOv2 classifier are separate paths until a real model passes the release gate.

**Tech Stack:** Python 3.12, pytest, PyTorch/DINOv2, IBM Cloud Object Storage and VPC GPU, local pilot HTTP server.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`; acquisition detail in `.specs/sub-tasks/train-watch-authenticity-model.feature/04c-data-acquisition.md` and `04d-source-acquisition-connectors.md`; reference detail in `04e-reference-intelligence.md`.

## Current execution map (checked 2026-11-17)

This parent plan retains the ten-stage design. The six focused plans below supersede its task text where the code has since advanced; an unchecked box in this document is **not** proof that its function is absent. Execute each focused plan in this order, one result at a time:

1. [Confirmed watch data](2026-11-17-confirmed-watch-data.md): collection yield, accepted labels, identity and real freeze. Existing planner, usability binding and quality gate have focused tests; actual training-ready confirmed data is still missing.
2. [Visual evidence](2026-11-17-visual-evidence.md): held-out angle review, cited reference facts and six-photo pilot comparison. Existing pilot proposes 116610LV only as a reference hypothesis.
3. [Pilot speed](2026-11-17-pilot-speed.md): comparable latency, honest distinct-image counts and browser recovery. Request-local visual reuse already exists.
4. [Classifier and release](2026-11-17-classifier-release.md): frozen independent sets, bounded IBM training and real release decision. Current archive cannot authorize training.
5. [Confidence report](2026-11-17-confidence-report.md): null confidence now; calibrated confidence only from an exact released artifact.
6. [Confirmed fake quality](2026-11-17-fake-quality.md): existing rubric and review gates; actual full inspection and scored fake remain absent.

**Dependency and truth gate:** Plans 2–3 can validate the local pilot without Plan 1. Plans 4–5 cannot claim a released classifier before Plan 1 provides a ready freeze and Plan 4 passes evaluation. Plan 6 can prove null output now; a numeric score requires an independently confirmed fake and complete exterior, opened-case and mechanical inspection. Existing archive labels never become ground truth by repetition. Each plan records two consecutive optimization passes without a substantial change after its last improvement.

| Cross-plan pass | Substantial improvement | Decision |
|---|---|---|
| 1 | Yes: current code already has planner yield, usability review, visual reuse, release gates and quality rubric. | Replace duplicate implementation work with real-data and final-route proof. |
| 2 | Yes: lock confidence and numeric quality behind separate released-model and complete-inspection gates. | Clarify Plans 5–6. |
| 3 | No: all six plans preserve the candidate/training/release boundaries. | Keep. |
| 4 | No: ownership, dependency order and measurable result are consistent across plans. | Stop after two consecutive unchanged cross-plan passes. |

## Global Constraints

- Initial coverage is Rolex Submariner 126610LN; a different reference must not become a fake label merely by differing from it.
- The archived Super Clone Luxury catalog reports 1,057 products and 3,672 image URLs; its `vendor_declared_replica` label is candidate evidence, `training_eligible` is false, and `quality_score_0_1000` is null.
- A fake's quality score from 0 to 1,000 is defined only after that watch is confirmed fake. It is distinct from confidence and from training grades 1–4.
- Admission retains 500 distinct images and 100 watch entities per class/angle, three source domains per genuine/fake label, 200 Tier-A gold images, and the other gates in `dataset/readiness.py`; no augmentation satisfies a count.
- Preserve rights/evidence checks, immutable receipts, revocation, grouped splits, and the existing IBM credit/cost guard. No paid AI API fallback.
- Release requires measured standard, cross-source, unseen-factory, temporal and gold slices; a local test or CI result never substitutes for real end-to-end proof.
- Implementation changes use a dedicated branch/one PR, targeted tests before review, then one final canonical preflight. The existing Python project has no Vercel Preview; report that gate as not applicable if still true at delivery.

## Review Focus

1. Seller calls a watch “superclone” but offers no independent confirmation: Task 2 keeps it candidate-only and leaves the 0–1,000 score empty.
2. One physical watch is reposted by different sellers or cropped differently: Task 4 keeps every related image in one split.
3. A green-dial Submariner is submitted as 126610LN: Task 9 proposes a compatible reference or inconclusive, never fake on color alone.
4. A high-quality replica appears from an unseen source: Task 7 measures false reassurance separately and blocks release if its bound fails.
5. Six clear photos take minutes in the pilot: Task 8 benchmarks the same machine and preserves conclusions while reducing repeated work.

## File Structure and Ownership

| Owner | Existing files | Planned responsibility |
|---|---|---|
| Acquisition | `acquisition/planner.py`, `acquisition/run.py`, `acquisition/supercloneluxury_archive.py` | Select missing cells and preserve candidate provenance; do not turn a seller claim into ground truth. |
| Evidence and data | `labeling/review.py`, `labeling/adjudicate.py`, `labeling/manifest.py`, `dataset/readiness.py`, `dataset/angle_review.py`, `dataset/entities.py`, `dataset/splits.py` | Admit verified labels, judge view quality, group watches, freeze independent splits. |
| Learning and release | `training/run.py`, `evaluation/evaluate.py`, `evaluation/release_gate.py` | Train in measured waves and reject source shortcuts, false reassurance and regressions. |
| Analysis | `pilot/server.py`, `pilot/watch_locator.py`, `inference/predict.py`, `reference_intelligence/compare.py` | Inspect usable regions, cite reference facts, abstain where evidence is missing. |
| Output | `pilot/server.py`, `core/types.py` | Keep label, inference confidence and confirmed-fake quality score separate. |

## Stage A — Establish useful examples

### Task 1: Optimize collection by eligible yield

**Files:** Modify `acquisition/planner.py` and `acquisition/run.py`; reuse `dataset/readiness.py`; test `tests/test_acquisition_planner.py`, `tests/test_acquisition_run.py` and `tests/test_dataset_readiness.py`.

**Interfaces:** Consume `ReadinessReport.cell_counts`, `CollectionWave`, source/seller entity counts and accepted records. Extend `plan_acquisition(..., *, source_entity_counts: Mapping[str, int], seller_entity_counts: Mapping[str, int], eligible_yield_by_source: Mapping[str, tuple[int, int]]) -> tuple[AcquisitionTarget, ...]`; each yield tuple means reviewed candidate entities and admitted independent entities, never image count. Give `AcquisitionTarget` an explicit `exploratory: bool` and `candidate_limit: int` (20 for an unknown source). The collection orchestrator selects a target/source before calling the existing `run_public_image_collection_once(...)`; that collector still enforces source authorization and immutable receipts.

- [ ] Record the current 126610LN count for every class/angle as distinct images and watch entities, plus rejected and candidate-only counts; label this baseline with the source snapshot/hash and date (a dataset freeze exists only after admission).
- [ ] Add a failing test where 1,000 reposted images from one seller lose priority to one underfilled cell with new eligible entities.
- [ ] Make `plan_acquisition` rank by *observed eligible new entities per bounded collection wave*, constrained by existing source/seller caps; never treat raw downloads as eligible yield. Give a source with no yield history one 20-candidate exploratory wave, not an invented yield of zero or 100%.
- [ ] In the existing collection runner, select the authorized source from the highest viable target and record the yield after evidence review; prioritize original candidates carrying item-level authentication/service evidence alongside replica candidates. Discovery metadata alone cannot fill an original training cell.
- [ ] Run `pytest tests/test_acquisition_planner.py tests/test_acquisition_run.py tests/test_dataset_readiness.py -q` and compare one real bounded collection receipt against the baseline; commit the result.

**Acceptance:** Every proposed wave names its class/angle deficit, eligible-source capacity and measured yield (or marks the one exploratory wave as unknown); a measured zero gain produces no repeated collection target. Do not lower readiness floors.

### Task 2: Make label and fake-quality decisions auditable

**Files:** Reuse `labeling/adjudicate.py` and `labeling/review.py`; create `labeling/quality.py` only for the separate confirmed-fake quality assessment; test `tests/test_label_review.py` and a focused `tests/test_fake_quality.py`.

**Interfaces:** Reuse `adjudicate(candidate: CandidateRecord, reviews: Sequence[ReviewDecision], reviewer_registry: Mapping[str, ReviewerRecord], *, gold_candidate_ids: frozenset[str] = frozenset()) -> EvidenceRecord`. Define `assess_fake_quality(evidence: EvidenceRecord, watch_entity_id: str, reviewer_scores: Mapping[str, int], quality_reviewers: Mapping[str, QualityReviewerRecord], rubric_version: str, inspection_evidence_sha256: str) -> FakeQualityAssessment | None`: return `None` unless accepted label is fake, evidence is eligible, the inspection evidence covers the rated attributes and two independent qualified reviewers scored it; keep this value out of `EvidenceRecord.accepted_grade`.

`QualityReviewerRecord` is separate from label-review authority and binds a reviewer ID, qualification artifact hash, validity window and contributor conflict groups. `FakeQualityAssessment` is an immutable record with `watch_entity_id: str`, `evidence_record_id: str`, `inspection_evidence_sha256: str`, `rubric_version: str`, `score_0_1000: int`, `reviewer_scores: tuple[tuple[str, int], ...]` and a digest of these fields. Scores differing by more than 100 points require an independent adjudicator; otherwise use their rounded midpoint. Missing mechanical/inside evidence cannot be silently treated as a perfect 1,000-point assessment: return unassessed if the rubric needs that evidence. Draft or unqualified scores stay unassessed.

- [ ] Verify the existing `adjudicate` tests for a seller-only “superclone”, an unsupported original claim and conflicting reviewers; add only missing regression cases, without reimplementing its existing gate.
- [ ] Add failing tests that a confirmed fake with one quality reviewer, two scores outside 0–1,000, disagreement over 100 points without adjudication, missing required inspection evidence or a nonfake evidence record has no published quality score.
- [ ] Define a written 0–1,000 scoring rubric with anchored examples and an `unassessed` state; require the same confirmed fake to receive two independent scores and adjudicate large disagreement before using the score. Do not calculate it from grade 1–4 or model confidence.
- [ ] Run `pytest tests/test_label_review.py tests/test_fake_quality.py -q`; commit the separate quality contract and rubric.

**Acceptance:** Every accepted label cites the underlying evidence and reviewer decision; a quality score also cites its own evidence and calibration anchors. All 1,057 archived products retain a null quality score until separately assessed.

## Stage B — Prepare independent, inspectable data

### Task 3: Measure usable views rather than trusting the selected angle

**Files:** Modify `dataset/angle_review.py`, `labeling/manifest.py`, `acquisition/types.py`, `training/data.py` and `dataset/freeze.py` only to carry the reviewed angle binding through the existing manifest/freeze; reuse `dataset/crop.py`; test `tests/test_angle_review.py`, `tests/test_label_review.py`, `tests/test_training_data.py` and `tests/test_crop.py`.

**Interfaces:** Keep `suggest_angle(...) -> AngleSuggestion`. Define `record_usability_review(candidate_id: str, image_sha256: str, angle: Angle, detail_visible: bool, usable: bool, reviewer_id: str, reviewed_at: datetime) -> AngleUsabilityReview` in `dataset/angle_review.py`. Bind its digest to the matching candidate derivative and accepted `TrainingRecord`; `build_training_record(...)` requires a matching usability review for non-rejection rows, and the freeze/training loader verifies the persisted digest. Reuse the existing `Angle` vocabulary.

- [ ] Establish a reviewed sample with examples of wrong dropdown angle, partly hidden detail, blur and valid macro; record the current suggestion accuracy.
- [ ] Add failing tests proving a selected “clasp” view without a visible clasp cannot count toward clasp readiness and an uncertain angle goes to review.
- [ ] In the first bounded work unit, implement the immutable usability review and prove mismatched image/angle hashes are rejected.
- [ ] In the second bounded work unit, require that review at `build_training_record`, carry its digest through freeze/load and preserve the full original image for audit.
- [ ] Run `pytest tests/test_angle_review.py tests/test_label_review.py tests/test_training_data.py tests/test_crop.py -q`; commit the work units independently.

**Acceptance:** Report view accuracy and usable-detail rate on a held-out reviewed sample; false angle claims cannot increase readiness.

### Task 4: Close cross-site watch-identity leakage

**Files:** Reuse `dataset/entities.py`, `dataset/splits.py`; modify them only if the cross-site fixture demonstrates a gap; test `tests/test_entities.py`, `tests/test_splits.py`.

**Interfaces:** Reuse `assign_watch_entities`, `build_splits` and `assert_no_connected_component_crosses_splits`; component identity is the shared watch, repost, seller or batch evidence already represented by records.

- [ ] Add a failing case with the same watch across two sites, one crop and one re-encoded photo, plus a visually similar but distinct watch.
- [ ] Confirm that existing content/entity grouping joins supported links and quarantines ambiguous ones; if the new fixture fails, repair the responsible shared function without merging visually similar but distinct watches.
- [ ] Assert every connected component belongs to one split, including gold and factory holdout; run `pytest tests/test_entities.py tests/test_splits.py -q` and commit.

**Acceptance:** No known shared watch crosses a split; ambiguous identity cannot inflate independent-entity counts.

### Task 5: Freeze a difficult test before training

**Files:** Reuse `dataset/splits.py`, `dataset/freeze.py`; modify only a demonstrated missing split/freeze invariant; test `tests/test_splits.py`, `tests/test_training_data.py`.

**Interfaces:** Keep `build_splits(...) -> DatasetSplits`; consume source, factory, time, capture style and entity groups. Freeze hashes bind the same rows later used by training and evaluation.

- [ ] Add failing tests for casual genuine/studio fake inversion, new factory, later collection period and attempts to reuse a holdout row in training.
- [ ] Verify the existing freeze of standard, cross-source, factory, temporal and Tier-A gold sets before any training wave; add a missing refusal only where the fixture fails.
- [ ] Run `pytest tests/test_splits.py tests/test_training_data.py -q`; commit the frozen split contract.

**Acceptance:** A readable split census lists entities per class, angle, source and holdout; no test photo trains the model or tunes its thresholds.

## Stage C — Learn and prove genuine watch discrimination

### Task 6: Spend GPU only on measurable learning

**Files:** Modify `training/run.py` and, if required, `evaluation/evaluate.py`; test `tests/test_training_run.py`, `tests/test_evaluate.py`.

**Interfaces:** Reuse the frozen DINOv2 backbone, per-angle heads, `training_wave_bundle(...)`, `learning_curve_report(...)` and the IBM worker/cost guard. A wave is keyed by the frozen dataset hash, model hash and independent entity set. A saved checkpoint is a *candidate*, never automatically a released model.

- [ ] Add failing tests for a second wave containing only duplicates, a wave with useful new entities and a worse validation result.
- [ ] Run a baseline wave only after `evaluate_readiness(...).ready` and rights checks pass; record GPU time, cost and balanced accuracy by class/angle. Bound one GPU attempt to 90 minutes plus 30 minutes for artifact verification and teardown; persist a resumable checkpoint if the attempt reaches its cap.
- [ ] Select the next wave from underperforming eligible cells and reviewed hard examples; keep test/gold entities out of all selection and stopping decisions.
- [ ] Stop when the learning curve shows no meaningful independent improvement or cost guard refuses another run; run targeted tests and commit.

**Acceptance:** Each additional GPU run has a predeclared data gain and comparison; no real train is claimed without a persisted checkpoint and evaluation on its exact freeze.

### Task 7: Gate release on false reassurance and coverage

**Files:** Modify `evaluation/release_gate.py`, `evaluation/evaluate.py`; test `tests/test_release_gate.py`, `tests/test_evaluate.py`.

**Interfaces:** Extend `EvaluationReport` and `release_gate(...) -> ReleaseDecision` with entity-level false-reassurance and abstention counts for the frozen high-quality-fake holdout. Retain all existing slice and calibration gates.

- [ ] Write failing tests for a model that passes average balanced accuracy but calls unseen high-quality fakes “no visible inconsistency”, and one that returns “inconclusive” for every watch.
- [ ] Freeze these pilot thresholds before opening the final holdout: one-sided 95% upper confidence bound for false reassurance among confirmed high-quality fake watches at most 5%; non-inconclusive coverage at least 20% separately among genuine and fake watch entities. Calculate by physical watch, not by photo. An insufficient holdout cannot pass the bound.
- [ ] Keep the current source, factory, temporal, gold and calibration checks; report the full error matrix and confidence intervals.
- [ ] Run `pytest tests/test_release_gate.py tests/test_evaluate.py -q`; commit.

**Acceptance:** The release decision cannot pass through high-confidence false reassurance, all-inconclusive output or a model that only answers for originals. Report numerator, denominator, interval and abstention rate by genuine/fake class and by grade; these pilot thresholds are frozen before final-holdout inspection and revisited only on a new version with a new unopened holdout.

## Stage D — Analyze only what is visible

### Task 8: Make six-photo inspection useful and timely

**Files:** Modify `pilot/server.py`, `pilot/watch_locator.py`; test `tests/test_pilot_server.py`, `tests/test_online_assessor.py`.

**Interfaces:** Preserve `/predict/batch` request/response fields and `PilotService.predict(body, content_type, angle)`. Reuse one loaded local vision model, normalize images once and cache repeated crops by content hash within the request; no paid API call. Count submitted photos and unique visual contents separately.

- [ ] Reproduce six-photo elapsed time on the same Mac and save its SHA, hardware and result text; the previous observed batch took 532 seconds but is not a comparable benchmark until rerun.
- [ ] Add a failing test for repeated photos in one batch and another for a clear watch whose selected angle does not match the located region.
- [ ] Remove repeated decode/inference work; report submitted photos, unique visual contents and re-used results separately so six duplicates do not masquerade as six independent inspections.
- [ ] Run `pytest tests/test_pilot_server.py tests/test_online_assessor.py -q`, rerun the same batch and commit.

**Acceptance:** At least 3× lower median latency on the same six-photo fixture and hardware, with equivalent findings and no falsely claimed inspection; otherwise report the measured bottleneck and keep the truthful version.

### Task 9: Check reference identity before calling a mismatch fake

**Files:** Modify `reference_intelligence/compare.py`, `inference/predict.py`, and the existing reference facts in `config/references/`; test `tests/test_reference_intelligence.py`, `tests/test_predict.py`.

**Interfaces:** Reuse `TechnicalClaim`, `VisualObservation`, `assess_reference(...)` and `InferenceResult`. A possible reference is a cited hypothesis; it is not an authentication or a fake verdict. If no trustworthy alternative-reference facts exist, return an unspecified different reference rather than naming one.

- [ ] Add failing tests for green dial / 126610LN, ambiguous bezel lighting, missing rehaut and conflicting online claims.
- [ ] Resolve reference-compatible alternatives only from versioned, cited claims; keep weaker forum content corroborating rather than authoritative. Do not expand the trained classifier's coverage when adding a reference hypothesis.
- [ ] Require a directly visible, reference-specific contradiction before reporting a mismatch; preserve `inconclusive` and `next_angle` when identity is uncertain.
- [ ] Run `pytest tests/test_reference_intelligence.py tests/test_predict.py -q`; commit.

**Acceptance:** The six green-dial photos yield “possible different reference” or inconclusive; color mismatch alone never yields fake.

### Task 10: Report evidence, uncertainty and fake quality separately

**Files:** Modify `pilot/server.py`, `inference/predict.py` and the existing output types in `core/types.py` only where the final trained route needs them; test `tests/test_pilot_server.py`, `tests/test_predict.py`, `tests/test_core_types.py`.

**Interfaces:** Preserve `fake_likely`, `no_visible_inconsistency`, `inconclusive` and rejection. Output must identify photos/details used, cited reference facts, observations not verified, calibrated confidence only when a released model exists, and a next useful angle. `quality_score_0_1000` is nullable and only for a separately confirmed fake.

- [ ] Write failing tests for an untrained pilot, a released model with conflicting evidence, a confirmed fake without quality assessment and a candidate-only “superclone”.
- [ ] Render the current pilot as evidence inspection without a fabricated fake probability; show model hash and calibration only when a released classifier is actually loaded.
- [ ] If a fake-quality rubric is validated on confirmed fakes, display its 0–1,000 result and evidence separately; otherwise show “não avaliada”.
- [ ] Run targeted tests and a real local upload with one clear and one insufficient image; commit.

**Acceptance:** The user can tell what was seen, what was inferred, what is unknown and which photo would help. No source term, training grade or confidence is silently converted into fake quality.

## Rollout and Proof

Tasks 1–5 can improve acquisition/data quality without a GPU. Task 6 is blocked until readiness and the IBM cost guard pass. Task 6 yields only a candidate checkpoint; Task 7 can admit it for release after all gates. Tasks 9 and 10 cannot claim trained-model success before that admission; Task 8 can improve the existing pilot independently. Each work unit is estimated before starting and capped at 120 minutes, including verification; split a larger item into independently testable slices first. A GPU attempt uses at most 90 minutes and reserves 30 minutes for teardown/report; a resumable checkpoint is progress, not completion. For each task: baseline or failing case → minimal change → targeted tests → self-review/required independent review → one final canonical preflight → one PR → real E2E for its observable result. A missing external data source or approved training set remains an explicit blocker, never a reason to substitute a mock result. Do not merge a change with a known failing gate.

## Optimization Loop

The plan is reviewed in cascading passes against the spec and current code. A pass counts as **substantial** only if it changes the order, safety boundary, measurable acceptance or owner of a task. Stop after two consecutive passes with none; record every pass and the decision below before execution.

| Pass | Scope | Substantial improvement | Decision |
|---|---|---|---|
| 1 | Collection and labels | Yes: prioritize independent eligible yield, give unknown sources a bounded exploratory wave, separate fake quality from confidence. | Revise Tasks 1–2. |
| 2 | Training and evaluation | Yes: freeze a false-reassurance bound and a minimum answer rate before opening holdout. | Revise Tasks 6–7. |
| 3 | Inference and output | Yes: count duplicate photos honestly, prevent unsupported alternative references and separate a candidate checkpoint from release. | Revise Tasks 8–10 and rollout. |
| 4 | Existing ownership | Yes: reuse current adjudication, grouping and split functions instead of rebuilding them. | Revise Tasks 2, 4–5. |
| 5 | Time and cost | Yes: cap a GPU attempt at 90 minutes and reserve 30 minutes for teardown/report within the two-hour work unit. | Revise Task 6 and rollout. |
| 6 | Quality-score qualification | Yes: label reviewers are not automatically qualified to score fake quality; require a separate registry and sufficient inspection evidence. | Revise Task 2. |
| 7 | End-to-end ownership and release gaming | Yes: connect acquisition priorities to the actual runner, bind usable-angle review to the training manifest, and require answer coverage for both genuine and fake watches. | Revise Tasks 1, 3, 7. |
| 8 | Full pass across Tasks 1–10 | No: each item has owner, test, measurable acceptance and retained evidence/cost boundaries. | Keep plan. |
| 9 | Independent full pass against spec and repository paths | No: all ten items map to existing owners or named new files; no missing path, placeholder or relaxed gate found. | Stop: two consecutive passes without substantial improvement. |
