# Step 09: Inference Orchestration (predict)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 4
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `06-backbone-heads`, `07a-rejection`, `07b-calibration`, `08a-quality-gate`, `08b-gradcam`
**Parallel with:** None
**Note:** Integration hub — wires every prior component into one ordered path. No new invariants; reuse `core`.

**Goal:** Orchestrate the full inference path so a prediction is a schema-valid `Prediction` with calibrated in-range confidence, a grade when fake, the grade-4 ceiling applied, and a heatmap — or a `Rejection` when the quality gate or OOD check fires first (CK-1,2,3,4,5,6).

Create `inference/predict.py`: order = `quality_gate` → `model.rejection` (OOD) → backbone+heads → `core.aggregate_angles` → `calibration.calibrate` (temperature → `core.apply_grade4_ceiling` → `core.clip_confidence`) → `explainability.grad_cam` → `core.Prediction`. Quality-gate failure and OOD short-circuit to `Rejection` with zero authenticity-model calls.

#### Expected Output

- `inference/predict.py` — `predict(image)` → `Prediction | Rejection`
- `tests/test_predict.py` (unit + integration)

#### Success Criteria

- [ ] `predict(valid_fixture)` returns a `Prediction` with confidence strictly in (0.001, 0.999) (CK-1)
- [ ] A fake prediction reports an associated grade 1-4 (CK-2); a genuine prediction has no grade
- [ ] Grade-4-consistent output has genuine-confidence ≤ ceiling and `ceiling_applied=True` (CK-3)
- [ ] OOD input returns `Rejection`, not a score (CK-4)
- [ ] A blurry fixture through the full pipeline causes ZERO calls to the authenticity model (CK-5 integration)
- [ ] Every `Prediction` includes a heatmap matching image dimensions (CK-6)
- [ ] `pytest tests/test_predict.py` passes

#### Subtasks

- [ ] Implement `predict(image)` ordered orchestration in `inference/predict.py`
- [ ] Short-circuit quality-gate + OOD before any scoring
- [ ] Reuse `core.aggregate_angles`, `calibration.calibrate`, `explainability.grad_cam`
- [ ] Write integration test asserting zero authenticity-model calls on a blurry input (CK-5)
- [ ] Write unit tests for confidence range, grade presence, ceiling, heatmap (CK-1,2,3,6)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Stage order wrong (score before quality gate) → CK-5 fail | High | Low | Gate + OOD first by construction; integration test counts model calls on blurry input |
| Risk | Raw softmax returned, bypassing calibrate/clip | High | Low | Single return path forces calibrate→ceiling→clip; CK-1 range test |
