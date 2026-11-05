# Step 07b: Temperature-Scaling Calibration + Ceiling Wrap

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 3
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `06-backbone-heads`
**Parallel with:** `07a-rejection`
**Note:** None

**Goal:** Calibrate reported confidence via temperature scaling fit on the held-out validation split (distinct from test), then wrap the output with the `core` grade-4 ceiling and `core` clip so no raw softmax is ever reported (CK-15, and CK-3 integration).

Create `calibration/temperature_scaling.py`: reimplement Guo et al. 2017 `fit_temperature(logits, labels) -> float` (SKILL.md:157-168 — do NOT depend on the abandoned `gpleiss` package). Fit on the validation split only, never test. Provide a `calibrate(logits, grade)` path that applies temperature, then `core.apply_grade4_ceiling`, then `core.clip_confidence` — reusing the single core implementations (CK-22).

#### Expected Output

- `calibration/temperature_scaling.py` — `fit_temperature`, `calibrate(...)` wrapping core ceiling + clip
- `tests/test_calibration.py`

#### Success Criteria

- [ ] `temperature_scale(logits, T=2.0)` yields a flatter (lower max) probability than `T=1.0` (CK-15)
- [ ] Property: for any logit vector and `T>1`, `max(softmax(logits/T)) <= max(softmax(logits))` (CK-15)
- [ ] Calibration fit reads only the validation split's manifest rows, never the test split's (CK-15 integration)
- [ ] Reported confidence passes through `core.apply_grade4_ceiling` then `core.clip_confidence` (CK-3, CK-1)
- [ ] `pytest tests/test_calibration.py` passes

#### Subtasks

- [ ] Reimplement `fit_temperature` (LBFGS NLL) in `calibration/temperature_scaling.py`
- [ ] Implement `calibrate(...)` chaining temperature → `core.apply_grade4_ceiling` → `core.clip_confidence`
- [ ] Write unit + property tests for temperature flattening/monotonicity (CK-15)
- [ ] Write integration test asserting fit uses validation split only (CK-15)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Temperature fit on test split leaks, miscalibration hidden | High | Low | Fit reads validation manifest only; integration test asserts split source |
| Risk | Ceiling/clip re-implemented here instead of reused → CK-22 | Med | Med | Import `core.apply_grade4_ceiling` + `core.clip_confidence`; reviewer checks at phase |
