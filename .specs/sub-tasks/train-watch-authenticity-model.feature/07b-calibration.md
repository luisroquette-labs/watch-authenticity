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

- [X] `temperature_scale(logits, T=2.0)` yields a flatter (lower max) probability than `T=1.0` (CK-15)
- [X] Property: for any logit vector and `T>1`, `max(softmax(logits/T)) <= max(softmax(logits))` (CK-15)
- [X] Calibration fit reads only the validation split's manifest rows, never the test split's (CK-15 integration)
- [X] Reported confidence passes through `core.apply_grade4_ceiling` then `core.clip_confidence` (CK-3, CK-1)
- [X] `pytest tests/test_calibration.py` passes

#### Subtasks

- [X] Reimplement `fit_temperature` (LBFGS NLL) in `calibration/temperature_scaling.py`
- [X] Implement `calibrate(...)` chaining temperature → `core.apply_grade4_ceiling` → `core.clip_confidence`
- [X] Write unit + property tests for temperature flattening/monotonicity (CK-15)
- [X] Write integration test asserting fit uses validation split only (CK-15)

**Implementation note:** this sandbox's Python (3.14) has no `torch==2.5.1` wheel, matching the
environment limitation `06-backbone-heads` already hit and documented. Followed the same pattern
(`pytest.importorskip` + `TYPE_CHECKING`-only import) in `tests/test_calibration.py` so the native
sandbox run documents an explicit skip (not a silent no-op). Additionally verified all 8 tests for
real, not just structurally: built a disposable Python 3.12 venv, installed the pinned
`torch==2.5.1`/`transformers==4.46.0`, and ran `pytest tests/test_calibration.py` plus
`tests/test_model_build.py`/`test_core_rules.py`/`test_core_types.py` — 50/50 passed. Venv was
discarded after verification; nothing added to the repo or its dependency pins.

Also added `fit_temperature_on_validation_split(splits, logits_by_path, labels_by_path)` alongside
the two functions named in Expected Output: the success criterion "fit reads only the validation
split's manifest rows, never the test's" needed a real call site to enforce (not just a test that
manually slices `splits[VAL]`), and `model/heads.py`'s own docstring already commits calibration
(this step) to reading `GENUINE_CLASS_INDEX` from it rather than re-deriving the class-index layout.

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Temperature fit on test split leaks, miscalibration hidden | High | Low | Fit reads validation manifest only; integration test asserts split source |
| Risk | Ceiling/clip re-implemented here instead of reused → CK-22 | Med | Med | Import `core.apply_grade4_ceiling` + `core.clip_confidence`; reviewer checks at phase |
