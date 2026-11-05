# Step 08a: Pre-Inference Photo Quality Gate

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 4
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `03-core-rules`
**Parallel with:** `08b-gradcam`
**Note:** Needs only `core` (returns `core.Rejection`) + opencv; placed in the inference phase it completes.

**Goal:** Reject blurry/badly-framed photos with a recapture instruction BEFORE the authenticity model runs (CK-5) — a bad photo is never scored.

Create `quality_gate/blur_light_check.py`: `sharpness_score(gray_image)` via `cv2.Laplacian(...).var()` (SKILL.md:174-175) and `quality_gate(image) -> Pass | core.Rejection`. The Laplacian threshold is **per-angle, tunable, commented** (no universal constant; polished casebacks false-positive — SKILL.md). A failing photo returns `Rejection("blurry", "recapture needed")`.

#### Expected Output

- `quality_gate/blur_light_check.py` — `sharpness_score`, `quality_gate(image)`
- `tests/test_quality_gate.py`

#### Success Criteria

- [X] `quality_gate(sharp_well_lit_fixture)` passes (CK-5)
- [X] `quality_gate(blurry_fixture_below_threshold)` returns `Rejection` with a recapture instruction (CK-5)
- [X] BVA around the threshold: thresh-1 rejects, thresh/thresh+1 behavior asserted (CK-5)
- [X] Threshold is commented as empirically tuned per angle, not a universal constant
- [X] `pytest tests/test_quality_gate.py` passes

#### Subtasks

- [X] Implement `sharpness_score` + `quality_gate` in `quality_gate/blur_light_check.py`
- [X] Return `core.Rejection("blurry","recapture needed")` on failure
- [X] Write unit tests incl. BVA around threshold (CK-5)

**Implementation note:** `quality_gate` returns a local `Pass` sentinel (defined in `quality_gate/blur_light_check.py`, not `core/`) alongside `core.types.Rejection`, matching the Contracts signature `quality_gate(image) -> Pass | Rejection` without touching the already-`[DONE]` `core/types.py`. Threshold is looked up via an `angle: str | None = None` keyword with a `DEFAULT_SHARPNESS_THRESHOLD` fallback, mirroring `dataset/crop.py`'s `DEFAULT_CROP_FRACTION` pattern.

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Flat polished caseback falsely rejected as blurry | Med | Med | Per-angle threshold; SKILL.md notes BRISQUE as secondary check if false-reject rate matters |
