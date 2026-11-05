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

- [ ] `quality_gate(sharp_well_lit_fixture)` passes (CK-5)
- [ ] `quality_gate(blurry_fixture_below_threshold)` returns `Rejection` with a recapture instruction (CK-5)
- [ ] BVA around the threshold: thresh-1 rejects, thresh/thresh+1 behavior asserted (CK-5)
- [ ] Threshold is commented as empirically tuned per angle, not a universal constant
- [ ] `pytest tests/test_quality_gate.py` passes

#### Subtasks

- [ ] Implement `sharpness_score` + `quality_gate` in `quality_gate/blur_light_check.py`
- [ ] Return `core.Rejection("blurry","recapture needed")` on failure
- [ ] Write unit tests incl. BVA around threshold (CK-5)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Flat polished caseback falsely rejected as blurry | Med | Med | Per-angle threshold; SKILL.md notes BRISQUE as secondary check if false-reject rate matters |
