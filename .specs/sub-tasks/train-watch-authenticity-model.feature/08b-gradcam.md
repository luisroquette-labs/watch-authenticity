# Step 08b: Grad-CAM Explainability

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 4
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `06-backbone-heads`
**Parallel with:** `08a-quality-gate`
**Note:** None

**Goal:** Produce a Grad-CAM heatmap for every prediction (CK-6) — dual role: user trust and source-bias audit (background lighting up = item-13 confound flagrant).

Create `explainability/gradcam.py`: use `jacobgil/pytorch-grad-cam` with the ViT target layer `model.blocks[-1].norm1` (SKILL.md:81-82). Return an image-dimensioned heatmap. Smoke-test it on a real trained checkpoint (from Step 06) to confirm the map is non-degenerate (not all-zero / not uniformly flat) — SKILL.md flags DINOv2 compatibility as plausible but not vendor-confirmed.

#### Expected Output

- `explainability/gradcam.py` — `grad_cam(model, image)` → heatmap
- `tests/test_gradcam.py`

#### Success Criteria

- [ ] `grad_cam(...)` returns a heatmap array matching image dimensions (CK-6)
- [ ] On a small-scale trained checkpoint the heatmap is non-degenerate (not all-zero) for a fixture sample (CK-6 integration)
- [ ] Target layer is `model.blocks[-1].norm1` (ViT-compatible)
- [ ] `pytest tests/test_gradcam.py` passes

#### Subtasks

- [ ] Implement `grad_cam` targeting `blocks[-1].norm1` in `explainability/gradcam.py`
- [ ] Write unit test: heatmap dimensions match image (CK-6)
- [ ] Write integration test: non-degenerate heatmap on a trained checkpoint (CK-6)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Grad-CAM target layer incompatible with DINOv2 → degenerate map | High | Med | Smoke-test on the real checkpoint (SKILL.md requirement); fall back to an alternate late-block norm if flat |
