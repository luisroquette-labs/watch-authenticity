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

- [X] `grad_cam(...)` returns a heatmap array matching image dimensions (CK-6) — `explainability/gradcam.py:35-68`, verified by `tests/test_gradcam.py::test_grad_cam_returns_heatmap_matching_image_dimensions`
- [X] On a small-scale trained checkpoint the heatmap is non-degenerate (not all-zero) for a fixture sample (CK-6 integration) — `tests/test_gradcam.py::test_grad_cam_is_non_degenerate_on_a_small_scale_trained_checkpoint` (real 20-step training loop + save/reload, asserts `heatmap.std() > 0.0`)
- [X] Target layer is `model.blocks[-1].norm1` (ViT-compatible) — `explainability/gradcam.py`'s `_resolve_target_layer`: tries `.blocks[-1].norm1` first, falls back to the real loaded backbone's actual shape (HF `Dinov2Model`'s `.encoder.layer[-1].norm1`), both paths covered by tests
- [X] `pytest tests/test_gradcam.py` passes — skips in this sandbox (`importorskip`, no torch wheel for Python 3.14); **independently verified for real** in a disposable venv with real `torch==2.15.0`/`grad-cam==1.5.7`: 5/5 passed, 0 skipped, 0 failed; `ruff check` and `mypy` on `explainability/gradcam.py` also run clean there and in this sandbox

#### Subtasks

- [X] Implement `grad_cam` targeting `blocks[-1].norm1` in `explainability/gradcam.py` (with a DINOv2-shaped fallback, see Blockers & Risks)
- [X] Write unit test: heatmap dimensions match image (CK-6) — plus 3 adjacent unit tests covering the DINOv2-style target-layer fallback, a fully-frozen model (the real deployment shape), and an ambient `torch.no_grad()` caller context (how `model.heads.AngleHeadModel.forward` actually runs)
- [X] Write integration test: non-degenerate heatmap on a trained checkpoint (CK-6)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Grad-CAM target layer incompatible with DINOv2 → degenerate map | High | Med | **Resolved.** SKILL.md's `model.blocks[-1].norm1` snippet assumes a timm-style ViT; the real backbone this project loads (`adapters.backbone.load_frozen_backbone`, HF `AutoModel.from_pretrained("facebook/dinov2-base")`) is a `Dinov2Model` with no `.blocks` attribute at all — its equivalent is `encoder.layer[-1].norm1`. `_resolve_target_layer` tries `.blocks` first, falls back to `.encoder.layer`, verified by `test_grad_cam_resolves_dinov2_style_target_layer_without_blocks_attribute` against a stub shaped exactly like the real `Dinov2Model`. Two further risks found and fixed only by running the real package (not just reading SKILL.md's snippet): (1) installed `grad-cam==1.5.7`'s `GradCAM.__call__` requires `targets` as a positional arg (no default) — fixed by passing `targets=None` explicitly, which the library's own `BaseCAM.forward` documents as "auto-pick each sample's argmax class". (2) a ViT block's target-layer output is `(batch, tokens, dim)`, not the `(batch, C, H, W)` `pytorch_grad_cam` assumes by default — added `reshape_transform=_vit_tokens_to_grid` (drops the CLS token, reshapes the square patch grid), matching jacobgil's own documented ViT Grad-CAM tutorial pattern, which SKILL.md's snippet omits. Deviation note: `torch`/`grad-cam` are not importable in this sandbox (Python 3.14, pinned `requirements.txt` wheels unavailable — same class of environment limitation as `06-backbone-heads`/`07a-rejection`). Followed the same instructed fallback: real production code in `explainability/gradcam.py` (no stub/mock outside tests), every test in `tests/test_gradcam.py` gated behind `pytest.importorskip`, and independently verified for real in a disposable venv (`pip install torch grad-cam numpy opencv-python pytest hypothesis ruff mypy`, deleted after use): 5/5 tests passed, 0 skipped, 0 failed; `ruff check`/`mypy` clean. No change to `requirements.txt` (`grad-cam==1.5.4` was already pinned there); nothing in the repo depends on the disposable venv. |
