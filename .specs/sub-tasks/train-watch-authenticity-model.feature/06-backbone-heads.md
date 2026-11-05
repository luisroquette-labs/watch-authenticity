# Step 06: Frozen DINOv2 Backbone + Per-Angle Heads

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 3
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `05c-augment-splits`
**Parallel with:** None
**Note:** Follows SKILL.md "Recommended Stack" verbatim — not a from-scratch design.

**Goal:** Build the architecture-conformant model: a frozen pretrained DINOv2 ViT-B/14 backbone with a separate per-angle linear-probe head, trainable only on the heads (CK-14). Aggregation stays the non-learned `core.aggregate_angles` rule.

Create `adapters/backbone.py`: load `facebook/dinov2-base` via `AutoModel.from_pretrained` and set `requires_grad=False` on all backbone params (SKILL.md:146-150). Create `model/heads.py`: `nn.Linear(768, NUM_CLASSES)` per guided angle, where `NUM_CLASSES` = genuine + grade1-4 + rejection. Provide a train entry that runs the per-angle heads on the frozen features (1-epoch smoke on the fixture split must work). Aggregation is `core.aggregate_angles` — do NOT add a learned aggregator (CK-19).

#### Expected Output

- `adapters/backbone.py` — frozen DINOv2 loader
- `model/heads.py` — per-angle `nn.Linear(768, NUM_CLASSES)` + head-training entry
- `tests/test_model_build.py`

#### Success Criteria

- [X] `build_model()` backbone parameters all have `requires_grad=False`; only per-angle head params are trainable (CK-14) — `model/heads.py::AngleHeadModel`/`build_model`, verified by `test_build_model_backbone_frozen_heads_trainable`
- [X] One `nn.Linear(768, NUM_CLASSES)` head exists per guided angle — `model/heads.py::AngleHeadModel.__init__`, verified by `test_build_model_has_one_linear_head_per_guided_angle`
- [X] `aggregate_angles` used for combination is the `core` non-learned rule (no trainable params) (CK-14/19) — `model/heads.py::combine_angle_scores` imports and calls `core.rules.aggregate_angles` directly (same function object, verified by `test_heads_module_reuses_the_core_aggregate_angles_object`)
- [X] A 1-epoch smoke training run on the fixture split completes and writes a checkpoint — `model/heads.py::run_smoke_training`, verified by `test_run_smoke_training_completes_and_writes_a_checkpoint`
- [X] `pytest tests/test_model_build.py` passes — passes in this sandbox via `importorskip` (torch unavailable here, see deviation note below); **independently verified for real** (all 6 tests executed, not skipped) in a disposable venv with real `torch==2.15.0`/`transformers==5.22.1` installed — 6 passed, 0 failed, 0 skipped

#### Subtasks

- [X] Implement frozen backbone loader in `adapters/backbone.py` (SKILL.md:146-150)
- [X] Implement per-angle heads in `model/heads.py`
- [X] Wire a 1-epoch fixture smoke training entry producing a checkpoint
- [X] Write `tests/test_model_build.py`: backbone frozen, heads trainable (CK-14)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Backbone left trainable (requires_grad True) → CK-14 fail | High | Low | Explicit freeze + test asserting zero trainable backbone params |
| Blocker | DINOv2 weights download unavailable in test env | Med | Med | Smoke test may use a tiny stub backbone; freeze-assertion test does not need real download |
| **Deviation (resolved, flagged per orchestrator instruction)** | **`torch`/`transformers` not importable at all in this execution sandbox** — Python 3.14 here is newer than the pinned `torch==2.5.1`/`transformers==4.46.0` wheels (`requirements.txt`) support; one level more severe than the anticipated blocker above (not just "no network for the checkpoint download" — the library itself has no wheel for this interpreter). Same class of environment limitation as `05c-augment-splits` (see `dataset/augment.py`'s docstring: "torch is not installed here"). | Med | High (confirmed) | Followed the instructed fallback: (1) wrote real, correct production code in `adapters/backbone.py`/`model/heads.py` using `transformers.AutoModel`/`torch.nn` exactly per SKILL.md — no stub/mock in production code; (2) gated every test in `tests/test_model_build.py` behind `pytest.importorskip("torch")`/`importorskip("transformers")` with an explicit reason string, so `pytest tests/test_model_build.py` exits 0 (skip, not failure) in this sandbox and real-collected/real-run wherever the pinned deps install; (3) every test uses a tiny stub backbone (`_StubBackbone`) or monkeypatches `AutoModel.from_pretrained`, so no test needs the network or the real ~350MB `facebook/dinov2-base` checkpoint even when torch IS present, per this row's own original mitigation; (4) **did not stop at "skip is enough"** — created a disposable venv (outside the repo, deleted after use) with `pip install torch transformers numpy pytest hypothesis` (unpinned, current versions — `torch==2.15.0`, `transformers==5.22.1`, since the pinned versions have no wheel for Python 3.14 either) and ran `tests/test_model_build.py` there for real: all 6 tests passed, 0 skipped, 0 failed, confirming the implementation is actually correct and not merely untested. No change was made to `requirements.txt`, no real checkpoint was downloaded, nothing in the repo depends on the disposable venv. |
