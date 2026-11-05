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

- [ ] `build_model()` backbone parameters all have `requires_grad=False`; only per-angle head params are trainable (CK-14)
- [ ] One `nn.Linear(768, NUM_CLASSES)` head exists per guided angle
- [ ] `aggregate_angles` used for combination is the `core` non-learned rule (no trainable params) (CK-14/19)
- [ ] A 1-epoch smoke training run on the fixture split completes and writes a checkpoint
- [ ] `pytest tests/test_model_build.py` passes

#### Subtasks

- [ ] Implement frozen backbone loader in `adapters/backbone.py` (SKILL.md:146-150)
- [ ] Implement per-angle heads in `model/heads.py`
- [ ] Wire a 1-epoch fixture smoke training entry producing a checkpoint
- [ ] Write `tests/test_model_build.py`: backbone frozen, heads trainable (CK-14)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Backbone left trainable (requires_grad True) → CK-14 fail | High | Low | Explicit freeze + test asserting zero trainable backbone params |
| Blocker | DINOv2 weights download unavailable in test env | Med | Med | Smoke test may use a tiny stub backbone; freeze-assertion test does not need real download |
