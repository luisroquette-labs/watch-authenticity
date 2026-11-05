# Step 05c: Augmentation + Dataset Splits

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `04a-manifest`
**Parallel with:** `05a-scrapers`, `05b-dedup-crop`
**Note:** None

**Goal:** Apply heavy lighting/blur/JPEG + photographic-style augmentation (CK-16, item 13 source-bias defense) and produce the six disjoint splits — including the two that make the generalization gates meaningful — with no leakage.

Create `dataset/augment.py`: `augment_pipeline(image)` applying at least one of {blur, JPEG-compression, lighting} plus photographic-style augmentation, returning a tensor different from the input. Create `dataset/splits.py`: build train / val / test / **cross-source** (genuine-in-replica-style vs fake-in-studio-style) / **held-out-factory** / frozen **gold** (~200 photos) splits from the manifest, enforcing that gold and factory-holdout never leak into train.

#### Expected Output

- `dataset/augment.py` — `augment_pipeline(image)`
- `dataset/splits.py` — six-way split builder over the manifest
- `tests/test_augment.py`, `tests/test_splits.py`

#### Success Criteria

- [ ] `augment_pipeline(fixture_image)` applies ≥1 of {blur, jpeg, lighting} and returns a tensor ≠ input (CK-16)
- [ ] `splits.py` produces train/val/test/cross-source/factory-holdout/gold splits from manifest rows
- [ ] Gold-set and factory-holdout photos never appear in the train split (no-leak test)
- [ ] Cross-source split pairs genuine-in-replica-style vs fake-in-studio-style
- [ ] `pytest tests/test_augment.py tests/test_splits.py` passes

#### Subtasks

- [ ] Implement `augment_pipeline` (lighting/blur/JPEG + style) in `dataset/augment.py`
- [ ] Implement six-way split builder in `dataset/splits.py`
- [ ] Write augment unit test (CK-16)
- [ ] Write splits no-leak integration test (gold + factory-holdout disjoint from train)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Split leakage silently inflates eval accuracy | High | Med | Explicit no-leak assertion test on gold + factory-holdout vs train |
| Risk | Cross-source split mislabeled, invalidating the item-13 gate | High | Med | Build it from manifest `source` + style tags; integration test in Phase 5 consumes it |
