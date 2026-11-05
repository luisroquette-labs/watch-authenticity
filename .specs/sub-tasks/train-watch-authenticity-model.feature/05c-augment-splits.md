# Step 05c: Augmentation + Dataset Splits

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `04a-manifest`
**Parallel with:** `05a-scrapers`, `05b-dedup-crop`
**Note:** Historical prototype completed. `04c-data-acquisition.md` Task 6 supersedes row-level/source-derived splitting with rights/evidence-gated group splits and quantitative readiness.

**Goal:** Apply heavy lighting/blur/JPEG + photographic-style augmentation (CK-16, item 13 source-bias defense) and produce the six disjoint splits — including the two that make the generalization gates meaningful — with no leakage.

Create `dataset/augment.py`: `augment_pipeline(image)` applying at least one of {blur, JPEG-compression, lighting} plus photographic-style augmentation, returning a tensor different from the input. Create `dataset/splits.py`: build train / val / test / **cross-source** (genuine-in-replica-style vs fake-in-studio-style) / **held-out-factory** / frozen **gold** (~200 photos) splits from the manifest, enforcing that gold and factory-holdout never leak into train.

#### Expected Output

- `dataset/augment.py` — `augment_pipeline(image)`
- `dataset/splits.py` — six-way split builder over the manifest
- `tests/test_augment.py`, `tests/test_splits.py`

#### Success Criteria

- [X] `augment_pipeline(fixture_image)` applies ≥1 of {blur, jpeg, lighting} and returns a tensor ≠ input (CK-16)
- [X] `splits.py` produces train/val/test/cross-source/factory-holdout/gold splits from manifest rows
- [X] Gold-set and factory-holdout photos never appear in the train split (no-leak test)
- [X] Cross-source split pairs genuine-in-replica-style vs fake-in-studio-style
- [X] Cross-source split photos never appear in the train split (no-leak test) — the rubric's heaviest-weighted criterion (0.25, generalization/regression rigor) depends on this pool being untainted
- [X] `pytest tests/test_augment.py tests/test_splits.py` passes

#### Subtasks

- [X] Implement `augment_pipeline` (lighting/blur/JPEG + style) in `dataset/augment.py`
- [X] Implement six-way split builder in `dataset/splits.py`
- [X] Write augment unit test (CK-16)
- [X] Write splits no-leak integration test (gold + factory-holdout + cross-source all disjoint from train)

**Implementation notes / deviations**:
- `augment_pipeline` returns a `numpy.ndarray`, not a `torch.Tensor`. `torch`/`torchvision` are pinned in `requirements.txt` but are not importable in this execution environment (Python 3.14 has no published torch wheel yet); `numpy` + `Pillow` + `opencv-python-headless` (already installed) satisfy CK-16's "returns a tensor" under the array sense of "tensor" without adding an uninstallable dependency. `Pillow`/`numpy` were added as explicit pins to `requirements.txt` since `dataset/augment.py` now imports them directly rather than only transitively.
- `ManifestRecord` has no dedicated photographic-style field, so the cross-source split's "genuine-in-replica-style vs fake-in-studio-style" pairing (item 13) is derived from `source`: `CASUAL_STYLE_SOURCES = {"watchuseek"}` (genuine owner-forum snapshots) paired against `STUDIO_STYLE_SOURCES = {"chinese_replica"}` (fake catalog/studio photos) in `dataset/splits.py`. This is a documented source->style lookup, not a measured/learned style classifier — extend the two sets as new scraper sources are added.
- `build_splits` takes `gold_paths` and `holdout_sources` as explicit caller-supplied sets rather than inferring them, since gold-set membership (hand-verified) and held-out-factory membership are decisions no manifest row encodes on its own.

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Split leakage silently inflates eval accuracy | High | Med | Explicit no-leak assertion test on gold + factory-holdout vs train |
| Risk | Cross-source split mislabeled, invalidating the item-13 gate | High | Med | Build it from manifest `source` + style tags; integration test in Phase 5 consumes it |
