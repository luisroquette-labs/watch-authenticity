# Step 05b: Perceptual-Hash Dedup + Studio Filter + Per-Angle Crop

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** None
**Parallel with:** `04a-manifest`, `04b-infra`, `05a-scrapers`, `05c-augment-splits`
**Note:** Historical prototype completed. `04c-data-acquisition.md` Task 4 adds physical-watch/repost grouping and replaces unconditional studio deletion with capture-style classification.

**Goal:** Keep source-style confounds and duplicates out of the training set, and isolate each guided angle so heads see the watch part, not the background (CK-11, CK-21) — the crop is a first-line defense against the source-bias risk (item 13).

Create `dataset/dedup.py`: perceptual-hash (pHash via `ImageHash`) near-duplicate removal with an **empirically-tuned, commented** Hamming threshold (no universal constant — SKILL.md), plus a studio/catalog-photo filter (near-perfect white background, repeated across listings). Create `dataset/crop.py`: `crop_angle(image, angle)` returning a region smaller than the full frame centered on the expected sub-region per guided angle (rehaut, cyclops, crown, clasp, dial, caseback).

#### Expected Output

- `dataset/dedup.py` — `dedup_filter`, `studio_photo_filter`, tunable commented threshold
- `dataset/crop.py` — `crop_angle(image, angle)`
- `tests/test_dedup.py`, `tests/test_crop.py`

#### Success Criteria

- [X] `dedup_filter([imgA, imgA_near_duplicate])` drops the duplicate by pHash distance below threshold (CK-11)
- [X] `studio_photo_filter(white_background_catalog_fixture)` flags the image for exclusion (CK-11)
- [X] Property: `hash_distance(a,b)==hash_distance(b,a)` (symmetry) (CK-11)
- [X] `crop_angle(full_frame, angle="rehaut")` returns a region smaller than the full frame centered on the expected sub-region (CK-21)
- [X] Hamming/threshold constants are commented as empirically-tuned-per-dataset, not universal
- [X] `pytest tests/test_dedup.py tests/test_crop.py` passes

#### Subtasks

- [X] Implement `dedup_filter` + `studio_photo_filter` in `dataset/dedup.py`
- [X] Implement `crop_angle` in `dataset/crop.py`
- [X] Write dedup unit + symmetry property tests (CK-11)
- [X] Write crop unit test (CK-21)

**Implementation note:** `ImageHash==4.3.1` (pinned in `requirements.txt`) was
not yet installed in this environment's Python 3.14 user site-packages;
installed it via `python3 -m pip install "ImageHash==4.3.1"` before running
tests. `crop_angle`/`dedup_filter`/`studio_photo_filter` all standardize on
a `PIL.Image`-like input (`.size`/`.crop()`/`.convert()`) — this step's two
files don't need `opencv` directly, only the shared `dataset/` dependency
column in the architecture table does collectively (likely for
`augment.py`/`splits.py` in the parallel `05c` step). Per-angle crop
fractions in `dataset/crop.py` are first-pass estimates (commented
`ponytail:`), not measured framing data, since the guided-capture app that
will produce these photos doesn't exist yet (Requirements item 11).

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | pHash threshold over/under-aggressive | Med | Med | Threshold tunable + commented; test uses a controlled near-duplicate fixture |
| Risk | Crop misses the sub-region, defeats source-bias defense | High | Med | Per-angle expected regions; Grad-CAM audit (Phase 4) flags background-dominated maps |
