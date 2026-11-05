# Step 05b: Perceptual-Hash Dedup + Studio Filter + Per-Angle Crop

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `04a-manifest`
**Parallel with:** `05a-scrapers`, `05c-augment-splits`
**Note:** None

**Goal:** Keep source-style confounds and duplicates out of the training set, and isolate each guided angle so heads see the watch part, not the background (CK-11, CK-21) — the crop is a first-line defense against the source-bias risk (item 13).

Create `dataset/dedup.py`: perceptual-hash (pHash via `ImageHash`) near-duplicate removal with an **empirically-tuned, commented** Hamming threshold (no universal constant — SKILL.md), plus a studio/catalog-photo filter (near-perfect white background, repeated across listings). Create `dataset/crop.py`: `crop_angle(image, angle)` returning a region smaller than the full frame centered on the expected sub-region per guided angle (rehaut, cyclops, crown, clasp, dial, caseback).

#### Expected Output

- `dataset/dedup.py` — `dedup_filter`, `studio_photo_filter`, tunable commented threshold
- `dataset/crop.py` — `crop_angle(image, angle)`
- `tests/test_dedup.py`, `tests/test_crop.py`

#### Success Criteria

- [ ] `dedup_filter([imgA, imgA_near_duplicate])` drops the duplicate by pHash distance below threshold (CK-11)
- [ ] `studio_photo_filter(white_background_catalog_fixture)` flags the image for exclusion (CK-11)
- [ ] Property: `hash_distance(a,b)==hash_distance(b,a)` (symmetry) (CK-11)
- [ ] `crop_angle(full_frame, angle="rehaut")` returns a region smaller than the full frame centered on the expected sub-region (CK-21)
- [ ] Hamming/threshold constants are commented as empirically-tuned-per-dataset, not universal
- [ ] `pytest tests/test_dedup.py tests/test_crop.py` passes

#### Subtasks

- [ ] Implement `dedup_filter` + `studio_photo_filter` in `dataset/dedup.py`
- [ ] Implement `crop_angle` in `dataset/crop.py`
- [ ] Write dedup unit + symmetry property tests (CK-11)
- [ ] Write crop unit test (CK-21)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | pHash threshold over/under-aggressive | Med | Med | Threshold tunable + commented; test uses a controlled near-duplicate fixture |
| Risk | Crop misses the sub-region, defeats source-bias defense | High | Med | Per-angle expected regions; Grad-CAM audit (Phase 4) flags background-dominated maps |
