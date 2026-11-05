# Step 04a: Versioned Dataset Manifest

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `03-core-rules`
**Parallel with:** `04b-infra`
**Note:** None

**Goal:** Provide the versioned manifest I/O that every scraped photo row flows through, so the dataset is reproducible and legally auditable (CK-10) without redistributing images.

Create `labeling/manifest.py`: read/write a versioned manifest where each row is a `core.types.ManifestRecord` (source, url, label, grade, phash, scraped_at, blur_score). It is the single call site into `core.grade_map` for grade assignment. Persist as newline-delimited JSON (or CSV) with a version marker so a new dataset version is a new file/record, not an overwrite.

#### Expected Output

- `labeling/manifest.py` — `write_row`/`append`, `read_manifest`, version marker; uses `core.ManifestRecord` + `core.grade_map`
- `tests/test_manifest.py`

#### Success Criteria

- [ ] A written row has non-empty source, url, label, grade, phash fields (CK-10)
- [ ] Grade is assigned via `core.grade_map` only (no inline grade logic — CK-22)
- [ ] Manifest is versioned (new version does not mutate the prior version in place)
- [ ] `pytest tests/test_manifest.py` passes

#### Subtasks

- [ ] Implement manifest read/write over `core.ManifestRecord` in `labeling/manifest.py`
- [ ] Route grade assignment through `core.grade_map`
- [ ] Add version marker handling
- [ ] Write `tests/test_manifest.py`: `manifest_row(fixture_photo)` has complete fields (CK-10)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Manifest row missing a required field slips through | High | Low | Construct rows only via `ManifestRecord`; test asserts every required field non-empty |
