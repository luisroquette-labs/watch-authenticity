# Step 04a: Versioned Dataset Manifest

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `03-core-rules`
**Parallel with:** `04b-infra`
**Note:** Historical prototype completed. `04c-data-acquisition.md` Tasks 1 and 5 supersede the flat source-labeled manifest with candidate, rights, evidence and accepted-training records.

**Goal:** Provide the versioned manifest I/O that every scraped photo row flows through, so the dataset is reproducible and legally auditable (CK-10) without redistributing images.

Create `labeling/manifest.py`: read/write a versioned manifest where each row is a `core.types.ManifestRecord` (source, url, label, grade, phash, scraped_at, blur_score). It is the single call site into `core.grade_map` for grade assignment. Persist as newline-delimited JSON (or CSV) with a version marker so a new dataset version is a new file/record, not an overwrite.

#### Expected Output

- `labeling/manifest.py` — `write_row`/`append`, `read_manifest`, version marker; uses `core.ManifestRecord` + `core.grade_map`
- `tests/test_manifest.py`

#### Success Criteria

- [X] A written row has non-empty source, url, label, grade, phash fields (CK-10)
- [X] Grade is assigned via `core.grade_map` only (no inline grade logic — CK-22)
- [X] Manifest is versioned (new version does not mutate the prior version in place)
- [X] `pytest tests/test_manifest.py` passes

#### Subtasks

- [X] Implement manifest read/write over `core.ManifestRecord` in `labeling/manifest.py`
- [X] Route grade assignment through `core.grade_map`
- [X] Add version marker handling
- [X] Write `tests/test_manifest.py`: `manifest_row(fixture_photo)` has complete fields (CK-10)

**Implementation note**: the version marker is the manifest file name itself
(`manifest_v{version}.jsonl` via `manifest_path()`), not a field inside each
row — a new version is structurally a new file, so it cannot mutate a prior
version's rows in place. `write_row` is an alias for `append` (both names
from Expected Output refer to one implementation). `build_fake_row()` is the
single call site into `core.grade_map`, verified in
`tests/test_manifest.py::test_build_fake_row_delegates_grade_assignment_to_core_grade_map`
by patching `core.grade_map` and asserting it fires exactly once.

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Manifest row missing a required field slips through | High | Low | Construct rows only via `ManifestRecord`; test asserts every required field non-empty |
