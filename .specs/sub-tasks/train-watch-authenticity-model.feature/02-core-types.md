# Step 02: Core Types (enums + value objects)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 1
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** None
**Parallel with:** `01-project-scaffolding`
**Note:** None

**Goal:** Define the framework-free domain vocabulary every stage and rule shares: enums and value objects, with zero `torch`/`opencv`/`boto3` imports. This is the inward-most layer (L1) — it imports nothing above it.

Create `core/types.py` with: `Grade` (1-4), `Label` enum (`GENUINE | FAKE | EXCLUDED`), `ManifestRecord` (path, source, url, label, grade:int|None, phash, scraped_at, blur_score:float|None — per the Contracts "Manifest record"), `Prediction` (confidence:float, genuine:bool, grade:int|None, ceiling_applied:bool, heatmap), and `Rejection` (reason, instruction). Use `dataclass`/`Enum` from stdlib only.

#### Expected Output

- `core/types.py` — `Grade`, `Label`, `ManifestRecord`, `Prediction`, `Rejection`
- `tests/test_core_types.py`

#### Success Criteria

- [X] `core/types.py` imports only the Python stdlib (no torch/opencv/boto3/numpy framework imports)
- [X] `ManifestRecord` has required fields source, url, label, grade, phash, scraped_at (grade nullable)
- [X] `Prediction` carries confidence, genuine, grade, ceiling_applied, heatmap; `Rejection` carries reason, instruction
- [X] `pytest tests/test_core_types.py` passes

#### Subtasks

- [X] Define `Grade`, `Label` enums in `core/types.py`
- [X] Define `ManifestRecord`, `Prediction`, `Rejection` value objects
- [X] Write `tests/test_core_types.py` asserting field presence and stdlib-only construction

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | `heatmap` field typed to a framework array leaks torch/numpy into core | Med | Med | Type `heatmap` as a generic sequence/`Any` in core; concrete array type stays in the adapter layer |
