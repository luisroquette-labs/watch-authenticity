# Step 07a: OOD / Rejection Classifier

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 3
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `06-backbone-heads`
**Parallel with:** `07b-calibration`
**Note:** None

**Goal:** Ensure out-of-distribution input (wrong watch model, no watch) returns a dedicated `Rejection`, never a fabricated genuine/fake score (CK-4).

Create `model/rejection.py`: `classify(photo)` returning `core.Rejection("not-a-submariner")` for a different watch model and `core.Rejection("not-evaluable")` when no watch is in frame. Use the rejection class of the per-angle heads and/or a Submariner-presence check; OOD input must short-circuit before any genuine/fake scoring.

#### Expected Output

- `model/rejection.py` — `classify(photo)` → `Rejection(...)` for OOD
- `tests/test_rejection.py`

#### Success Criteria

- [ ] `classify(photo_of_different_watch_model)` returns `Rejection("not-a-submariner")` (CK-4)
- [ ] `classify(no_watch_in_frame)` returns `Rejection("not-evaluable")` (CK-4)
- [ ] OOD input never yields a `Prediction` with a genuine/fake score
- [ ] `pytest tests/test_rejection.py` passes

#### Subtasks

- [ ] Implement `classify(photo)` OOD detection in `model/rejection.py` returning `core.Rejection`
- [ ] Write `tests/test_rejection.py`: wrong-watch and no-watch cases (CK-4)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | OOD input scored instead of rejected | High | Med | Rejection short-circuits before scoring; both CK-4 cases tested with fixtures |
