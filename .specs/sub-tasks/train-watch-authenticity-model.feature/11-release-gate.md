# Step 11: Release Gates (cross-source + gold-set regression)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 5
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `10-evaluate`
**Parallel with:** None
**Note:** Gates must REFUSE publish, not warn. Predicates live in `core` (Step 03) — this step wires them to the evaluation output.

**Goal:** Wire the evaluation output to the two publish-blocking gates so a source-biased or regressed candidate cannot be published (CK-8, CK-12).

Create `evaluation/release_gate.py`: call `core.publish_gate(standard_acc, cross_source_acc, max_delta)` and `core.goldset_regression_gate(current_acc, baseline_acc)` against the Step 10 report and a frozen previous-version baseline. If either returns block, the release is refused (an exception / non-zero result), not a printed warning.

#### Expected Output

- `evaluation/release_gate.py` — gate orchestration over `core` predicates + frozen baseline
- `tests/test_release_gate.py`

#### Success Criteria

- [ ] `publish_gate(standard_acc=0.94, cross_source_acc=0.60, max_delta=0.10)` blocks publish (CK-8)
- [ ] `publish_gate(standard_acc=0.94, cross_source_acc=0.91, max_delta=0.10)` allows publish (CK-8)
- [ ] Publish is refused (not warned) when gold-set accuracy regresses vs the frozen baseline (CK-12 integration)
- [ ] Gate predicates are the `core` implementations, not re-coded here (CK-22)
- [ ] `pytest tests/test_release_gate.py` passes

#### Subtasks

- [ ] Implement `evaluation/release_gate.py` calling `core.publish_gate` + `core.goldset_regression_gate`
- [ ] Load the frozen previous-version gold-set baseline
- [ ] Make a failing gate refuse (raise/return block), never warn-and-continue
- [ ] Write unit tests (CK-8) + gold-set regression integration test (CK-12)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Gate logs a warning but still allows publish | High | Med | Gate raises/returns block; integration test asserts publish refused on regression |
| Blocker | No frozen baseline exists for the first version | Med | High | First version records the baseline; gate treated as pass-and-record when no prior baseline |
