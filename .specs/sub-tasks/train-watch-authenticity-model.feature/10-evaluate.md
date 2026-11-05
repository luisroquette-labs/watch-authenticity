# Step 10: Three-Split Evaluation

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 5
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `09-inference-predict`
**Parallel with:** None
**Note:** None

**Goal:** Report standard-test, cross-source-test, AND held-out-factory-test accuracy as separate breakdowns — not one aggregate number — with per-grade accuracy on the factory-disjoint set (CK-7, CK-9). This is the highest-weight rubric (Generalization & Regression, 0.25).

Create `evaluation/evaluate.py`: run `inference.predict` over the standard-test, cross-source, and held-out-factory splits built in Step 05c; compute accuracy per split and per-grade for the factory-holdout split. Produce a structured report carrying all three breakdowns (standard, cross-source delta, per-grade factory-holdout).

#### Expected Output

- `evaluation/evaluate.py` — three-split evaluator + per-grade report
- `tests/test_evaluate.py`

#### Success Criteria

- [ ] Harness computes standard-test AND cross-source-test accuracy separately from the manifest-labeled fixture set (CK-7)
- [ ] Harness reports per-grade accuracy on a held-out split whose factories/sources are absent from the training manifest (CK-9)
- [ ] Report contains all three breakdowns, not a single aggregate number
- [ ] `pytest tests/test_evaluate.py` passes

#### Subtasks

- [ ] Implement three-split evaluation + per-grade factory-holdout in `evaluation/evaluate.py`
- [ ] Emit a structured report (standard / cross-source / per-grade factory)
- [ ] Write integration test: standard + cross-source computed separately (CK-7)
- [ ] Write integration test: per-grade factory-disjoint accuracy reported (CK-9)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Factory-holdout not truly disjoint from train | High | Med | Consumes Step 05c no-leak splits; test asserts holdout factories absent from training manifest |
