# Step 03: Core Rules (safety invariants, single source of truth)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 1
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `02-core-types`
**Parallel with:** None
**Note:** This is the highest-leverage, rubric-tested module. Every invariant lives here ONCE (CK-22) — no stage may re-implement any of these.

**Goal:** Implement every pure safety rule the task's credibility rests on, framework-free, each exactly once. The rubric tests these directly; duplicating any of them across stages fails CK-22.

Implement in `core/rules.py` (stdlib only, no torch):
- `clip_confidence(p) -> float` → clamps to `[0.001, 0.999]` (CK-1).
- `apply_grade4_ceiling(raw_genuine, grade) -> float` → when `grade == 4`, return `min(raw_genuine, GRADE4_GENUINE_CEILING≈0.60)`, else unchanged (CK-3).
- `grade_map(...) -> int` → single factory/price → grade mapping (used by manifest).
- `source_label_rule(source, *, subreddit=None, marketplace=None, listing_text=None, permission_on_file=False) -> Label` → OLX/ML `listing_text="original"` → `EXCLUDED` never `GENUINE` (CK-13/18); Reddit/r-rolex → `EXCLUDED` unless `permission_on_file=True` then `GENUINE` (CK-27/29); Chrono24/Watchuseek → `GENUINE`; replica sources → `FAKE`.
- `aggregate_angles(scores_per_angle) -> dict` → simple non-learned rule, ZERO trainable parameters (CK-19).
- `publish_gate(standard_acc, cross_source_acc, max_delta) -> bool` → blocks when `standard_acc - cross_source_acc > max_delta` (CK-8).
- `goldset_regression_gate(current_acc, baseline_acc) -> bool` → blocks when `current_acc < baseline_acc` (CK-12).

Expose `GRADE4_GENUINE_CEILING` as a named constant with a `ponytail:`-style comment noting it is a tunable policy value.

#### Expected Output

- `core/rules.py` — all seven pure functions + `GRADE4_GENUINE_CEILING`
- `tests/test_core_rules.py` (unit + property-based via hypothesis)

#### Success Criteria

- [X] `core/rules.py` imports only stdlib (no torch/opencv/boto3)
- [X] `clip_confidence(0.0)==0.001`, `clip_confidence(1.0)==0.999`, `clip_confidence(0.5)==0.5` (CK-1)
- [X] `apply_grade4_ceiling(0.95,4) <= 0.60`, `apply_grade4_ceiling(0.40,4)==0.40` (CK-3)
- [X] `source_label_rule(listing_text="original", marketplace="olx")==EXCLUDED` (CK-13/18)
- [X] `source_label_rule(source="reddit", subreddit="r/rolex", permission_on_file=False)==EXCLUDED`; `...True)==GENUINE` (CK-27)
- [X] `aggregate_angles` has no trainable parameters (plain function) (CK-19)
- [X] `publish_gate(0.94,0.60,0.10)` blocks; `publish_gate(0.94,0.91,0.10)` allows (CK-8)
- [X] Property: for any `p∈[0,1]`, `clip_confidence(p)∈[0.001,0.999]`; for any `raw_genuine∈[0,1]`, `apply_grade4_ceiling(raw_genuine,4)<=0.60`
- [X] `pytest tests/test_core_rules.py` passes

#### Subtasks

- [X] Implement `clip_confidence`, `apply_grade4_ceiling` + `GRADE4_GENUINE_CEILING`
- [X] Implement `grade_map`, `source_label_rule` (OLX-original + Reddit-permission gates)
- [X] Implement `aggregate_angles` (simple rule), `publish_gate`, `goldset_regression_gate`
- [X] Write unit tests for every rule's documented cases (CK-1,3,8,12,13,18,19,27)
- [X] Write hypothesis property tests for clip bounds and grade-4 ceiling

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | A later stage re-implements a rule (grade map / clamp) inline → CK-22 fail | High | Med | Every stage imports from `core.rules`; code-reviewer checks for duplicated logic at phase review |
| Risk | `source_label_rule` branch order lets an "original" OLX ad fall through to GENUINE | High | Low | EXCLUDED checks evaluated first; explicit unit test for the pitfall (CK-18) |
