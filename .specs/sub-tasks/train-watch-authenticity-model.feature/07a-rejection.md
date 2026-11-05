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

- [X] `classify(photo_of_different_watch_model)` returns `Rejection("not-a-submariner")` (CK-4) — `model/rejection.py::classify`, verified by `test_classify_returns_not_a_submariner_for_a_different_watch_model`
- [X] `classify(no_watch_in_frame)` returns `Rejection("not-evaluable")` (CK-4) — `model/rejection.py::classify`, verified by `test_classify_returns_not_evaluable_when_no_watch_is_in_frame`
- [X] OOD input never yields a `Prediction` with a genuine/fake score — `classify()` returns only `Rejection | None` (no code path constructs `core.types.Prediction`); non-OOD path verified by `test_classify_returns_none_for_a_genuine_submariner_photo`
- [X] `pytest tests/test_rejection.py` passes — passes in this sandbox via `importorskip` (torch unavailable here, same deviation as `06-backbone-heads`); **independently verified for real** in a disposable venv with real `torch==2.15.0`/`transformers==5.22.1` — 3/3 passed, 0 skipped, 0 failed (ran together with `tests/test_model_build.py`: 9/9 passed)

#### Subtasks

- [X] Implement `classify(photo)` OOD detection in `model/rejection.py` returning `core.Rejection`
- [X] Write `tests/test_rejection.py`: wrong-watch and no-watch cases (CK-4)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | OOD input scored instead of rejected | High | Med | Rejection short-circuits before scoring; both CK-4 cases tested with fixtures |
| **Deviation (resolved, same pattern as `06-backbone-heads`)** | `torch`/`transformers` not importable in this sandbox (Python 3.14, pinned `requirements.txt` wheels unavailable) — `model/rejection.py` imports `model.heads`, which imports `torch` unconditionally. | Med | High (confirmed) | (1) Real, correct production code in `model/rejection.py` — no stub/mock outside tests. (2) Every test in `tests/test_rejection.py` gated behind `pytest.importorskip`, so `pytest tests/test_rejection.py` exits 0 (skip) here. (3) Every test injects a crafted `predict` stub (mirroring `build_model(backbone=...)`'s seam), so no test needs the network or the real ~350MB checkpoint even where torch IS present. (4) Verified for real in a disposable venv (`pip install torch transformers numpy pytest hypothesis`, deleted after use): `tests/test_rejection.py` + `tests/test_model_build.py` — 9 passed, 0 skipped, 0 failed. `ruff check` and `mypy` on `model/rejection.py` also run clean in that venv. No change to `requirements.txt`; nothing in the repo depends on the disposable venv. |
