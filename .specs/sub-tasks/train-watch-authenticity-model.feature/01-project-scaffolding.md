# Step 01: Project Scaffolding (deps + preflight)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 1
**Model:** haiku
**Agent:** haiku
**Depends on:** None
**Parallel with:** `02-core-types`
**Note:** This is the shared prerequisite every later step imports/runs against, so it is a deliberate small standalone step.

**Goal:** Stand up the greenfield Python project so every later step has pinned dependencies and one canonical preflight command (`ruff` + `mypy` + `pytest`). No CI/preflight convention exists in this repo yet.

Create `requirements.txt` with the exact version pins from SKILL.md:120-124 (`torch==2.5.1`, `torchvision==0.20.1`, `transformers==4.46.0`, `grad-cam==1.5.4`, `ImageHash==4.3.1`, `opencv-python==4.10.0.84`) plus `pytest`, `hypothesis`, `ruff`, `mypy`, `boto3`, `huggingface_hub`, `requests`. Create `pyproject.toml` configuring ruff + mypy + pytest, and a single preflight entry point (`scripts/preflight.sh` or a `pyproject` script) running `ruff check` → `mypy` → `pytest`. Create the package skeleton dirs with `__init__.py` so imports resolve: `core/`, `scrapers/`, `labeling/`, `dataset/`, `adapters/`, `model/`, `calibration/`, `quality_gate/`, `explainability/`, `inference/`, `evaluation/`, `infra/`, `release/`, `tests/`.

#### Expected Output

- `requirements.txt` (pinned, from SKILL.md)
- `pyproject.toml` (ruff + mypy + pytest config)
- `scripts/preflight.sh` (or equivalent) running ruff → mypy → pytest
- Package directories with `__init__.py` and a `tests/` dir

#### Success Criteria

- [ ] `requirements.txt` contains every SKILL.md:120-124 pin verbatim plus test/lint/infra deps
- [ ] Preflight command runs `ruff` + `mypy` + `pytest` and exits 0 on an empty/placeholder test
- [ ] `python -c "import core, dataset, model, evaluation"` resolves (dirs + `__init__.py` exist)

#### Subtasks

- [ ] Write `requirements.txt` with SKILL.md pins + pytest/hypothesis/ruff/mypy/boto3/huggingface_hub/requests
- [ ] Write `pyproject.toml` with ruff + mypy + pytest config
- [ ] Write `scripts/preflight.sh` chaining ruff → mypy → pytest
- [ ] Create all package dirs with `__init__.py` and an empty `tests/` package
- [ ] Add one trivial passing `tests/test_smoke_import.py` so preflight is green

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Heavy ML deps (torch) slow/unavailable in CI env | Med | Med | Preflight only needs ruff/mypy/pytest import-resolvable; torch imports confined to adapter layer, not `core/` tests |
| Risk | Pin incompatibility with local CUDA | Low | Low | SKILL.md note: pin torch to the CUDA version at deploy time; comment this in `requirements.txt` |
