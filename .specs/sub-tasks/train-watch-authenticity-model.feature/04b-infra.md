# Step 04b: Training Infra (confirm-gated AWS + S3)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `03-core-rules`
**Parallel with:** `04a-manifest`
**Note:** Independent lane — needs only `core`, not the dataset steps. Runs concurrently with the Phase 2 dataset work.

**Goal:** Make every non-trivial GPU provisioning call impossible without an explicit cost confirmation (CK-20), and wire S3 photo storage. Real money per round ($5-20); irreversible spend if triggered blindly.

Create `infra/aws_train.(py|sh)` exposing `launch_training(instance_type, *, confirm: bool)` which **refuses/raises when `confirm=False`** and only provisions when `confirm=True`. Default instance type `g6.xlarge` (AD-2), and surface the g5-vs-g6 choice + live-priced estimate to the owner at the confirmation point (never resolve silently). Create `infra/s3_sync.py` for photo upload/download via boto3.

#### Expected Output

- `infra/aws_train.py` — `launch_training(instance_type, *, confirm)`, g6.xlarge default, cost surfaced
- `infra/s3_sync.py` — boto3 S3 photo sync
- `tests/test_infra.py`

#### Success Criteria

- [ ] `launch_training(instance_type="g5.xlarge", confirm=False)` raises/refuses to provision (CK-20)
- [ ] `launch_training(instance_type="g5.xlarge", confirm=True)` proceeds to provisioning (mockable)
- [ ] Default instance type is `g6.xlarge`; the g5-vs-g6 divergence from spec is surfaced, not silent (AD-2)
- [ ] `pytest tests/test_infra.py` passes (boto3 mocked — no real spend in tests)

#### Subtasks

- [ ] Implement `launch_training` with `confirm` gate and g6.xlarge default in `infra/aws_train.py`
- [ ] Surface g5-vs-g6 + estimated cost at the confirmation boundary
- [ ] Implement `infra/s3_sync.py` photo sync
- [ ] Write `tests/test_infra.py`: confirm=False refuses, confirm=True proceeds (CK-20)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Blocker | AWS creds unavailable in test env | Med | High | Mock boto3 in tests; the confirm-gate logic is pure and testable without AWS |
| Risk | A test accidentally provisions a real instance | High | Low | Tests use mocked boto3 only; `confirm=True` path exercised against the mock |
