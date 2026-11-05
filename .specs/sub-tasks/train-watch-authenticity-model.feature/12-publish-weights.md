# Step 12: Publish Weights + Model Card (gated)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 5
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `11-release-gate`
**Parallel with:** None
**Note:** Final step — also carries the full-suite coverage meta-gates (CK-23/24/25/26) and the exported-artifact smoke test.

**Goal:** Prepare open weights (~400MB) and a model card for publication ONLY when the release gates pass, with raw dataset photos excluded from what is published (CK-17) — and close out the test-suite coverage gates with the exported-artifact smoke test.

Create `release/publish_weights.py`: publish weights + model card via `huggingface_hub` only if `evaluation.release_gate` passes; the published bundle contains weights + card, never raw dataset photos (LAION-style URLs-only posture). The model card documents architecture, calibration, the three-split evaluation, and the source/legal posture (Reddit excluded by default). Add the smoke test loading the exported artifact and asserting a schema-valid prediction on one gold-set image.

#### Expected Output

- `release/publish_weights.py` — gated publish (weights + card, raw photos excluded)
- model card (markdown)
- `tests/test_publish_smoke.py` (smoke) + coverage confirmation

#### Success Criteria

- [ ] Publish proceeds only when `release_gate` passes; blocked gate → no publish (CK-17, CK-8/12 wired)
- [ ] Published bundle contains weights (~400MB) + model card; raw dataset photos are excluded (CK-17)
- [ ] [smoke] exported weights load and return a schema-valid prediction (confidence in range, grade/rejection field present, heatmap present) for one fixture gold-set image (CK-23/24/25/26)
- [ ] Every Test Matrix type (unit/integration/smoke/property-based) has ≥1 test across the suite (CK-23/24)
- [ ] Every Test Cases to Cover entry has an implemented, passing test; no orphan checklist items (CK-25/26)
- [ ] `pytest` full suite passes via preflight

#### Subtasks

- [ ] Implement gated `release/publish_weights.py` (weights + card, raw photos excluded)
- [ ] Write the model card (architecture, calibration, 3-split eval, Reddit/legal posture)
- [ ] Write `tests/test_publish_smoke.py` loading the exported artifact (CK-23/26)
- [ ] Verify full-suite coverage: every Test Matrix row + Test Cases to Cover entry has a test (CK-24/25)

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Raw dataset photos leak into the published bundle | High | Low | Publish bundle whitelists weights + card only; test asserts no image files present |
| Risk | A Test Cases to Cover entry has no test (orphan) → CK-25/26 fail | High | Med | Final coverage pass maps every entry to a test before publish |
