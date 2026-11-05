# Step 05a: Per-Source Scrapers (legal gates)

**Task File:** `.specs/tasks/todo/train-watch-authenticity-model.feature.md`

> The task file moves between `.specs/tasks/{draft,todo,in-progress,done}/` as work progresses; if it is not at this path, resolve it by its filename under `.specs/tasks/`.

**Phase:** Phase 2
**Model:** sonnet
**Agent:** sdd:developer
**Depends on:** `04a-manifest`
**Parallel with:** `05b-dedup-crop`, `05c-augment-splits`
**Note:** Seven source files share ONE pattern (fetch → `core.source_label_rule` → write via `labeling.manifest`). Label derivation is never per-scraper logic.

**Goal:** Ingest photos per source and derive each row's label strictly via `core.source_label_rule`, with the two legally load-bearing gates enforced at the adapter boundary: Mercado Livre via official API only (CK-28) and Reddit default-excluded (CK-27/29).

Create `scrapers/<source>.py` for olx, mercadolivre, chrono24, reptime, watchuseek, reddit, chinese_replica. Each fetches photos and writes manifest rows with labels from `core.source_label_rule` (never ad-hoc). `scrapers/mercadolivre.py` exposes `mercadolivre_ingest(method)` where `method="scrape"` raises/refuses and `method="official_api"` proceeds (CK-28). `scrapers/reddit.py` passes `source="reddit"` through `source_label_rule` so it is EXCLUDED unless a documented permission record flips `permission_on_file=True` (CK-27/29).

#### Expected Output

- `scrapers/{olx,mercadolivre,chrono24,reptime,watchuseek,reddit,chinese_replica}.py`
- `tests/test_scrapers.py`

#### Success Criteria

- [ ] Every scraper derives label via `core.source_label_rule` (no inline label logic)
- [ ] `mercadolivre_ingest(method="scrape")` raises; `mercadolivre_ingest(method="official_api")` proceeds (CK-28)
- [ ] Reddit ingestion yields EXCLUDED rows unless a documented permission record exists (CK-27/29)
- [ ] An OLX/ML "original" listing is never written as GENUINE (CK-13/18)
- [ ] `pytest tests/test_scrapers.py` passes (network mocked/fixtured)

#### Subtasks

- [ ] Implement the seven scrapers over the shared fetch→label→write pattern
- [ ] Implement `mercadolivre_ingest(method)` with scrape-path refusal (CK-28)
- [ ] Route reddit through `source_label_rule` permission gate (CK-27/29)
- [ ] Write `tests/test_scrapers.py`: CK-28 both methods, CK-27 both permission states, CK-13 original-ad exclusion

#### Blockers & Risks

| Type | Item | Impact | Likelihood | Mitigation / Resolution |
|------|------|--------|------------|-------------------------|
| Risk | Reddit content enters training without permission | High | Low | Default EXCLUDED via `source_label_rule`; owner sign-off record required to flip; pitfall test CK-29 |
| Risk | Per-source ToS/anti-bot variance | Med | High | Scraping mechanics isolated per adapter; only label derivation is centralized; ML sourcing via official API |
