# Source Acquisition Connectors Implementation Plan

> Execute each task in an isolated branch. No task may run longer than two hours.

**Goal:** Fill the gap between approved source discovery and the existing evidence-gated collector without scraping prohibited sources or treating marketplace claims as labels.

**Architecture:** Versioned source-access evidence authorizes discovery separately from download and training. Connectors emit immutable metadata into a resumable discovery ledger. The existing collector downloads only rights-cleared URLs from approved public hosts, while partner folders use the same image validation and receipt lineage. Angle suggestions remain advisory until reviewed, and acquisition priorities come from readiness deficits rather than raw file counts.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`

## Constraints

- No connector may perform network I/O without a current official API, licensed feed, rights-cleared export or owner-authorized upload.
- API availability does not grant image-training rights; discovery, download and training are separate decisions.
- Reddit remains blocked without written Reddit and rightsholder permission.
- Mercado Livre uses only its official API; raw scraping always refuses before network access.
- Chrono24, OLX, Watchuseek, RepTime and replica-vendor connectors remain input-only until an approved feed/export exists.
- No CAPTCHA bypass, proxy rotation, browser evasion or paid AI API.
- Every external URL is untrusted; private/internal destinations and unapproved redirects are rejected.
- Each task is timeboxed to two hours and gets focused tests before the final review and one canonical preflight.

## Source order

| Priority | Source | Initial state | Activation evidence |
|---|---|---|---|
| 1 | Partner/owner upload | enabled with rights record | signed/export declaration tied to every file |
| 2 | Mercado Livre | discovery-only | official API access; separate image rightsholder permission for download/train |
| 3 | Chrono24/OLX/Watchuseek | unavailable | licensed API/feed/export plus rightsholder permission |
| 4 | RepTime/replica vendors | unavailable | rights-cleared export plus explicit replica/factory evidence |
| blocked | Reddit | blocked | written Reddit approval and permission from applicable content rightsholders |

## Task 1: Bind source access to verifiable evidence

**Files:** `acquisition/types.py`, `acquisition/source_policy.py`, `config/source_policies.json`, `tests/test_source_policy.py`

- Add explicit access state, evidence URL/hash, verified timestamp, expiry, allowed image hosts and conservative rate limit.
- Refuse discovery/download when access evidence is absent, expired or incompatible with the requested method.
- Keep the deny-all fallback complete and immutable.
- Test missing evidence, expiry boundary, unknown host, and official-API discovery without training rights.

## Task 2: Add a resumable discovery ledger

**Files:** `acquisition/discovery.py`, `tests/test_discovery.py`

- Persist canonical source record IDs, listing URL, seller group, reference, image URLs, cursor and observation time.
- Make replay idempotent and reject divergent payloads for the same source record/version.
- Advance a cursor only after its complete page is durably recorded.
- Test duplicate pages, crash/resume, cursor forks and input-order determinism.

## Task 3: Complete owner-authorized partner ingestion

**Files:** `acquisition/partner_import.py`, `acquisition/collect.py`, `tests/test_acquisition_collect.py`

- Accept a folder containing a manifest, declared images, rights evidence and multiple photos per physical watch/listing.
- Reuse the existing sanitizer, hashes, immutable receipts and COS object layout.
- Reject undeclared files, traversal, symlinks, swapped files, missing evidence and mixed rights.
- Prove a 20–30 image fixture batch can resume without duplicate objects or receipts.

## Task 4: Implement Mercado Livre official-API discovery

**Files:** `scrapers/mercadolivre.py`, `acquisition/discovery.py`, `tests/test_mercadolivre_discovery.py`

- Query exact Submariner references through an injected official API client and paginate deterministically.
- Store metadata only until a separate rights record authorizes image download and training.
- Preserve item, seller and picture IDs; never infer genuine/fake from listing claims.
- Test pagination, token omission, rate limiting, malformed payloads and raw-scrape refusal without real network calls.

## Task 5: Activate licensed feeds one source at a time

**Files:** `scrapers/{chrono24,olx,watchuseek,reptime,chinese_replica}.py`, source-specific tests

- Reuse the common discovery contract; do not create source-specific downloaders.
- Implement only the source whose real approved feed/export contract is available in the task.
- Keep every other adapter returning `SourceUnavailable`.
- Test schema drift, source mismatch, missing rights and deterministic normalization.

## Task 6: Harden external image downloads

**Files:** `acquisition/collect.py`, `acquisition/source_policy.py`, `tests/test_acquisition_collect.py`

- Require HTTPS and an exact approved hostname for every image and redirect hop.
- Resolve and reject loopback, private, link-local, multicast, reserved and unspecified destinations before each connection.
- Preserve size, MIME, magic-byte, image-decoding, retry and immutable-hash checks.
- Test hostname tricks, credentials in URLs, ports, redirects, DNS rebinding boundary and CDN allowlists.

## Task 7: Bootstrap reviewed angle suggestions

**Files:** `dataset/angle_review.py`, `tests/test_angle_review.py`

- Record human-reviewed angle labels for dial, cyclops, rehaut, crown, clasp and caseback.
- Refuse automatic suggestions until every angle has at least 100 reviewed examples and include an unknown result.
- Keep suggestions advisory; only reviewed angles may enter readiness counts.
- Test threshold boundaries, low confidence, unknown views and reviewer/hash lineage.

## Task 8: Preserve watch/evidence lineage end to end

**Files:** integration tests across `labeling/review.py`, `dataset/entities.py`, `dataset/dedup.py`

- Verify discovered candidates cannot become training rows without Tier A/B evidence and current rights.
- Keep one listing, physical watch, seller, content cluster and batch in one split family.
- Quarantine conflicting labels for duplicate content instead of voting.
- Test multi-angle listing grouping and cross-source reposts.

## Task 9: Plan acquisition from readiness deficits

**Files:** `acquisition/planner.py`, `tests/test_acquisition_planner.py`

- Convert readiness cell deficits into ranked collection targets by class, grade, angle, source and entity.
- Enforce source maximum 40%, seller maximum 10% and one-source-at-a-time rollout.
- Stop recommending more of a saturated distribution after the existing two-wave learning-curve rule.
- Test raw-volume traps, missing angles, source dominance and deterministic ranking.

## Task 10: Verify and roll out in bounded waves

**Files:** targeted tests, one final `scripts/preflight.sh`, immutable rollout receipts

- Run focused tests after each task and independent review before the single full preflight.
- Run synthetic local E2E first, then 20–30 authorized images, 500, 5,000 and readiness scale.
- Never provision IBM GPU until the frozen dataset and independent IBM cost gates pass.
- Keep real API/feed, COS and training proof `PENDENTE E2E` until observed externally.

## Completion proof

- A connector cannot discover without current access evidence or download without separate current rights.
- A repeated or interrupted discovery/collection run resumes without duplication or cursor loss.
- A crafted URL cannot reach an internal destination or an unapproved host.
- A source/listing claim never becomes an accepted authenticity label.
- Acquisition priorities name exact readiness deficits instead of chasing a raw photo total.

## Optimization ledger

- Round 1: separated discovery permission from image download/training rights.
- Round 2: replaced immediate angle automation with reviewed bootstrap and an unknown class.
- Round 3: added host/redirect/IP controls, durable cursors and replay invariants.
- Round 4: replaced volume goals with readiness-deficit priorities and saturation stops.
- Round 5: clean pass; no substantial gap remained across access, safety, lineage, angles, scale or proof.
- Round 6: second consecutive clean pass; no substantial improvement or simplification remained.
