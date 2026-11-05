# Evidence-Gated Dataset Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace source-as-truth prototype labeling with a lawful, evidence-gated acquisition pipeline that produces a leakage-safe, auditable Rolex Submariner training dataset and a repeatable admission gate for later watch references.

**Architecture:** Approved APIs, partner exports and owner-authorized uploads create immutable candidate receipts; a default-deny rights gate decides whether bytes may be downloaded and used for training. Evidence review converts only confirmed candidates into training rows, then entity grouping, deduplication and coverage gates freeze content-addressed splits. DINOv2 training consumes only those frozen A/B-evidence splits; source reputation alone never creates ground truth.

**Tech Stack:** Python 3.10, dataclasses/enums, `requests`, Pillow, ImageHash, IBM Cloud Object Storage through `infra.ibm_cos`, JSON/JSONL ledgers, pytest + Hypothesis.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`

## Global Constraints

- MVP scope is Rolex Submariner; another reference cannot enter training until Task 8's admission gate passes.
- No source name alone may yield `GENUINE` or `FAKE`; it may only create a candidate and proposed label.
- Tier A/B evidence is training-eligible; Tier C and inconclusive records remain discovery/review data and never enter train, validation, test or gold.
- A valid, unexpired rights record is required before image download and again before split freeze; missing, blocked or revoked rights fail closed.
- Reddit remains excluded without written Reddit and rightsholder permission; Mercado Livre uses its official API only; no adapter bypasses a source policy.
- Original marketplace claims from OLX/Mercado Livre are never accepted as genuine evidence.
- Raw images, evidence documents, serial numbers and contributor identity remain private; public artifacts contain weights, model card and aggregate statistics only.
- Each physical watch, content-repost cluster, seller/account and acquisition batch stays in one split family; uncertainty means group together, never split apart.
- Price is candidate metadata, not sufficient evidence of fake grade; named factory is a proposed grade until evidence review accepts it.
- IBM COS objects are content-addressed and write-once; manifests/splits record hashes, not mutable object names.
- Training launch remains blocked until IBM credit/expiry/SKU/cost controls in `04b-infra.md` pass independently.

## Review Focus

1. A high-reputation dealer lists an unauthenticated watch: store as Tier C candidate, never genuine training data.
2. One photo is syndicated by several sellers: one content cluster and one split family, never apparent independent watches.
3. A rights grant expires or is revoked after collection: exclude affected assets from the next freeze and emit an impacted-artifact report.
4. A confirmed studio-shot replica is needed for cross-source testing: retain it in evaluation instead of deleting all white-background images.
5. A new watch reference has many photos from one seller but little entity/source diversity: fail admission despite raw photo count.

---

### Task 1: Define acquisition, evidence and rights contracts

**Files:**
- Modify: `core/types.py`
- Create: `acquisition/types.py`
- Modify: `core/__init__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_acquisition_types.py`

**Interfaces:**
- Consumes: existing `Label`, `Grade`, `ManifestRecord`.
- Consumes: `infra.ibm_types.ObjectRef` from `04b-infra.md` Task 1.
- Produces: `EvidenceTier`, `RightsStatus`, `ReviewStatus`, `Angle`, `SourcePolicy`, `RightsRecord`, `CandidateRecord`, `ReviewDecision`, `EvidenceRecord`, `TrainingRecord`, `BatchReceipt`, `DatasetFreeze`, `ImpactedArtifactReport`, `DataDispositionReport`, `FreezeResult`.
- Test factories: `candidate`, `rights`, `expired_rights`, `object_ref`, `valid_contract_bundle`; each returns the real frozen types, never dict substitutes.

- [ ] **Step 1: Write failing enum and immutability tests**

```python
def test_only_a_and_b_are_training_eligible():
    assert EvidenceTier.A.training_eligible
    assert EvidenceTier.B.training_eligible
    assert not EvidenceTier.C.training_eligible
    assert not EvidenceTier.INCONCLUSIVE.training_eligible

def test_candidate_is_immutable(candidate):
    with pytest.raises(FrozenInstanceError):
        candidate.source_url = "https://changed.invalid"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_acquisition_types.py -v`

Expected: FAIL because the acquisition contracts do not exist.

- [ ] **Step 3: Implement exact value objects**

```python
class EvidenceTier(Enum):
    A = "a"
    B = "b"
    C = "c"
    INCONCLUSIVE = "inconclusive"

    @property
    def training_eligible(self) -> bool:
        return self in {EvidenceTier.A, EvidenceTier.B}

@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    acquisition_batch_id: str
    source_policy_id: str
    source_record_id: str
    source_url: str
    collected_at: datetime
    declared_reference: str
    proposed_label: Label | None
    proposed_grade: int | None
    factory: str | None
    price_usd: Decimal | None
    raw_sha256: str
    phash: str
    rights_record_id: str

@dataclass(frozen=True)
class TrainingRecord:
    asset_id: str
    candidate_id: str
    object_ref: ObjectRef
    reference_id: str
    watch_entity_id: str
    content_cluster_id: str
    seller_group_id: str
    acquisition_batch_id: str
    angle: Angle
    source_domain: str
    capture_style: str
    label: Label
    grade: int | None
    factory: str | None
    evidence_tier: EvidenceTier
    evidence_record_id: str
    rights_record_id: str
    image_sha256: str
    phash: str
```

`ReviewDecision` stores candidate/reviewer IDs, proposed label/grade/tier, evidence type, supporting hashes, review timestamp and decision hash. `EvidenceRecord` stores `candidate_id`, accepted label/grade/reference, evidence tier, evidence type, reviewer IDs, review timestamps, supporting-artifact hashes and append-only decision hash. `RightsRecord` stores source, acquisition method, allowed purposes, effective/expiry timestamps, required disposition (`retain`, `quarantine`, `delete`), disposition deadline, rightsholder, evidence hash and status. `BatchReceipt` stores batch/policy/rights hashes plus accepted/refused candidate IDs. `DatasetFreeze` stores manifest hash, split hash, policy-set hash, eligible asset IDs, excluded asset IDs and reason counts; `FreezeResult` pairs it with optional impacted-artifact and data-disposition reports. Add only Task 1's listed pytest factories in `tests/conftest.py` using deterministic IDs, UTC timestamps and syntactically valid SHA-256 values. Each later task adds its own factories after the corresponding production type exists.

- [ ] **Step 4: Add validation tests**

```python
@pytest.mark.parametrize(
    "mutation",
    ["empty_id", "bad_sha256", "naive_timestamp", "genuine_with_grade",
     "fake_without_grade", "candidate_evidence_mismatch", "tier_c_training"],
)
def test_contract_rejects_invalid_state(valid_contract_bundle, mutation):
    with pytest.raises(ValueError):
        valid_contract_bundle.mutated(mutation).validate()
```

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_acquisition_types.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add core/types.py core/__init__.py acquisition/types.py tests/conftest.py tests/test_acquisition_types.py
git commit -m "feat: define dataset evidence contracts"
```

### Task 2: Enforce source access and training rights before download

**Files:**
- Create: `acquisition/source_policy.py`
- Create: `config/source_policies.json`
- Modify: `core/rules.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_source_policy.py`

**Interfaces:**
- Consumes: `SourcePolicy`, `RightsRecord`, current UTC time.
- Produces: `load_source_policies(path: Path) -> dict[str, SourcePolicy]`, `get_source_policy(policies: Mapping[str, SourcePolicy], source: str) -> SourcePolicy`, `authorize_candidate(policy: SourcePolicy, rights: RightsRecord, *, now: datetime) -> AcquisitionDecision`.
- Test factory: `policy_case`.

- [ ] **Step 1: Write default-deny and expiry tests**

```python
def test_unknown_source_cannot_download_or_train(policy_set, rights):
    decision = authorize_candidate(get_source_policy(policy_set, "unknown"), rights, now=NOW)
    assert not decision.may_download
    assert not decision.may_train

def test_expired_grant_fails_closed(chrono24_policy, expired_rights):
    decision = authorize_candidate(chrono24_policy, expired_rights, now=NOW)
    assert decision.reasons == ("rights expired",)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_source_policy.py -v`

Expected: FAIL because the policy loader and decision do not exist.

- [ ] **Step 3: Create the versioned policy registry**

Each JSON entry contains `policy_id`, `source`, `allowed_methods`, `may_download`, `may_train`, `required_rights_purposes`, `requires_rightsholder_permission` and `policy_evidence_sha256`. `get_source_policy` returns an immutable deny-all policy for any absent key. Seed explicit entries for partner upload, owner-authorized upload, Mercado Livre official API, Reddit blocked, and all existing web sources as candidate-only until a concrete rights record authorizes training. Do not encode `Chrono24` or `Watchuseek` as ground-truth labels.

- [ ] **Step 4: Replace source-as-label with candidate proposal**

Add `propose_source_label(...) -> Label | None`. It may propose `FAKE` for an explicitly replica-declared listing and must return `None` for original claims, generic owner posts and unknown sources. Keep the existing `source_label_rule` unchanged only as a temporary compatibility symbol so this commit remains green; no new acquisition module may import it. Task 3 migrates every existing caller/test and deletes it atomically. Only Task 5 evidence adjudication produces the accepted label.

- [ ] **Step 5: Test current legal boundaries**

```python
@pytest.mark.parametrize(
    ("case", "allowed"),
    [("reddit_missing_rightsholder", False), ("mercadolivre_scrape", False),
     ("olx_original_claim", False), ("partner_export_with_grant", True),
     ("missing_policy_hash", False), ("purpose_mismatch", False),
     ("revoked", False), ("expires_exactly_now", False)],
)
def test_source_policy_matrix(policy_case, case, allowed):
    decision = policy_case(case)
    assert decision.may_download is allowed
    assert decision.may_train is allowed
```

- [ ] **Step 6: Run focused tests and commit**

Run: `pytest tests/test_source_policy.py tests/test_core_rules.py -v`

Expected: PASS after callers/tests use `propose_source_label`.

```bash
git add acquisition/source_policy.py config/source_policies.json core/rules.py tests/conftest.py tests/test_source_policy.py tests/test_core_rules.py
git commit -m "feat: gate dataset sources by rights"
```

### Task 3: Collect approved batches with immutable receipts

**Files:**
- Create: `acquisition/collect.py`
- Create: `acquisition/partner_import.py`
- Modify: `scrapers/{olx,mercadolivre,chrono24,reptime,watchuseek,reddit,chinese_replica}.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_acquisition_collect.py`

**Interfaces:**
- Consumes: `AcquisitionDecision`, pre-fetched API/feed/export records, `infra.ibm_cos.upload_verified`.
- Produces: `collect_batch(records, policy, rights, *, http_client, cos_client, bucket, batch_id, now) -> BatchReceipt`, `import_partner_export(path: Path, declaration: RightsRecord) -> list[CandidateRecord]`.
- Test factory: `collector_fixture`.

- [ ] **Step 1: Write refusal-before-network tests**

```python
def test_blocked_policy_makes_no_http_or_cos_call(blocked_policy, rights, http_client, cos):
    with pytest.raises(AcquisitionRefused):
        collect_batch([], blocked_policy, rights, http_client=http_client, cos_client=cos, bucket="private", batch_id="b1", now=NOW)
    http_client.get.assert_not_called()
    cos.put_object.assert_not_called()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_acquisition_collect.py -v`

Expected: FAIL because batch collection does not exist.

- [ ] **Step 3: Implement bounded, idempotent collection**

For each authorized record, stream with 10-second connect/30-second read timeouts, require `image/*`, enforce a 25 MiB maximum, hash bytes while streaming and reject HTML/error bodies. Verify magic bytes and decode with Pillow under a 100-megapixel limit; accept only single-frame JPEG/PNG/WebP, reject SVG, polyglots, truncated/multi-frame files and every decompression-bomb warning/error, then reload and normalize the derivative to RGB. Store the raw object under `datasets/raw/{sha256}` and append a receipt under `datasets/receipts/{batch_id}/{candidate_id}.json`. A rerun reuses an object only when remote length/hash metadata match. Retries are bounded to three for 429/5xx and never change a refused policy into an allowed one.

- [ ] **Step 4: Sanitize training derivatives**

Strip EXIF/GPS, contributor names and serial-number metadata from the derivative; retain the unmodified raw object and supporting documents in separate restricted prefixes. The training manifest references only the sanitized derivative hash. Tests prove a GPS fixture disappears and the pixel content remains readable.

- [ ] **Step 5: Convert source modules into adapters, not label authorities**

Each adapter returns normalized candidate metadata from an approved API/feed/export. Migrate every `source_label_rule` caller/test to candidate proposal/policy flow, then delete the legacy function in the same commit. Remove placeholder/example endpoints, direct manifest writes and source-to-final-label behavior. If no approved endpoint or export exists, the adapter returns a typed `SourceUnavailable` result rather than inventing data. `rg -n "source_label_rule" --glob '*.py'` must return no matches before the Task 3 commit.

- [ ] **Step 6: Test interruption and resume**

```python
@pytest.mark.parametrize(
    "payload_case",
    ["partial", "hash_mismatch", "oversized", "non_image", "mime_magic_mismatch",
     "truncated", "polyglot", "multi_frame", "decompression_bomb"],
)
def test_unsafe_or_incomplete_payload_never_reaches_cos(payload_case, collector_fixture, cos):
    with pytest.raises((IntegrityError, UnsafeImage)):
        collector_fixture.collect(payload_case)
    cos.put_object.assert_not_called()

def test_repeated_batch_reuses_only_verified_object(collector_fixture, cos):
    first = collector_fixture.collect("valid")
    second = collector_fixture.collect("valid")
    assert second == first
    assert cos.put_object.call_count == 1
```

Partner imports accept a JSONL manifest plus files under one declared root; a parameterized traversal test rejects absolute paths, `..`, symlinks and undeclared files rather than extracting arbitrary archives. Separate tests assert 429 exhaustion, missing rights and receipt collision fail without a final object.

- [ ] **Step 7: Run focused tests and commit**

Run: `pytest tests/test_acquisition_collect.py tests/test_scrapers.py -v`

Expected: PASS.

```bash
git add acquisition/collect.py acquisition/partner_import.py scrapers tests/conftest.py tests/test_acquisition_collect.py tests/test_scrapers.py
git commit -m "feat: collect authorized image batches"
```

### Task 4: Resolve reposts and physical-watch groups conservatively

**Files:**
- Create: `dataset/entities.py`
- Modify: `dataset/dedup.py`
- Modify: `dataset/crop.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_entities.py`
- Test: `tests/test_dedup.py`

**Interfaces:**
- Consumes: `CandidateRecord`, sanitized image, listing/account/batch evidence.
- Produces: `GroupingDecision`, `build_content_clusters(records) -> dict[str, str]`, `assign_watch_entities(records, content_clusters) -> dict[str, str]`, `build_grouping_decision(candidate, clusters, entities) -> GroupingDecision`, `classify_capture_style(image, metadata) -> str`.
- Test factories: `grouping`, `grouped_records`.

- [ ] **Step 1: Write leakage-oriented grouping tests**

```python
def test_syndicated_listing_photos_share_content_cluster(records):
    clusters = build_content_clusters(records)
    assert clusters[records[0].candidate_id] == clusters[records[1].candidate_id]

def test_uncertain_same_watch_is_grouped_not_split(records):
    entities = assign_watch_entities(records, build_content_clusters(records))
    assert entities["front"] == entities["clasp"]
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_entities.py tests/test_dedup.py -v`

Expected: FAIL because entity grouping does not exist.

- [ ] **Step 3: Build two-stage grouping**

First group exact SHA-256 in a dictionary. For 64-bit pHash, split each hash into `HAMMING_DEDUP_THRESHOLD + 1` indexed blocks; by the pigeonhole rule, hashes within the threshold share at least one block. Compare Hamming distance only inside the resulting candidate sets, then build content clusters. A property test compares indexed results with brute force on small random fixtures, and a 10,000-hash test asserts distance-call count stays below 5% of all pairs. Then group photos into physical-watch entities using explicit listing ID, seller/account, batch, declared serial hash and reviewer link; visual similarity alone may merge content copies but cannot assert a physical identity. Ambiguous records receive the same conservative split-family ID when leakage is plausible.

If one exact/near-duplicate content cluster receives conflicting proposed or accepted labels/grades, quarantine the whole cluster before readiness and emit the candidate/evidence IDs for adjudication. Neither majority vote nor source reputation resolves a collision automatically.

- [ ] **Step 4: Fix studio-photo handling**

Change `studio_photo_filter` from unconditional deletion to `classify_capture_style`. Exclude repeated catalog images with uncertain provenance; retain rights-cleared, confirmed studio replicas for the cross-source evaluation pool. Record style independently of source domain.

- [ ] **Step 5: Validate crops without destroying evidence**

Keep the sanitized full image content-addressed. Store each angle crop as a derivative with parent hash, crop coordinates and algorithm version. Reject a crop outside image bounds or missing the expected watch region.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_entities.py tests/test_dedup.py tests/test_crop.py -v`

Expected: PASS.

```bash
git add dataset/entities.py dataset/dedup.py dataset/crop.py tests/conftest.py tests/test_entities.py tests/test_dedup.py tests/test_crop.py
git commit -m "feat: group watch entities without leakage"
```

### Task 5: Review and adjudicate labels from evidence

**Files:**
- Create: `labeling/review.py`
- Create: `labeling/adjudicate.py`
- Create: `config/reviewer_registry.json`
- Modify: `labeling/manifest.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_label_review.py`

**Interfaces:**
- Consumes: candidate, rights decision, supporting artifact hashes and append-only reviewer decisions.
- Produces: `ReviewerRecord`, `load_reviewer_registry(path: Path) -> dict[str, ReviewerRecord]`, `submit_review(candidate: CandidateRecord, reviewer: ReviewerRecord, *, proposed_label: Label | None, proposed_grade: int | None, proposed_tier: EvidenceTier, evidence_type: str, supporting_artifact_hashes: tuple[str, ...], reviewed_at: datetime) -> ReviewDecision`, `adjudicate(candidate: CandidateRecord, reviews: Sequence[ReviewDecision]) -> EvidenceRecord`, `build_training_record(candidate: CandidateRecord, evidence: EvidenceRecord, grouping: GroupingDecision, rights: RightsRecord, *, object_ref: ObjectRef, angle: Angle, capture_style: str) -> TrainingRecord`.
- Test factories: `make_review`, `make_decision`, `adjudication_fixture`, `review_store`.

- [ ] **Step 1: Write evidence-tier tests**

```python
def test_dealer_reputation_alone_is_tier_c_and_not_trainable(candidate, make_review, grouping, rights, object_ref):
    evidence = adjudicate(candidate, [make_review(evidence_type="source_reputation")])
    assert evidence.tier is EvidenceTier.C
    with pytest.raises(NotTrainingEligible):
        build_training_record(
            candidate, evidence, grouping, rights,
            object_ref=object_ref, angle=Angle.DIAL, capture_style="casual",
        )

def test_conflicting_reviews_become_inconclusive(candidate, make_review):
    evidence = adjudicate(candidate, [make_review(label=Label.GENUINE), make_review(label=Label.FAKE)])
    assert evidence.tier is EvidenceTier.INCONCLUSIVE
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_label_review.py -v`

Expected: FAIL because evidence review does not exist.

- [ ] **Step 3: Implement the evidence matrix**

Tier A requires controlled physical examination, service/authentication record, or equivalent evidence with artifact hash. Tier B requires an explicit replica transaction/factory record or a verified partner's authenticated-original declaration tied to the specific watch. Tier C covers source reputation, seller text, price and uncorroborated owner claims. Inconclusive covers conflict, missing watch identity or unsupported reference.

- [ ] **Step 4: Implement review rules**

One qualified review can accept Tier B. Tier A and every gold-set record require two independent reviewers; disagreement requires a third adjudicator. The versioned reviewer registry records pseudonymous ID, allowed evidence types, qualification-artifact hash, effective/expiry timestamps and prohibited contributor/source groups. Unknown, expired, conflicted or out-of-scope reviewers fail closed. A reviewer cannot approve their own contributed batch. Any paid reviewer engagement remains separately cost-authorized before work begins.

Persist each decision as an immutable COS object at `datasets/reviews/{candidate_id}/{reviewer_id}/{decision_hash}.json`; never append to or overwrite one shared file. Repeating the same hash is idempotent. Two different hashes from the same reviewer/candidate quarantine that reviewer decision until adjudication. Dataset freeze builds a sorted canonical projection from the immutable objects, so concurrent reviews cannot lose data and require no database.

- [ ] **Step 5: Stop price-only grading**

Factory/order evidence may support grades 3–4; inspection evidence may support any grade. Price-only grade guesses remain candidate metadata and cannot populate `TrainingRecord.grade`. Unknown grade routes to review instead of defaulting.

- [ ] **Step 6: Test tampering and append-only behavior**

```python
@pytest.mark.parametrize(
    "case",
    ["changed_evidence_hash", "self_approval", "duplicate_decision", "revoked_rights",
     "genuine_with_grade", "fake_without_grade"],
)
def test_adjudication_refuses_invalid_or_conflicted_state(case, adjudication_fixture):
    with pytest.raises((EvidenceTampered, ReviewerConflict, ValueError)):
        adjudication_fixture.run(case)

def test_disagreement_requires_third_adjudicator(candidate, disagreeing_reviews):
    assert adjudicate(candidate, disagreeing_reviews).tier is EvidenceTier.INCONCLUSIVE

def test_two_decisions_from_same_reviewer_are_preserved_and_quarantined(review_store, make_decision):
    review_store.put(make_decision(hash="a" * 64))
    review_store.put(make_decision(hash="b" * 64))
    assert len(review_store.list()) == 2
    assert review_store.project().status is ReviewStatus.QUARANTINED
```

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/test_label_review.py tests/test_manifest.py -v`

Expected: PASS.

```bash
git add labeling/review.py labeling/adjudicate.py labeling/manifest.py config/reviewer_registry.json tests/conftest.py tests/test_label_review.py tests/test_manifest.py
git commit -m "feat: adjudicate evidence-backed labels"
```

### Task 6: Gate dataset readiness and freeze leakage-safe splits

**Files:**
- Create: `dataset/readiness.py`
- Modify: `dataset/splits.py`
- Create: `dataset/freeze.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_dataset_readiness.py`
- Modify: `tests/test_splits.py`

**Interfaces:**
- Consumes: accepted `TrainingRecord`s, source policy set and previous freeze.
- Produces: `evaluate_readiness(records: Sequence[TrainingRecord], reference_id: str) -> ReadinessReport`, `build_splits(records: Sequence[TrainingRecord], *, seed: int, held_out_factories: Collection[str]) -> DatasetSplits`, `freeze_dataset(records: Sequence[TrainingRecord], splits: DatasetSplits, policies: Mapping[str, SourcePolicy], rights: Mapping[str, RightsRecord], *, previous_freeze: DatasetFreeze | None, published_artifact_hashes: Collection[str], now: datetime) -> FreezeResult`.
- Test factory: `make_balanced_records`.

- [ ] **Step 1: Write quantitative readiness tests**

```python
def test_many_photos_from_one_watch_fail_diversity_gate(records_one_entity):
    report = evaluate_readiness(records_one_entity, "rolex-submariner")
    assert not report.ready
    assert "distinct watch entities" in report.reasons

def test_one_source_over_forty_percent_fails(records_source_dominated):
    report = evaluate_readiness(records_source_dominated, "rolex-submariner")
    assert not report.ready
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_dataset_readiness.py tests/test_splits.py -v`

Expected: FAIL because readiness and entity-group splits do not exist.

- [ ] **Step 3: Implement explicit MVP gates**

For genuine and each fake grade at every required angle, require at least 500 distinct, pre-augmentation eligible assets and 100 distinct watch entities. Augmented derivatives, multiple crops of one asset and reposts never increase readiness counts. For rejection, require at least 500 owned/licensed negatives, 250 distinct subjects, both `no-watch` and `non-Submariner` subtypes, and at least five non-Submariner references; no angle matrix applies. Require at least three independent source domains per authenticity label, no source domain above 40% of train entities, no seller group above 10%, at least 200 Tier-A gold images spanning all six outputs, and non-empty confirmed studio/casual examples for genuine and fake. Report all deficits; never downsample a deficit into apparent readiness.

- [ ] **Step 4: Split by groups, not rows**

Build a graph whose edges join records sharing `watch_entity_id`, `content_cluster_id`, `seller_group_id` or `acquisition_batch_id`; each transitive connected component is one indivisible split family. Compute the temporal cutoff as the 80th percentile of component-level first-seen timestamps and persist it in the split artifact; callers cannot supply or move it. Reserve gold first, then unseen factories, then all components at/after that cutoff for temporal holdout, then cross-source, then group-stratified train/validation/test. Assert no component or constituent group ID crosses a split boundary.

- [ ] **Step 5: Freeze content-addressed artifacts**

Canonicalize and hash the accepted manifest, exclusions, policy set and split assignments. A revoked/expired rights record immediately makes affected objects unavailable to new jobs, produces a new candidate freeze plus `ImpactedArtifactReport`, and never mutates the prior freeze or silently republishes weights. `DataDispositionReport` lists raw/derivative/evidence object refs, required action and deadline. `quarantine` removes normal read access; `delete` requires the owner's destructive-action confirmation, then verified COS absence while retaining only a non-reversible hash tombstone. If confirmation is pending, state `BLOQUEADO — DISPOSIÇÃO DE DADOS NECESSÁRIA`; never treat exclusion from training as completed deletion.

- [ ] **Step 6: Add property and boundary tests**

```python
@pytest.mark.parametrize(
    ("images", "entities", "ready"),
    [(500, 100, True), (499, 100, False), (500, 99, False)],
)
def test_readiness_boundaries(make_balanced_records, images, entities, ready):
    report = evaluate_readiness(make_balanced_records(images, entities), "rolex-submariner")
    assert report.ready is ready

def test_all_group_relations_are_split_disjoint(grouped_records):
    splits = build_splits(grouped_records, seed=42, held_out_factories={"vsf"})
    assert_no_connected_component_crosses_splits(splits)
```

Additional named tests cover exactly 40%/40%+1 source share, deterministic seed, empty class-angle cell, rights expiry at freeze time, revocation impacting a published checkpoint, quarantine denying reads, and deletion remaining pending until COS absence is observed.

- [ ] **Step 7: Run tests and commit**

Run: `pytest tests/test_dataset_readiness.py tests/test_splits.py -v`

Expected: PASS.

```bash
git add dataset/readiness.py dataset/splits.py dataset/freeze.py tests/conftest.py tests/test_dataset_readiness.py tests/test_splits.py
git commit -m "feat: freeze evidence-ready dataset splits"
```

### Task 7: Train in waves and prove the model learned the watch

**Files:**
- Modify: `training/data.py`
- Modify: `training/run.py`
- Modify: `evaluation/evaluate.py`
- Modify: `evaluation/release_gate.py`
- Modify: `tests/conftest.py`
- Test: `tests/test_training_data.py`
- Modify: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: one `DatasetFreeze`, verified COS objects and existing DINOv2 per-angle heads.
- Produces: `evaluate_source_probe(embeddings, *, permutations: int, seed: int) -> SourceProbeReport`, `collection_decision(waves: Sequence[CollectionWave]) -> CollectionDecision`, learning-curve report, per-slice evaluation and release decision.
- Test factories: `source_biased_embeddings`, `wave`.

- [ ] **Step 1: Write freeze and eligibility refusal tests**

```python
def test_training_refuses_unfrozen_or_tier_c_rows(loader, tier_c_freeze):
    with pytest.raises(TrainingDataRefused):
        loader.load(tier_c_freeze)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_training_data.py tests/test_evaluate.py -v`

Expected: FAIL until the loader enforces the acquisition contracts.

- [ ] **Step 3: Train incremental waves**

Train identical seeded probes on 25%, 50%, 75% and 100% of eligible watch entities, balanced by class/angle/entity rather than raw photo count. Record macro balanced accuracy, worst required-slice accuracy, calibration error and cross-source delta for each wave. After readiness, a collection wave must add at least 20% new physical-watch entities. Stop collecting more of the same distribution after two consecutive such waves each improve macro balanced accuracy by less than 1 percentage point and the worst required slice by less than 2 points. Continue only for an explicit readiness deficit or a named weak class/angle/source/factory slice; never chase a raw target of millions of files.

- [ ] **Step 4: Add shortcut diagnostics**

Evaluate standard, cross-source, unseen-factory, temporal and gold sets. Train a diagnostic source-domain probe on frozen embeddings and compare it with 1,000 shuffled-source permutations using the same folds. Block release when the real probe's lower 95% bootstrap confidence bound exceeds the shuffled probe's upper 95% bound by more than 0.10; return the source slices that drove the result. Grad-CAM remains a required per-prediction artifact and manual bias audit, not a numerical gate without foreground-mask ground truth.

```python
def test_source_signal_above_permutation_gate_blocks_release(source_biased_embeddings):
    report = evaluate_source_probe(source_biased_embeddings, permutations=1_000, seed=42)
    assert report.real_ci_low > report.shuffled_ci_high + 0.10
    assert not release_gate(report).allowed

def test_two_saturated_collection_waves_stop_same_distribution():
    decision = collection_decision([
        wave(new_entity_gain=0.20, macro_gain_pp=0.8, worst_slice_gain_pp=1.5),
        wave(new_entity_gain=0.21, macro_gain_pp=0.7, worst_slice_gain_pp=1.2),
    ])
    assert decision.stop_same_distribution
```

- [ ] **Step 5: Preserve honest uncertainty**

Report per-class/angle/source/entity metrics and confidence intervals. Keep the existing grade-4 ceiling, rejection class and calibration gates. Do not convert a weak slice into an aggregate accuracy claim.

- [ ] **Step 6: Run tests and commit**

Run: `pytest tests/test_training_data.py tests/test_evaluate.py tests/test_release_gate.py -v`

Expected: PASS.

```bash
git add training/data.py training/run.py evaluation/evaluate.py evaluation/release_gate.py tests/conftest.py tests/test_training_data.py tests/test_evaluate.py tests/test_release_gate.py
git commit -m "feat: gate training on dataset evidence"
```

### Task 8: Admit later references without weakening the MVP

**Files:**
- Create: `dataset/reference_catalog.py`
- Create: `config/reference_catalog.json`
- Modify: `tests/conftest.py`
- Test: `tests/test_reference_catalog.py`

**Interfaces:**
- Consumes: per-reference `ReadinessReport`, reference identity evidence and trained Submariner baseline.
- Produces: `admit_reference(reference_id, report, catalog) -> ReferenceAdmission`, immutable catalog entry and separate per-reference head configuration.
- Test factory: `catalog_fixture`.

- [ ] **Step 1: Write admission refusal tests**

```python
def test_photo_volume_cannot_bypass_reference_readiness(catalog, dominated_report):
    decision = admit_reference("rolex-datejust-126334", dominated_report, catalog)
    assert not decision.allowed

def test_submariner_rows_cannot_train_an_unadmitted_head(catalog):
    with pytest.raises(ReferenceNotAdmitted):
        catalog.head_for("rolex-datejust-126334")
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_reference_catalog.py -v`

Expected: FAIL because reference admission does not exist.

- [ ] **Step 3: Implement the admission contract**

A reference needs normalized brand/family/reference identity, required guided angles, a passing Task 6 readiness report and its own gold/holdout slices. Reuse the frozen DINOv2 backbone; create a separate per-reference head and calibration artifact. Do not build a universal authenticity head or advertise unsupported references.

- [ ] **Step 4: Test catalog versioning and compatibility**

```python
@pytest.mark.parametrize(
    "case",
    ["alias_collision", "generation_mismatch", "changed_angles", "revoked_admission",
     "wrong_reference_checkpoint", "catalog_hash_mismatch"],
)
def test_reference_catalog_refuses_incompatible_state(case, catalog_fixture):
    with pytest.raises(ReferenceCatalogError):
        catalog_fixture.load(case)
```

- [ ] **Step 5: Run tests and commit**

Run: `pytest tests/test_reference_catalog.py -v`

Expected: PASS.

```bash
git add dataset/reference_catalog.py config/reference_catalog.json tests/conftest.py tests/test_reference_catalog.py
git commit -m "feat: gate additional watch references"
```

## Operational Rollout

1. Run the full pipeline first on synthetic/local fixtures; no network or IBM resource is required.
2. Run one owner-authorized partner batch of 20–30 images through rights, receipt, review, grouping and freeze-preview; this proves mechanics but cannot satisfy readiness.
3. Enable one approved source at a time and inspect exclusion/review/source-dominance reports before the next source.
4. Freeze the first trainable dataset only when every Task 6 gate passes; otherwise state the exact deficit and keep training blocked.
5. Provision IBM GPU only after `04b-infra.md` cost/watchdog gates and the frozen dataset hashes pass; real acquisition/training E2E remains `PENDENTE E2E` until externally observed.

## Definition of Done

- The source name is never the sole producer of an accepted authenticity label.
- Every training asset has current rights, Tier A/B evidence, immutable hashes and one physical-watch/content/seller/batch split family.
- Quantitative class/angle/entity/source gates pass for the Submariner dataset and are persisted in a content-addressed freeze.
- Standard, cross-source, unseen-factory, temporal and gold evaluations consume the same frozen hashes.
- A later reference cannot obtain a head or coverage claim without its own passing admission report.
- Raw data and evidence documents remain private; only explicitly authorized artifacts can be published.

## Requirements Traceability

| Requirement | Owning task | Authoritative test/proof |
|---|---|---|
| CK-10/31 | Tasks 1, 2 and 5 | `test_acquisition_types`, `test_source_policy`, `test_label_review`: complete rights/evidence/entity lineage is required to build `TrainingRecord` |
| CK-11/33 | Task 4 and Task 6 | `test_entities`, `test_dedup`, `test_splits`: repost/watch/seller/batch families cannot cross splits; confirmed studio images are retained by style |
| CK-13/32 | Tasks 2 and 5 | `test_source_policy`, `test_label_review`: original claims/source reputation/price create no accepted label or grade |
| CK-27/28 | Tasks 2 and 3 | `test_source_policy`, `test_acquisition_collect`: Reddit permission and Mercado Livre official-API gates refuse before network |
| CK-34 | Task 6 | `test_dataset_readiness`: 500/100 class-angle boundaries, rejection quotas and source/seller dominance gates |
| CK-35 | Task 6 | `test_dataset_readiness`: revocation creates a new freeze and impacted-artifact report without mutating history |
| CK-36 | Task 8 | `test_reference_catalog`: no head or coverage claim before per-reference admission |

## Optimization Ledger

- **Round 1 — substantive findings applied:** source reputation was incorrectly treated as ground truth; added A/B/C/inconclusive evidence, rights and candidate-vs-training contracts.
- **Round 2 — substantive findings applied:** adapters assumed JSON endpoints and wrote labels directly without real download, immutable receipt or resume semantics; added approved-adapter collection, content-addressed COS storage and refusal-before-network behavior.
- **Round 3 — substantive findings applied:** pHash dedup did not prevent the same physical watch, seller or batch crossing splits, while the white-background filter deleted the studio replicas needed for cross-source evaluation; added conservative entity grouping and style classification.
- **Round 4 — substantive findings applied:** raw photo counts could hide one-source/one-watch dominance and the plan offered no repeatable expansion rule; added quantitative readiness, learning-curve waves and per-reference admission.
- **Round 5 — substantive findings applied:** the main spec still preserved source-as-truth, unconditional studio deletion, row-level splits and three-view evaluation in architecture/rubric/test sections. Replaced those contracts with rights/evidence lineage, style-aware retention, grouped splits and five-view evaluation.
- **Round 6 — substantive findings applied:** unknown-source fallback, `ObjectRef` ownership, test-helper names, rejection readiness and shortcut-diagnostic thresholds were implicit or inconsistent. Added exact producers, deny-all lookup, consistent signatures, a rejection-specific quota and a permutation/CI gate; removed an ungrounded Grad-CAM numeric threshold.
- **Round 7 — substantive finding applied:** the review interface returned the final `EvidenceRecord` for both an individual decision and adjudication, and a training test called an undefined freeze helper. Added the distinct `ReviewDecision` contract and converted the test input to an explicit fixture.
- **Round 8 — substantive finding applied:** the new CK-31–36 behaviors were present but lacked an explicit requirement-to-task-to-test map, and Task 5 consumed an unnamed grouping value. Added traceability and the `GroupingDecision` producer/constructor.
- **Round 9 — substantive findings applied:** pHash dedup still implied quadratic comparison at acquisition scale, split families ignored transitive relations, and reviewer qualification had no authoritative producer. Added exact multi-index pHash candidate generation with a complexity test, connected-component split families and a versioned fail-closed reviewer registry.
- **Round 10 — substantive findings applied:** readiness could be inflated by augmented/cropped copies, and a caller-controlled temporal cutoff could cherry-pick evaluation. Restricted quotas to distinct pre-augmentation assets and made the 80th-percentile component cutoff deterministic and freeze-bound.
- **Round 11 — substantive finding applied:** the refusal-before-network test referenced an absent rights fixture and could not inject the HTTP client it asserted against. Added both to the public collection contract and test.
- **Round 12 — substantive finding applied:** `submit_review(...)` and `freeze_dataset(...)` were placeholder interfaces, leaving required inputs and revocation output ambiguous. Replaced them with complete typed signatures and named `FreezeResult`/`ImpactedArtifactReport` contracts.
- **Round 13 — substantive finding applied:** learning curves had no objective acquisition stop rule, so the plan could still justify endless same-distribution collection. Added 20%-new-entity waves and a two-wave saturation rule using macro and worst-slice gains.
- **Round 14 — substantive finding applied:** Task 2 removed `source_label_rule` before Task 3 migrated its callers, making the intermediate commit fail imports/tests. Converted it to a two-commit compatibility migration and required zero legacy references before deletion.
- **Round 15 — clean pass 1:** re-read all eight tasks as a fresh executor; verified task files/interfaces, RED/GREEN/commit boundaries, compatibility migration, rights/evidence producers, grouped split order, quantitative gates, IBM dependencies and CK-10/11/13/27/28/31–36 traceability. No substantial improvement or optimization remained.
- **Round 16 — substantive findings applied:** untrusted downloads relied on MIME/size without safe decoding, and duplicate images with conflicting labels had no quarantine path. Added bounded single-frame raster validation, path-safe partner imports and conflict-cluster quarantine without majority/source resolution.
- **Round 17 — substantive finding applied:** the main CK-27 still required only Reddit permission while the canonical plan requires Reddit plus rightsholder permission, and CK-13 conflated retained candidate metadata with training inclusion. Aligned both acceptance boundaries.
- **Round 18 — clean pass 1:** rechecked the main spec and subplan for source-as-truth remnants, rights/permission alignment, task/test traceability, quantitative thresholds, dedup scale, split connectivity, temporal determinism, acquisition saturation and IBM dependency boundaries. No substantial improvement or optimization remained.
- **Round 19 — substantive finding applied:** several steps named failure cases without concrete test bodies, violating `writing-plans`, and Task 7 lacked explicit tests for its source-probe and collection-saturation gates. Added parameterized test skeletons and exact assertions.
- **Round 20 — substantive finding applied:** the concrete tests referenced shared factories and two Task 7 functions without declared producers. Added real-type pytest factory ownership in Task 1 and exact source-probe/collection-decision signatures.
- **Round 21 — clean pass 1:** reviewed all eight tasks as an executor after distributing test factories by production ownership; every task has files, interfaces, concrete RED/GREEN tests, a green commit boundary and traceable acceptance coverage. No substantial improvement or optimization remained.
- **Round 22 — substantive finding applied:** rights revocation removed assets from future training but left raw/derivative/evidence retention and deletion proof undefined. Added fail-closed quarantine, deadline-bound disposition reports, owner-confirmed verified deletion and hash-only tombstones.
- **Round 23 — clean pass 1:** rechecked the complete spec/subplan for requirement coverage, concrete tests, contract ownership, rights and data disposition, adversarial inputs, label conflicts, dedup scale, leakage, temporal/source bias, acquisition saturation, reference admission and IBM cost boundaries. No substantial improvement or optimization remained.
- **Round 24 — substantive finding applied:** append-only review decisions lacked concurrent-write semantics and could be overwritten or duplicated. Replaced shared append with immutable per-candidate/reviewer/hash objects, deterministic freeze projection and conflict quarantine without adding a database.
- **Round 25 — clean pass 1:** revalidated all task contracts, concrete tests, immutable review concurrency, acquisition safety, evidence/rights gates, data disposition, quantitative readiness, grouped/time/source holdouts, saturation and expansion boundaries. No substantial improvement or optimization remained.
- **Round 26 — clean pass 2:** performed the final handoff review across canonical spec, historical-plan supersession notes, eight task boundaries, CK-31–36 acceptance mapping, external authorization gates and `PENDENTE E2E` states. No substantial improvement or optimization remained; convergence goal satisfied.
