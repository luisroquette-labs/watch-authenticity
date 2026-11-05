# IBM Cloud Training Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete AWS adapter with a reproducible IBM Cloud training plane that consumes the CF Gauss promotional credit without silently creating cash overage, and that can actually train, evaluate and preserve a Rolex Submariner model.

**Architecture:** A local controller authorizes one immutable run, IBM VPC executes one digest-pinned GPU container, IBM COS is the append-only object/receipt store, and a pre-deployed IBM Code Engine cron watchdog deletes expired workers. The VSI has no inbound path or floating IP: its trusted profile obtains short-lived credentials through Instance Metadata, and it reaches private IBM service endpoints plus package repositories through an outbound-only Public Gateway. Pure cost/lifecycle rules are SDK-free; IBM SDK adapters sit at the boundary.

**Tech Stack:** Python 3.10, PyTorch 2.5.1/CUDA 12.4, `ibm-vpc==0.36.1`, `ibm-cos-sdk==2.16.2`, `ibm-platform-services==0.77.2`, IBM VPC `gx3-16x80x1l4`, IBM COS, IBM Container Registry, IBM Code Engine cron, IAM trusted profiles, IBM Cloud Logs/Monitoring.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`

## Global Constraints

- Production profile is exactly `gx3-16x80x1l4`: 1 NVIDIA L4 24 GB, 16 vCPU and 80 GiB RAM.
- L4 host bootstrap must prove NVIDIA driver >=550 and CUDA 12.4 before the container starts.
- The applied promo code is not balance proof. Production launch requires console/API evidence of remaining credit, expiry and eligible IBM-native SKUs observed within 24 hours.
- A run must finish at least 48 hours before promo expiry.
- Default safety reserve is USD 500. Lowering it is a financial decision requiring the owner's explicit approval in that conversation.
- Spending notifications are evidence only; IBM sends 80/90/100% alerts and does not promise a hard stop.
- The estimate includes every metered resource used by the run: VSI, boot volume, Public Gateway, COS storage/operations, Container Registry storage, Code Engine, Logs/Monitoring and the USD 500 lag reserve. An unpriced line item blocks launch.
- No AWS, S3, external GPU, paid AI API or automatic paid fallback.
- No raw scraped image is published. COS buckets are private and deny public access.
- No credential is committed, copied into the VSI, baked into a container layer or written to logs/receipts.
- Every run is tied to code SHA, image digest, manifest SHA-256, split SHA-256 and unique run ID.
- Ambiguous create/delete responses are reconciled before retry. Never create a second VSI automatically.
- The watchdog must be deployed and pass a no-delete dry run before a GPU instance can be created.
- The VSI has no floating IP and its security group allows no inbound traffic. Outbound access is limited to DNS, HTTPS, IBM private/direct service endpoints and the package repositories required by the pinned bootstrap.
- Every metered resource has a lease owner and terminal policy. Boot volumes and the dedicated Public Gateway are deleted with each run; persistent COS/Registry/Logs resources require a verified export-and-delete or explicitly funded post-promo plan before the first run.
- The current repo has no `main`, initial commit or remote; commit/PR steps remain blocked until repository history is established without swallowing unrelated work.
- After repository history exists, each numbered task is one review/commit boundary. Do not combine tasks; the exact commit messages are listed under Execution Commit Boundaries.

## Review Focus

1. IBM accepts create but the response is lost: reconcile deterministic instance name before any retry.
2. Usage data lags behind real consumption: price every metered line item, retain USD 500 and prevent back-to-back runs until the prior receipt is reconciled.
3. Promo expires during a run: require `now + TTL + 48h <= expires_at`.
4. Dataset contains placeholder URLs, missing objects or no rejection examples: block before GPU provisioning.
5. Container or checkpoint changes under the same name: require OCI digest and SHA-256 verification, never mutable tags alone.

## File Structure

- Create `infra/ibm_types.py`: immutable credit, usage, pricing, run plan and receipt value objects.
- Create `infra/ibm_evidence.py`: validate runtime-only promo/price captures and normalize IBM Usage Reports.
- Create `infra/ibm_cost_guard.py`: pure authorization decision; no IBM SDK imports.
- Create `infra/ibm_cos.py`: COS upload/download/head verification and receipt writer.
- Create `infra/ibm_vpc.py`: thin `ibm-vpc` adapter and deterministic reconciliation.
- Create `infra/ibm_train.py`: lifecycle orchestrator with `finally` teardown.
- Create `infra/ibm_watchdog.py`: stale-instance reconciliation used by Code Engine.
- Create `infra/bootstrap_host.sh`: two-stage driver 550/CUDA 12.4 bootstrap, reboot and digest-pinned container launch.
- Create `Containerfile.train`: digest-pinned training runtime.
- Create `training/run.py`: real deterministic head-training entrypoint.
- Create `training/data.py`: manifest/COS loader, target mapping and class/split validation.
- Modify `tests/conftest.py`: valid IBM value-object and SDK-response factories shared by focused tests.
- Create `tests/test_ibm_{types,evidence,cost_guard,cos,vpc,train,watchdog}.py`.
- Create `tests/test_training_run.py`.
- Modify `core/types.py`: define the single six-class output vocabulary and a trainable rejection/OOD label.
- Modify `core/rules.py`: accept only explicitly owned/licensed negative sources for the rejection label.
- Modify `dataset/splits.py`: group-split rejection examples without leakage.
- Modify `requirements.txt`: replace AWS dependency with pinned IBM SDKs.
- Modify `.gitignore`: exclude `.runtime/ibm/evidence/`, recovery files and runtime receipts.
- Delete `infra/aws_train.py`, `infra/s3_sync.py` and superseded AWS tests only after IBM tests pass.

### Task 1: Pin IBM and six-class domain contracts

**Files:**
- Create: `infra/ibm_types.py`
- Modify: `core/types.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: `tests/test_ibm_types.py`

**Interfaces:**
- Produces: `CreditEvidence`, `UsageEvidence`, `PriceLine`, `PricingEvidence`, `ResourceLease`, `PostPromoPlan`, `RunSpec`, `AuthorizationDecision`, `AuthorizedRun`, `RunEvent`, `InstanceRef`, `ModelClass`.
- `CreditEvidence` records `observed_at, remaining_usd, expires_at, eligible_service_plans, evidence_sha256`.
- `UsageEvidence` records `observed_at, billing_period_start, observed_usage_usd, unreconciled_run_ids, evidence_sha256`.
- `PriceLine` records `service, plan, unit, estimated_quantity, unit_price_usd`; `PricingEvidence` records `observed_at, currency, run_spec_sha256, lines, evidence_sha256`.
- `RunSpec` contains `run_id, code_sha, image_digest, manifest_sha256, splits_sha256, profile, region, zone, max_hours, bucket, prefix, cos_service_instance_id, network_id, subnet_id, public_gateway_id, security_group_id, trusted_profile_id`.
- `AuthorizationDecision` records `allowed, reasons, estimated_usd, usable_credit_usd, evidence_hashes`; `AuthorizedRun` records the exact `RunSpec`, estimate and evidence hashes.
- `RunEvent` records `event_id, run_id, sequence, state, observed_at, instance_id, lineage_sha256, previous_event_sha256, payload_sha256`; `event_id` is a fresh UUID and state is one of `planned, authorized, creating, running, artifact_sync, succeeded, failed, deleting, deleted, quarantined`.
- `InstanceRef` records `id, name, profile, zone, vpc_id, subnet_id, image_id, status`; `ObjectRef` records `bucket, key, sha256, length, etag`.
- `ResourceLease` records `resource_crn, service, run_id, created_at, expires_at, terminal_policy`; `PostPromoPlan` records each persistent service, export destination hash, delete/downgrade deadline and owner authorization hash.

- [ ] **Step 1: Write failing tests for immutable identifiers and OOD target**

```python
def test_run_spec_rejects_mutable_image_tag():
    with pytest.raises(ValueError, match="sha256"):
        replace(valid_run_spec(), image_digest="icr.io/watch/train:latest")

def test_model_classes_include_trained_rejection():
    assert ModelClass.REJECTION.value == 5

def test_rejection_is_a_manifest_training_label():
    assert Label.REJECTION.value == "rejection"
```

- [ ] **Step 2: Run the tests and verify failure**

Run: `pytest tests/test_ibm_types.py -v`
Expected: FAIL because the types do not exist.

- [ ] **Step 3: Implement frozen dataclasses with constructor validation**

```python
class ModelClass(IntEnum):
    GENUINE = 0
    GRADE_1 = 1
    GRADE_2 = 2
    GRADE_3 = 3
    GRADE_4 = 4
    REJECTION = 5

class Label(Enum):
    GENUINE = "genuine"
    FAKE = "fake"
    REJECTION = "rejection"
    EXCLUDED = "excluded"

@dataclass(frozen=True)
class RunSpec:
    run_id: UUID
    code_sha: str
    image_digest: str
    manifest_sha256: str
    splits_sha256: str
    profile: str
    zone: str
    max_hours: Decimal
    bucket: str
    prefix: str
```

Validate 40-character Git SHA, 64-hex content hashes, `@sha256:` image reference, exact profile, IBM CRN/UUID identifiers, non-empty region/zone/bucket/prefix and `0 < max_hours <= 24`. Reject a shared/unowned Public Gateway because the run cannot safely delete it. `tests/conftest.py` defines `valid_run_spec()`, `valid_credit()`, `valid_usage()`, `valid_pricing()`, `valid_post_promo_plan()`, `vpc_instance()` and `safe_user_data()` with complete valid values; every mutation test uses `dataclasses.replace`. Define `ModelClass` once in `core/types.py`; `model/heads.py`, `model/rejection.py` and `training/data.py` import its indices rather than recomputing them.

- [ ] **Step 4: Pin dependencies**

```text
ibm-vpc==0.36.1
ibm-cos-sdk==2.16.2
ibm-platform-services==0.77.2
ibm-cloud-sdk-core==3.28.1
```

Add the IBM pins here; Task 9 exclusively owns removal of regular `boto3` after the IBM path passes, avoiding a broken intermediate commit. `ibm_boto3` is supplied by `ibm-cos-sdk`.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_ibm_types.py tests/test_core_types.py -v`
Expected: PASS.

### Task 2: Capture evidence and build the fail-closed promotional-credit gate

**Files:**
- Create: `infra/ibm_evidence.py`
- Create: `infra/ibm_cost_guard.py`
- Test: `tests/test_ibm_evidence.py`
- Test: `tests/test_ibm_cost_guard.py`

**Interfaces:**
- Produces: `load_credit_evidence(record_path: Path, console_capture_path: Path, *, now: datetime) -> CreditEvidence`, `load_pricing_evidence(record_path: Path, quote_capture_path: Path, spec: RunSpec, *, now: datetime) -> PricingEvidence`, `load_post_promo_plan(record_path: Path, authorization_artifact: Path, *, now: datetime) -> PostPromoPlan`, `fetch_usage_evidence(client: UsageReportsV4, account_id: str, billing_period: str, pending_run_ids: tuple[UUID, ...], *, now: datetime) -> UsageEvidence`.
- Consumes: `CreditEvidence, UsageEvidence, PricingEvidence, RunSpec, tuple[RunEvent, ...]`.
- Produces: `authorize_run(credit: CreditEvidence, usage: UsageEvidence, pricing: PricingEvidence, post_promo: PostPromoPlan, spec: RunSpec, events: tuple[RunEvent, ...], *, now: datetime) -> AuthorizationDecision`; it never provisions. `AuthorizedRun.from_decision(spec: RunSpec, decision: AuthorizationDecision, evidence_hashes: tuple[str, ...]) -> AuthorizedRun` refuses a denied decision and freezes the evidence hashes used by the launch.

- [ ] **Step 1: Write failing capture-integrity tests**

The promo, price and post-promo records are runtime JSON beside the owner-captured console/quote/authorization artifact. Reject a missing artifact, SHA-256 mismatch, future timestamp, non-USD value, unknown service plan, account mismatch or a financial capture older than 24 hours. A post-promo plan must cover exactly `{cos-storage, container-registry-storage, logs-monitoring}`, set every delete/downgrade/export deadline no later than `promo_expires_at - 48h`, and cannot itself authorize the later destructive deletion. `fetch_usage_evidence` uses an injected `UsageReportsV4` client and rejects pagination gaps, duplicate resources, non-USD currency and an incomplete billing period response.

- [ ] **Step 2: Implement the evidence loaders**

The loader parses with `json.loads`, requires an exact schema version, computes artifact hashes itself, and returns frozen types. It never OCRs a screenshot or infers a number: the owner-recorded balance/expiry/eligible plans must match the attached authoritative console capture and remain `PENDENTE E2E` until that capture exists. Usage collection uses the active IBM identity; no API key/token is read from the JSON or persisted.

- [ ] **Step 3: Run evidence tests**

Run: `pytest tests/test_ibm_evidence.py -v`
Expected: PASS.

- [ ] **Step 4: Write parameterized failing authorization tests**

```python
@pytest.mark.parametrize("age", [timedelta(hours=24, seconds=1), timedelta(days=2)])
def test_authorize_run_rejects_stale_credit(age):
    now = datetime(2026, 11, 5, tzinfo=UTC)
    credit = replace(valid_credit(now), observed_at=now - age)
    decision = authorize_run(
        credit,
        valid_usage(now),
        valid_pricing(now),
        valid_post_promo_plan(now),
        valid_run_spec(),
        (),
        now=now,
    )
    assert not decision.allowed
    assert "credit evidence is stale" in decision.reasons
```

Add one explicit `dataclasses.replace` test for stale pricing, ineligible service plan, expiry window, reserve breach, missing required service line, unreconciled prior run, non-USD currency and pricing bound to a different `run_spec_sha256`; do not hide those cases behind an undefined mutation helper.

- [ ] **Step 5: Verify RED**

Run: `pytest tests/test_ibm_cost_guard.py -v`
Expected: FAIL because `authorize_run` is absent.

- [ ] **Step 6: Implement the pure calculation**

```python
estimated = sum(line.estimated_quantity * line.unit_price_usd for line in pricing.lines)
usable = credit.remaining_usd - Decimal("500.00")
allowed = (
    credit.age(now) <= timedelta(hours=24)
    and pricing.age(now) <= timedelta(hours=24)
    and all(line.plan in credit.eligible_service_plans for line in pricing.lines)
    and not usage.unreconciled_run_ids
    and pricing.covers(REQUIRED_METERED_LINES)
    and post_promo.covers(PERSISTENT_METERED_SERVICES, credit.expires_at)
    and now + timedelta(seconds=int(spec.max_hours * 3600) + 48 * 3600) <= credit.expires_at
    and estimated <= usable
)
```

`REQUIRED_METERED_LINES` is the exact frozen set `{vpc-vsi, vpc-boot-volume, vpc-public-gateway, cos-storage, cos-operations, container-registry-storage, code-engine, logs-monitoring}`. A line advertised as free still has an explicit zero price; absence is not treated as free.

Return all refusal reasons together. Never raise merely for a financial refusal; the caller prints the decision without reaching IBM.

- [ ] **Step 7: Add boundary/property tests**

Cover exactly 24h vs 24h+1s, exactly USD 500 reserve vs one cent below, Decimal-only arithmetic, timezone-aware timestamps, negative/NaN values, duplicate line items, a missing free-tier price, usage newer than credit evidence, and a prior run whose provider usage has not reached a terminal reconciliation event.

- [ ] **Step 8: Run focused tests**

Run: `pytest tests/test_ibm_cost_guard.py -v`
Expected: PASS.

### Task 3: Make COS the content-addressed source of truth

**Files:**
- Create: `infra/ibm_cos.py`
- Test: `tests/test_ibm_cos.py`

**Interfaces:**
- Produces: `upload_verified(client: Any, path: Path, bucket: str, key: str, expected_sha256: str) -> ObjectRef`, `download_verified(client: Any, bucket: str, key: str, destination: Path, expected_sha256: str) -> Path`, `append_event(client: Any, bucket: str, prefix: str, event: RunEvent) -> ObjectRef`.

- [ ] **Step 1: Write failing tests for checksum and write-once behavior**

```python
def test_upload_rejects_head_checksum_mismatch(fake_cos, file):
    fake_cos.head_object.return_value = {"Metadata": {"sha256": "wrong"}}
    with pytest.raises(IntegrityError):
        upload_verified(fake_cos, file, "bucket", "objects/x", sha256(file))

def test_same_sequence_with_two_event_ids_is_a_fork(fake_cos, run_spec):
    first = run_event(event_id=UUID(int=1), sequence=3)
    second = replace(first, event_id=UUID(int=2), payload_sha256="b" * 64)
    append_event(fake_cos, "private-bucket", "runs/abc", first)
    append_event(fake_cos, "private-bucket", "runs/abc", second)
    with pytest.raises(EventChainFork):
        validate_event_chain((first, second), run_spec)
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ibm_cos.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement IBM COS client injection**

Create the production client with `auth = VPCInstanceAuthenticator(iam_profile_id=spec.trusted_profile_id)` and `ibm_boto3.client("s3", ibm_service_instance_id=spec.cos_service_instance_id, token_manager=auth.token_manager, config=Config(signature_version="oauth"))`; the installed `ibm-cos-sdk` accepts a token manager, not an `Authenticator` object. Tests assert that no API key/HMAC argument is supplied and inject a fake client after client construction is covered. Content-addressed objects use the digest in the final key and are idempotent only when the existing metadata hash and length match. IBM COS `PutObject` in SDK 2.16.2 has no `IfNoneMatch`; events therefore use collision-resistant immutable keys `events/{sequence:06d}-{event_id}.json`. Two objects with one sequence are preserved as a detectable fork and quarantine the run; no event key is reused or overwritten.

- [ ] **Step 4: Test interrupted multipart upload and corrupt download**

Assert temporary objects are removed, final key is absent on failure, a corrupt download is deleted locally, two event IDs at one sequence quarantine the run, and concurrent writers preserve both branches rather than replacing an event.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_ibm_cos.py -v`
Expected: PASS.

### Task 4: Implement deterministic VPC creation and ambiguity reconciliation

**Files:**
- Create: `infra/ibm_vpc.py`
- Test: `tests/test_ibm_vpc.py`

**Interfaces:**
- Produces: `build_instance_prototype(spec: RunSpec, user_data: str) -> dict[str, object]`, `ensure_single_instance(vpc: VpcV1, spec: RunSpec, prototype: dict[str, object]) -> InstanceRef`, `wait_running(vpc: VpcV1, instance_id: str, *, deadline: float) -> InstanceRef`, `delete_and_wait(vpc: VpcV1, instance_id: str, *, deadline: float) -> None`.

- [ ] **Step 1: Write failing ambiguity tests**

```python
def test_lost_create_response_reconciles_by_exact_name(fake_vpc, spec):
    fake_vpc.create_instance.side_effect = ReadTimeout()
    fake_vpc.list_instances.return_value = [vpc_instance(name=f"watch-auth-{spec.run_id}")]
    prototype = build_instance_prototype(spec, safe_user_data())
    assert ensure_single_instance(fake_vpc, spec, prototype).id == "vsi-1"
    assert fake_vpc.create_instance.call_count == 1

def test_multiple_name_matches_quarantine_without_delete(fake_vpc, spec):
    fake_vpc.list_instances.return_value = [vpc_instance(id="a"), vpc_instance(id="b")]
    with pytest.raises(AmbiguousProvisioning):
        ensure_single_instance(fake_vpc, spec, build_instance_prototype(spec, safe_user_data()))
    fake_vpc.delete_instance.assert_not_called()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ibm_vpc.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement exact instance prototype**

Use `VpcV1.create_instance` with deterministic name, dedicated resource group, VPC, zone, subnet, pinned Ubuntu image ID, profile `gx3-16x80x1l4`, encrypted boot volume with `delete_volume_on_instance_delete=True`, the exact trusted profile, Instance Metadata enabled, no floating IP, an inbound-deny security group and user-data. The subnet attaches one dedicated, leased Public Gateway. Persist a canonical SHA-256 of the intended prototype before create. On reconciliation, exact name alone is insufficient: resource group, profile, zone, VPC, subnet and image must all match inspectable provider fields; apply project/run tags immediately after adoption. Resolve IDs before authorization; do not accept IDs from dataset content.

- [ ] **Step 4: Implement bounded polling**

Poll explicit terminal states with monotonic deadlines and bounded exponential backoff with jitter. Unknown state, 429 exhaustion or timeout persists an ambiguous event and blocks automatic recreate. A name match with a different prototype is never adopted or deleted automatically.

- [ ] **Step 5: Test delete reconciliation**

404 after delete counts as deleted; timeout requires a later read; never report deletion until `get_instance` returns 404. After observed VSI deletion, verify its boot volume is 404, detach/delete the dedicated Public Gateway, observe its 404 and emit separate resource-deletion events.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_ibm_vpc.py -v`
Expected: PASS.

### Task 5: Build and prove the immutable GPU runtime

**Files:**
- Create: `Containerfile.train`
- Create: `infra/bootstrap_host.sh`
- Create: `requirements-train.lock`
- Create: `tests/test_training_image_contract.py`

**Interfaces:**
- Consumes: `RunSpec.image_digest`, `RunSpec.region`, `RunSpec.trusted_profile_id` and the immutable COS run key.
- Produces: a `linux/amd64` OCI manifest digest, `/var/lib/watch-auth/bootstrap-proof.json`, a healthy training container and the first hash-linked `running`/`failed` event.

- [ ] **Step 1: Resolve and record base-image digest**

Run: `docker buildx imagetools inspect nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
Expected: a platform-specific `sha256:` digest. Put that exact digest in `FROM`; mutable-only tags fail the contract test.

- [ ] **Step 2: Write failing static contract tests**

Assert a `linux/amd64` digest-pinned `FROM`, non-root user, no `ARG/ENV` secret names, `pip install --require-hashes -r requirements-train.lock`, the PyTorch 2.5.1 CUDA 12.4 wheel index/artifact hash and an OCI healthcheck. The contract test fails if `requirements.txt` is copied directly into the training image.

- [ ] **Step 3: Implement host bootstrap**

Implement an idempotent two-stage systemd bootstrap. Stage 1 verifies signed APT repository fingerprints recorded in the script, installs exact NVIDIA 550 driver/Container Toolkit package versions, writes no secret, enables Stage 2 and reboots once. Stage 2 proves cloud-init success, driver >=550, L4 memory, toolkit and CUDA 12.4 compatibility before requesting a short-lived instance-identity token from Instance Metadata. It logs into `private.${IBM_REGION}.icr.io` with username `iamtoken` and the token on stdin, pulls the exact digest, logs out, and launches the container with `--gpus all --network host --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp`. Host networking is required only so the container can use `VPCInstanceAuthenticator`; the security group still has no inbound rule. End with:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
docker run --rm --gpus all "${TRAIN_IMAGE_DIGEST}" python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The container proof additionally asserts `torch.__version__.startswith("2.5.1")`, `torch.version.cuda == "12.4"`, device name contains `L4`, and total memory is at least 22 GiB. Write proof to `/var/lib/watch-auth/bootstrap-proof.json` and upload its hash as the first run event; never echo or persist tokens. A failed package signature, required reboot, failed identity exchange, unexpected GPU, digest mismatch or healthcheck blocks training and proceeds to teardown.

- [ ] **Step 4: Build, scan and push**

Generate `requirements-train.lock` once for `linux/amd64` with every direct/transitive artifact pinned and SHA-256 hashed, then prove it with `python -m pip install --dry-run --require-hashes -r requirements-train.lock` inside the same pinned base image. Build the `linux/amd64` image into `TRAIN_IMAGE_LOCAL=watch-train:${CODE_SHA}` and run `docker scout cves --exit-code --only-severity critical,high "$TRAIN_IMAGE_LOCAL"` when Docker Scout is installed; otherwise run `trivy image --exit-code 1 --severity HIGH,CRITICAL "$TRAIN_IMAGE_LOCAL"`. If neither scanner exists, stop with `BLOQUEADO — INTERVENÇÃO NECESSÁRIA`; do not silently skip scanning or install a paid service. Push to IBM Container Registry only after external-write/cost authorization, then resolve the manifest digest. `RunSpec.image_digest` must match `^private\.[a-z0-9-]+\.icr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}$`; registry retention and storage price are included in `PricingEvidence`.

- [ ] **Step 5: Run contract tests**

Run: `pytest tests/test_training_image_contract.py -v`
Expected: PASS without requiring a GPU.

### Task 6: Add the missing real training producer

**Files:**
- Create: `training/data.py`
- Create: `training/run.py`
- Modify: `core/rules.py`
- Modify: `model/heads.py`
- Modify: `dataset/splits.py`
- Test: `tests/test_training_run.py`

**Interfaces:**
- Consumes: immutable COS manifest/splits and `RunSpec`.
- Produces: `load_training_data(cos: Any, spec: RunSpec) -> TrainingBundle`, `run_training(bundle: TrainingBundle, spec: RunSpec, output_dir: Path) -> RunResult`, plus `checkpoint.pt`, `metrics.json`, `class-map.json`, `run-result.json`.

- [ ] **Step 1: Write failing preflight tests**

Reject placeholder/example domains, missing COS objects, hash mismatches, split overlap, empty class, missing OOD/rejection examples, rejection examples without an owned/licensed provenance record, fewer than two source domains per authenticity label, and a manifest/split digest different from RunSpec.

- [ ] **Step 2: Make rejection trainable**

Consume the Task 1 `Label.REJECTION` and map only owner-supplied or explicitly licensed negatives for no-watch and non-Submariner watches to `ModelClass.REJECTION`. Do not scrape a new source implicitly. Group-split them by provenance record and subject so near-duplicates never cross train/eval. The head imports `len(ModelClass)` and must never expose a class with zero training examples; the existing `model.rejection.REJECTION_CLASS_INDEX` imports `ModelClass.REJECTION.value`.

- [ ] **Step 3: Implement deterministic training**

Seed Python/NumPy/Torch, enable deterministic algorithms where supported, load the frozen backbone once, optimize only heads, use class-weighted cross entropy, early stopping on validation NLL, and checkpoint the best epoch plus optimizer-independent class map.

- [ ] **Step 4: Add resume semantics**

Resume only when code/image/manifest/splits/class-map hashes match. Save only plain tensor state dictionaries plus JSON metadata; load with `torch.load(path, weights_only=True, map_location="cpu")`. A mismatched checkpoint or serialized Python object is refused, not partially loaded.

- [ ] **Step 5: Test CPU fixture end-to-end**

Run: `pytest tests/test_training_run.py -v`
Expected: two tiny epochs produce finite loss, immutable output hashes and a loadable checkpoint with no backbone weights.

### Task 7: Orchestrate one authorized run and always tear it down

**Files:**
- Create: `infra/ibm_train.py`
- Test: `tests/test_ibm_train.py`

**Interfaces:**
- Consumes: `IBMClients`, `AuthorizedRun` and the Task 3/4 adapters. `IBMClients` is a frozen dataclass containing injected `vpc: VpcV1` and `cos: Any`; it does not wrap their APIs.
- Produces: `launch_training(clients: IBMClients, run: AuthorizedRun, *, confirm: bool, now: datetime) -> RunEvent`; return value is the observed `deleted` event, not merely a submitted create response.

- [ ] **Step 1: Write the lifecycle test**

```python
def test_success_uploads_terminal_event_then_deletes(ibm_clients, authorized_run):
    event = launch_training(
        ibm_clients, authorized_run, confirm=True, now=datetime(2026, 11, 5, tzinfo=UTC)
    )
    assert ibm_clients.cos.terminal_written_before_delete
    ibm_clients.vpc.delete_instance.assert_called_once_with("vsi-1")
    assert event.state == "deleted"
```

Add failure cases for `confirm=False`, denied authorization, bootstrap failure, training failure, COS failure, Ctrl-C and timeout. Every post-create case reaches deletion.

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ibm_train.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement lifecycle state machine**

```text
planned -> authorized -> creating -> running -> artifact_sync
        -> terminal_receipt -> deleting -> deleted
        -> quarantined (only when remote outcome cannot be proven)
```

Persist each transition as an immutable COS event (`events/{sequence:06d}-{event_id}.json`) whose payload hashes the previous event. Before launch and on every poll, `validate_event_chain(events: tuple[RunEvent, ...], spec: RunSpec) -> None` requires one event per contiguous sequence, one run ID, matching lineage hashes and an exact previous-event hash; a fork, gap or regression quarantines the run. The terminal success event is written only after artifact hashes verify. The deletion event is separate and requires observed 404. The local controller never needs SSH: user-data starts the container, the container reads the immutable RunSpec from COS through its trusted profile, and controller/container communicate only through append-only COS events.

- [ ] **Step 4: Implement signal-safe cleanup**

Catch `BaseException` only to persist failure and enter `finally`; re-raise after cleanup. If COS is unavailable, deletion still runs and a local recovery record is written atomically under `.runtime/ibm/recovery/<run_id>.json` for the watchdog/controller to reconcile. Never swallow KeyboardInterrupt/SystemExit.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_ibm_train.py -v`
Expected: PASS.

### Task 8: Add an independent IBM watchdog and observability

**Files:**
- Create: `infra/ibm_watchdog.py`
- Create: `scripts/deploy_ibm_watchdog.sh`
- Test: `tests/test_ibm_watchdog.py`

**Interfaces:**
- Consumes: `RunSpec`, `ResourceLease`, validated `RunEvent` chains, injected VPC/COS clients and current UTC time.
- Produces: `reconcile_expired_resources(vpc: VpcV1, cos: Any, leases: tuple[ResourceLease, ...], events: tuple[RunEvent, ...], *, now: datetime) -> tuple[RunEvent, ...]` plus an idempotently deployed Code Engine job/cron.

- [ ] **Step 1: Write failing stale-resource tests**

Cover expired tagged VSI, active valid VSI, untagged VSI, malformed TTL, duplicate matches and delete timeout. Only an exact project/run identity with expired TTL may be deleted.

- [ ] **Step 2: Implement reconcile-only watchdog**

The watchdog never creates compute. It lists the dedicated resource group and reads append-only run events. When an instance ID is recorded, that ID plus resource group, deterministic name, project tag and run ID must match. For a lost create response with no ID, it may adopt/delete only one exact match on the inspectable intended prototype; zero or multiple matches are quarantined. After VSI/volume 404, it deletes the run-owned Public Gateway only when no other subnet/resource references it. It writes one observed terminal event per resource; shared or unowned resources are quarantined, never deleted.

- [ ] **Step 3: Define Code Engine cron**

`scripts/deploy_ibm_watchdog.sh` uses `ibmcloud ce job create|update` with a digest-pinned image, then `ibmcloud ce sub cron create|update --destination-type job --destination watch-auth-watchdog --schedule '*/15 * * * *'`. The script reads project/service-account names from required environment variables, checks the selected account/resource group/project before mutation, and never creates those scopes implicitly. Bind a service ID to the dedicated project resource group and COS bucket; IBM IAM cannot express a COS-prefix or tag predicate, so bucket/resource-group scope is the provider-side blast-radius boundary and application-level prefix/identity checks are mandatory. Before any GPU create, the provider E2E must create an owner-authorized cheapest eligible CPU VSI canary with a 15-minute TTL, observe Code Engine delete it, observe VPC 404 and persist the watchdog event. A dry run proves read access but not delete permission and is insufficient. Include canary and Code Engine ongoing cost in the authorization decision; remove the cron and job after the final run only through a separately recorded teardown step.

- [ ] **Step 4: Wire logs, metrics and spending notifications**

Emit structured fields `run_id, state, vsi_id_hash, age_seconds, estimated_usd, observed_usage_usd`; redact tokens, signed URLs, object contents and raw instance IDs. Route Code Engine platform logs and VPC platform metrics to the IBM Cloud Logs/Monitoring instances through the account's platform integration; do not install an agent that requires a persistent ingestion key. If the chosen region/account cannot prove credentialless platform routing, launch remains blocked rather than weakening the no-secret invariant. Spending notifications remain secondary alerts, never the overage control. The usage collector stores the raw Usage Reports response hash and blocks a new run until the previous run has a terminal provider-usage reconciliation.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_ibm_watchdog.py -v`
Expected: PASS.

### Task 9: Remove AWS and prove the IBM path without spending

**Files:**
- Delete: `infra/aws_train.py`
- Delete: `infra/s3_sync.py`
- Modify: `tests/test_infra.py`
- Modify: `scripts/preflight.sh`

**Interfaces:**
- Consumes: all runtime Python sources and the Task 1–8 adapters.
- Produces: one IBM-only `launch_training` path, a failing static AWS regression on forbidden imports/client calls, and a mocked E2E ending in terminal artifact plus per-resource deletion events.

- [ ] **Step 1: Add a static no-AWS regression**

```python
def test_runtime_has_no_aws_provider():
    forbidden = ("import boto3", "client(\"ec2\")", "client(\"s3\")")
    assert not any(token in runtime_sources() for token in forbidden)
```

- [ ] **Step 2: Remove superseded code/dependency**

Delete AWS modules/tests and regular `boto3`. `rg -n "upload_photo|download_photo" --glob '*.py'` must first identify every caller; migrate each found caller to `upload_verified`/`download_verified`, then remove the old names. If no caller exists, delete without a compatibility wrapper.

- [ ] **Step 3: Fix the existing preflight bypass**

Remove `|| true` from `mypy`; a type failure must fail the canonical preflight.

- [ ] **Step 4: Run the canonical local gate**

Run: `mac-gate ./scripts/preflight.sh`
Expected: ruff, mypy and all tests PASS. No `next build`.

- [ ] **Step 5: Run a zero-cost mocked E2E**

Execute plan -> authorization -> mocked VPC -> mocked bootstrap/training -> verified COS outputs -> terminal receipt -> deletion receipt. Assert one create and one delete.

- [ ] **Step 6: Stop before real provisioning**

Real IBM launch remains `PENDENTE E2E` until balance/expiry/SKU eligibility, IAM, zone capacity, current pricing, image digest, dataset hashes and the post-promo disposition of COS/Registry/Logs are captured for the same run. A real canary requires exact target/cost/TTL authorization immediately before create. Automatic deletion of persistent data is forbidden: before promo expiry, export and verify it first, then obtain the owner's destructive-action confirmation or record separately funded retention.

## Execution Commit Boundaries

| Task | Commit after focused tests pass |
|---|---|
| 1 | `feat: define IBM run contracts` |
| 2 | `feat: capture and gate IBM credit evidence` |
| 3 | `feat: add verified IBM COS storage` |
| 4 | `feat: reconcile IBM VPC lifecycle` |
| 5 | `feat: build immutable GPU runtime` |
| 6 | `feat: add production training runner` |
| 7 | `feat: orchestrate IBM training teardown` |
| 8 | `feat: add independent IBM watchdog` |
| 9 | `refactor: remove AWS training path` |

Each implementation checkbox is executed as a RED/GREEN micro-loop: add one named test, run that single node ID, implement only the symbol exercised by it, rerun that node ID, then proceed. A step containing several named failure modes is a checklist of separate 2–5 minute loops, not one bulk edit.

## Definition of Done

- All focused tests and `mac-gate ./scripts/preflight.sh` pass on one SHA.
- Phase 2 review is repeated; the AWS review is not reused.
- No runtime AWS reference or paid-provider fallback remains.
- Credit evidence and pricing evidence are runtime-only, fresh and hashed into the receipt.
- Every metered service has an explicit USD price line; missing or non-USD pricing fails closed.
- The VSI authenticates only through its trusted profile, has no inbound path/floating IP, and the watchdog is proven live before GPU creation.
- VSI, boot volume and run-owned Public Gateway all reach observed 404; persistent services have a dated, authorized post-promo disposition.
- Dataset has real retrievable objects and trained rejection examples; placeholder endpoints cannot reach GPU.
- Mocked E2E proves exactly one create, artifact integrity, terminal receipt and observed deletion.
- Real provider E2E remains `PENDENTE E2E` until explicitly authorized and externally observed.

## Optimization Ledger

- **Round 1 — substantive findings applied:** missing real training producer; untrained rejection class; illustrative scrapers could feed no real dataset; no immutable dataset/code/image lineage; AWS placeholder lifecycle lacked complete launch/delete semantics. Added Tasks 1–9 and fail-closed gates.
- **Round 2 — substantive findings applied:** the first draft did not define secretless registry/COS authentication, priced only VSI plus an opaque ancillary budget, used a race-prone receipt write, assumed a one-pass driver install without reboot, adopted instances by name alone, omitted the no-SSH execution protocol, overstated tag-level IAM enforcement, and allowed a watchdog that had not been proven live. Added trusted-profile/metadata auth, per-service price lines and usage reconciliation, conditional hash-chained events, two-stage bootstrap, prototype fingerprinting, COS-only control flow, dedicated-resource-group IAM boundaries and a pre-create watchdog dry run.
- **Round 3 — substantive findings applied:** the main spec still exposed the obsolete AWS-shaped launch signature and an incomplete IBM file map; subplan names drifted (`RunReceipt` vs `RunEvent`, profile vs service-plan eligibility), test snippets used undefined ellipses/helpers, cost quantities and adapter signatures were underspecified, lost-create watchdog reconciliation was impossible without an instance ID, scanner behavior was vague, and a watchdog dry run could not prove delete permission. Unified the main/subplan contracts, defined complete factories/signatures and required service lines, specified inspectable reconciliation, made scanner absence fail closed, added the real training interfaces, and required a low-cost provider deletion canary before GPU use.
- **Round 4 — substantive findings applied:** the Code Engine deployment artifact had no executable provider contract, COS-prefix IAM scope was overstated, the bootstrap contained an invalid command and unpinned repository trust, event-chain integrity and safe checkpoint loading were absent, core value objects were still only names, observability could have required a persisted ingestion secret, and task commit/micro-loop boundaries were missing. Replaced YAML with an idempotent CLI deployment script, corrected provider blast-radius claims, fixed and pinned bootstrap/scanning, added event-chain and `weights_only` checks, completed value-object fields, restricted observability to credentialless platform routing, and added exact commit/RED-GREEN boundaries.
- **Round 5 — substantive findings applied:** the authorization gate consumed typed promo, pricing and usage evidence, but no task produced those values or bound them to authoritative artifacts; a hand-created JSON could therefore authorize spend without proof. Added runtime-only console/quote capture validation, IBM Usage Reports normalization, account/currency/pagination checks, injected-client tests and a strict rule that loaders never OCR or infer financial values.
- **Round 6 — substantive finding applied:** source inspection of `ibm-cos-sdk==2.16.2` and `ibm-cloud-sdk-core==3.28.1` showed that `ibm_boto3` accepts a custom token manager, not `VPCInstanceAuthenticator` directly, and token-authenticated COS also needs its service-instance ID. Added `cos_service_instance_id` to `RunSpec`, the verified `auth.token_manager` client construction and a regression test forbidding API-key/HMAC fallback.
- **Round 7 — clean pass 1:** rechecked spec coverage, producer-to-receipt lifecycle, financial fail-closed rules, IAM/network boundaries, type/signature consistency, AWS removal path and all five Review Focus cases. No substantial improvement or optimization remained.
- **Round 8 — substantive finding applied:** lifecycle closure covered the GPU VSI but could leave its boot volume, outbound Public Gateway and persistent COS/Registry/Logs resources accruing charges after the promotional credit expired. Added per-resource leases, auto-delete plus observed-404 checks for boot volumes, run-owned Public Gateway teardown, watchdog orphan handling and a mandatory authorized post-promo export/delete or funded-retention plan without permitting automatic data destruction.
- **Round 9 — substantive finding applied:** Tasks 5, 8 and 9 lacked the mandatory `Interfaces` blocks from `writing-plans`, leaving runtime image, watchdog and migration handoffs implicit. Added exact consumed contracts and produced artifacts/functions for all three tasks.
- **Round 10 — substantive finding applied:** inspection of the pinned `ibm-cos-sdk-core==2.16.2` service model proved that `PutObject` does not support `IfNoneMatch`, invalidating the proposed conditional event write. Replaced it with never-reused UUID event keys; concurrent events at one sequence are both preserved and the chain validator treats them as a fork that quarantines the run.
- **Round 11 — substantive finding applied:** the image task named a lock file without requiring hash enforcement or proving that PyTorch 2.5.1 was the CUDA 12.4 build for `linux/amd64`. Added `--require-hashes`, transitive artifact hashes, same-base dry-run resolution and runtime assertions for Torch version, CUDA version, L4 identity and usable VRAM.
- **Round 12 — clean pass 1:** reviewed security boundaries and failure injection across evidence capture, trusted-profile auth, COS concurrency, ambiguous VPC create/delete, bootstrap/reboot, signal cleanup, watchdog permissions, per-resource expiry and image supply chain. No substantial improvement or optimization remained.
- **Round 13 — substantive finding applied:** `Label.REJECTION` was owned by both Task 1 and Task 6, so an executor could define the six-class vocabulary twice or invalidate Task 1's tests later. Consolidated `Label`/`ModelClass` ownership and tests in Task 1; Task 6 now only consumes that vocabulary while owning data mapping/training.
- **Round 14 — clean pass 1:** reviewed ownership, task ordering, contract handoffs, migration safety and smallest-complete-system scope. No substantial improvement or optimization remained. One non-substantive wording cleanup assigned `boto3` removal solely to Task 9 so intermediate commits stay runnable.
- **Round 15 — clean pass 2:** re-read the main spec and complete implementation plan from a fresh executor perspective; rechecked every task's files/interfaces, type names, Review Focus test ownership, provider-SDK facts, cost/expiry invariants, deletion proof and forbidden fallback paths. No substantial improvement or optimization remained.
