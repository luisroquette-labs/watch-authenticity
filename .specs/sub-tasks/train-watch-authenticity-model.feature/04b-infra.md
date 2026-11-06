# IBM Cloud Training Plane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete AWS adapter with a reproducible IBM Cloud training plane that consumes the CF Gauss promotional credit without silently creating cash overage, and that can actually train, evaluate and preserve a Rolex Submariner model.

**Architecture:** A local controller authorizes one immutable run, IBM VPC executes one digest-pinned GPU container, IBM COS is the append-only object/receipt store, and a pre-deployed IBM Code Engine cron watchdog deletes expired workers. The VSI has no inbound path or floating IP. Its hash-bound host bootstrap installs a local forward proxy; the training container runs on an isolated bridge that can reach only the exact regional COS endpoint and IAM through that proxy. The trusted host installs signed packages before mounting any dataset, obtains short-lived credentials through Instance Metadata and reaches the internet through the outbound-only Public Gateway. Pure cost/lifecycle rules are SDK-free; IBM SDK adapters sit at the boundary.

**Tech Stack:** Python 3.10, PyTorch 2.5.1/CUDA 12.4, `ibm-vpc==0.36.1`, `ibm-cos-sdk==2.16.2`, `ibm-platform-services==0.77.1`, `ibm-cloud-sdk-core==3.28.0`, IBM VPC `gx3-16x80x1l4`, IBM COS, IBM Container Registry, IBM Code Engine cron, IAM trusted profiles, IBM Cloud Logs/Monitoring.

**Spec:** `.specs/tasks/in-progress/train-watch-authenticity-model.feature.md`

## Global Constraints

- Production profile is exactly `gx3-16x80x1l4`: 1 NVIDIA L4 24 GB, 16 vCPU and 80 GiB RAM.
- L4 host bootstrap must prove NVIDIA driver >=550 and CUDA 12.4 before the container starts.
- The applied promo code is not balance proof. Production launch requires console/API evidence of remaining credit, expiry and eligible IBM-native SKUs observed within 24 hours.
- A run must finish at least 48 hours before promo expiry.
- Default safety reserve is USD 500. Lowering it is a financial decision requiring the owner's explicit approval in that conversation.
- Spending notifications are evidence only; IBM sends 80/90/100% alerts and does not promise a hard stop.
- The estimate includes every metered resource used by the run: VSI, boot volume, Public Gateway, its dedicated Floating IP, COS storage/operations, Container Registry storage, Code Engine, Logs/Monitoring and the USD 500 lag reserve. An unpriced line item blocks launch.
- No AWS, S3, external GPU, paid AI API or automatic paid fallback.
- No raw scraped image is published. COS buckets are private and deny public access.
- No credential is committed, copied into the VSI, baked into a container layer or written to logs/receipts.
- Every run is tied to code SHA, image digest, bootstrap SHA-256, manifest SHA-256, split SHA-256 and unique run ID.
- Ambiguous create/delete responses are reconciled before retry. Never create a second VSI automatically.
- The watchdog must be deployed and pass a no-delete dry run before a GPU instance can be created.
- The VSI has no floating IP and its security group allows no inbound traffic. The host security group permits only DNS and HTTPS; the hash-bound bootstrap adds the mandatory domain-filtering proxy and blocks direct container egress and container access to other host services. Before any dataset is mounted, the trusted host may reach the signed package repositories required by bootstrap. The training container can reach only Instance Metadata, `iam.cloud.ibm.com` and `s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud`.
- Every metered resource has a lease owner and terminal policy. Boot volumes, the dedicated Public Gateway and its dedicated Floating IP are deleted with each run; persistent COS/Registry/Logs resources require a verified export-and-delete or explicitly funded post-promo plan before the first run.
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
- `RunSpec` contains exactly `run_id, account_id, code_sha, image_digest, bootstrap_sha256, manifest_sha256, splits_sha256, profile, region, zone, cos_endpoint, max_hours, boot_volume_gb, cos_storage_gb_month, cos_operations, registry_storage_gb_month, code_engine_job_hours, logs_monitoring_gb, bucket, prefix, cos_service_instance_id, resource_group_id, network_id, subnet_id, primary_ipv4_address, image_id, public_gateway_id, public_gateway_floating_ip_id, security_group_id, trusted_profile_id, subnet_owned_by_run, public_gateway_owned_by_run, cos_private_access_only, cos_versioning_enabled`.
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
    account_id: str
    code_sha: str
    image_digest: str
    bootstrap_sha256: str
    manifest_sha256: str
    splits_sha256: str
    profile: str
    region: str
    zone: str
    cos_endpoint: str
    max_hours: Decimal
    boot_volume_gb: Decimal
    cos_storage_gb_month: Decimal
    cos_operations: Decimal
    registry_storage_gb_month: Decimal
    code_engine_job_hours: Decimal
    logs_monitoring_gb: Decimal
    bucket: str
    prefix: str
    cos_service_instance_id: str
    resource_group_id: str
    network_id: str
    subnet_id: str
    primary_ipv4_address: str
    image_id: str
    public_gateway_id: str
    public_gateway_floating_ip_id: str
    security_group_id: str
    trusted_profile_id: str
    subnet_owned_by_run: bool
    public_gateway_owned_by_run: bool
    cos_private_access_only: bool
    cos_versioning_enabled: bool
```

Validate 40-character Git SHA, 64-hex content hashes including `bootstrap_sha256`, `@sha256:` image reference, exact profile, IBM CRN/UUID identifiers, one RFC1918 IPv4 address reserved for this run, the exact regional COS direct endpoint, non-empty region/zone/bucket/prefix and `0 < max_hours <= 24`. Reject loopback, link-local, unspecified, reserved or public IPs. Reject a shared/unowned subnet, Public Gateway or Floating IP because the run cannot safely delete it. `build_instance_prototype` computes SHA-256 over the exact UTF-8 user-data bytes and refuses a value different from `RunSpec.bootstrap_sha256`. `tests/conftest.py` defines `valid_run_spec()`, `valid_credit()`, `valid_usage()`, `valid_pricing()`, `valid_post_promo_plan()`, `vpc_instance()` and `safe_user_data()` with complete valid values; every mutation test uses `dataclasses.replace`. Define `ModelClass` once in `core/types.py`; `model/heads.py`, `model/rejection.py` and `training/data.py` import its indices rather than recomputing them.

- [ ] **Step 4: Pin dependencies**

```text
ibm-vpc==0.36.1
ibm-cos-sdk==2.16.2
ibm-platform-services==0.77.1
ibm-cloud-sdk-core==3.28.0
```

`ibm-vpc==0.36.1` requires `python-dateutil<3`; newer `ibm-cloud-sdk-core==3.28.1` requires `python-dateutil>=3`, so that combination is intentionally rejected as unsatisfiable.

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

`REQUIRED_METERED_LINES` is the exact frozen set `{vpc-vsi, vpc-boot-volume, vpc-public-gateway, vpc-floating-ip, cos-storage, cos-operations, container-registry-storage, code-engine, logs-monitoring}`. A line advertised as free still has an explicit zero price; absence is not treated as free.

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
- Produces: `build_instance_prototype(spec: RunSpec, user_data: str) -> dict[str, object]`, `ensure_single_instance(vpc: VpcV1, spec: RunSpec, prototype: dict[str, object], *, tagger: GlobalTaggingV1, deadline: float | None = None) -> InstanceRef`, `recover_created_instance_id(spec: RunSpec, prototype: dict[str, object]) -> str`, `wait_running(vpc: VpcV1, instance_id: str, *, deadline: float) -> InstanceRef`, `delete_and_wait(vpc: VpcV1, spec: RunSpec, instance_id: str, *, deadline: float, tagger: GlobalTaggingV1) -> None`.

- [ ] **Step 1: Write failing ambiguity tests**

```python
def test_lost_create_response_reconciles_by_exact_name(fake_vpc, spec):
    fake_vpc.create_instance.side_effect = ReadTimeout()
    fake_vpc.list_instances.return_value = [vpc_instance(name=f"watch-auth-{spec.run_id}")]
    prototype = build_instance_prototype(spec, safe_user_data())
    assert ensure_single_instance(fake_vpc, spec, prototype, tagger=fake_tagger).id == "vsi-1"
    assert fake_vpc.create_instance.call_count == 1

def test_multiple_name_matches_quarantine_without_delete(fake_vpc, spec):
    fake_vpc.list_instances.return_value = [vpc_instance(id="a"), vpc_instance(id="b")]
    with pytest.raises(AmbiguousProvisioning):
        ensure_single_instance(
            fake_vpc,
            spec,
            build_instance_prototype(spec, safe_user_data()),
            tagger=fake_tagger,
        )
    fake_vpc.delete_instance.assert_not_called()
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_ibm_vpc.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement exact instance prototype**

Use `VpcV1.create_instance` with deterministic name, dedicated resource group, VPC, zone, run-owned subnet, exact RFC1918 primary IP with `auto_delete=False`, pinned Ubuntu image ID, profile `gx3-16x80x1l4`, provider-managed encrypted boot volume with `delete_volume_on_instance_delete=True`, the exact trusted profile, Instance Metadata enabled over its SDK-default link-local HTTP endpoint with response hop limit 2, no VSI floating IP, an inbound-deny security group and user-data whose SHA-256 equals `RunSpec.bootstrap_sha256`. The provider's exclusive reserved-IP binding is the cross-controller create mutex: a simultaneous create conflicts, and the non-auto-deleted reservation blocks a later replay after VSI deletion. Controllers are separate synchronous processes; each invokes the SDK lifecycle on its main thread so the Unix alarm enforces a hard total deadline even if HTTP keeps producing bytes. Final deletion of the run-owned subnet removes the reservation and also makes any later replay fail before create because the authorized subnet ID is absent. Cap the local lease, every provider call and every reconciliation sleep by the same monotonic deadline. The subnet attaches one dedicated, leased Public Gateway backed by one dedicated, leased Floating IP whose exact ID is in `RunSpec`. Persist the full canonical prototype plus its SHA-256 before create. After an accepted or reconciled create, durably persist the instance ID before reading any other provider field; if the response omits boot-volume context, perform bounded reads by that exact ID. Then atomically persist the complete instance, volume, gateway and Floating IP cleanup context before any later validation so Task 7 can tear them down. If hydration stays ambiguous, retain the immutable accepted-ID record and block recreate until reconciliation. A 2xx response without ID enters the same exact-name reconciliation path as a lost response. Deletion independently validates ownership before mutation. Instance, subnet, Public Gateway and Floating IP CRNs must exactly bind the authorized zone, account, resource type and resource ID before their tags are trusted. On reconciliation, exact name alone is insufficient: resource group, profile, zone, VPC, subnet, primary IP and image must all match inspectable provider fields; apply project/run tags immediately after adoption. Resolve IDs before authorization; do not accept IDs from dataset content.

- [ ] **Step 4: Implement bounded polling**

Poll explicit terminal states with monotonic deadlines and bounded exponential backoff with jitter. Every read must return the requested instance ID; a different or missing ID is ambiguous. Unknown state, 429 exhaustion or timeout persists an ambiguous event and blocks automatic recreate. A name match with a different prototype is never adopted or deleted automatically.

- [ ] **Step 5: Test delete reconciliation**

404 after DELETE counts as deleted; an initial pre-DELETE 404 requires a second consecutive 404 separated by bounded backoff so create visibility lag cannot produce a false receipt. 408, 409, 429 and 5xx mutation responses are reconciled by polling rather than treated as terminal failure; timeout requires a later controller/watchdog attempt. Never mutate dependent volume/network resources unless the VSI has confirmed 404; deterministic VSI delete failure returns immediately. Authentication/authorization failures and other deterministic 4xx responses still fail immediately. If any post-create instance validation rejects the provider representation, exact durable create intent plus instance ID and CRN still authorize deleting only that locally created VSI; volume, subnet, gateway and Floating IP retain independent ownership validation. Valid-JSON and malformed cleanup ledgers are quarantined and reported while immutable created/accepted fallbacks continue. If a partial accepted response never yields a volume ID and the VSI later disappears, list volumes by the exact run tag and zone, revalidate resource group, zone, capacity, encryption and the complete exclusive ownership tag set, delete every exact run-owned match, and require a final empty run-owned inventory; unrelated zone-volume churn cannot prevent closure. After observed VSI 404, attempt the boot volume, subnet, Public Gateway and Floating IP independently so one provider failure cannot strand the other billable resources. Delete the run-owned subnet so IBM atomically removes its gateway association, then delete and observe 404 for the dedicated Public Gateway and Floating IP. The preliminary Floating IP read has a separate earlier deadline and cannot consume the gateway window. Never call the subnet gateway-unset endpoint, because it cannot condition on the expected gateway ID. A missing Floating IP never counts as a missing gateway; require each applicable 404 and emit separate resource-deletion events.

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
- Consumes: `RunSpec.image_digest`, `RunSpec.bootstrap_sha256`, `RunSpec.region`, `RunSpec.trusted_profile_id` and the immutable COS run key.
- Produces: a `linux/amd64` OCI manifest digest, `/var/lib/watch-auth/bootstrap-proof.json`, a healthy training container and the first hash-linked `running`/`failed` event.

- [ ] **Step 1: Resolve and record base-image digest**

Run: `docker buildx imagetools inspect nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`
Expected: a platform-specific `sha256:` digest. Put that exact digest in `FROM`; mutable-only tags fail the contract test.

- [ ] **Step 2: Write failing static contract tests**

Assert a `linux/amd64` digest-pinned `FROM`, non-root user, no `ARG/ENV` secret names, `pip install --require-hashes -r requirements-train.lock`, the PyTorch 2.5.1 CUDA 12.4 wheel index/artifact hash and an OCI healthcheck. The contract test fails if `requirements.txt` is copied directly into the training image. It also hashes `infra/bootstrap_host.sh` bytes and requires equality with `RunSpec.bootstrap_sha256`; rejects `--network host`; requires the isolated bridge plus `DOCKER-USER` forwarding and `INPUT` host-service default-drop rules; requires IPv6 disabled on that bridge or equivalent default-drop IPv6 rules and negative proof; requires Squid to bind only the bridge host IP and allow the two exact proxy hosts; and rejects wildcard/suffix proxy allowlists.

- [ ] **Step 3: Implement host bootstrap**

Implement an idempotent two-stage systemd bootstrap. Stage 1 verifies signed APT repository fingerprints recorded in the script, installs exact NVIDIA 550 driver/Container Toolkit and Squid package versions, writes no secret, mounts no dataset, enables Stage 2 and reboots once. Stage 2 proves cloud-init success, driver >=550, L4 memory, toolkit and CUDA 12.4 compatibility before requesting a short-lived instance-identity token from Instance Metadata. Configure Squid to bind only the `watch-auth-isolated` bridge host IP, accept only CONNECT on port 443 and allow only the exact hosts `iam.cloud.ibm.com` and `s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud`; disable cache, redact query strings and deny every other destination. Create the dedicated Docker bridge with IPv6 disabled; if the platform cannot prove IPv6 is disabled, install equivalent `ip6tables`/nftables default-drop forwarding and host-input rules. Install IPv4 `DOCKER-USER` rules that allow forwarded container traffic only to the link-local metadata address and drop every other direct destination, and `INPUT` rules that allow bridge traffic only to Squid port 3128 while rejecting every other host service. Prove a direct-IP HTTPS request, another host port, a disallowed domain and IPv6 egress all fail while Instance Metadata and the configured regional COS endpoint succeed. Any missing IPv4/IPv6 firewall or proxy proof blocks dataset mounting and training.

After the network proof, log into `private.${IBM_REGION}.icr.io` with username `iamtoken` and the short-lived token on stdin, pull the exact digest, log out, and launch the container with `--gpus all --network watch-auth-isolated --read-only --cap-drop=ALL --security-opt=no-new-privileges --tmpfs /tmp`, `HTTPS_PROXY=http://host.docker.internal:3128`, `NO_PROXY=169.254.169.254` and `--add-host=host.docker.internal:host-gateway`. `DOCKER-USER` permits only forwarded link-local metadata traffic; `INPUT` permits only the proxy on the bridge host; every other direct destination or host service is rejected. The trusted-profile token remains short-lived and is never written to disk, baked into the image or exposed in logs. End with:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
docker run --rm --gpus all "${TRAIN_IMAGE_DIGEST}" python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
```

The container proof additionally asserts `torch.__version__.startswith("2.5.1")`, `torch.version.cuda == "12.4"`, device name contains `L4`, total memory is at least 22 GiB, direct egress is blocked and the proxy rejects an unlisted domain. Write proof to `/var/lib/watch-auth/bootstrap-proof.json` and upload its hash as the first run event; never echo or persist tokens. A failed package signature, required reboot, failed identity exchange, unexpected GPU, digest mismatch, proxy/firewall proof or healthcheck blocks training and proceeds to teardown.

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
- Consumes: `IBMClients`, `AuthorizedRun` and the Task 3/4 adapters. `IBMClients` is a frozen dataclass containing injected `vpc: VpcV1`, `tagger: GlobalTaggingV1` and `cos: Any`; it does not wrap their APIs.
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

The watchdog never creates compute. It lists the dedicated resource group and reads append-only run events. When an instance ID is recorded, that ID plus resource group, deterministic name, project tag and run ID must match. For a lost create response with no ID, it may adopt/delete only one exact match on the inspectable intended prototype; zero or multiple matches are quarantined. After VSI/volume 404, it deletes the run-owned Public Gateway only when no other subnet/resource references it, then independently deletes and confirms 404 for the run-owned Floating IP. It writes one observed terminal event per resource; shared or unowned resources are quarantined, never deleted.

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
- VSI, boot volume, run-owned Public Gateway and its dedicated Floating IP all reach observed 404; persistent services have a dated, authorized post-promo disposition.
- Dataset has real retrievable objects and trained rejection examples; placeholder endpoints cannot reach GPU.
- Mocked E2E proves exactly one create, artifact integrity, terminal receipt and observed deletion.
- Real provider E2E remains `PENDENTE E2E` until explicitly authorized and externally observed.

## Optimization Ledger

- **Round 1 — substantive findings applied:** missing real training producer; untrained rejection class; illustrative scrapers could feed no real dataset; no immutable dataset/code/image lineage; AWS placeholder lifecycle lacked complete launch/delete semantics. Added Tasks 1–9 and fail-closed gates.
- **Round 2 — substantive findings applied:** the first draft did not define secretless registry/COS authentication, priced only VSI plus an opaque ancillary budget, used a race-prone receipt write, assumed a one-pass driver install without reboot, adopted instances by name alone, omitted the no-SSH execution protocol, overstated tag-level IAM enforcement, and allowed a watchdog that had not been proven live. Added trusted-profile/metadata auth, per-service price lines and usage reconciliation, conditional hash-chained events, two-stage bootstrap, prototype fingerprinting, COS-only control flow, dedicated-resource-group IAM boundaries and a pre-create watchdog dry run.
- **Round 3 — substantive findings applied:** the main spec still exposed the obsolete AWS-shaped launch signature and an incomplete IBM file map; subplan names drifted (`RunReceipt` vs `RunEvent`, profile vs service-plan eligibility), test snippets used undefined ellipses/helpers, cost quantities and adapter signatures were underspecified, lost-create watchdog reconciliation was impossible without an instance ID, scanner behavior was vague, and a watchdog dry run could not prove delete permission. Unified the main/subplan contracts, defined complete factories/signatures and required service lines, specified inspectable reconciliation, made scanner absence fail closed, added the real training interfaces, and required a low-cost provider deletion canary before GPU use.
- **Round 4 — substantive findings applied:** the Code Engine deployment artifact had no executable provider contract, COS-prefix IAM scope was overstated, the bootstrap contained an invalid command and unpinned repository trust, event-chain integrity and safe checkpoint loading were absent, core value objects were still only names, observability could have required a persisted ingestion secret, and task commit/micro-loop boundaries were missing. Replaced YAML with an idempotent CLI deployment script, corrected provider blast-radius claims, fixed and pinned bootstrap/scanning, added event-chain and `weights_only` checks, completed value-object fields, restricted observability to credentialless platform routing, and added exact commit/RED-GREEN boundaries.
- **Round 5 — substantive findings applied:** the authorization gate consumed typed promo, pricing and usage evidence, but no task produced those values or bound them to authoritative artifacts; a hand-created JSON could therefore authorize spend without proof. Added runtime-only console/quote capture validation, IBM Usage Reports normalization, account/currency/pagination checks, injected-client tests and a strict rule that loaders never OCR or infer financial values.
- **Round 6 — substantive finding applied:** source inspection of `ibm-cos-sdk==2.16.2` and the compatible `ibm-cloud-sdk-core==3.28.0` showed that `ibm_boto3` accepts a custom token manager, not `VPCInstanceAuthenticator` directly, and token-authenticated COS also needs its service-instance ID. Added `cos_service_instance_id` to `RunSpec`, the verified `auth.token_manager` client construction and a regression test forbidding API-key/HMAC fallback.
- **Round 7 — clean pass 1:** rechecked spec coverage, producer-to-receipt lifecycle, financial fail-closed rules, IAM/network boundaries, type/signature consistency, AWS removal path and all five Review Focus cases. No substantial improvement or optimization remained.
- **Round 8 — substantive finding applied:** lifecycle closure covered the GPU VSI but could leave its boot volume, outbound Public Gateway and persistent COS/Registry/Logs resources accruing charges after the promotional credit expired. Added per-resource leases, auto-delete plus observed-404 checks for boot volumes, run-owned Public Gateway teardown, watchdog orphan handling and a mandatory authorized post-promo export/delete or funded-retention plan without permitting automatic data destruction.
- **Round 9 — substantive finding applied:** Tasks 5, 8 and 9 lacked the mandatory `Interfaces` blocks from `writing-plans`, leaving runtime image, watchdog and migration handoffs implicit. Added exact consumed contracts and produced artifacts/functions for all three tasks.
- **Round 10 — substantive finding applied:** inspection of the pinned `ibm-cos-sdk-core==2.16.2` service model proved that `PutObject` does not support `IfNoneMatch`, invalidating the proposed conditional event write. Replaced it with never-reused UUID event keys; concurrent events at one sequence are both preserved and the chain validator treats them as a fork that quarantines the run.
- **Round 11 — substantive finding applied:** the image task named a lock file without requiring hash enforcement or proving that PyTorch 2.5.1 was the CUDA 12.4 build for `linux/amd64`. Added `--require-hashes`, transitive artifact hashes, same-base dry-run resolution and runtime assertions for Torch version, CUDA version, L4 identity and usable VRAM.
- **Round 12 — clean pass 1:** reviewed security boundaries and failure injection across evidence capture, trusted-profile auth, COS concurrency, ambiguous VPC create/delete, bootstrap/reboot, signal cleanup, watchdog permissions, per-resource expiry and image supply chain. No substantial improvement or optimization remained.
- **Round 13 — substantive finding applied:** `Label.REJECTION` was owned by both Task 1 and Task 6, so an executor could define the six-class vocabulary twice or invalidate Task 1's tests later. Consolidated `Label`/`ModelClass` ownership and tests in Task 1; Task 6 now only consumes that vocabulary while owning data mapping/training.
- **Round 14 — clean pass 1:** reviewed ownership, task ordering, contract handoffs, migration safety and smallest-complete-system scope. No substantial improvement or optimization remained. One non-substantive wording cleanup assigned `boto3` removal solely to Task 9 so intermediate commits stay runnable.
- **Round 15 — clean pass 2:** re-read the main spec and complete implementation plan from a fresh executor perspective; rechecked every task's files/interfaces, type names, Review Focus test ownership, provider-SDK facts, cost/expiry invariants, deletion proof and forbidden fallback paths. No substantial improvement or optimization remained.
- **Round 16 — substantive findings applied:** independent implementation review proved that a DNS/HTTPS-only security group still allowed a compromised training container to exfiltrate to arbitrary HTTPS destinations, and that user-data was not bound by `RunSpec`. Replaced host networking with a hash-bound local Squid proxy, isolated Docker bridge and `DOCKER-USER` default-drop policy; limited the container to the exact IAM and regional COS hosts; required network-negative tests; and added `bootstrap_sha256` to authorization and prototype validation without adding another metered IBM resource.
- **Round 17 — clean pass 1:** rechecked host-before-dataset ordering, proxy destination scope, metadata access, container capabilities, TLS verification, credential persistence, cost-line completeness and Task 4/5 ownership. No substantial improvement or optimization remained.
- **Round 18 — clean pass 2:** re-read Tasks 1, 4, 5 and 7 as a fresh executor; verified `bootstrap_sha256` type/signature consistency, exact-host allowlisting, direct-egress negative proof, no extra IBM SKU, teardown invariants and the absence of placeholder steps. No substantial improvement or optimization remained.
- **Round 19 — substantive findings applied:** independent SHA review found that the installed trusted-profile authenticator defaults to link-local HTTP while the VSI required HTTPS/hop limit 1; `DOCKER-USER` does not filter traffic addressed to the host; the proxy allowed only the direct COS hostname while `RunSpec` also accepted private; and post-create validation could fail before preserving the billed VSI ID. Aligned metadata with the installed SDK and bridge hop, required host `INPUT` isolation plus bridge-bound Squid, narrowed `RunSpec` to the direct endpoint, and durably recorded the instance ID immediately after create.
- **Round 20 — clean pass 1:** rechecked SDK metadata defaults, bridge routing, host-versus-forwarded firewall chains, exact COS host binding, create-response durability and teardown handoff. No substantial improvement remained.
- **Round 21 — clean pass 2:** re-read Tasks 1, 4, 5 and 7 with the corrected contracts; verified no new IBM resource or paid service, no broader egress, no credential persistence, and a recoverable post-create ID before every later validation. No substantial improvement remained.
- **Round 22 — substantive findings applied:** independent SHA reviews found that reconciled creates skipped durable ID recording, 2xx responses without an ID skipped reconciliation, cleanup context was persisted only after deletion began, polling trusted a response for another VSI, and the bridge plan did not close IPv6. Unified accepted/reconciled create handling, persisted ID and cleanup context before later failures, bound polling to the requested ID, and required IPv6 disablement or equivalent default-drop rules with negative proof.
- **Round 23 — clean pass 1:** rechecked every accepted, lost-response and malformed-response create path, the pre-return cleanup handoff, poll identity binding and dual-stack network boundary. No substantial improvement remained.
- **Round 24 — clean pass 2:** re-read Tasks 4, 5 and 7 after the lifecycle changes; verified one durable ID/cleanup path, no automatic duplicate create/delete, exact prototype ownership and fail-closed IPv4/IPv6 behavior. No substantial improvement remained.
- **Round 25 — substantive findings applied:** independent SHA reviews found that cleanup context could still be absent after an early post-create validation failure, a delete-time 409 stopped reconciliation, tag lookup trusted a CRN belonging to another resource, and unpriced user-managed KMS encryption was accepted. Persisted candidate cleanup context before remote validation while retaining delete-time ownership checks; reconciled 409 by polling; bound every tagged VPC CRN to account/type/ID; and limited the boot volume to provider-managed encryption.
- **Round 26 — clean pass 1:** rechecked early-failure recovery, absent-VSI cleanup, 409/404 transitions, CRN structure and the priced storage contract. No substantial improvement remained.
- **Round 27 — clean pass 2:** re-read create through teardown with corrupted provider records and in-progress deletion responses; verified no unpriced KMS dependency, no cross-resource tag trust and no terminal success before observed 404. No substantial improvement remained.
- **Round 28 — substantive finding applied:** independent SHA review found that ID persistence preceded prototype/duplicate checks but candidate volume/gateway persistence did not, leaving an orphan-cost interval. Unified post-create/adoption persistence so ID and candidate cleanup context are durable together before the first later validation.
- **Round 29 — clean pass 1:** rechecked direct create, timeout reconciliation, 2xx-without-ID reconciliation, existing adoption and duplicate detection; every path now records the same teardown context first. No substantial improvement remained.
- **Round 30 — clean pass 2:** re-read the recovery files and absent-VSI resume path; verified candidate IDs are only used after provider ownership revalidation and no unsafe deletion permission was introduced. No substantial improvement remained.
- **Round 31 — substantive finding applied:** independent SHA review proved that missing tags could fall back to a matching name/prototype intent without binding the current provider ID to the durable accepted-create ID. Required exact ID equality with the tamper-checked created record before intent can authorize cleanup.
- **Round 32 — clean pass 1:** rechecked missing-tag cleanup with same-name replacement records, forged CRNs and mismatched IDs. No substantial improvement remained.
- **Round 33 — clean pass 2:** re-read all ownership fallbacks; verified tags cannot be borrowed across CRNs and local intent cannot authorize a resource other than the exact durably recorded VSI. No substantial improvement remained.
- **Round 34 — substantive findings applied:** independent SHA review found that adoption could overwrite an earlier accepted-create ID, ID and cleanup were separate crash points, and detach-time 409 used one read instead of bounded polling. Made the created record idempotent and immutable, stored ID/volume/gateway atomically in that one record with cleanup fallback, and added deadline-bounded detach polling for transient responses.
- **Round 35 — clean pass 1:** rechecked crash points, same-name replacement adoption, cleanup fallback and detach transition timing. No substantial improvement remained.
- **Round 36 — clean pass 2:** re-read all recovery writes and gateway detach branches; verified no recorded identity can be overwritten, partial context remains sufficient for independently validated cleanup, and transient detach never reports success before observed detachment. No substantial improvement remained.
- **Round 37 — substantive finding applied:** independent SHA review found that deterministic create-time 4xx responses entered ambiguity reconciliation and left the pre-create intent blocking a retry after IAM correction. Restricted reconciliation to transport failures, 408/409/429 and 5xx; deterministic 4xx now remove the unconsumed intent durably and propagate unchanged.
- **Round 38 — clean pass 1:** rechecked create response classification, recovery-directory fsync and retry behavior after authorization correction. No substantial improvement remained.
- **Round 39 — clean pass 2:** re-read every create exception branch; verified accepted or uncertain creates remain single-shot while proven rejections cannot strand an uncreated run. No substantial improvement remained.
- **Round 40 — substantive findings applied:** independent SHA review checked IBM's real CRN examples and found zonal VPC resources use `spec.zone`, not the parent region; it also reproduced a 503 reconciliation read ending after one attempt. Bound instance/subnet/gateway CRNs to the zone and retried bounded transport, 408/409/429 and 5xx reconciliation reads while keeping malformed local/provider records fail-closed.
- **Round 41 — clean pass 1:** rechecked IBM CRN shape, zone/account/type/ID binding and transient-versus-malformed reconciliation classification. No substantial improvement remained.
- **Round 42 — clean pass 2:** re-read every real-SDK/provider response boundary; verified deterministic mismatches stop immediately while temporary read failures cannot abandon an accepted create after one observation. No substantial improvement remained.
- **Round 43 — substantive findings applied:** two independent SHA reviews found that a locally created VSI could become undeletable after post-create prototype drift and that a preexisting Floating IP attached to the Public Gateway could remain allocated after gateway teardown. Allowed exact durable create intent plus ID/CRN to authorize only the mismatched VSI deletion, retained independent validation for every dependent resource, added the dedicated Floating IP to `RunSpec`, pricing, immutable recovery, ownership/CRN checks and observed-404 teardown, including resume after an already deleted gateway.
- **Round 44 — clean pass 1:** rechecked accepted create, post-create rejection, missing tags, replacement IDs, gateway detach, already-absent gateway and Floating IP deletion. Every metered VPC resource now has an authorized ID, independent ownership proof, durable recovery context and an observed-404 terminal state. No substantial improvement remained.
- **Round 45 — clean pass 2:** re-read the financial evidence, RunSpec hash binding, recovery records, SDK boundary and teardown order from a fresh executor perspective. Missing Floating IP price eligibility fails before launch; foreign or malformed Floating IP state fails closed; resume cannot report completion while the dedicated IP remains allocated. No substantial improvement remained.
- **Round 46 — substantive findings applied:** independent final-SHA review found that a successful create response could contain a VSI ID but omit boot-volume context, causing failure before the only durable ID record, and that the plan's `RunSpec` inventory diverged from the implemented dataclass. Added an immutable accepted-ID record before cleanup-context hydration, bounded exact-ID reads for eventually complete provider state, recovery from the accepted record when hydration stays ambiguous, and one exact full `RunSpec` inventory shared by the interface and implementation snippet.
- **Round 47 — clean pass 1:** rechecked direct create, lost-response reconciliation, partial 2xx responses, eventual 404/transport reads, stale accepted IDs and concurrent controllers. Every accepted VSI ID is durable before another provider read; incomplete cleanup context blocks recreate and remains recoverable. No substantial improvement remained.
- **Round 48 — clean pass 2:** re-read the complete `RunSpec`, evidence hash binding, accepted/created/cleanup recovery chain and mismatch teardown as a fresh executor. The plan and dataclass fields match exactly, no nullable cleanup ID authorizes deletion, and dependent resources retain independent provider ownership validation. No substantial improvement remained.
- **Round 49 — substantive findings applied:** two independent final-SHA reviews found that trusted-profile/network/storage validation failures could still strand a locally created VSI, a missing nested Floating IP could be mistaken for an absent gateway, and partial-create recovery with no volume ID could abort before cleaning known network resources. Exact intent plus accepted ID and instance CRN now authorize deletion of the created VSI across every post-create validation failure; gateway existence is validated separately from Floating IP existence; accepted recovery stores known network IDs immediately; and an unknown boot-volume ID is recovered through a fully validated run-tag inventory that must end empty.
- **Round 50 — clean pass 1:** rechecked every instance validation failure, partial create/read exhaustion, instance disappearance, volume discovery pagination, gateway detach, missing nested Floating IP and already-absent gateway. Each mutation requires its own exact ownership proof and one resource's absence cannot falsely prove another resource absent. No substantial improvement remained.
- **Round 51 — clean pass 2:** re-read creation, immutable recovery and teardown from the perspective of both billing containment and destructive safety. Accepted identity is durable before hydration, only exact run-owned volumes are discoverable/deletable, gateway and Floating IP 404 states are distinguished, and no provider absence is inferred from another resource's response. No substantial improvement remained.
- **Round 52 — substantive finding applied:** independent final-SHA review reproduced an eventually consistent volume inventory that listed an ID whose detail read already returned 404, aborting before gateway and Floating IP cleanup. Discovery now treats detail 404 as an already absent candidate, retains duplicate-ID detection across all pages, repeats the authoritative inventory check and continues independently owned network cleanup.
- **Round 53 — clean pass 1:** rechecked stale volume-list entries, deleted-volume detail reads, duplicate IDs, pagination and final empty inventory before network teardown. A stale volume entry cannot block known gateway or Floating IP cleanup and cannot be mistaken for a deletable object. No substantial improvement remained.
- **Round 54 — clean pass 2:** re-read all absence transitions across VSI, volume, gateway and Floating IP. Every 404 is scoped to the resource operation that produced it; no nested or stale response proves another resource deleted. No substantial improvement remained.
- **Round 55 — substantive findings applied:** two independent final-SHA reviews found that cleanup context could overwrite the original accepted instance identity and that a stale subnet reference could block Floating IP cleanup after the gateway itself reached 404. Cleanup context is now immutable and rejects any differing rewrite; gateway-detail 404 is terminal for that gateway even when subnet propagation lags, skips detach mutation and continues independently validated Floating IP cleanup.
- **Round 56 — clean pass 1:** rechecked accepted/created/cleanup record precedence, replacement IDs, corrupted JSON and idempotent rewrites. No later provider observation can replace the original cleanup identity or make the loader ignore the valid immutable record. No substantial improvement remained.
- **Round 57 — clean pass 2:** re-read gateway/subnet/Floating IP transitions under eventual consistency. A stale subnet reference cannot resurrect a deleted gateway, trigger an unsafe detach, or block inspection and deletion of the independently identified Floating IP. No substantial improvement remained.
- **Round 58 — substantive findings applied:** independent final-SHA review found that a cleanup record could target a replacement VSI despite an immutable accepted ID, a subnet 404 could block independent gateway/Floating IP cleanup, a reassociated Floating IP could be deleted without validating its live target, ambiguous volume discovery stopped cleanup of independently owned network resources, and a later VSI read omitting CRN could strand an already accepted create. Bound cleanup to the accepted ID whenever that record exists, treated subnet absence independently, required the exact authorized gateway target before Floating IP mutation, deferred ambiguous-volume failure until after network cleanup, and persisted the validated VSI CRN for later teardown.
- **Round 59 — clean pass 1:** rechecked accepted/created/cleanup identity binding, durable CRN fallback, subnet absence, Floating IP reassociation and ambiguous-volume isolation. Each mutation remains independently authorized and one unresolved resource no longer prevents safe cleanup of another. No substantial improvement remained.
- **Round 60 — clean pass 2:** re-read create recovery and teardown from crash-safety, destructive-safety and billing-containment perspectives. Cleanup cannot switch VSI identity, missing provider fields cannot erase durable ownership evidence, and network cleanup continues without weakening volume validation. No substantial improvement remained.
- **Round 61 — substantive findings applied:** two independent final-SHA reviews found that a known-volume failure still aborted independent network cleanup, a transient Floating IP read was re-raised after gateway deletion without retry, and an accepted create with volume but no CRN could return before durable recovery existed. Cleanup now reserves deadline for later resources and aggregates volume, gateway and Floating IP failures only after attempting every independently authorized teardown; transient Floating IP reads are retried while ownership mismatches remain fail-closed; and the immutable accepted account/zone/ID identity deterministically yields the deletion CRN while successful hydration still waits for the provider CRN.
- **Round 62 — clean pass 1:** rechecked malformed versus absent CRNs, accepted-record immutability, known and discovered volume failures, gateway read/deadline failures, permanent Floating IP ownership mismatches and transient provider reads. Each resource retains independent cleanup without authorizing deletion from an unbound identity. No substantial improvement remained.
- **Round 63 — clean pass 2:** re-read the accepted-create and teardown paths under partial responses, retry exhaustion, deadline exhaustion and eventual consistency. A missing provider field cannot erase the durable deletion address, deterministic mismatches are never retried into authorization, and one resource's failure no longer strands another independently validated resource. No substantial improvement remained.
- **Round 64 — substantive findings applied:** two independent final-SHA reviews found that retryable DELETE responses were followed only by GET polling, gateway cleanup could consume the Floating IP deadline, multiple teardown failures exposed only the first error, transient provider reads ended polling too early, and a Floating IP reassociation race remained between validation and DELETE. DELETE is now reissued after retryable outcomes until 404 or its resource deadline; lifecycle reads retry bounded transient states; VSI, volume, gateway and Floating IP have reserved deadline quarters; the Floating IP is revalidated immediately before every DELETE; and simultaneous resource failures are returned as one exception group.
- **Round 65 — clean pass 1:** rechecked retry classification, repeated DELETE semantics, 404 confirmation, per-resource deadline boundaries, exact target revalidation and multi-error preservation. Deterministic ownership mismatches remain non-retryable and no stale validation authorizes a Floating IP mutation. No substantial improvement remained.
- **Round 66 — clean pass 2:** re-read teardown under 409/429/5xx, exhausted deadlines, gateway/Floating IP races and concurrent independent failures. Every mutation is bounded, later resources retain an execution window, and the caller receives every unresolved metered-resource state. No substantial improvement remained.
- **Round 67 — substantive findings applied:** two independent final-SHA reviews found that a Floating IP could be reassociated after the first DELETE conflict but before a retry, and that built-in `ExceptionGroup` violated the declared Python 3.10 minimum. Added a pre-delete callback executed before every Floating IP DELETE attempt, so each retry revalidates exact ownership and target, and replaced the 3.11-only built-in with a project-owned `CleanupFailures` carrying the complete immutable tuple of errors.
- **Round 68 — clean pass 1:** rechecked first-attempt and retry-time reassociation, transient DELETE/read ordering, deterministic target mismatch and Python 3.10 syntax/runtime ownership. No retry can reuse stale Floating IP authorization and no unsupported built-in remains. No substantial improvement remained.
- **Round 69 — clean pass 2:** re-read every multi-error and Floating IP mutation path. Individual errors remain directly raised, multiple errors remain inspectable in order through `CleanupFailures.errors`, and every DELETE attempt has a fresh exact-target proof. No substantial improvement remained.
- **Round 70 — substantive finding applied:** two independent final-SHA reviews found that a transient provider failure inside the per-attempt Floating IP validator escaped before the DELETE retry loop could act. The callback now treats 404 as terminal absence, retries timeout/409/429/5xx within the same resource deadline, and still fails immediately on ownership mismatch or deterministic 4xx before any mutation.
- **Round 71 — clean pass 1:** rechecked callback outcomes before first DELETE and every retry: absent, transient, deterministic denial, malformed ownership and reassociated target. Only fresh exact ownership reaches mutation. No substantial improvement remained.
- **Round 72 — clean pass 2:** re-read validation, pause, DELETE and GET ordering under repeated transient failures. The Floating IP retains its reserved deadline, transient reads cannot strand it after one attempt, and no retry bypasses target validation. No substantial improvement remained.
- **Round 73 — substantive findings applied:** two independent final-SHA reviews found that a 404 from the separate Global Tagging API could be mistaken for a VPC resource 404 and persist a false deletion receipt, and that recovery identity did not bind the IBM account before deriving a CRN. Direct VPC GET 404 now becomes a distinct `ResourceAbsent` signal; tag-service 404 remains retryable/fail-closed and can never prove deletion. Intent, accepted, created and cleanup records now bind `account_id`, so a recovery cannot switch accounts.
- **Round 74 — clean pass 1:** rechecked every 404 source across subnet, gateway, Floating IP, tag pagination, DELETE confirmation and recovery loading. Only the direct resource GET establishes absence; auxiliary-service failures cannot create a terminal receipt. No substantial improvement remained.
- **Round 75 — clean pass 2:** re-read immutable identity under account, zone, ID, prototype and CRN changes. Account drift fails before derived-CRN use or mutation, while same-account partial-create recovery remains available. No substantial improvement remained.
- **Round 76 — substantive findings applied:** independent final-SHA reviews found that a gateway could remain billable on resume after its Floating IP had already reached 404, and that VSI, volume and gateway DELETE retries reused stale ownership validation. Gateway ownership now permits only an absent nested Floating IP after the direct Floating IP GET proved 404, while any foreign ID remains rejected. VSI, volume and gateway now use per-attempt pre-delete validators; gateway retries also recheck an empty attachment inventory.
- **Round 77 — clean pass 1:** rechecked resume with missing versus foreign gateway Floating IP links, and retry-time changes to instance tags, volume ownership tags and gateway subnet attachments. Every retry obtains fresh provider ownership immediately before mutation. No substantial improvement remained.
- **Round 78 — clean pass 2:** re-read dependency order and independent teardown after partial prior success. A confirmed-absent Floating IP cannot strand its gateway, and a changed VSI, volume or gateway cannot be deleted from stale authorization. No substantial improvement remained.
- **Round 79 — substantive finding applied:** an independent final-SHA review found that local intent allowed fallback to the durable instance CRN even when the live provider record contained a different non-empty CRN. Initial cleanup and every retry now use the durable CRN only when the live CRN field is absent; any present account/zone/type/ID mismatch blocks before DELETE.
- **Round 80 — clean pass 1:** rechecked absent, valid and conflicting live CRNs under local intent, missing tags, initial cleanup and retry. Missing data remains recoverable, while contradictory data is never normalized into authorization. No substantial improvement remained.
- **Round 81 — clean pass 2:** re-read durable identity fallback against replacement-account and malformed-provider records. The accepted ID can supply missing identity evidence but cannot override a present conflict. No substantial improvement remained.
- **Round 82 — substantive findings applied:** independent final-SHA reviews found that a present but empty/null/non-string CRN was still treated as absent, that pre-delete validation could consume the remaining deadline before mutation, and that one discovered-volume failure stopped later owned volumes. Fallback now requires the CRN key to be absent; any present value must validate. Every loop rechecks the deadline immediately before mutation, and discovered volumes are attempted independently with final authoritative inventory reconciliation.
- **Round 83 — clean pass 1:** rechecked missing-key versus malformed-value semantics in initial cleanup and retries, deadline expiry during validation, and multi-volume partial failures. No malformed live identity reaches DELETE, no mutation occurs after its resource deadline, and a failed volume cannot suppress later attempts. No substantial improvement remained.
- **Round 84 — clean pass 2:** re-read final volume receipts and error aggregation. The empty-inventory receipt is written only after an authoritative empty list; unresolved inventories retain their errors without a false terminal record. No substantial improvement remained.
- **Round 85 — substantive findings applied:** two independent final-SHA reviews found that volume discovery still validated inline and could abort before later IDs, while an empty tag-filtered final list cleared prior per-ID failures even if a volume merely lost its tag. Discovery now collects every unique paginated ID without detail validation; cleanup validates and attempts each independently. A failed ID is resolved only by direct `get_volume(id)` 404, never by absence from the tag-filtered inventory.
- **Round 86 — clean pass 1:** rechecked multi-page discovery, first-ID mismatch, later-ID deletion, tag loss, stale list entries and direct per-ID 404 reconciliation. Every discovered ID reaches an independent decision and no tag query substitutes for resource absence. No substantial improvement remained.
- **Round 87 — clean pass 2:** re-read volume terminal receipts and error ordering. The inventory-empty receipt requires both an empty final list and no unresolved per-ID failures; otherwise all unresolved failures remain visible. No substantial improvement remained.
- **Round 88 — substantive findings applied:** two independent final-SHA reviews found that a stale non-empty volume page could be converted into an authoritative-empty receipt after per-ID 404s, and that duplicated tag names or subnet IDs across pages could hide conflicting ownership or gateway attachments. Final volume reconciliation now re-lists with bounded backoff until the provider returns an actually empty inventory; tag and subnet pagination reject missing or repeated identities before authorization.
- **Round 89 — clean pass 1:** rechecked stale volume pages, direct per-ID 404s, deadline exhaustion, receipt timing and duplicated ownership identities across page boundaries. No stale list can produce an empty-inventory receipt, and pagination cannot deduplicate conflicting evidence into authorization. No substantial improvement remained.
- **Round 90 — clean pass 2:** re-read volume, tag and subnet reconciliation from billing-containment and destructive-safety perspectives. Terminal evidence requires the provider's empty list, while malformed or repeated inventory remains fail-closed without broadening any delete permission. No substantial improvement remained.
- **Round 91 — substantive findings applied:** two independent final-SHA reviews found that a valid boot volume first visible during final reconciliation was reported but never deleted, consecutive empty tag-index reads had no stabilization delay, mutable pagination counts could hide ownership or gateway attachments, and the SDK's default request timeout could consume later resource windows. Final reconciliation now runs every newly visible volume through the same validated delete path, requires two empty observations separated by backoff, rejects page-count drift across every VPC/tag inventory, and caps each provider request to the smaller of five seconds or its current resource deadline.
- **Round 92 — clean pass 1:** rechecked late volume discovery, stale/non-empty and consecutive-empty inventories, tag loss, mutable page counts, per-call timeouts, deadline reservations and direct-GET 404 evidence. Newly visible owned volumes are deleted, terminal inventory requires a stabilized empty provider view, and no individual SDK call can inherit the sixty-second default timeout. No substantial improvement remained.
- **Round 93 — clean pass 2:** re-read teardown from partial-create, eventual-consistency, pagination-race and blocked-provider perspectives. Destructive authorization remains fail-closed, later billable resources retain bounded execution time, and final receipts cannot precede a stable empty inventory. No substantial improvement remained.
- **Round 94 — substantive findings applied:** two independent final-SHA reviews found that instance, volume and gateway ownership callbacks still skipped the first DELETE attempt; volume discovery depended on an eventually consistent tag-filtered list; and discovered failure IDs existed only in memory. Every DELETE attempt now revalidates ownership immediately before mutation, discovery scans the complete zone inventory and filters exact run tags locally, and every candidate ID is durably recorded before validation/deletion then retained until direct GET 404.
- **Round 95 — clean pass 1:** rechecked first-attempt ownership/attachment races, unfiltered volume discovery, tag-index delay, crash between discovery and error return, tag loss after discovery and resume from durable candidate IDs. No mutation uses stale authorization, and a previously seen volume cannot disappear from cleanup merely because a later list or process does not remember it. No substantial improvement remained.
- **Round 96 — clean pass 2:** re-read terminal receipts and retry ordering under process death, provider-list lag and concurrent ownership changes. A receipt requires stabilized empty unfiltered inventory plus an empty durable candidate set; all known IDs independently reach direct 404, while changed resources fail closed before DELETE. No substantial improvement remained.
- **Round 97 — substantive findings applied:** two independent final-SHA reviews found that same-count tag and volume pagination could combine changing snapshots, first-page volume candidates were not durable until the last page, a corrupt candidate ledger blocked provider cleanup, request timeouts leaked through a shared SDK client, instance-receipt failure stopped dependent teardown, and multiple volume receipts overwrote one another. Destructive tag authorization now requires two identical complete snapshots; terminal volume evidence requires identical complete zone inventories; candidates persist page by page; corrupt-ledger errors are retained while provider cleanup continues; provider calls are serialized with timeout restoration; receipt failures aggregate after independent cleanup; and volume receipt names include the resource-ID hash.
- **Round 98 — clean pass 1:** rechecked same-count shifts across tag and volume pages, crash after the first volume page, corrupt/truncated local evidence, concurrent SDK-client use, receipt fsync failure and multiple run-owned volumes. Ownership snapshots cannot be mixed into authorization, every observed candidate is durable before the next page, and evidence failures no longer strand independently deletable metered resources. No substantial improvement remained.
- **Round 99 — clean pass 2:** re-read first and retry DELETE paths plus terminal evidence under concurrent provider mutation and local I/O failure. Every destructive authorization uses stable ownership, complete inventory stabilization includes unrelated zone IDs, SDK timeout state is restored, and per-resource deletion evidence remains distinct. No substantial improvement remained.
- **Round 100 — substantive findings applied:** two independent final-SHA reviews found that waiting for the shared SDK lock could expire a deadline before mutation, timeout injection dropped active HTTP controls, corrupt or unwritable candidate ledgers could abort discovered-volume cleanup, one volume receipt failure stopped later volumes, an initial transient VSI read stopped every dependent cleanup, Floating-IP receipt failure masked prior errors, and subnet pagination lacked stable snapshots. Deadlines are now recalculated under the provider-call lock; timeout config is merged and restored; corrupt ledgers are quarantined and write failures retained without stopping validated deletion; receipt failures aggregate after independent resources; initial VSI reads reconcile within their reserved window and fall back to durable cleanup context; and destructive subnet checks require identical complete snapshots.
- **Round 101 — clean pass 1:** rechecked lock waiting past deadline, proxy/TLS config preservation, disk-full and corrupt-ledger paths, two-volume receipt failure, transient initial VSI reads, Floating-IP evidence failure and same-count subnet churn. No late mutation reaches the provider, SDK controls remain active, and one evidence failure cannot suppress cleanup of another billable resource. No substantial improvement remained.
- **Round 102 — clean pass 2:** re-read all provider-call, local-persistence and pagination boundaries from destructive-safety and billing-containment perspectives. Mutations require current time plus stable ownership/attachment evidence; provider and local failures remain aggregated after every independently safe teardown attempt. No substantial improvement remained.
- **Round 103 — substantive findings applied:** one final independent review found that instance DELETE failure still stopped dependent cleanup, exhausted initial VSI reads were never retried after the reserved quarter, a corrupt cleanup record blocked valid immutable fallbacks, and lifecycle lease acquisition could exceed the public deadline. Instance deletion errors now aggregate after volume/network teardown; unknown VSI state is retried through the overall deadline and successful late deletion re-runs failed volume cleanup; corrupt cleanup records are quarantined while created/accepted fallbacks continue; and lease acquisition uses deadline-aware nonblocking flock retries.
- **Round 104 — clean pass 1:** rechecked persistent instance DELETE denial, quarter-window read exhaustion followed by provider recovery, corrupt cleanup plus valid created fallback, late instance deletion with attached volume retry, and a lease held beyond deadline. Independent resources are still attempted, exact-ID VSI cleanup regains the remaining window, corrupted evidence remains reportable, and callers cannot wait past their authorized deadline for a local lock. No substantial improvement remained.
- **Round 105 — clean pass 2:** re-read instance, volume, gateway and Floating-IP teardown ordering under simultaneous provider, lock and local-ledger failures. Every safely recoverable resource is attempted before aggregated failure returns, without weakening exact ownership or mutation deadlines. No substantial improvement remained.
- **Round 106 — substantive findings applied:** final reviews found that the provider-call lock itself could wait past deadline, a corrupt accepted ledger blocked valid created/cleanup fallbacks, and a successful late VSI reconciliation only retried volume cleanup after an earlier error. Provider-lock acquisition now uses the remaining deadline; corrupt accepted evidence is quarantined and reported while immutable fallbacks continue; and every late VSI resolution triggers a fresh full volume reconciliation, including paths that previously recorded an empty inventory.
- **Round 107 — clean pass 1:** rechecked a stuck cross-run provider call, unreadable accepted identity with valid created/cleanup records, and a boot volume becoming visible only after the VSI is resolved late. Calls fail at their own deadline, corruption cannot suppress independently validated teardown, and late VSI closure invalidates and repeats prior volume conclusions. No substantial improvement remained.
- **Round 108 — clean pass 2:** re-read deadline acquisition, identity fallback and late-discovery ordering against orphaned GPU and volume billing. The final state preserves exact ownership, bounded waiting and post-VSI volume proof without adding provider resources. No substantial improvement remained.
- **Round 109 — substantive findings applied:** two independent SHA reviews found that the shared SDK lock could still wait past the public deadline, a syntactically valid but inconsistent accepted ledger blocked valid immutable cleanup fallbacks, and a known volume returning 404 could suppress discovery of a different late-visible boot volume. SDK-lock acquisition now uses the remaining deadline; every rejected accepted ledger is quarantined and retained as an error while created/cleanup fallbacks continue; and a known-volume 404 triggers a complete zone reconciliation before terminal evidence.
- **Round 110 — clean pass 1:** rechecked SDK-lock contention, valid-but-conflicting accepted records, direct known-volume 404, late zone inventory visibility and independent network cleanup. Calls remain deadline-bounded, contradictory local evidence cannot authorize deletion, and late volumes cannot disappear behind stale known IDs. No substantial improvement remained.
- **Round 111 — clean pass 2:** re-read teardown from durable identity loading through final volume and network receipts, including repeated empty inventories and late VSI absence reconciliation. Every mutation still requires exact current ownership, every billable volume receives full discovery, and local evidence failures remain visible without suppressing independently safe cleanup. No substantial improvement remained.
- **Round 112 — substantive findings applied:** two independent SHA reviews found that local-only launch locking allowed two controllers to create duplicate GPU VSIs, create/hydration calls had no shared deadline, subnet gateway detachment could race and detach a replacement, and unrelated zone-volume churn could block a correct teardown. Added one exact private primary IP to the immutable prototype so IBM's exclusive reserved-IP binding is the provider-side create mutex; bounded the create lease, calls and retries; made the subnet run-owned and delete it instead of calling the gateway-unset endpoint; and stabilize only the complete run-owned volume inventory after each internally consistent zone scan.
- **Round 113 — substantive self-review finding applied:** deleting the run-owned subnet and its Public Gateway inside one prior gateway window could let subnet retries consume all time and strand the billable gateway. Split the remaining gateway window at entry so subnet deletion receives only the first half and Public Gateway deletion retains the second half; the Floating IP still keeps its independent final window.
- **Round 114 — clean pass 1:** rechecked provider-side create uniqueness, conflict adoption, exact primary-IP prototype matching, bounded local/provider waits, subnet ownership, replacement-gateway races and reserved teardown windows. A second controller cannot allocate the same primary IP, and deletion of the run-owned subnet never uses an unconditioned gateway-detach mutation. No substantial improvement remained.
- **Round 115 — clean pass 2:** re-read create through recovery and teardown under lost responses, 409 conflicts, process separation, provider lag, unrelated volume churn and subnet/gateway partial success. Every metered resource retains a bounded cleanup opportunity, exact ownership remains mandatory, and no added paid IBM service or fallback is required. No substantial improvement remained.
- **Round 116 — substantive findings applied:** two independent SHA reviews found that one pre-delete VSI 404 could create a false terminal receipt during delayed create visibility, syntactically valid but inconsistent cleanup evidence blocked immutable fallbacks, a Floating IP probe could consume the gateway window, an active VSI delete failure did not block destructive network cleanup, the auto-deleted fixed IP allowed sequential cross-controller replay, and generic `is_private` accepted unusable addresses. VSI absence now requires two consecutive reads before dependent teardown; deterministic VSI failure returns before volume/network mutation; every invalid ledger is quarantined while valid created/accepted evidence continues; the Floating IP probe has its own earlier deadline; the primary IP is RFC1918-only and remains reserved until deletion of the run-owned subnet, which is the durable provider tombstone against replay.
- **Round 117 — clean pass 1:** rechecked delayed create visibility, repeated 404 evidence, deterministic instance denial, cleanup retries, semantic ledger corruption, probe/gateway deadline isolation and simultaneous versus sequential controllers. No dependent mutation precedes confirmed VSI absence, and no second controller can reuse the run IP while the authorized subnet still exists. No substantial improvement remained.
- **Round 118 — clean pass 2:** re-read the complete provider lifecycle with the official reserved-IP and subnet deletion semantics, including failed create, accepted-lost-response, VSI deletion, retained reservation and final subnet removal. The reservation blocks replay while useful, subnet deletion removes it and invalidates the authorized subnet ID, and every remaining metered resource keeps independent cleanup/error evidence after VSI 404. No substantial improvement remained.
- **Round 119 — substantive findings applied:** two independent SHA reviews found that a single pre-delete 404 could still create false terminal evidence for a known volume, run subnet, Public Gateway or Floating IP during provider consistency delay. Every first 404 before any DELETE now requires a second consecutive 404 separated by bounded backoff; if the resource reappears, ownership validation and deletion resume. A 404 after a DELETE attempt remains terminal because the mutation has already been issued.
- **Round 120 — clean pass 1:** rechecked initial versus post-mutation 404 handling for VSI, boot volumes, subnet, Public Gateway and Floating IP, including a resource reappearing between reads. No pre-delete terminal receipt can be produced from one absence observation, and no added provider resource or paid service is required.
- **Round 121 — clean pass 2:** re-read all deletion paths against eventual consistency, retry deadlines, per-attempt ownership checks and independent cleanup ordering. Two separated absence observations are required before skipping an unattempted DELETE, while an observed 404 after an attempted DELETE remains the smallest sufficient terminal proof. No substantial improvement or optimization remained.
- **Round 122 — substantive findings applied:** two independent final-SHA reviews found that socket inactivity timeouts did not bound a slow-trickle response, an accepted VSI could be deleted before its missing boot-volume ID was resolved, Floating-IP absence confirmation errors escaped before independent network cleanup, deletion-receipt I/O failures could strand the Public Gateway, and an already confirmed absent subnet was re-probed. Provider calls now run synchronously on the controller main thread under a hard Unix alarm plus the SDK timeout; accepted cleanup resolves and persists the exact attachment before VSI deletion; probe and receipt failures are aggregated after independent safe mutations; and confirmed absence is reused without redundant reads.
- **Round 123 — clean pass 1:** rechecked slow response bodies, deadline and SDK-lock restoration, separate controller processes, partial accepted records, attachment ownership, `404 -> deterministic error` probes, receipt fsync failures and subnet/gateway ordering. No call can outlive its total deadline, no known boot volume is replaced by an empty-list inference, and evidence failure does not suppress a later independently safe mutation. No substantial improvement remained.
- **Round 124 — clean pass 2:** re-read create and teardown from accepted-ID persistence through volume, subnet, Public Gateway and Floating-IP terminal evidence. Hard deadlines fail closed outside the synchronous controller contract; every destructive call retains current ownership checks; and the correction adds no IBM resource, paid provider or speculative abstraction. No substantial improvement or optimization remained.
- **Round 125 — non-substantive preflight correction:** the canonical type check rejected a nullable tuple inference and a default-argument lambda. Added the already-proven non-null assertion and used the enclosing volume ID directly; behavior and lifecycle contracts are unchanged.
