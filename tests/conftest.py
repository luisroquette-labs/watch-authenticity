"""Shared factories for IBM training-plane contract tests."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from PIL import Image, PngImagePlugin

from acquisition.collect import collect_batch
from acquisition.source_policy import authorize_candidate, get_source_policy, load_source_policies
from acquisition.types import (
    Angle,
    CandidateRecord,
    EvidenceRecord,
    EvidenceTier,
    RightsRecord,
    RightsStatus,
    TrainingRecord,
)
from core.types import Label
from dataset.reference_catalog import (
    ReferenceCatalog,
    catalog_document_sha256,
    parse_reference_catalog,
)
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import (
    CreditEvidence,
    ObjectRef,
    PostPromoEntry,
    PostPromoPlan,
    PriceLine,
    PricingEvidence,
    RunSpec,
    UsageEvidence,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
_SAFE_USER_DATA = "#!/bin/bash\nset -euo pipefail\n/opt/watch-auth/bootstrap-host\n"


class _ReferenceCatalogFixture:
    def __init__(self):
        self.document = json.loads(Path("config/reference_catalog.json").read_text())
        self.catalog = self._parse(self.document)
        self.admitted_document = self._admit_submariner(self.document)
        self.admitted_catalog = self._parse(self.admitted_document)

    @staticmethod
    def _parse(
        document: dict[str, Any], previous: ReferenceCatalog | None = None
    ) -> ReferenceCatalog:
        return parse_reference_catalog(
            document,
            expected_sha256=catalog_document_sha256(document),
            previous=previous,
        )

    @staticmethod
    def _admit_submariner(document: dict[str, Any]) -> dict[str, Any]:
        admitted = copy.deepcopy(document)
        admitted["references"][0].update(
            {
                "identity_evidence_sha256": "1" * 64,
                "gold_freeze_sha256": "2" * 64,
                "holdout_freeze_sha256": "3" * 64,
                "status": "admitted",
                "head": {
                    "reference_id": "rolex-submariner-126610ln",
                    "generation": 1,
                    "checkpoint_sha256": "4" * 64,
                    "calibration_sha256": "5" * 64,
                },
            }
        )
        return admitted

    def load(self, case: str) -> ReferenceCatalog:
        admitted_cases = {"changed_angles", "revoked_admission", "wrong_reference_checkpoint"}
        document = copy.deepcopy(
            self.admitted_document if case in admitted_cases else self.document
        )
        previous = None
        if case == "alias_collision":
            document["references"][1]["aliases"] = ["rolex-submariner-126610ln"]
        elif case == "generation_mismatch":
            document["generation"] = 2
        elif case == "changed_angles":
            document["references"][0]["required_angles"] = ["dial"]
            previous = self.admitted_catalog
        elif case == "revoked_admission":
            document["references"][0]["status"] = "pending"
            document["references"][0]["head"] = None
            previous = self.admitted_catalog
        elif case == "wrong_reference_checkpoint":
            document["references"][0]["head"]["reference_id"] = "rolex-datejust-126334"
        expected = "0" * 64 if case == "catalog_hash_mismatch" else catalog_document_sha256(document)
        return parse_reference_catalog(document, expected_sha256=expected, previous=previous)


@pytest.fixture
def catalog_fixture():
    return _ReferenceCatalogFixture()


class _Response:
    def __init__(self, body: bytes, content_type: str, status_code: int = 200, length=None):
        self.body = body
        self.status_code = status_code
        self.headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body) if length is None else length),
        }

    def iter_content(self, chunk_size: int):
        yield from (self.body[index : index + chunk_size] for index in range(0, len(self.body), chunk_size))

    def close(self):
        return None


class _NotFound(Exception):
    def __init__(self):
        self.response = {"Error": {"Code": "NoSuchKey"}}


class _CollectorCOS:
    def __init__(self):
        self.bodies: dict[str, bytes] = {}
        self.metadata: dict[str, dict[str, str]] = {}
        self.write_count = 0
        self.receipt_write_barrier = None
        self._watch_auth_bucket = "private"
        self._watch_auth_prefix = "datasets"
        self._watch_auth_configuration_verified = True

    def head_object(self, *, Bucket: str, Key: str):
        del Bucket
        if Key not in self.bodies:
            raise _NotFound
        return {
            "ContentLength": len(self.bodies[Key]),
            "Metadata": self.metadata[Key],
            "ETag": f'"etag-{hashlib.sha256(self.bodies[Key]).hexdigest()}"',
        }

    def get_object(self, *, Bucket: str, Key: str):
        del Bucket
        if Key not in self.bodies:
            raise _NotFound
        return {"Body": io.BytesIO(self.bodies[Key])}

    def list_objects_v2(self, *, Bucket: str, Prefix: str):
        del Bucket
        return {
            "Contents": [{"Key": key} for key in sorted(self.bodies) if key.startswith(Prefix)],
            "IsTruncated": False,
        }

    def put_object(
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        Metadata=None,
        ContentType: str | None = None,
    ):
        del Bucket, ContentType
        self.bodies[Key] = Body
        self.metadata[Key] = Metadata or {}
        self.write_count += 1
        if Key.startswith("datasets/receipts/") and self.receipt_write_barrier is not None:
            self.receipt_write_barrier.wait(timeout=5)

    def upload_file(self, filename: str, bucket: str, key: str, ExtraArgs):
        del bucket
        self.bodies[key] = Path(filename).read_bytes()
        self.metadata[key] = ExtraArgs["Metadata"]
        self.write_count += 1

    def download_file(self, bucket: str, key: str, filename: str):
        del bucket
        Path(filename).write_bytes(self.bodies[key])

    def copy_object(self, *, Bucket: str, Key: str, CopySource, MetadataDirective: str):
        del Bucket, MetadataDirective
        source = CopySource["Key"]
        self.bodies[Key] = self.bodies[source]
        self.metadata[Key] = self.metadata[source]

    def delete_object(self, *, Bucket: str, Key: str):
        del Bucket
        self.bodies.pop(Key, None)
        self.metadata.pop(Key, None)


def _image_bytes(fmt: str = "PNG", *, metadata: bool = False, frames: int = 1) -> bytes:
    output = io.BytesIO()
    images = [Image.new("RGB", (8, 8), color) for color in ("red", "blue")[:frames]]
    pnginfo = None
    if metadata:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("contributor", "Private Person")
        pnginfo.add_text("serial", "12345678")
    images[0].save(
        output,
        format=fmt,
        save_all=frames > 1,
        append_images=images[1:],
        pnginfo=pnginfo,
    )
    return output.getvalue()


@pytest.fixture
def jpeg_with_gps() -> bytes:
    exif = Image.Exif()
    exif[0x010E] = "Private contributor"
    exif[0x8825] = {
        1: "N",
        2: (33.9249, 0.0, 0.0),
        3: "E",
        4: (18.4241, 0.0, 0.0),
    }
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="JPEG", exif=exif)
    return output.getvalue()


@pytest.fixture
def cos():
    return _CollectorCOS()


@pytest.fixture
def http_client():
    return MagicMock()


@pytest.fixture
def allowed_policy(policy_set):
    return get_source_policy(policy_set, "partner-upload")


@pytest.fixture
def blocked_policy(policy_set):
    return get_source_policy(policy_set, "reddit")


@pytest.fixture
def collector_fixture(allowed_policy, rights, http_client, cos):
    class CollectorFixture:
        def __init__(self):
            self.http_client = http_client
            self.statuses = [200]

        def set_statuses(self, *statuses: int):
            self.statuses = list(statuses)

        def collect(self, payload_case: str):
            body = _image_bytes(metadata=payload_case == "metadata")
            content_type = "image/png"
            length = None
            expected_sha256 = hashlib.sha256(body).hexdigest()
            if payload_case == "partial":
                length = len(body) + 10
            elif payload_case == "hash_mismatch":
                expected_sha256 = "0" * 64
            elif payload_case == "oversized":
                length = 25 * 1024 * 1024 + 1
            elif payload_case == "non_image":
                body, content_type = b"<html>error</html>", "text/html"
            elif payload_case == "mime_magic_mismatch":
                content_type = "image/jpeg"
            elif payload_case == "truncated":
                body = body[:-12]
            elif payload_case == "polyglot":
                body += b"<script>alert(1)</script>"
            elif payload_case == "multi_frame":
                body = _image_bytes(frames=2)
            elif payload_case == "decompression_bomb":
                body = bytearray(body)
                body[16:24] = (10001).to_bytes(4, "big") + (10000).to_bytes(4, "big")
                body = bytes(body)

            responses = [
                _Response(body, content_type, status_code=status, length=length)
                for status in self.statuses
            ]
            self.http_client.get.side_effect = responses
            record = {
                "source_record_id": "record-1",
                "source_url": "https://partner.example/record-1",
                "image_url": "https://partner.example/image-1.png",
                "declared_reference": "rolex-submariner-126610ln",
                "expected_sha256": expected_sha256,
            }
            return collect_batch(
                [record],
                allowed_policy,
                rights,
                http_client=self.http_client,
                cos_client=cos,
                bucket="private",
                batch_id="b1",
                now=NOW,
            )

    return CollectorFixture()


@pytest.fixture
def candidate() -> CandidateRecord:
    return CandidateRecord(
        candidate_id="candidate-1",
        acquisition_batch_id="batch-1",
        source_policy_id="policy-1",
        source_record_id="listing-1",
        source_url="https://partner.example/listing-1",
        collected_at=NOW,
        declared_reference="rolex-submariner-126610ln",
        proposed_label=Label.GENUINE,
        proposed_grade=None,
        factory=None,
        price_usd=Decimal("12000.00"),
        raw_sha256="a" * 64,
        derivative_sha256="b" * 64,
        phash="0123456789abcdef",
        rights_record_id="rights-1",
    )


@pytest.fixture
def grouped_records(candidate: CandidateRecord) -> list[CandidateRecord]:
    return [
        replace(
            candidate,
            candidate_id="front",
            source_record_id="listing-front",
            raw_sha256="1" * 64,
            phash="0123456789abcdef",
        ),
        replace(
            candidate,
            candidate_id="repost",
            source_record_id="listing-repost",
            raw_sha256="2" * 64,
            phash="0123456789abcdee",
        ),
        replace(
            candidate,
            candidate_id="clasp",
            source_record_id="listing-clasp",
            raw_sha256="3" * 64,
            phash="fedcba9876543210",
        ),
    ]


@pytest.fixture
def object_ref() -> ObjectRef:
    return ObjectRef(
        bucket="watch-auth-private",
        key="datasets/derivatives/" + "b" * 64,
        sha256="b" * 64,
        length=1024,
        etag="etag-1",
    )


@pytest.fixture
def rights() -> RightsRecord:
    return RightsRecord(
        rights_record_id="rights-1",
        source="partner-upload",
        acquisition_method="partner-export",
        allowed_purposes=("download", "train"),
        effective_at=NOW,
        expires_at=NOW + timedelta(days=30),
        required_disposition="quarantine",
        disposition_deadline=NOW + timedelta(days=60),
        rightsholder="Partner Example",
        evidence_sha256="c" * 64,
        status=RightsStatus.ACTIVE,
    )


@pytest.fixture
def expired_rights() -> RightsRecord:
    return RightsRecord(
        rights_record_id="rights-expired",
        source="partner-upload",
        acquisition_method="partner-export",
        allowed_purposes=("download", "train"),
        effective_at=NOW - timedelta(days=60),
        expires_at=NOW - timedelta(days=30),
        required_disposition="delete",
        disposition_deadline=NOW + timedelta(days=1),
        rightsholder="Expired Partner",
        evidence_sha256="d" * 64,
        status=RightsStatus.EXPIRED,
    )


@pytest.fixture
def policy_set():
    return load_source_policies(Path("config/source_policies.json"))


@pytest.fixture
def policy_case(policy_set, rights):
    def decide(case: str):
        policy = get_source_policy(policy_set, "partner-upload")
        case_rights = rights
        if case == "reddit_missing_rightsholder":
            policy = get_source_policy(policy_set, "reddit")
            case_rights = replace(
                rights,
                source="reddit",
                acquisition_method="official-api",
                rightsholder="Reddit",
            )
        elif case == "mercadolivre_scrape":
            policy = get_source_policy(policy_set, "mercadolivre")
            case_rights = replace(
                rights,
                source="mercadolivre",
                acquisition_method="raw-scrape",
            )
        elif case == "olx_original_claim":
            policy = get_source_policy(policy_set, "olx")
            case_rights = replace(
                rights,
                source="olx",
                acquisition_method="seller-original-claim",
            )
        elif case == "missing_policy_hash":
            policy = get_source_policy(policy_set, "unregistered-source")
        elif case == "purpose_mismatch":
            case_rights = replace(rights, allowed_purposes=("download",))
        elif case == "revoked":
            case_rights = replace(rights, status=RightsStatus.REVOKED)
        elif case == "expires_exactly_now":
            case_rights = replace(
                rights,
                effective_at=NOW - timedelta(days=1),
                expires_at=NOW,
            )
        elif case != "partner_export_with_grant":
            raise AssertionError(f"unknown policy case: {case}")
        return authorize_candidate(policy, case_rights, now=NOW)

    return decide


@pytest.fixture
def valid_contract_bundle(candidate, rights, object_ref):
    evidence = EvidenceRecord(
        evidence_record_id="evidence-1",
        candidate_id=candidate.candidate_id,
        candidate_sha256="0" * 64,
        accepted_label=Label.GENUINE,
        accepted_grade=None,
        reference_id="rolex-submariner-126610ln",
        tier=EvidenceTier.A,
        evidence_type="controlled-examination",
        reviewer_ids=("reviewer-1",),
        review_timestamps=(NOW,),
        supporting_artifact_hashes=("e" * 64,),
        review_decision_sha256s=("f" * 64,),
        decision_sha256="f" * 64,
    )
    training = TrainingRecord(
        asset_id="asset-1",
        candidate_id=candidate.candidate_id,
        object_ref=object_ref,
        reference_id=evidence.reference_id,
        watch_entity_id="watch-1",
        content_cluster_id="content-1",
        seller_group_id="seller-1",
        acquisition_batch_id=candidate.acquisition_batch_id,
        angle=Angle.DIAL,
        collected_at=candidate.collected_at,
        original_sha256=candidate.raw_sha256,
        is_augmented=False,
        is_gold=True,
        rejection_subtype=None,
        rejection_reference_id=None,
        source_domain="partner.example",
        capture_style="casual",
        label=Label.GENUINE,
        grade=None,
        factory=None,
        evidence_tier=EvidenceTier.A,
        evidence_record_id=evidence.evidence_record_id,
        rights_record_id=rights.rights_record_id,
        image_sha256=object_ref.sha256,
        phash="fedcba9876543210",
    )
    return candidate, rights, evidence, training


def _sku(service: str) -> str:
    return f"ibm-{service}-us-south-rate-2026"


def valid_run_spec() -> RunSpec:
    return RunSpec(
        run_id=UUID("11111111-1111-4111-8111-111111111111"),
        account_id="account-1",
        code_sha="a" * 40,
        image_digest="private.us-south.icr.io/watch/train@sha256:" + "b" * 64,
        bootstrap_sha256=hashlib.sha256(_SAFE_USER_DATA.encode()).hexdigest(),
        manifest_sha256="c" * 64,
        splits_sha256="d" * 64,
        profile="gx3-16x80x1l4",
        region="us-south",
        zone="us-south-1",
        cos_endpoint="https://s3.direct.us-south.cloud-object-storage.appdomain.cloud",
        max_hours=Decimal("6"),
        boot_volume_gb=Decimal("100"),
        cos_storage_gb_month=Decimal("50"),
        cos_operations=Decimal("1000"),
        registry_storage_gb_month=Decimal("10"),
        code_engine_job_hours=Decimal("1"),
        logs_monitoring_gb=Decimal("1"),
        bucket="watch-auth-private",
        prefix="runs/11111111-1111-4111-8111-111111111111",
        cos_service_instance_id=(
            "crn:v1:bluemix:public:cloud-object-storage:global:a/account-1:cos::"
        ),
        resource_group_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        network_id="r006-22222222-2222-4222-8222-222222222222",
        subnet_id="0717-33333333-3333-4333-8333-333333333333",
        primary_ipv4_address="10.240.0.15",
        image_id="r006-88888888-8888-4888-8888-888888888888",
        public_gateway_id="r006-44444444-4444-4444-8444-444444444444",
        public_gateway_floating_ip_id="r006-99999999-9999-4999-8999-999999999999",
        security_group_id="r006-55555555-5555-4555-8555-555555555555",
        trusted_profile_id="66666666-6666-4666-8666-666666666666",
        subnet_owned_by_run=True,
        public_gateway_owned_by_run=True,
        cos_private_access_only=True,
        cos_versioning_enabled=True,
    )


def valid_credit(now: datetime = NOW) -> CreditEvidence:
    return CreditEvidence(
        account_id="account-1",
        observed_at=now,
        remaining_usd=Decimal("18000.00"),
        expires_at=now + timedelta(days=90),
        eligible_service_plans=(
            ("vpc-vsi", "gx3-16x80x1l4", _sku("vpc-vsi"), "us-south"),
            ("vpc-boot-volume", "standard", _sku("vpc-boot-volume"), "us-south"),
            ("vpc-public-gateway", "standard", _sku("vpc-public-gateway"), "us-south"),
            ("vpc-floating-ip", "standard", _sku("vpc-floating-ip"), "us-south"),
            ("cos-storage", "standard", _sku("cos-storage"), "us-south"),
            ("cos-operations", "standard", _sku("cos-operations"), "us-south"),
            ("container-registry-storage", "standard", _sku("container-registry-storage"), "us-south"),
            ("code-engine", "standard", _sku("code-engine"), "us-south"),
            ("logs-monitoring", "standard", _sku("logs-monitoring"), "us-south"),
        ),
        evidence_sha256="1" * 64,
        provider_verified=True,
    )


def valid_usage(now: datetime = NOW) -> UsageEvidence:
    return UsageEvidence(
        account_id="account-1",
        observed_at=now,
        billing_period_start=now.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
        observed_usage_usd=Decimal("0.00"),
        resource_count=0,
        unreconciled_run_ids=(),
        reconciled_run_ids=(),
        evidence_sha256="2" * 64,
        provider_verified=True,
    )


def valid_pricing(now: datetime = NOW, spec: RunSpec | None = None) -> PricingEvidence:
    spec = spec or valid_run_spec()
    return PricingEvidence(
        account_id=spec.account_id,
        observed_at=now,
        currency="USD",
        run_spec_sha256=run_spec_sha256(spec),
        lines=(
            PriceLine(
                service="vpc-vsi",
                plan="gx3-16x80x1l4",
                sku_id=_sku("vpc-vsi"),
                region="us-south",
                unit="instance-hour",
                estimated_quantity=Decimal("6"),
                unit_price_usd=Decimal("2.00"),
            ),
            PriceLine("vpc-boot-volume", "standard", _sku("vpc-boot-volume"), "us-south", "gb-hour", Decimal("600"), Decimal("0.001")),
            PriceLine("vpc-public-gateway", "standard", _sku("vpc-public-gateway"), "us-south", "gateway-hour", Decimal("6"), Decimal("0.10")),
            PriceLine("vpc-floating-ip", "standard", _sku("vpc-floating-ip"), "us-south", "ip-hour", Decimal("6"), Decimal("0.02")),
            PriceLine("cos-storage", "standard", _sku("cos-storage"), "us-south", "gb-month", Decimal("50"), Decimal("0.02")),
            PriceLine("cos-operations", "standard", _sku("cos-operations"), "us-south", "request", Decimal("1000"), Decimal("0.00001")),
            PriceLine("container-registry-storage", "standard", _sku("container-registry-storage"), "us-south", "gb-month", Decimal("10"), Decimal("0.00")),
            PriceLine("code-engine", "standard", _sku("code-engine"), "us-south", "job-hour", Decimal("1"), Decimal("0.00")),
            PriceLine("logs-monitoring", "standard", _sku("logs-monitoring"), "us-south", "gb", Decimal("1"), Decimal("0.00")),
        ),
        evidence_sha256="4" * 64,
        provider_verified=True,
    )


def valid_post_promo_plan(now: datetime = NOW) -> PostPromoPlan:
    return PostPromoPlan(
        account_id="account-1",
        observed_at=now,
        promo_expires_at=now + timedelta(days=90),
        entries=(
            PostPromoEntry("cos-storage", "5" * 64, now + timedelta(days=30)),
            PostPromoEntry("container-registry-storage", "6" * 64, now + timedelta(days=30)),
            PostPromoEntry("logs-monitoring", "7" * 64, now + timedelta(days=30)),
        ),
        owner_authorization_sha256="8" * 64,
        evidence_sha256="9" * 64,
    )


def vpc_instance(**overrides: object) -> dict[str, object]:
    instance: dict[str, object] = {
        "id": "r006-77777777-7777-4777-8777-777777777777",
        "crn": "crn:v1:bluemix:public:is:us-south-1:a/account-1::instance:r006-77777777-7777-4777-8777-777777777777",
        "name": "watch-auth-11111111-1111-4111-8111-111111111111",
        "profile": {"name": "gx3-16x80x1l4"},
        "zone": {"name": "us-south-1"},
        "vpc": {"id": "r006-22222222-2222-4222-8222-222222222222"},
        "image": {"id": "r006-88888888-8888-4888-8888-888888888888"},
        "resource_group": {"id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"},
        "primary_network_interface": {
            "id": "0717-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "subnet": {"id": "0717-33333333-3333-4333-8333-333333333333"},
            "primary_ip": {"address": "10.240.0.15"},
        },
        "network_interfaces": [
            {"id": "0717-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}
        ],
        "network_attachments": [],
        "metadata_service": {"enabled": True, "protocol": "http", "response_hop_limit": 2},
        "boot_volume_attachment": {
            "id": "r006-cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "volume": {"id": "r006-bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"}
        },
        "status": "running",
    }
    instance.update(overrides)
    return instance


def safe_user_data() -> str:
    return _SAFE_USER_DATA
