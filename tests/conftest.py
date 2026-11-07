"""Shared factories for IBM training-plane contract tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

from infra.ibm_evidence import run_spec_sha256
from infra.ibm_types import (
    CreditEvidence,
    PostPromoEntry,
    PostPromoPlan,
    PriceLine,
    PricingEvidence,
    RunSpec,
    UsageEvidence,
)

NOW = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)
_SAFE_USER_DATA = "#!/bin/bash\nset -euo pipefail\n/opt/watch-auth/bootstrap-host\n"


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
