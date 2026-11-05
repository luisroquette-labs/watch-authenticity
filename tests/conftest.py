"""Shared factories for IBM training-plane contract tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

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


def valid_run_spec() -> RunSpec:
    return RunSpec(
        run_id=UUID("11111111-1111-4111-8111-111111111111"),
        code_sha="a" * 40,
        image_digest="private.us-south.icr.io/watch/train@sha256:" + "b" * 64,
        manifest_sha256="c" * 64,
        splits_sha256="d" * 64,
        profile="gx3-16x80x1l4",
        region="us-south",
        zone="us-south-1",
        max_hours=Decimal("6"),
        bucket="watch-auth-private",
        prefix="runs/11111111-1111-4111-8111-111111111111",
        cos_service_instance_id="crn:v1:bluemix:public:cloud-object-storage:global:a/abc:cos::",
        network_id="r006-22222222-2222-4222-8222-222222222222",
        subnet_id="0717-33333333-3333-4333-8333-333333333333",
        public_gateway_id="r006-44444444-4444-4444-8444-444444444444",
        security_group_id="r006-55555555-5555-4555-8555-555555555555",
        trusted_profile_id="66666666-6666-4666-8666-666666666666",
        public_gateway_owned_by_run=True,
    )


def valid_credit(now: datetime = NOW) -> CreditEvidence:
    return CreditEvidence(
        observed_at=now,
        remaining_usd=Decimal("18000.00"),
        expires_at=now + timedelta(days=90),
        eligible_service_plans=("gx3-16x80x1l4", "standard"),
        evidence_sha256="1" * 64,
    )


def valid_usage(now: datetime = NOW) -> UsageEvidence:
    return UsageEvidence(
        observed_at=now,
        billing_period_start=now - timedelta(days=4),
        observed_usage_usd=Decimal("0.00"),
        unreconciled_run_ids=(),
        evidence_sha256="2" * 64,
    )


def valid_pricing(now: datetime = NOW) -> PricingEvidence:
    return PricingEvidence(
        observed_at=now,
        currency="USD",
        run_spec_sha256="3" * 64,
        lines=(
            PriceLine(
                service="vpc-vsi",
                plan="gx3-16x80x1l4",
                unit="instance-hour",
                estimated_quantity=Decimal("6"),
                unit_price_usd=Decimal("2.00"),
            ),
        ),
        evidence_sha256="4" * 64,
    )


def valid_post_promo_plan(now: datetime = NOW) -> PostPromoPlan:
    return PostPromoPlan(
        entries=(
            PostPromoEntry("cos-storage", "5" * 64, now + timedelta(days=30)),
            PostPromoEntry("container-registry-storage", "6" * 64, now + timedelta(days=30)),
            PostPromoEntry("logs-monitoring", "7" * 64, now + timedelta(days=30)),
        ),
        owner_authorization_sha256="8" * 64,
    )


def vpc_instance(**overrides: object) -> dict[str, object]:
    instance: dict[str, object] = {
        "id": "r006-77777777-7777-4777-8777-777777777777",
        "name": "watch-auth-11111111-1111-4111-8111-111111111111",
        "profile": {"name": "gx3-16x80x1l4"},
        "zone": {"name": "us-south-1"},
        "vpc": {"id": "r006-22222222-2222-4222-8222-222222222222"},
        "image": {"id": "r006-88888888-8888-4888-8888-888888888888"},
        "status": "running",
    }
    instance.update(overrides)
    return instance


def safe_user_data() -> str:
    return "#!/bin/bash\nset -euo pipefail\n/opt/watch-auth/bootstrap-host\n"
