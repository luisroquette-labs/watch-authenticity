"""Integrity tests for runtime-only IBM financial evidence."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from uuid import UUID

import pytest

from infra.ibm_evidence import (
    fetch_usage_evidence,
    load_credit_evidence,
    load_post_promo_plan,
    load_pricing_evidence,
    run_spec_sha256,
)
from tests.conftest import NOW, valid_run_spec


def _artifact_and_record(
    tmp_path: Path,
    name: str,
    record: dict[str, Any],
) -> tuple[Path, Path]:
    evidence_dir = tmp_path / ".runtime" / "ibm" / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    artifact = evidence_dir / f"{name}.capture"
    artifact.write_bytes(f"authoritative-{name}".encode())
    record[f"{name}_capture_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    record_path = evidence_dir / f"{name}.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return record_path, artifact


def _credit_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observed_at": NOW.isoformat(),
        "currency": "USD",
        "account_id": "account-1",
        "remaining_usd": "18000.00",
        "expires_at": (NOW + timedelta(days=90)).isoformat(),
        "eligible_service_plans": [
            {
                "service": service,
                "plan": plan,
                "sku_id": f"ibm-{service}-us-south-rate-2026",
                "region": "us-south",
            }
            for service, plan in (
                ("vpc-vsi", "gx3-16x80x1l4"),
                ("vpc-boot-volume", "standard"),
                ("vpc-public-gateway", "standard"),
                ("vpc-floating-ip", "standard"),
                ("cos-storage", "standard"),
                ("cos-operations", "standard"),
                ("container-registry-storage", "standard"),
                ("code-engine", "standard"),
                ("logs-monitoring", "standard"),
            )
        ],
    }


def _pricing_record() -> dict[str, Any]:
    services = (
        ("vpc-vsi", "gx3-16x80x1l4", "instance-hour", "6", "2.00"),
        ("vpc-boot-volume", "standard", "gb-hour", "600", "0.001"),
        ("vpc-public-gateway", "standard", "gateway-hour", "6", "0.10"),
        ("vpc-floating-ip", "standard", "ip-hour", "6", "0.02"),
        ("cos-storage", "standard", "gb-month", "50", "0.02"),
        ("cos-operations", "standard", "request", "1000", "0.00001"),
        ("container-registry-storage", "standard", "gb-month", "10", "0.00"),
        ("code-engine", "standard", "job-hour", "1", "0.00"),
        ("logs-monitoring", "standard", "gb", "1", "0.00"),
    )
    return {
        "schema_version": 1,
        "observed_at": NOW.isoformat(),
        "account_id": "account-1",
        "currency": "USD",
        "run_spec_sha256": run_spec_sha256(valid_run_spec()),
        "lines": [
            {
                "service": service,
                "plan": plan,
                "sku_id": f"ibm-{service}-us-south-rate-2026",
                "region": "us-south",
                "unit": unit,
                "estimated_quantity": quantity,
                "unit_price_usd": price,
            }
            for service, plan, unit, quantity, price in services
        ],
    }


def _post_promo_record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observed_at": NOW.isoformat(),
        "account_id": "account-1",
        "promo_expires_at": (NOW + timedelta(days=90)).isoformat(),
        "entries": [
            {
                "service": service,
                "export_destination_sha256": digit * 64,
                "delete_or_downgrade_deadline": (NOW + timedelta(days=30)).isoformat(),
            }
            for service, digit in (
                ("cos-storage", "5"),
                ("container-registry-storage", "6"),
                ("logs-monitoring", "7"),
            )
        ],
    }


def test_credit_loader_binds_runtime_record_to_authoritative_capture(tmp_path: Path):
    record, capture = _artifact_and_record(tmp_path, "console", _credit_record())

    credit = load_credit_evidence(record, capture, now=NOW)

    assert credit.remaining_usd == Decimal("18000.00")
    assert credit.account_id == "account-1"
    assert any(
        item[:2] == ("vpc-vsi", "gx3-16x80x1l4")
        for item in credit.eligible_service_plans
    )
    assert len(credit.evidence_sha256) == 64
    assert not credit.provider_verified


def test_credit_loader_rejects_versioned_evidence_path(tmp_path: Path):
    artifact = tmp_path / "console.capture"
    artifact.write_bytes(b"authoritative-console")
    payload = _credit_record()
    payload["console_capture_sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    record = tmp_path / "credit.json"
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        load_credit_evidence(record, artifact, now=NOW)


@pytest.mark.parametrize("age", [timedelta(hours=24, seconds=1), timedelta(days=2)])
def test_credit_loader_rejects_stale_capture(tmp_path: Path, age: timedelta):
    payload = _credit_record()
    payload["observed_at"] = (NOW - age).isoformat()
    record, capture = _artifact_and_record(tmp_path, "console", payload)

    with pytest.raises(ValueError, match="stale"):
        load_credit_evidence(record, capture, now=NOW)


def test_credit_loader_rejects_missing_or_mismatched_capture(tmp_path: Path):
    record, capture = _artifact_and_record(tmp_path, "console", _credit_record())
    capture.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_credit_evidence(record, capture, now=NOW)
    capture.unlink()
    with pytest.raises(ValueError, match="artifact"):
        load_credit_evidence(record, capture, now=NOW)


def test_credit_loader_rejects_future_non_usd_and_unknown_plan(tmp_path: Path):
    for field, value, message in (
        ("observed_at", (NOW + timedelta(seconds=1)).isoformat(), "future"),
        ("currency", "BRL", "USD"),
        (
            "eligible_service_plans",
            [
                {
                    "service": "vpc-vsi",
                    "plan": "mystery-plan",
                    "sku_id": "ibm-vpc-vsi-us-south-rate-2026",
                    "region": "us-south",
                }
            ],
            "unknown service plan",
        ),
    ):
        payload = _credit_record()
        payload[field] = value
        record, capture = _artifact_and_record(tmp_path / field, "console", payload)
        with pytest.raises(ValueError, match=message):
            load_credit_evidence(record, capture, now=NOW)


def test_pricing_loader_requires_exact_spec_and_quote_hash(tmp_path: Path):
    record, capture = _artifact_and_record(tmp_path, "quote", _pricing_record())

    pricing = load_pricing_evidence(record, capture, valid_run_spec(), now=NOW)

    assert pricing.currency == "USD"
    assert pricing.account_id == "account-1"
    assert pricing.run_spec_sha256 == run_spec_sha256(valid_run_spec())
    assert len(pricing.lines) == 9
    assert not pricing.provider_verified


def test_pricing_loader_rejects_wrong_spec_unknown_plan_and_float_money(tmp_path: Path):
    mutations = (
        ("wrong spec", lambda value: value.update(run_spec_sha256="f" * 64), "run spec"),
        (
            "unknown plan",
            lambda value: value["lines"][0].update(plan="unknown"),
            "unknown service plan",
        ),
        (
            "float money",
            lambda value: value["lines"][0].update(unit_price_usd=1.2),
            "decimal string",
        ),
    )
    for name, mutate, message in mutations:
        payload = _pricing_record()
        mutate(payload)
        record, capture = _artifact_and_record(tmp_path / name, "quote", payload)
        with pytest.raises((TypeError, ValueError), match=message):
            load_pricing_evidence(record, capture, valid_run_spec(), now=NOW)


def test_post_promo_loader_requires_exact_services_and_safe_deadlines(tmp_path: Path):
    payload = _post_promo_record()
    record, capture = _artifact_and_record(tmp_path, "authorization", payload)
    plan = load_post_promo_plan(record, capture, now=NOW)
    assert plan.observed_at == NOW
    assert plan.account_id == "account-1"
    assert plan.promo_expires_at == NOW + timedelta(days=90)
    assert len(plan.evidence_sha256) == 64
    assert {entry.service for entry in plan.entries} == {
        "cos-storage",
        "container-registry-storage",
        "logs-monitoring",
    }
    assert plan.owner_authorization_sha256 == hashlib.sha256(capture.read_bytes()).hexdigest()

    payload = _post_promo_record()
    payload["entries"][0]["delete_or_downgrade_deadline"] = (
        NOW + timedelta(days=89)
    ).isoformat()
    record, capture = _artifact_and_record(tmp_path / "late", "authorization", payload)
    with pytest.raises(ValueError, match="48 hours"):
        load_post_promo_plan(record, capture, now=NOW)


class _Response:
    def __init__(self, result: dict[str, object]) -> None:
        self._result = result

    def get_result(self) -> dict[str, object]:
        return self._result


def _usage_resource(instance_id: str, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "account_id": "account-1",
        "resource_instance_id": instance_id,
        "currency_code": "USD",
        "month": "2026-11",
        "pending": False,
        "usage": [{"metric": "hours", "cost": 1.25}],
    }
    value.update(overrides)
    return value


def test_usage_loader_fetches_every_page_and_hashes_raw_response():
    client = Mock()
    client.get_resource_usage_account.side_effect = (
        _Response(
            {
                "limit": 1,
                "count": 1,
                "next": {"offset": "page-2"},
                "resources": [_usage_resource("instance-1")],
            }
        ),
        _Response(
            {
                "limit": 1,
                "count": 1,
                "resources": [_usage_resource("instance-2")],
            }
        ),
    )
    pending = (UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),)

    usage = fetch_usage_evidence(client, "account-1", "2026-11", pending, now=NOW)

    assert usage.observed_usage_usd == Decimal("2.50")
    assert usage.resource_count == 2
    assert usage.account_id == "account-1"
    assert usage.billing_period_start.isoformat() == "2026-11-01T00:00:00+00:00"
    assert usage.unreconciled_run_ids == pending
    assert usage.provider_verified
    assert len(usage.evidence_sha256) == 64
    assert client.get_resource_usage_account.call_count == 2
    assert client.get_resource_usage_account.call_args_list[1].kwargs["start"] == "page-2"


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            {"limit": 1, "count": 1, "resources": [_usage_resource("x", account_id="other")]},
            "account",
        ),
        (
            {"limit": 1, "count": 1, "resources": [_usage_resource("x", currency_code="BRL")]},
            "USD",
        ),
        (
            {"limit": 1, "count": 1, "resources": [_usage_resource("x", pending=True)]},
            "incomplete",
        ),
        (
            {"limit": 1, "count": 1, "next": {}, "resources": [_usage_resource("x")]},
            "pagination",
        ),
        (
            {
                "limit": 2,
                "count": 2,
                "resources": [_usage_resource("x"), _usage_resource("x")],
            },
            "duplicate",
        ),
    ],
)
def test_usage_loader_rejects_untrustworthy_provider_response(
    result: dict[str, object], message: str
):
    client = Mock()
    client.get_resource_usage_account.return_value = _Response(result)
    with pytest.raises(ValueError, match=message):
        fetch_usage_evidence(client, "account-1", "2026-11", (), now=NOW)


def test_usage_loader_rejects_future_month_and_accepts_missing_pending():
    client = Mock()
    resource = _usage_resource("instance-1")
    resource.pop("pending")
    client.get_resource_usage_account.return_value = _Response(
        {"limit": 1, "count": 1, "resources": [resource]}
    )
    assert fetch_usage_evidence(client, "account-1", "2026-11", (), now=NOW).account_id == "account-1"
    with pytest.raises(ValueError, match="future"):
        fetch_usage_evidence(client, "account-1", "2026-12", (), now=NOW)
    resource["month"] = "2026-10"
    assert fetch_usage_evidence(client, "account-1", "2026-10", (), now=NOW).resource_count == 1


def test_usage_loader_uses_utc_month_at_timezone_boundary():
    client = Mock()
    resource = _usage_resource("instance-1", month="2026-12")
    client.get_resource_usage_account.return_value = _Response(
        {"limit": 1, "count": 1, "resources": [resource]}
    )
    instant_utc = datetime(2026, 12, 1, 0, 30, tzinfo=timezone.utc)
    instant_local = instant_utc.astimezone(timezone(timedelta(hours=-3)))

    usage = fetch_usage_evidence(client, "account-1", "2026-12", (), now=instant_local)

    assert usage.billing_period_start == datetime(2026, 12, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="billing period mismatch"):
        fetch_usage_evidence(client, "account-1", "2026-11", (), now=instant_local)


def test_financial_loaders_preserve_account_binding(tmp_path: Path):
    pricing_payload = _pricing_record()
    pricing_payload["account_id"] = "account-2"
    pricing_record, pricing_capture = _artifact_and_record(
        tmp_path / "pricing", "quote", pricing_payload
    )
    with pytest.raises(ValueError, match="account"):
        load_pricing_evidence(
            pricing_record, pricing_capture, valid_run_spec(), now=NOW
        )

    post_payload = _post_promo_record()
    post_payload["account_id"] = "account-2"
    post_record, post_capture = _artifact_and_record(
        tmp_path / "post", "authorization", post_payload
    )
    plan = load_post_promo_plan(post_record, post_capture, now=NOW)
    assert plan.account_id == "account-2"
