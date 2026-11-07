"""Documented readbacks cannot substitute for credentialless source provenance."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from infra.ibm_routing import RoutingProofUnavailable, main, verify_routing_probes
from tests.conftest import NOW


@pytest.fixture
def documented_monitoring_response():
    # Sanitized IBM "Extract platform metrics" sample; current timestamp does not
    # add producer authentication identity to data[].d/t.
    # https://cloud.ibm.com/docs/monitoring?topic=monitoring-metrics_api
    timestamp = int(NOW.timestamp())
    return {
        "data": [{"d": ["11111111-1111-4111-8111-111111111111", 120, 18809015.575], "t": timestamp}],
        "start": timestamp - 60,
        "end": timestamp,
    }


@pytest.fixture
def documented_logs_archive_row():
    # Actual Parquet column names and JSON-string encoding from IBM's archive schema:
    # https://github.com/ibm-cloud-docs/cloud-logs/blob/master/query-archive-data-bucket.md
    return {
        "src_obj__event_metadata": json.dumps({
            "timestamp": NOW.isoformat(), "severity": "Info",
            "priorityclass": "low", "logid": "11111111-1111-4111-8111-111111111111",
        }),
        "src_obj__event_labels": json.dumps({
            "applicationname": "watch-auth", "subsystemname": "watchdog",
        }),
        "src_obj__user_data": json.dumps({
            "_container_id": "sanitized-container", "host": "sanitized-host",
            "short_message": "watchdog routing probe",
        }),
    }


def test_documented_monitoring_response_does_not_prove_producer_credentials(documented_monitoring_response):
    with pytest.raises(RoutingProofUnavailable, match="producer credential mode"):
        verify_routing_probes(documented_monitoring_response)


def test_documented_logs_archive_row_does_not_prove_producer_credentials(documented_logs_archive_row):
    with pytest.raises(RoutingProofUnavailable, match="service-account binding"):
        verify_routing_probes(documented_logs_archive_row)


def test_untrusted_log_payload_cannot_self_assert_authentication(documented_logs_archive_row):
    documented_logs_archive_row["src_obj__user_data"] = json.dumps({
        "authentication": "trusted_profile", "service_account": "watchdog",
        "destination_crn": "crn:v1:bluemix:public:logs:us-south:a/account-1:logs-1::",
    })
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(documented_logs_archive_row)


@pytest.mark.parametrize("proof", [
    {"credentialless": True, "logs_probe": "delivered", "monitoring_probe": "delivered"},
    {"records": [{"id": "probe-1", "authentication": "trusted_profile", "timestamp": NOW.isoformat()}]},
    {"probe_id": "probe-1", "logs": {}, "monitoring": {}},
])
def test_old_local_flags_and_invented_records_never_authorize_deployment(proof):
    with pytest.raises(RoutingProofUnavailable, match="PENDENTE E2E"):
        verify_routing_probes(proof)


def test_main_exits_with_concrete_blocker():
    with pytest.raises(SystemExit, match="Code Engine integration/IAM"):
        main()


def test_deploy_stops_before_any_ibm_cli_call(tmp_path):
    marker = tmp_path / "provider-called"
    fake_cli = tmp_path / "ibmcloud"
    fake_cli.write_text('#!/bin/sh\nprintf invoked > "$WATCH_TEST_CALLS"\nexit 0\n')
    fake_cli.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    environment = {
        "PATH": f"{tmp_path}:{Path(sys.executable).parent}:{os.defpath}",
        "PYTHONPATH": str(root), "WATCH_TEST_CALLS": str(marker),
        "WATCH_AUTH_ACCOUNT_ID": "account-1", "WATCH_AUTH_RESOURCE_GROUP": "watch-auth",
        "WATCH_AUTH_RESOURCE_GROUP_ID": "rg-1", "WATCH_AUTH_REGION": "us-south",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_CE_PROJECT": "watch-auth", "WATCH_AUTH_COS_BUCKET": "watch-private",
        "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": "watchdog",
        "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": "private.us-south.icr.io/watch/watchdog@sha256:" + "a" * 64,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
        "WATCH_AUTH_PLATFORM_LOGS_CRN": "crn:v1:bluemix:public:logs:us-south:a/account-1:logs-1::",
        "WATCH_AUTH_PLATFORM_MONITORING_CRN": "crn:v1:bluemix:public:sysdig-monitor:us-south:a/account-1:monitor-1::",
    }
    result = subprocess.run(
        ["bash", "scripts/deploy_ibm_watchdog.sh"], cwd=root, env=environment,
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "PENDENTE E2E" in result.stderr
    assert "destination CRNs, source identity and fresh delivery records" in result.stderr
    assert not marker.exists()
