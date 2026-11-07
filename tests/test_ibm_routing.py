"""Routing proof must come from fresh authenticated destination records."""

from copy import deepcopy
from datetime import timedelta
from unittest.mock import Mock

import pytest

from infra.ibm_routing import verify_routing_probes
from tests.conftest import NOW

LOGS = "crn:v1:bluemix:public:logs:us-south:a/account-1:logs-1::"
MONITORING = "crn:v1:bluemix:public:sysdig-monitor:us-south:a/account-1:monitor-1::"


def _inputs():
    proof = {
        "probe_id": "probe-1",
        "logs": {"record_id": "log-1", "record_url": "https://logs-1.api.us-south.logs.cloud.ibm.com/v1/records/log-1"},
        "monitoring": {"record_id": "metric-1", "record_url": "https://us-south.monitoring.cloud.ibm.com/api/records/metric-1"},
    }
    records = {}
    for kind, crn in (("logs", LOGS), ("monitoring", MONITORING)):
        records[proof[kind]["record_url"]] = {"records": [{
            "id": proof[kind]["record_id"], "probe_id": "probe-1", "destination_crn": crn,
            "account_id": "account-1", "region": "us-south", "project": "watch-auth",
            "service_account": "watchdog", "authentication": "trusted_profile",
            "timestamp": NOW.isoformat(),
        }]}
    return proof, records


def _verify(proof, read):
    verify_routing_probes(
        proof, read, account="account-1", region="us-south", logs_crn=LOGS,
        monitoring_crn=MONITORING, project="watch-auth", service_account="watchdog", now=NOW,
    )


def test_fresh_provider_read_records_are_required_and_accepted():
    proof, records = _inputs()
    read = Mock(side_effect=lambda url: deepcopy(records[url]))
    _verify(proof, read)
    assert read.call_count == 2


def test_self_authored_local_flags_are_rejected_without_read():
    read = Mock()
    with pytest.raises(ValueError, match="local flags"):
        _verify({"credentialless": True, "logs_probe": "delivered", "monitoring_probe": "delivered"}, read)
    read.assert_not_called()


@pytest.mark.parametrize("field,value", [
    ("destination_crn", "wrong"), ("id", "missing"), ("probe_id", "wrong"),
    ("timestamp", (NOW - timedelta(minutes=16)).isoformat()),
    ("timestamp", (NOW + timedelta(minutes=1)).isoformat()),
    ("authentication", "api_key"), ("service_account", "another"),
])
def test_wrong_stale_or_noncredentialless_provider_records_are_rejected(field, value):
    proof, records = _inputs()
    records[proof["logs"]["record_url"]]["records"][0][field] = value
    with pytest.raises(ValueError):
        _verify(proof, lambda url: records[url])


@pytest.mark.parametrize("url", [
    "file:///tmp/proof.json", "https://evil.example/v1/records/log-1",
    "https://logs-1.api.us-south.logs.cloud.ibm.com.evil.example/v1/log-1",
    "http://logs-1.api.us-south.logs.cloud.ibm.com/v1/log-1",
    "https://user@logs-1.api.us-south.logs.cloud.ibm.com/v1/log-1",
])
def test_provider_url_is_bound_to_destination_host(url):
    proof, _records = _inputs()
    proof["logs"]["record_url"] = url
    read = Mock()
    with pytest.raises(ValueError, match="URL"):
        _verify(proof, read)
    read.assert_not_called()


def test_unknown_provider_response_is_rejected():
    proof, _records = _inputs()
    with pytest.raises(ValueError, match="unrecognized"):
        _verify(proof, lambda _: {"delivered": True})


@pytest.mark.parametrize("status", [200, 302, 401])
def test_main_reads_with_short_lived_iam_and_never_follows_redirects(monkeypatch, tmp_path, status):
    import json
    from types import SimpleNamespace

    from infra.ibm_routing import main

    proof, records = _inputs()
    path = tmp_path / "references.json"
    path.write_text(json.dumps(proof))
    for name, value in {
        "WATCH_AUTH_PLATFORM_ROUTING_PROOF": str(path),
        "WATCH_AUTH_ACCOUNT_ID": "account-1", "WATCH_AUTH_REGION": "us-south",
        "WATCH_AUTH_PLATFORM_LOGS_CRN": LOGS, "WATCH_AUTH_PLATFORM_MONITORING_CRN": MONITORING,
        "WATCH_AUTH_CE_PROJECT": "watch-auth", "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": "watchdog",
    }.items():
        monkeypatch.setenv(name, value)
    token_read = Mock(return_value=SimpleNamespace(stdout='{"iam_token":"Bearer short-lived"}'))
    monkeypatch.setattr("infra.ibm_routing.subprocess.run", token_read)
    clock = Mock()
    clock.now.return_value = NOW
    clock.fromisoformat.side_effect = __import__("datetime").datetime.fromisoformat
    monkeypatch.setattr("infra.ibm_routing.datetime", clock)

    def provider_get(url, *, headers, timeout, allow_redirects):
        assert headers == {"Authorization": "Bearer short-lived"}
        assert timeout == 30 and allow_redirects is False
        return SimpleNamespace(status_code=status, json=lambda: records[url])

    monkeypatch.setattr("infra.ibm_routing.requests.get", provider_get)
    if status == 200:
        main()
    else:
        with pytest.raises(ValueError, match="read failed"):
            main()
    token_read.assert_called_once_with(
        ["ibmcloud", "iam", "oauth-tokens", "--output", "json"],
        check=True, capture_output=True, text=True,
    )
