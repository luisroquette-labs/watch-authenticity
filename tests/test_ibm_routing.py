"""Sanitized official CE v2 readbacks and a provider-filtered raw Logs query."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from infra import ibm_routing as routing
from infra.ibm_routing import (
    RoutingProbe,
    RoutingProofUnavailable,
    collect_routing_probes,
    main,
    verify_routing_probes,
)
from tests.conftest import NOW


@pytest.fixture
def probe():
    return RoutingProbe(
        account_id="account-1", region="us-south",
        project_id="11111111-1111-4111-8111-111111111111", project_name="watch-auth",
        job_name="watch-auth-watchdog", job_run_name="watch-auth-watchdog-jobrun-abcde",
        service_account="watchdog", trusted_profile_id="Profile-watchdog",
        image_digest="private.us-south.icr.io/watch/watchdog@sha256:" + "a" * 64,
        logs_crn="crn:v1:bluemix:public:logs:us-south:a/account-1:22222222-2222-4222-8222-222222222222::",
        nonce="33333333-3333-4333-8333-333333333333",
    )


@pytest.fixture
def provider_evidence(probe):
    # Fields, including optional run_* readbacks, come from IBM's Job/JobRun models:
    # https://github.com/IBM/code-engine-python-sdk/blob/main/ibm_code_engine_sdk/code_engine_v2.py
    profile = {"name": "WATCH_AUTH_TRUSTED_PROFILE_ID", "type": "literal",
               "value": probe.trusted_profile_id}
    source = {"project_id": probe.project_id, "region": probe.region,
              "image_reference": probe.image_digest, "run_service_account": probe.service_account,
              "run_compute_resource_token_enabled": True, "run_env_variables": [profile]}
    job = {**copy.deepcopy(source), "name": probe.job_name}
    run = {**copy.deepcopy(source), "name": probe.job_run_name, "job_name": probe.job_name,
           "created_at": (NOW - timedelta(minutes=1)).isoformat()}
    run["run_env_variables"].append({"name": "WATCH_AUTH_ROUTING_PROBE_NONCE", "type": "literal",
                                     "value": probe.nonce})
    return {
        "project": {"id": probe.project_id, "name": probe.project_name,
                    "crn": f"crn:v1:bluemix:public:codeengine:{probe.region}:a/{probe.account_id}:{probe.project_id}::"},
        "job": job, "job_run": run,
        # Native IAM Identity V1 readbacks; env/profile flags cannot substitute for these.
        "trusted_profile": {
            "id": probe.trusted_profile_id, "account_id": probe.account_id,
            "iam_id": "iam-" + probe.trusted_profile_id,
            "crn": f"crn:v1:bluemix:public:iam-identity:us-south:a/{probe.account_id}::profile:{probe.trusted_profile_id}",
        },
        "trusted_profile_rules": {"rules": []},
        "trusted_profile_links": {"links": [{
            "id": "link-1", "cr_type": "CE", "is_cross_account": False,
            "link": {
                "crn": f"crn:v1:bluemix:public:codeengine:{probe.region}:a/{probe.account_id}:{probe.project_id}::",
                "component_type": "job", "component_name": probe.job_name,
            },
        }]},
        # logs-raw is intentionally opaque. Provider-side equality filters bind its metadata.
        "logs_output": probe.marker + "\n",
    }


@pytest.fixture
def providers(monkeypatch, provider_evidence):
    def cli(*args, **kwargs):
        if args[0][1:3] == ["iam", "oauth-tokens"]:
            output = json.dumps({"iam_token": "Bearer ephemeral-test-token"})
        else:
            output = provider_evidence["logs_output"]
        return SimpleNamespace(stdout=output, stderr="", returncode=0)

    cli_mock = Mock(side_effect=cli)
    http = Mock(side_effect=[SimpleNamespace(status_code=200, text=json.dumps(provider_evidence[key]))
                             for key in ("project", "job", "job_run", "trusted_profile",
                                         "trusted_profile_rules", "trusted_profile_links")])
    monkeypatch.setattr(routing.subprocess, "run", cli_mock)
    monkeypatch.setattr(routing.requests, "get", http)
    return cli_mock, http


def test_official_chain_passes_without_authentication_fields_in_logs(probe, provider_evidence):
    verify_routing_probes(provider_evidence, probe=probe, now=NOW)
    assert "authentication" not in provider_evidence["logs_output"]
    assert "service_account" not in provider_evidence["logs_output"]


@pytest.mark.parametrize(("section", "key", "value"), [
    ("project", "id", "different"), ("project", "name", "different"),
    ("project", "crn", "crn:v1:bluemix:public:codeengine:us-south:a/other:project::"),
    ("job", "name", "other"), ("job", "project_id", "other"), ("job", "region", "eu-de"),
    ("job", "run_compute_resource_token_enabled", False),
    ("job", "run_compute_resource_token_enabled", "true"),
    ("job", "run_service_account", "other"), ("job", "image_reference", "other"),
    ("job_run", "name", "other"), ("job_run", "job_name", "other"),
    ("job_run", "project_id", "other"), ("job_run", "region", "eu-de"),
    ("job_run", "run_compute_resource_token_enabled", False),
    ("job_run", "run_compute_resource_token_enabled", 1),
    ("job_run", "run_compute_resource_token_enabled", "true"),
    ("job_run", "run_service_account", "other"), ("job_run", "image_reference", "other"),
    ("job_run", "created_at", (NOW - timedelta(minutes=16)).isoformat()),
    ("job_run", "created_at", (NOW + timedelta(seconds=1)).isoformat()),
    ("job_run", "created_at", "2026-11-05T12:00:00"),
    ("job_run", "created_at", None), ("job_run", "run_env_variables", []),
])
def test_divergent_or_missing_link_blocks(probe, provider_evidence, section, key, value):
    provider_evidence[section][key] = value
    with pytest.raises(RoutingProofUnavailable, match="PENDENTE E2E"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("section", ["project", "job", "job_run", "trusted_profile",
                                      "trusted_profile_rules", "trusted_profile_links"])
def test_missing_readback_blocks(probe, provider_evidence, section):
    del provider_evidence[section]
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("section", ["job", "job_run"])
@pytest.mark.parametrize("mutation", ["missing", "duplicate", "secret", "different"])
def test_profile_environment_must_be_unique_literal(probe, provider_evidence, section, mutation):
    env = provider_evidence[section]["run_env_variables"]
    if mutation == "missing":
        env.pop(0)
    elif mutation == "duplicate":
        env.append(dict(env[0]))
    elif mutation == "secret":
        env[0]["type"] = "secret_key_reference"
    else:
        env[0]["value"] = "Profile-other"
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


def test_provider_run_nonce_must_match(probe, provider_evidence):
    provider_evidence["job_run"]["run_env_variables"][-1]["value"] = "different"
    with pytest.raises(RoutingProofUnavailable, match="nonce differs"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("raw", ["", "no matching logs", None, "{nonce}-suffix", "prefix-{nonce}", "unrelated\n{nonce}",
                                 "{nonce}\n{nonce}", "{nonce} {nonce}"])
def test_zero_ambiguous_or_partial_nonce_blocks(probe, provider_evidence, raw):
    provider_evidence["logs_output"] = raw.format(nonce=probe.nonce) if raw is not None else None
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("evidence", [None, {"credentialless": True},
    {"records": [{"authentication": "trusted_profile", "service_account": "watchdog"}]},
    {"data": [{"d": ["id", 120, 18809015.575], "t": int(NOW.timestamp())}]},
    {"src_obj__user_data": '{"authentication":"trusted_profile"}'},
])
def test_local_flags_invented_records_and_monitoring_do_not_authorize(evidence):
    with pytest.raises(RoutingProofUnavailable, match="PENDENTE E2E"):
        verify_routing_probes(evidence)


@pytest.mark.parametrize(("field", "value"), [
    ("job_run_name", "run' || true"), ("project_id", "other"), ("nonce", "not-a-nonce"),
    ("nonce", "33333333-3333-1333-8333-333333333333"),
    ("logs_crn", "crn:v1:bluemix:public:logs:us-south:a/other:22222222-2222-4222-8222-222222222222::"),
    ("logs_crn", "crn:v1:bluemix:public:logs:us-south:a/account-1:attacker.example::"),
    ("image_digest", "image:latest"),
])
def test_unbound_or_injectable_probe_is_rejected(probe, field, value):
    with pytest.raises(RoutingProofUnavailable):
        replace(probe, **{field: value})


def test_collector_reads_official_ce_api_and_queries_exact_provider_metadata(probe, providers):
    cli, http = providers
    result = collect_routing_probes(probe, now=NOW)
    assert result["logs_output"] == probe.marker + "\n"
    assert [call.args[0] for call in http.call_args_list] == [
        f"https://api.us-south.codeengine.cloud.ibm.com/v2/projects/{probe.project_id}{suffix}"
        for suffix in ("", f"/jobs/{probe.job_name}", f"/job_runs/{probe.job_run_name}")
    ] + [f"https://iam.cloud.ibm.com/v1/profiles/{probe.trusted_profile_id}{suffix}"
         for suffix in ("", "/rules", "/links")]
    for call in http.call_args_list:
        assert call.kwargs["headers"]["Authorization"] == "Bearer ephemeral-test-token"
        assert call.kwargs["allow_redirects"] is False
        assert call.kwargs["timeout"] == 20
    command = cli.call_args_list[-1].args[0]
    assert command[:3] == ["ibmcloud", "logs", "query"]
    assert command[command.index("--service-url") + 1] == probe.logs_url
    assert command[command.index("--output") + 1] == "logs-raw"
    query = command[command.index("--query") + 1]
    for expression in ("$d.app == 'codeengine'", "$d.originator == 'user'",
                       "$d.codeengine.componentType == 'job_run'",
                       f"$d.codeengine.projectGuid == '{probe.project_id}'",
                       f"$d.codeengine.component == '{probe.job_name}'",
                       "$d.codeengine.subcomponentType == 'job_run'",
                       f"$d.codeengine.subcomponent == '{probe.job_run_name}'",
                       f"$d.message.message == '{probe.marker}'"):
        assert expression in query
    assert "--start-date" in command and "--end-date" in command
    assert "--limit" in command and command[command.index("--limit") + 1] == "2"


@pytest.mark.parametrize("error", [requests.Timeout("secret"), requests.ConnectionError("secret")])
def test_http_error_fails_closed_without_leaking_provider_exception(probe, providers, error):
    cli, http = providers
    http.side_effect = error
    with pytest.raises(RoutingProofUnavailable) as exc:
        collect_routing_probes(probe, now=NOW)
    assert "secret" not in str(exc.value)
    assert cli.call_count == 1


@pytest.mark.parametrize("status", [301, 401, 403, 404, 500])
def test_provider_http_failure_never_reaches_logs(probe, providers, status):
    cli, http = providers
    http.side_effect = None
    http.return_value = SimpleNamespace(status_code=status, text="secret")
    with pytest.raises(RoutingProofUnavailable):
        collect_routing_probes(probe, now=NOW)
    assert cli.call_count == 1


@pytest.mark.parametrize("raw", ['{"iam_token":"Bearer x","iam_token":"Bearer y"}',
                                  '{"iam_token":"secret"}', 'not-json'])
def test_malformed_or_ambiguous_session_token_blocks(probe, providers, raw):
    cli, http = providers
    cli.side_effect = None
    cli.return_value = SimpleNamespace(stdout=raw, stderr="", returncode=0)
    with pytest.raises(RoutingProofUnavailable):
        collect_routing_probes(probe, now=NOW)
    http.assert_not_called()


@pytest.mark.parametrize("stage", ["token", "logs"])
@pytest.mark.parametrize("error", [subprocess.TimeoutExpired("ibmcloud", 30),
                                   subprocess.CalledProcessError(1, "ibmcloud", stderr="secret"),
                                   FileNotFoundError("secret")])
def test_cli_error_fails_closed(probe, providers, stage, error):
    cli, _ = providers
    if stage == "token":
        cli.side_effect = error
    else:
        cli.side_effect = [SimpleNamespace(stdout='{"iam_token":"Bearer ephemeral-test-token"}', stderr=""), error]
    with pytest.raises(RoutingProofUnavailable) as exc:
        collect_routing_probes(probe, now=NOW)
    assert "secret" not in str(exc.value)


def test_logs_cli_warning_is_not_success(probe, providers):
    cli, _ = providers
    cli.side_effect = [SimpleNamespace(stdout='{"iam_token":"Bearer ephemeral-test-token"}', stderr=""),
                       SimpleNamespace(stdout=probe.marker, stderr="query incomplete")]
    with pytest.raises(RoutingProofUnavailable, match="ambiguous"):
        collect_routing_probes(probe, now=NOW)


def test_invalid_source_prevents_logs_query(probe, provider_evidence, providers):
    provider_evidence["job"]["run_compute_resource_token_enabled"] = False
    cli, http = providers
    http.side_effect = [SimpleNamespace(status_code=200, text=json.dumps(provider_evidence[key]))
                        for key in ("project", "job", "job_run", "trusted_profile",
                                    "trusted_profile_rules", "trusted_profile_links")]
    with pytest.raises(RoutingProofUnavailable):
        collect_routing_probes(probe, now=NOW)
    assert cli.call_count == 1


def test_main_blocks_without_canary_configuration(monkeypatch):
    monkeypatch.delenv("WATCH_AUTH_ACCOUNT_ID", raising=False)
    with pytest.raises(SystemExit, match="missing routing probe configuration"):
        main()


def test_main_keeps_monitoring_proof_independent(monkeypatch, probe, provider_evidence):
    environment = {
        "WATCH_AUTH_ACCOUNT_ID": probe.account_id, "WATCH_AUTH_REGION": probe.region,
        "WATCH_AUTH_CE_PROJECT_ID": probe.project_id, "WATCH_AUTH_CE_PROJECT": probe.project_name,
        "WATCH_AUTH_ROUTING_JOB_RUN": probe.job_run_name,
        "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": probe.service_account,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": probe.trusted_profile_id,
        "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": probe.image_digest,
        "WATCH_AUTH_PLATFORM_LOGS_CRN": probe.logs_crn, "WATCH_AUTH_ROUTING_PROBE_NONCE": probe.nonce,
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    collector = Mock(return_value=provider_evidence)
    monkeypatch.setattr(routing, "collect_routing_probes", collector)
    with pytest.raises(SystemExit, match="Monitoring routing proof is still unavailable"):
        main()
    collector.assert_called_once()


def test_deploy_missing_nonce_stops_before_any_ibm_cli_call(tmp_path):
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
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1", "WATCH_AUTH_CE_PROJECT": "watch-auth",
        "WATCH_AUTH_CE_PROJECT_ID": "11111111-1111-4111-8111-111111111111",
        "WATCH_AUTH_ROUTING_JOB_RUN": "watchdog-run",
        "WATCH_AUTH_COS_BUCKET": "watch-private", "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": "watchdog",
        "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": "private.us-south.icr.io/watch/watchdog@sha256:" + "a" * 64,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1", "WATCH_AUTH_PLATFORM_LOGS_CRN": "logs",
        "WATCH_AUTH_PLATFORM_MONITORING_CRN": "monitoring",
    }
    result = subprocess.run(["bash", "scripts/deploy_ibm_watchdog.sh"], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "WATCH_AUTH_ROUTING_PROBE_NONCE" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("key", ["run_compute_resource_token_enabled", "run_service_account",
                                  "image_reference"])
def test_run_snapshot_cannot_rely_on_mutable_current_job(probe, provider_evidence, key):
    del provider_evidence["job_run"][key]
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize(("field", "value"), [
    ("id", "Profile-other"), ("account_id", "other-account"), ("iam_id", None),
    ("iam_id", ""), ("crn", "crn:v1:bluemix:public:iam-identity:us-south:a/other::profile:Profile-watchdog"),
])
def test_iam_profile_identity_and_account_must_match(probe, provider_evidence, field, value):
    provider_evidence["trusted_profile"][field] = value
    with pytest.raises(RoutingProofUnavailable, match="IAM trusted profile"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize(("field", "value"), [
    ("crn", "crn:v1:bluemix:public:codeengine:us-south:a/other:project::"),
    ("component_type", "app"), ("component_name", "other-job"),
    ("component_name", "*"), ("crn", None), ("component_type", None),
])
def test_iam_ce_link_must_bind_exact_project_and_job(probe, provider_evidence, field, value):
    provider_evidence["trusted_profile_links"]["links"][0]["link"][field] = value
    with pytest.raises(RoutingProofUnavailable, match="expected project and job"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cross-account", "wrong-type", "no-id"])
def test_iam_ce_link_must_be_unique_and_same_account(probe, provider_evidence, mutation):
    links = provider_evidence["trusted_profile_links"]["links"]
    if mutation == "missing":
        links.clear()
    elif mutation == "duplicate":
        links.append(copy.deepcopy(links[0]))
    elif mutation == "cross-account":
        links[0]["is_cross_account"] = True
    elif mutation == "wrong-type":
        links[0]["cr_type"] = "VSI"
    else:
        del links[0]["id"]
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("rules", [None, [{}], [{"type": []}], [{"type": "Profile-CR", "cr_type": []}],
    [{"type": "Profile-CR"}],
    [{"type": "Profile-CR", "cr_type": "CE", "conditions": []}],
    [{"type": "Profile-CR", "cr_type": "CE", "conditions": [
        {"claim": "project", "operator": "EQUALS", "value": "other-project"}]}],
])
def test_unverified_or_incompatible_ce_rule_never_substitutes_for_direct_link(probe, provider_evidence, rules):
    provider_evidence["trusted_profile_rules"]["rules"] = rules
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


def test_env_only_profile_reproduction_is_now_blocked(probe, provider_evidence):
    for key in ("trusted_profile", "trusted_profile_rules", "trusted_profile_links"):
        provider_evidence.pop(key)
    with pytest.raises(RoutingProofUnavailable, match="IAM trusted profile"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_authenticated_iam_failure_never_queries_logs(probe, providers, status):
    cli, http = providers
    http.side_effect = list(http.side_effect)[:3] + [SimpleNamespace(status_code=status, text="private")]
    with pytest.raises(RoutingProofUnavailable, match="trusted_profile readback failed"):
        collect_routing_probes(probe, now=NOW)
    assert cli.call_count == 1


@pytest.mark.parametrize("links", [None, [{}], [{"cr_type": []}], [{"cr_type": "CE", "link": None}]])
def test_malformed_iam_links_fail_with_concrete_blocker(probe, provider_evidence, links):
    provider_evidence["trusted_profile_links"]["links"] = links
    with pytest.raises(RoutingProofUnavailable, match="IAM"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)
