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
        monitoring_crn="crn:v1:bluemix:public:sysdig-monitor:us-south:a/account-1:44444444-4444-4444-8444-444444444444::",
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
    target = {
        "id": "55555555-5555-4555-8555-555555555555",
        "crn": f"crn:v1:bluemix:public:metrics-router:{probe.region}:a/{probe.account_id}::target:55555555-5555-4555-8555-555555555555",
        "destination_crn": probe.monitoring_crn, "target_type": "sysdig_monitor",
        "write_status": {"status": "success"}, "managed_by": "account",
    }
    instances = {label: {"guid": crn.split(":")[7], "crn": crn, "account_id": probe.account_id,
                         "region_id": probe.region, "state": "active"}
                 for label, crn in (("logs_instance", probe.logs_crn),
                                   ("monitoring_instance", probe.monitoring_crn))}
    return {
        **instances,
        "metrics_targets": {"targets": [target]},
        "metrics_routes": {"routes": []},
        "metrics_settings": {"default_targets": [{key: target[key] for key in ("id", "crn", "target_type")}],
                             "permitted_target_regions": [probe.region]},
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
                                         "trusted_profile_rules", "trusted_profile_links", "logs_instance",
                                         "monitoring_instance", "metrics_targets", "metrics_routes", "metrics_settings")])
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
         for suffix in ("", "/rules", "/links")] + [
        "https://resource-controller.cloud.ibm.com/v2/resource_instances/" + crn.split(":")[7]
        for crn in (probe.logs_crn, probe.monitoring_crn)
    ] + [f"https://{probe.region}.metrics-router.cloud.ibm.com/api/v3/{name}"
         for name in ("targets", "routes", "settings")]
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


def test_main_accepts_provider_verified_control_plane(monkeypatch, probe, provider_evidence):
    environment = {
        "WATCH_AUTH_ACCOUNT_ID": probe.account_id, "WATCH_AUTH_REGION": probe.region,
        "WATCH_AUTH_CE_PROJECT_ID": probe.project_id, "WATCH_AUTH_CE_PROJECT": probe.project_name,
        "WATCH_AUTH_ROUTING_JOB_RUN": probe.job_run_name,
        "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": probe.service_account,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": probe.trusted_profile_id,
        "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": probe.image_digest,
        "WATCH_AUTH_PLATFORM_LOGS_CRN": probe.logs_crn,
        "WATCH_AUTH_PLATFORM_MONITORING_CRN": probe.monitoring_crn, "WATCH_AUTH_ROUTING_PROBE_NONCE": probe.nonce,
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    collector = Mock(return_value=provider_evidence)
    monkeypatch.setattr(routing, "collect_routing_probes", collector)
    main()
    collector.assert_called_once()


@pytest.mark.parametrize("blocker", ["missing-nonce", "invalid-monitoring"])
def test_deploy_invalid_proof_stops_before_any_ibm_cli_call(tmp_path, blocker):
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
    if blocker == "invalid-monitoring":
        environment["WATCH_AUTH_ROUTING_PROBE_NONCE"] = "33333333-3333-4333-8333-333333333333"
        environment["WATCH_AUTH_PLATFORM_LOGS_CRN"] = "crn:v1:bluemix:public:logs:us-south:a/account-1:22222222-2222-4222-8222-222222222222::"
    result = subprocess.run(["bash", "scripts/deploy_ibm_watchdog.sh"], cwd=root, env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    expected = "WATCH_AUTH_ROUTING_PROBE_NONCE" if blocker == "missing-nonce" else "observability destination"
    assert expected in result.stderr
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


@pytest.mark.parametrize("section", ["logs_instance", "monitoring_instance", "metrics_targets",
                                      "metrics_routes", "metrics_settings"])
def test_missing_monitoring_control_plane_readback_blocks(probe, provider_evidence, section):
    del provider_evidence[section]
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("section", ["logs_instance", "monitoring_instance"])
@pytest.mark.parametrize(("field", "value"), [
    ("guid", "other-instance"), ("crn", "other-crn"), ("account_id", "other-account"),
    ("region_id", "eu-de"), ("state", "removed"), ("state", None),
])
def test_observability_instances_are_exact_active_account_and_region(probe, provider_evidence, section, field, value):
    provider_evidence[section][field] = value
    with pytest.raises(RoutingProofUnavailable, match="observability instance"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize(("field", "value"), [
    ("id", None), ("id", "other-instance"), ("crn", "other-account-or-region"),
    ("destination_crn", "other-monitoring-instance"), ("target_type", "logs"),
    ("write_status", None), ("write_status", {"status": "failed", "reason": "secret"}),
    ("write_status", {"status": True}),
])
def test_monitoring_target_identity_and_integration_are_provider_verified(probe, provider_evidence, field, value):
    provider_evidence["metrics_targets"]["targets"][0][field] = value
    with pytest.raises(RoutingProofUnavailable) as error:
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "duplicate-id", "malformed"])
def test_missing_or_ambiguous_monitoring_target_blocks(probe, provider_evidence, mutation):
    targets = provider_evidence["metrics_targets"]["targets"]
    if mutation == "missing":
        targets.clear()
    elif mutation == "duplicate":
        targets.append(copy.deepcopy(targets[0]))
    elif mutation == "duplicate-id":
        targets.append({**targets[0], "destination_crn": "other"})
    else:
        targets.append(None)
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("defaults", [None, [], [None], [{"id": "wrong"}], [{"id": "x"}, {"id": "x"}]])
def test_missing_or_ambiguous_default_target_blocks(probe, provider_evidence, defaults):
    provider_evidence["metrics_settings"]["default_targets"] = defaults
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.fixture
def explicit_route(probe, provider_evidence):
    target = provider_evidence["metrics_settings"]["default_targets"][0]
    rule = {"action": "send", "targets": [copy.deepcopy(target)], "inclusion_filters": [
        {"operand": "location", "operator": "is", "values": [probe.region]},
        {"operand": "service_name", "operator": "is", "values": ["is"]},
    ]}
    route = {"id": "66666666-6666-4666-8666-666666666666",
             "crn": f"crn:v1:bluemix:public:metrics-router:global:a/{probe.account_id}::route:66666666-6666-4666-8666-666666666666",
             "rules": [rule]}
    provider_evidence["metrics_routes"]["routes"] = [route]
    provider_evidence["metrics_settings"]["default_targets"] = []
    return route


@pytest.mark.parametrize("filters", [[], ["location"], ["service_name"], ["location", "service_name"]])
def test_single_official_send_route_covers_platform_vpc_metrics(probe, provider_evidence, explicit_route, filters):
    rule = explicit_route["rules"][0]
    rule["inclusion_filters"] = [item for item in rule["inclusion_filters"] if item["operand"] in filters]
    verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("mutation", ["drop", "other-account", "other-region", "other-service", "resource-filter",
    "duplicate-filter", "multiple-rules", "multiple-routes", "other-target", "multiple-targets"])
def test_divergent_or_complex_routes_remain_blocked(probe, provider_evidence, explicit_route, mutation):
    rule = explicit_route["rules"][0]
    if mutation == "drop":
        rule["action"] = "drop"
    elif mutation == "other-account":
        explicit_route["crn"] = explicit_route["crn"].replace(probe.account_id, "other")
    elif mutation == "other-region":
        rule["inclusion_filters"][0]["values"] = ["eu-de"]
    elif mutation == "other-service":
        rule["inclusion_filters"][1]["values"] = ["codeengine"]
    elif mutation == "resource-filter":
        rule["inclusion_filters"].append({"operand": "resource", "operator": "is", "values": ["another-vsi"]})
    elif mutation == "duplicate-filter":
        rule["inclusion_filters"].append(copy.deepcopy(rule["inclusion_filters"][0]))
    elif mutation == "multiple-rules":
        explicit_route["rules"].append(copy.deepcopy(rule))
    elif mutation == "multiple-routes":
        provider_evidence["metrics_routes"]["routes"].append(copy.deepcopy(explicit_route))
    elif mutation == "other-target":
        rule["targets"][0]["id"] = "other"
    else:
        rule["targets"].append(copy.deepcopy(rule["targets"][0]))
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("permitted", [None, "us-south", ["eu-de"], [None]])
def test_metrics_permitted_regions_fail_closed(probe, provider_evidence, permitted):
    provider_evidence["metrics_settings"]["permitted_target_regions"] = permitted
    with pytest.raises(RoutingProofUnavailable):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)


@pytest.mark.parametrize("stage", range(6, 11))
@pytest.mark.parametrize("failure", [401, 403, 404, 500, 302, requests.Timeout("secret")])
def test_monitoring_provider_failure_prevents_logs_query(probe, providers, stage, failure):
    cli, http = providers
    responses = list(http.side_effect)
    responses[stage] = (failure if isinstance(failure, Exception)
                        else SimpleNamespace(status_code=failure, text="secret"))
    http.side_effect = responses
    with pytest.raises(RoutingProofUnavailable) as error:
        collect_routing_probes(probe, now=NOW)
    assert "secret" not in str(error.value)
    assert cli.call_count == 1


@pytest.mark.parametrize("change", [
    lambda crn: crn.replace("account-1", "other-account"),
    lambda crn: crn.replace("us-south", "eu-de"),
    lambda crn: crn.replace("sysdig-monitor", "logs"),
    lambda crn: crn.replace("44444444-4444-4444-8444-444444444444", "attacker.example"),
])
def test_monitoring_destination_cannot_be_self_declared_elsewhere(probe, change):
    with pytest.raises(RoutingProofUnavailable):
        replace(probe, monitoring_crn=change(probe.monitoring_crn))


def test_returned_monitoring_proof_never_contains_session_bearer(probe, providers):
    proof = collect_routing_probes(probe, now=NOW)
    assert "ephemeral-test-token" not in json.dumps(proof)
    assert "Authorization" not in json.dumps(proof)


@pytest.mark.parametrize(("field", "value"), [("crn", "other-account"), ("target_type", "logs")])
def test_explicit_route_target_descriptor_cannot_diverge(probe, provider_evidence, explicit_route, field, value):
    explicit_route["rules"][0]["targets"][0][field] = value
    with pytest.raises(RoutingProofUnavailable, match="destination"):
        verify_routing_probes(provider_evidence, probe=probe, now=NOW)
