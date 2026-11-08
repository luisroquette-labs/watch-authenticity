"""Provider-shaped bootstrap tests: no cloud credentials or writes."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from infra.ibm_registry_bootstrap import IMAGE, JOB, SECRET, SERVICE, BootstrapBlocked, bootstrap

ENV = {
    "WATCH_AUTH_ACCOUNT_ID": "account-1", "WATCH_AUTH_RESOURCE_GROUP": "watch-auth",
    "WATCH_AUTH_RESOURCE_GROUP_ID": "rg-1", "WATCH_AUTH_REGION": "us-south",
    "WATCH_AUTH_CE_PROJECT": "watch-auth", "WATCH_AUTH_CE_PROJECT_ID": "project-1",
    "WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT": "none", "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": IMAGE,
    "WATCH_AUTH_TRUSTED_PROFILE_ID": "Profile-watchdog", "WATCH_AUTH_COS_BUCKET": "private-bucket",
    "WATCH_AUTH_COS_ENDPOINT": "https://s3.private.us-south.cloud-object-storage.appdomain.cloud",
    "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
}
PROJECT_CRN = "crn:v1:bluemix:public:codeengine:us-south:a/account-1:project-1::"
SERVICE_ID = {"id": "ServiceId-pull", "iam_id": "iam-ServiceId-pull", "name": SERVICE,
              "account_id": "account-1"}
API_KEY_ID = "ApiKey-11111111-2222-3333-4444-555555555555"


class Provider:
    def __init__(self):
        self.calls = []
        self.target = {"api_endpoint": "https://cloud.ibm.com", "account": {"guid": "account-1"},
                       "user": {"user_email": "owner@example.com"},
                       "resource_group": {"name": "watch-auth"},
                       "region": {"name": "us-south"}}
        self.target["account"]["owner"] = "owner@example.com"
        self.resource_group_id = "rg-1"
        self.selected = {"guid": "project-1", "name": "watch-auth"}
        self.project = {"id": "project-1", "name": "watch-auth", "crn": PROJECT_CRN}
        self.instance = {"account_id": "account-1", "resource_group_id": "rg-1",
                         "crn": PROJECT_CRN, "state": "active"}
        self.ids = []
        self.policy_items = []
        self.keys = []
        self.links = []
        self.rules = []
        self.groups = [{"id": "AccessGroupId-PublicAccess", "name": "Public Access"}]
        self.public_access_policies = {"v1": [], "v2": []}
        self.operator_policies = []
        self.secret = None
        self.job = None
        self.subscriptions = {
            kind: {f"{kind}_subscriptions": [], "limit": 0}
            for kind in ("cron", "cos", "kafka")
        }
        self.job_runs = {"job_runs": []}
        self.fail = None
        self.key_path = None
        self.leaked = "sensitive-fixture-key-never-log"

    def cli(self, *args):
        self.calls.append(args)
        if args[0] == "target":
            return json.dumps(self.target)
        if args[:2] == ("resource", "group"):
            return self.resource_group_id + "\n"
        if args[:3] == ("ce", "project", "current"):
            return json.dumps(self.selected)
        if args[:2] == ("iam", "oauth-tokens"):
            return json.dumps({"iam_token": "Bearer bootstrap-session"})
        if args[:2] == ("iam", "access-group-policies"):
            version = args[args.index("--api-version") + 1]
            return json.dumps(self.public_access_policies[version])
        if args[:2] == ("iam", "user-policies"):
            assert args[2] == "owner@example.com"
            return json.dumps(self.operator_policies)
        if args[:2] == ("ce", "sub") and args[3] == "list":
            return json.dumps(self.subscriptions[args[2]])
        if args[1] == "service-id-create":
            self.ids = [copy.deepcopy(SERVICE_ID)]
        elif args[1] == "user-policy-create":
            assert args == (
                "iam", "user-policy-create", "owner@example.com", "--roles", "Writer",
                "--service-name", "codeengine", "--service-instance", "project-1",
                "--output", "JSON",
            )
            self.operator_policies = [{"resources": [{"attributes": [
                {"name": "accountId", "value": "account-1"},
                {"name": "serviceName", "value": "codeengine"},
                {"name": "serviceInstance", "value": "project-1"},
            ]}], "roles": [{"role_id": "crn:v1:bluemix:public:iam::::serviceRole:Writer"}]}]
        elif args[1] == "service-policy-create":
            raise AssertionError("policy creation must use the authenticated IAM API")
        elif args[1] == "service-api-key-create":
            self.key_path = Path(args[args.index("--file") + 1])
            assert self.key_path.stat().st_mode & 0o777 == 0o600
            self.key_path.write_text(json.dumps({"apikey": self.leaked}))
            self.keys = [{"name": SERVICE, "id": API_KEY_ID, "iam_id": SERVICE_ID["iam_id"]}]
        elif args[:3] == ("ce", "registry", "list"):
            return json.dumps({"metadata": {"resourceVersion": "fixture"},
                               "items": [] if self.secret is None else [self.secret]})
        elif args[:3] == ("ce", "registry", "create"):
            path = Path(args[args.index("--password-from-json-file") + 1])
            assert path == self.key_path and path.stat().st_mode & 0o777 == 0o600
            assert json.loads(path.read_text())["apikey"] == self.leaked
            if self.fail == "secret":
                raise BootstrapBlocked("secret create failed")
            self.secret = {"name": SECRET, "project_id": "project-1", "format": "registry",
                           "data": {"server": "us.icr.io", "username": "iamapikey", "password": self.leaked}}
            if self.fail == "secret-response-lost":
                raise BootstrapBlocked("secret response lost")
            if self.fail == "secret-not-persisted":
                self.secret = None
        elif args[:2] == ("ce", "job"):
            assert self.secret is not None
            assert self.key_path is None or not self.key_path.exists()
            assert args[args.index("--image") + 1] == IMAGE
            assert args[args.index("--registry-secret") + 1] == SECRET
            self.job = {"image_reference": IMAGE, "image_secret": SECRET,
                        "run_service_account": "none", "run_compute_resource_token_enabled": True,
                        "run_commands": ["python3"], "run_arguments": ["-m", "infra.ibm_watchdog"],
                        "run_env_variables": [
                            {"name": value.split("=", 1)[0], "type": "literal", "value": value.split("=", 1)[1]}
                            for key, value in zip(args, args[1:]) if key == "--env"
                        ]}
        elif args[1] == "trusted-profile-link-create":
            self.links = [{"id": "link-1", "cr_type": "CE", "link": {
                "crn": PROJECT_CRN, "component_type": "job", "component_name": JOB,
            }}]
        else:
            raise AssertionError(args)
        return "suppressed CLI output " + self.leaked

    def get(self, url, **kwargs):
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"]["Authorization"] == "Bearer bootstrap-session"
        if url.endswith("/resource_instances/project-1"):
            result = self.instance
        elif url.endswith("/projects/project-1"):
            result = self.project
        elif url.endswith("/profiles/Profile-watchdog"):
            result = {"id": "Profile-watchdog", "account_id": "account-1"}
        elif url.endswith("/rules"):
            result = {"rules": self.rules}
        elif url.endswith("/links"):
            result = {"links": self.links}
        elif url.endswith("/serviceids/"):
            result = {"serviceids": self.ids}
        elif url.endswith("/groups"):
            result = {"groups": self.groups, "limit": 50, "offset": 0,
                      "total_count": len(self.groups)}
        elif url.endswith("/v1/policies"):
            assert kwargs["params"]["account_id"] == "account-1"
            result = {"policies": self.policy_items}
        elif url.endswith("/apikeys"):
            result = {"apikeys": self.keys}
        elif url.endswith("/apikeys/details"):
            assert kwargs["headers"]["IAM-ApiKey"] == self.leaked
            result = {"name": SERVICE, "id": API_KEY_ID, "account_id": "account-1",
                      "iam_id": SERVICE_ID["iam_id"]}
        elif url.endswith(f"/secrets/{SECRET}"):
            result = self.secret
        elif url.endswith(f"/jobs/{JOB}"):
            result = self.job
        elif url.endswith("/job_runs"):
            assert kwargs["params"]["job_name"] == JOB
            result = self.job_runs
        else:
            raise AssertionError(url)
        return SimpleNamespace(status_code=404 if result is None else 200, text=json.dumps(result))

    def post(self, url, **kwargs):
        self.calls.append(("http", "post", url))
        assert url == "https://iam.cloud.ibm.com/v1/policies"
        assert kwargs["allow_redirects"] is False and kwargs["timeout"] == 30
        assert kwargs["headers"] == {
            "Authorization": "Bearer bootstrap-session", "Accept": "application/json",
            "Content-Type": "application/json",
        }
        body = kwargs["json"]
        assert body["subjects"] == [{"attributes": [
            {"name": "iam_id", "value": SERVICE_ID["iam_id"]},
        ]}]
        assert body["roles"] == [{
            "role_id": "crn:v1:bluemix:public:iam::::serviceRole:Reader",
        }]
        assert {item["name"]: (item["value"], item["operator"])
                for item in body["resources"][0]["attributes"]} == {
            "accountId": ("account-1", "stringEquals"),
            "serviceName": ("container-registry", "stringEquals"),
            "region": ("us-south", "stringEquals"),
            "resourceType": ("namespace", "stringEquals"),
            "resource": ("watch-authenticity-3235996", "stringEquals"),
        }
        if self.fail == "policy":
            return SimpleNamespace(status_code=403, text="sensitive-provider-error")
        self.policy_items = [copy.deepcopy(body)]
        return SimpleNamespace(status_code=201, text=json.dumps(body))

    def delete(self, url, **kwargs):
        self.calls.append(("http", "delete", url))
        assert kwargs["allow_redirects"] is False and kwargs["timeout"] == 30
        assert kwargs["headers"]["Authorization"] == "Bearer bootstrap-session"
        if self.fail == "key-delete":
            return SimpleNamespace(status_code=403, text="sensitive-provider-error")
        assert url == "https://iam.cloud.ibm.com/v1/apikeys/" + API_KEY_ID
        self.keys = []
        return SimpleNamespace(status_code=204, text="")


def run(provider, tmp_path, env=None):
    tmp_path.chmod(0o700)
    bootstrap(env or ENV, tmp_path / "key.json", cli=provider.cli, http=provider)


def mutations(provider):
    return [call for call in provider.calls if not (
        call[0] == "target" or call[:3] == ("ce", "project", "current")
        or call[:2] == ("resource", "group")
        or call[:2] == ("iam", "oauth-tokens")
        or call[:2] == ("iam", "access-group-policies")
        or call[:2] == ("iam", "user-policies")
        or (call[:2] == ("ce", "sub") and call[3] == "list")
        or call[:3] == ("ce", "registry", "list")
    )]


@pytest.mark.parametrize("kind", ["cron", "cos", "kafka"])
def test_existing_subscription_blocks_bootstrap_before_any_mutation(tmp_path, kind):
    provider = Provider()
    provider.subscriptions[kind] = {f"{kind}_subscriptions": [{"spec": {"sink": {
        "ref": {"kind": "Job", "name": JOB},
    }}}], "limit": 1}
    with pytest.raises(BootstrapBlocked, match="subscription"):
        run(provider, tmp_path)
    assert mutations(provider) == []
    assert not (tmp_path / "key.json").exists()


@pytest.mark.parametrize("status", ["running", "pending", "unknown", None])
def test_active_or_incomplete_job_run_blocks_bootstrap_before_any_mutation(tmp_path, status):
    provider = Provider()
    provider.job_runs = {"job_runs": [{"job_name": JOB, "status": status}]}
    with pytest.raises(BootstrapBlocked, match="job run"):
        run(provider, tmp_path)
    assert mutations(provider) == []
    assert not (tmp_path / "key.json").exists()


@pytest.mark.parametrize("inventory", [
    {}, {"cron_subscriptions": None}, {"cron_subscriptions": [None]},
    {"cron_subscriptions": [{}]},
    {"cron_subscriptions": [], "metadata": {"continue": "next-page"}},
    {"cron_subscriptions": [{"spec": {"sink": {
        "uri": "https://unknown-destination",
    }}}]},
])
def test_missing_malformed_or_paginated_subscription_readback_blocks_all_mutation(tmp_path, inventory):
    provider = Provider()
    provider.subscriptions["cron"] = inventory
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    assert mutations(provider) == []


@pytest.mark.parametrize("inventory", [
    {}, {"job_runs": None}, {"job_runs": [{}]}, {"job_runs": [], "next": {"start": "next"}},
    {"job_runs": [{"job_name": JOB, "status": "succeeded"}]},
    {"job_runs": [{"job_name": JOB, "status": "failed", "status_details": {
        "completion_time": "2026-11-08T10:00:00Z", "running": 1,
    }}]},
])
def test_missing_malformed_or_paginated_run_readback_blocks_all_mutation(tmp_path, inventory):
    provider = Provider()
    provider.job_runs = inventory
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    assert mutations(provider) == []


@pytest.mark.parametrize("status", ["succeeded", "failed"])
def test_terminal_run_and_unrelated_subscription_allow_inert_bootstrap(tmp_path, status):
    provider = Provider()
    provider.job_runs = {"job_runs": [{"job_name": JOB, "status": status, "status_details": {
        "completion_time": "2026-11-08T10:00:00Z", "running": 0, "pending": 0, "unknown": 0,
    }}]}
    provider.subscriptions["cron"] = {"cron_subscriptions": [{"spec": {"sink": {
        "ref": {"kind": "Job", "name": "unrelated-job"},
    }}}], "limit": 1}
    run(provider, tmp_path)
    assert provider.job is not None


def test_full_deploy_routing_failure_blocks_before_any_provider_mutation(tmp_path, monkeypatch):
    provider = Provider()
    tmp_path.chmod(0o700)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)

    def failed_gate():
        raise SystemExit("routing verification blocked")

    monkeypatch.setattr("infra.ibm_registry_bootstrap.verify_routing", failed_gate)
    with pytest.raises(SystemExit, match="routing verification blocked"):
        bootstrap(ENV, tmp_path / "key.json", cli=provider.cli, http=provider, deploy=True)
    assert provider.calls == []


def test_full_deploy_cannot_reuse_routing_gate_for_another_target(tmp_path, monkeypatch):
    provider = Provider()
    tmp_path.chmod(0o700)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("WATCH_AUTH_CE_PROJECT_ID", "other-project")
    with pytest.raises(BootstrapBlocked, match="routing gate environment"):
        bootstrap(ENV, tmp_path / "key.json", cli=provider.cli, http=provider, deploy=True)
    assert provider.calls == []


@pytest.mark.parametrize("active_run", [False, True])
def test_full_deploy_existing_trigger_requires_real_gate_and_still_blocks_active_run(
    tmp_path, monkeypatch, active_run,
):
    provider = Provider()
    tmp_path.chmod(0o700)
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    provider.subscriptions["cron"] = {"cron_subscriptions": [{"spec": {"sink": {
        "ref": {"kind": "Job", "name": JOB},
    }}}], "limit": 1}
    if active_run:
        provider.job_runs = {"job_runs": [{"job_name": JOB, "status": "running"}]}
    gates = []
    monkeypatch.setattr("infra.ibm_registry_bootstrap.verify_routing", lambda: gates.append("PASS"))
    if active_run:
        with pytest.raises(BootstrapBlocked, match="active or incomplete"):
            bootstrap(ENV, tmp_path / "key.json", cli=provider.cli, http=provider, deploy=True)
        assert mutations(provider) == []
    else:
        bootstrap(ENV, tmp_path / "key.json", cli=provider.cli, http=provider, deploy=True)
        assert provider.job is not None
    assert gates == ["PASS"]


@pytest.mark.parametrize("field", ["account", "resource_group", "region", "api_endpoint", "project", "group"])
def test_target_mismatch_stops_before_any_mutation(tmp_path, field):
    provider = Provider()
    if field == "project":
        provider.selected["guid"] = "other"
    elif field == "group":
        provider.instance["resource_group_id"] = "other"
    else:
        provider.target[field] = {} if field != "api_endpoint" else "https://private.cloud.ibm.com"
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    assert all(args[0] == "target" or args[:3] == ("ce", "project", "current")
               or args[:2] == ("resource", "group")
               or args[:2] == ("iam", "oauth-tokens") for args in provider.calls)


@pytest.mark.parametrize("observed_id", ["", "other", "rg-1\nother"])
def test_resource_group_readback_must_bind_selected_target_before_mutation(tmp_path, observed_id):
    provider = Provider()
    provider.resource_group_id = observed_id
    with pytest.raises(BootstrapBlocked, match="resource group"):
        run(provider, tmp_path)
    assert mutations(provider) == []


@pytest.mark.parametrize("image", [IMAGE.replace("bf169b", "af169b"), IMAGE + ":tag", "us.icr.io/watch/watchdog:latest"])
def test_only_exact_approved_digest_is_accepted(tmp_path, image):
    provider = Provider()
    with pytest.raises(BootstrapBlocked, match="approved digest"):
        run(provider, tmp_path, {**ENV, "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST": image})
    assert provider.calls == []


def test_bootstrap_reconciles_without_rotation_and_deletes_local_key_before_job(tmp_path, capsys):
    provider = Provider()
    run(provider, tmp_path)
    assert not (tmp_path / "key.json").exists()
    first = list(provider.calls)
    provider.calls.clear()
    provider.job["run_env_variables"].reverse()
    run(provider, tmp_path)
    assert mutations(provider) == []
    create_key = next(call for call in first if call[1] == "service-api-key-create")
    assert create_key[3] == SERVICE and "--file" in create_key and provider.leaked not in str(first)
    assert ("http", "post", "https://iam.cloud.ibm.com/v1/policies") in first
    assert all(call[3] == "list" for call in first if call[:2] == ("ce", "sub"))
    assert not any("submit" in call for call in first)
    assert provider.leaked not in str(capsys.readouterr())


def test_policy_creation_failure_stops_before_key_secret_and_job(tmp_path, capsys):
    provider = Provider()
    provider.fail = "policy"
    with pytest.raises(BootstrapBlocked, match="policy creation failed"):
        run(provider, tmp_path)
    assert provider.keys == [] and provider.secret is None and provider.job is None
    assert "sensitive-provider-error" not in str(capsys.readouterr())


@pytest.mark.parametrize("failure", ["secret", "secret-not-persisted"])
def test_secret_failure_replaces_orphan_key_once_before_retrying_secret(tmp_path, failure):
    provider = Provider()
    provider.fail = failure
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    assert not any(call[:2] == ("ce", "job") for call in provider.calls)
    (tmp_path / "key.json").unlink(missing_ok=True)  # shell trap in production
    provider.fail = None
    run(provider, tmp_path)
    assert sum(call[:2] == ("http", "delete") for call in provider.calls) == 1
    assert sum(call[1] == "service-api-key-create" for call in provider.calls) == 2
    assert provider.job is not None


def test_orphan_key_deletion_failure_blocks_before_new_key_secret_or_job(tmp_path):
    provider = Provider()
    provider.fail = "secret"
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    (tmp_path / "key.json").unlink(missing_ok=True)  # shell trap in production
    provider.fail = "key-delete"
    with pytest.raises(BootstrapBlocked, match="key deletion failed"):
        run(provider, tmp_path)
    assert sum(call[1] == "service-api-key-create" for call in provider.calls) == 1
    assert provider.secret is None and provider.job is None


@pytest.mark.parametrize("change", ["role", "scope", "operator", "duplicate", "broad"])
def test_divergent_or_duplicate_code_engine_operator_policy_blocks(tmp_path, change):
    provider = Provider()
    exact = {"resources": [{"attributes": [
        {"name": "accountId", "operator": "stringEquals", "value": "account-1"},
        {"name": "serviceName", "operator": "stringEquals", "value": "codeengine"},
        {"name": "serviceInstance", "operator": "stringEquals", "value": "project-1"},
    ]}], "roles": [{"role_id": "crn:v1:bluemix:public:iam::::serviceRole:Writer"}]}
    provider.operator_policies = [copy.deepcopy(exact)]
    if change == "role":
        provider.operator_policies[0]["roles"][0]["role_id"] = "crn:v1:bluemix:public:iam::::serviceRole:Manager"
    elif change == "scope":
        provider.operator_policies[0]["resources"][0]["attributes"].append(
            {"name": "resourceGroupId", "operator": "stringEquals", "value": "rg-1"})
    elif change == "operator":
        provider.operator_policies[0]["resources"][0]["attributes"][0]["operator"] = "stringNotEquals"
    elif change == "duplicate":
        provider.operator_policies.append(copy.deepcopy(exact))
    else:
        provider.operator_policies[0]["resources"][0]["attributes"].pop()
    with pytest.raises(BootstrapBlocked, match="operator polic|duplicate Code Engine|broad Code Engine"):
        run(provider, tmp_path)
    assert not any(call[1] == "service-api-key-create" for call in provider.calls)


@pytest.mark.parametrize("value,operator", [
    (None, "stringEquals"), ("", "stringEquals"), ("*", "stringEquals"),
    ("other-project", "stringNotEquals"),
])
def test_nonconcrete_code_engine_policy_scope_blocks(tmp_path, value, operator):
    provider = Provider()
    provider.operator_policies = [{"resources": [{"attributes": [
        {"name": "accountId", "operator": "stringEquals", "value": "account-1"},
        {"name": "serviceName", "operator": "stringEquals", "value": "codeengine"},
        {"name": "serviceInstance", "operator": operator, "value": value},
    ]}], "roles": [{"role_id": "crn:v1:bluemix:public:iam::::serviceRole:Writer"}]}]
    with pytest.raises(BootstrapBlocked, match="broad Code Engine"):
        run(provider, tmp_path)
    assert not any(call[1] == "service-api-key-create" for call in provider.calls)


def test_non_owner_session_blocks_operator_policy_creation(tmp_path):
    provider = Provider()
    provider.target["user"]["user_email"] = "member@example.com"
    with pytest.raises(BootstrapBlocked, match="not the account owner"):
        run(provider, tmp_path)
    assert mutations(provider) == []


@pytest.mark.parametrize("bad_id", ["", "ApiKey-pull", "../profiles/target", API_KEY_ID + "/../other"])
def test_malformed_or_pathlike_orphan_key_id_blocks_deletion(tmp_path, bad_id):
    provider = Provider()
    provider.fail = "secret"
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    (tmp_path / "key.json").unlink(missing_ok=True)  # shell trap in production
    provider.keys[0]["id"] = bad_id
    provider.fail = None
    with pytest.raises(BootstrapBlocked, match="unexpected registry API keys"):
        run(provider, tmp_path)
    assert not any(call[:2] == ("http", "delete") for call in provider.calls)


def test_registry_cli_and_api_inventory_divergence_blocks_key_mutation(tmp_path):
    provider = Provider()
    original_cli = provider.cli

    def divergent_cli(*args):
        if args[:3] == ("ce", "registry", "list"):
            provider.calls.append(args)
            return json.dumps({"metadata": {"resourceVersion": "fixture"},
                               "items": [{"name": SECRET}]})
        return original_cli(*args)

    provider.cli = divergent_cli
    with pytest.raises(BootstrapBlocked, match="inventories differ"):
        run(provider, tmp_path)
    assert not any(call[1] == "service-api-key-create" for call in provider.calls)


@pytest.mark.parametrize("inventory", [
    [], {}, {"items": None}, {"items": [None]}, {"items": []},
    {"items": [], "metadata": []}, {"items": [], "metadata": {}},
    {"items": [{}], "metadata": {"resourceVersion": "fixture"}},
    {"items": [{"name": "../wrong"}], "metadata": {"resourceVersion": "fixture"}},
    {"items": [], "metadata": {"resourceVersion": "fixture", "continue": "next-page"}},
])
def test_malformed_or_paginated_registry_inventory_blocks_key_mutation(tmp_path, inventory):
    provider = Provider()
    provider.fail = "secret"
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    (tmp_path / "key.json").unlink(missing_ok=True)  # shell trap in production
    provider.fail = None
    original_cli = provider.cli

    def malformed_cli(*args):
        if args[:3] == ("ce", "registry", "list"):
            provider.calls.append(args)
            return json.dumps(inventory)
        return original_cli(*args)

    provider.cli = malformed_cli
    with pytest.raises(BootstrapBlocked, match="inventory"):
        run(provider, tmp_path)
    assert provider.keys == [{"name": SERVICE, "id": API_KEY_ID,
                              "iam_id": SERVICE_ID["iam_id"]}]
    assert not any(call[:2] == ("http", "delete") for call in provider.calls)
    assert sum(call[1] == "service-api-key-create" for call in provider.calls) == 1


def test_lost_secret_create_response_is_reconciled_without_a_second_key(tmp_path):
    provider = Provider()
    provider.fail = "secret-response-lost"
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    (tmp_path / "key.json").unlink()  # shell trap in production
    provider.fail = None
    run(provider, tmp_path)
    assert sum(call[1] == "service-api-key-create" for call in provider.calls) == 1
    assert provider.job is not None and len(provider.links) == 1


def test_stale_job_is_updated_only_after_bound_secret_readback(tmp_path):
    provider = Provider()
    run(provider, tmp_path)
    provider.calls.clear()
    provider.job["image_reference"] = "old-image"
    run(provider, tmp_path)
    assert [call[:3] for call in mutations(provider)] == [("ce", "job", "update")]
    assert provider.job["image_reference"] == IMAGE


def test_cli_failure_never_exposes_provider_output_or_trace(monkeypatch, capsys):
    from infra.ibm_registry_bootstrap import _cli

    def fail(_args, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["env"]["IBMCLOUD_TRACE"] == "false"
        return SimpleNamespace(returncode=1, stdout="secret-value", stderr="secret-value")

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(BootstrapBlocked, match="IBM CLI command failed") as error:
        _cli("iam", "service-api-key-create")
    assert "secret-value" not in str(error.value) + str(capsys.readouterr())
    assert "iam/service-api-key-create" in str(error.value)


@pytest.mark.parametrize("change", ["wide-policy", "operator", "duplicate-subject", "v2-policy",
                                    "inherited-group",
                                    "public-policy-v1", "public-policy-v2", "other-job-link",
                                    "ce-rule"])
def test_existing_broad_permissions_or_trust_fail_closed(tmp_path, change):
    provider = Provider()
    run(provider, tmp_path)
    provider.calls.clear()
    if change == "wide-policy":
        provider.policy_items[0]["resources"][0]["attributes"].pop()
    elif change == "operator":
        provider.policy_items[0]["resources"][0]["attributes"][0]["operator"] = "stringNotEquals"
    elif change == "duplicate-subject":
        provider.policy_items[0]["subjects"][0]["attributes"].insert(
            0, {"name": "iam_id", "value": "other-principal"})
    elif change == "v2-policy":
        # IAM v1 inventories include placeholders for v2 policies; they cannot pass Reader validation.
        provider.policy_items.append({"href": "https://iam.cloud.ibm.com/v2/policies/policy-2"})
    elif change == "inherited-group":
        provider.groups = [{"id": "broad-group"}]
    elif change.startswith("public-policy-"):
        provider.public_access_policies[change.removeprefix("public-policy-")] = [{"id": "wide"}]
    elif change == "other-job-link":
        provider.links[0]["link"]["component_name"] = "other-job"
    else:
        provider.rules = [{"type": "Profile-CR", "cr_type": "CE"}]
    with pytest.raises(BootstrapBlocked):
        run(provider, tmp_path)
    assert mutations(provider) == []


@pytest.mark.parametrize("exit_code", [0, 1])
def test_shell_trap_removes_sensitive_file(tmp_path, exit_code):
    executable = tmp_path / "python3"
    executable.write_text('#!/bin/sh\nprintf sensitive > "$3"\nls -l "$3" | cut -c1-10 > "$WATCH_TEST_MODE"\n'
                          f'exit {exit_code}\n')
    executable.chmod(0o755)
    mode = tmp_path / "mode"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", "scripts/bootstrap_ibm_watchdog.sh"], cwd=root, env={
        **os.environ, "PATH": f"{tmp_path}:{os.defpath}", "TMPDIR": str(tmp_path),
        "WATCH_TEST_MODE": str(mode),
    }, capture_output=True, text=True)
    assert result.returncode == exit_code and mode.read_text().strip() == "-rw-------"
    assert list(tmp_path.glob("watch-auth-registry.*")) == []
    assert "sensitive" not in result.stdout + result.stderr
