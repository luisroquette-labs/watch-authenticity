"""Bootstrap private image access and an inert watchdog job; never submit or schedule it."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from infra.ibm_routing import RoutingProofUnavailable, _json
from infra.ibm_routing import main as verify_routing

IMAGE = ("us.icr.io/watch-authenticity-3235996/watchdog@sha256:"
         "bf169b1164176e04d76b9d8cc1a86c1a552de218b8503a6dfe540c47e283f398")
SERVICE = "watch-auth-watchdog-pull"
SECRET = "watch-auth-watchdog-icr"
JOB = "watch-auth-watchdog"
IAM = "https://iam.cloud.ibm.com"
ENV_NAMES = (
    "ACCOUNT_ID", "RESOURCE_GROUP", "RESOURCE_GROUP_ID", "REGION", "CE_PROJECT",
    "CE_PROJECT_ID", "WATCHDOG_SERVICE_ACCOUNT", "WATCHDOG_IMAGE_DIGEST",
    "TRUSTED_PROFILE_ID", "COS_BUCKET", "COS_ENDPOINT", "COS_SERVICE_INSTANCE_ID",
)


class BootstrapBlocked(RuntimeError):
    """The observed provider scope cannot safely be reconciled."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BootstrapBlocked(message)


def _cli(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["ibmcloud", *arguments], capture_output=True, text=True, timeout=120,
            env={**os.environ, "IBMCLOUD_TRACE": "false"},
        )
        _require(result.returncode == 0 and not result.stderr.strip(), "IBM CLI command failed")
        return result.stdout
    except (OSError, subprocess.SubprocessError):
        # CLI output, exceptions and request bodies can contain credentials.
        raise BootstrapBlocked("IBM CLI command failed or timed out") from None


def _items(document: dict[str, Any], field: str) -> list[dict[str, Any]]:
    items = document.get(field)
    _require(isinstance(items, list) and all(isinstance(item, dict) for item in items),
             f"malformed {field} inventory")
    _require(not any(document.get(key) for key in ("next", "next_url", "next_page_token")),
             f"paginated {field} inventory needs explicit reconciliation")
    assert isinstance(items, list)
    return items


def bootstrap(env: Mapping[str, str], key_file: Path, *,
              cli: Callable[..., str] = _cli, http: Any = requests, deploy: bool = False) -> None:
    if deploy:
        _require(all(env.get(f"WATCH_AUTH_{name}") == os.environ.get(f"WATCH_AUTH_{name}")
                     for name in ENV_NAMES),
                 "full deployment target differs from the routing gate environment")
        # Full deployment may reconcile an existing trigger only after the real routing gate.
        verify_routing()
    v = {}
    for name in ENV_NAMES:
        value = env.get(f"WATCH_AUTH_{name}", "")
        _require(bool(value) and "\n" not in value, f"missing or invalid WATCH_AUTH_{name}")
        v[name] = value
    _require(v["WATCHDOG_IMAGE_DIGEST"] == IMAGE, "watchdog image differs from approved digest")
    _require(v["REGION"] == "us-south", "registry bootstrap requires approved us-south region")
    _require(v["WATCHDOG_SERVICE_ACCOUNT"] == "none", "watchdog requires service account none")
    for name in ("ACCOUNT_ID", "RESOURCE_GROUP_ID", "CE_PROJECT_ID", "TRUSTED_PROFILE_ID"):
        _require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", v[name])), "invalid target identifier")
    _require(not key_file.exists() and key_file.parent.stat().st_mode & 0o777 == 0o700,
             "key must use a new file in a private temporary directory")
    target = _json(cli("target", "--output", "json"))
    _require(target.get("api_endpoint") == "https://cloud.ibm.com"
             and target.get("account", {}).get("guid") == v["ACCOUNT_ID"]
             and target.get("resource_group", {}).get("name") == v["RESOURCE_GROUP"]
             and target.get("region", {}).get("name") == v["REGION"], "selected IBM target differs")
    resource_group_id = cli("resource", "group", v["RESOURCE_GROUP"], "--id").strip()
    _require(resource_group_id == v["RESOURCE_GROUP_ID"], "selected IBM resource group differs")
    selected = _json(cli("ce", "project", "current", "--output", "json"))
    _require(selected.get("guid") == v["CE_PROJECT_ID"]
             and selected.get("name") == v["CE_PROJECT"], "selected Code Engine project differs")
    token = _json(cli("iam", "oauth-tokens", "--output", "json")).get("iam_token")
    _require(isinstance(token, str) and token.startswith("Bearer "), "missing IBM session token")
    ce = f"https://api.us-south.codeengine.cloud.ibm.com/v2/projects/{v['CE_PROJECT_ID']}"
    project_crn = f"crn:v1:bluemix:public:codeengine:us-south:a/{v['ACCOUNT_ID']}:{v['CE_PROJECT_ID']}::"

    def get(url: str, *, optional: bool = False, params: dict[str, str] | None = None,
            extra_headers: dict[str, str] | None = None) -> dict[str, Any] | None:
        try:
            response = http.get(
                url, headers={"Authorization": token, "Accept": "application/json",
                              **(extra_headers or {})},
                params={"version": "2026-09-04", **(params or {})} if url.startswith(ce) else params,
                timeout=30, allow_redirects=False,
            )
            if optional and response.status_code == 404:
                return None
            _require(response.status_code == 200, "IBM provider readback failed")
            return _json(response.text)
        except requests.RequestException:
            raise BootstrapBlocked("IBM provider readback failed") from None

    project = get(ce)
    _require(project is not None and project.get("id") == v["CE_PROJECT_ID"]
             and project.get("name") == v["CE_PROJECT"] and project.get("crn") == project_crn,
             "Code Engine project identity differs")
    instance = get("https://resource-controller.cloud.ibm.com/v2/resource_instances/" + v["CE_PROJECT_ID"])
    _require(instance is not None and instance.get("account_id") == v["ACCOUNT_ID"]
             and instance.get("resource_group_id") == v["RESOURCE_GROUP_ID"]
             and instance.get("crn") == project_crn and instance.get("state") == "active",
             "Code Engine resource group or state differs")
    for kind in ("cron", "cos", "kafka"):
        inventory = _json(cli("ce", "sub", kind, "list", "--output", "json"))
        subscriptions = _items(inventory, f"{kind}_subscriptions")
        metadata = inventory.get("metadata", {})
        _require(isinstance(metadata, dict) and not metadata.get("continue"),
                 "malformed or paginated subscription inventory")
        for subscription in subscriptions:
            spec = subscription.get("spec")
            _require(isinstance(spec, dict), "malformed subscription spec")
            assert isinstance(spec, dict)
            sink = spec.get("sink")
            _require(isinstance(sink, dict), "malformed subscription sink")
            assert isinstance(sink, dict)
            ref = sink.get("ref", {})
            _require(not sink.get("uri") and isinstance(ref, dict)
                     and ref.get("kind") in {"Job", "Service"}
                     and isinstance(ref.get("name"), str) and bool(ref["name"]),
                     "malformed subscription destination")
            _require(deploy or ref["kind"] != "Job" or ref["name"] != JOB,
                     "existing subscription targets the watchdog job")
    runs = _items(get(ce + "/job_runs", params={"job_name": JOB, "limit": "100"}) or {}, "job_runs")
    for run in runs:
        details = run.get("status_details", {})
        _require(run.get("job_name") == JOB and run.get("status") in {"succeeded", "failed"}
                 and isinstance(details, dict) and isinstance(details.get("completion_time"), str)
                 and bool(details["completion_time"])
                 and all(details.get(field, 0) == 0 for field in ("running", "pending", "unknown")),
                 "active or incomplete watchdog job run")
    profile_url = f"{IAM}/v1/profiles/{v['TRUSTED_PROFILE_ID']}"
    profile = get(profile_url)
    _require(profile is not None and profile.get("id") == v["TRUSTED_PROFILE_ID"]
             and profile.get("account_id") == v["ACCOUNT_ID"], "trusted profile identity differs")
    rules = _items(get(profile_url + "/rules") or {}, "rules")
    _require(not any(rule.get("type") == "Profile-CR" and rule.get("cr_type") == "CE"
                     for rule in rules), "additional CE claim rule needs explicit reconciliation")
    expected_link = {"crn": project_crn, "component_type": "job", "component_name": JOB}

    def trust_links() -> list[dict[str, Any]]:
        links = _items(get(profile_url + "/links") or {}, "links")
        ce_links = [item for item in links if item.get("cr_type") == "CE"]
        _require(len(ce_links) <= 1 and all(item.get("link") == expected_link
                 and item.get("is_cross_account", False) is False for item in ce_links),
                 "unexpected or duplicate CE trusted-profile link")
        return ce_links

    links = trust_links()
    secret = get(ce + f"/secrets/{SECRET}", optional=True)
    job = get(ce + f"/jobs/{JOB}", optional=True)

    def service_ids() -> list[dict[str, Any]]:
        return _items(get(IAM + "/v1/serviceids/", params={
            "account_id": v["ACCOUNT_ID"], "name": SERVICE, "pagesize": "100",
        }) or {}, "serviceids")

    ids = service_ids()
    _require(len(ids) <= 1, "duplicate registry service IDs")
    if not ids:
        _require(secret is None, "existing registry secret has no dedicated service ID")
        cli("iam", "service-id-create", SERVICE, "--description", "Watch authenticity ICR pull only",
            "--output", "json")
        ids = service_ids()
    _require(len(ids) == 1 and ids[0].get("name") == SERVICE
             and ids[0].get("account_id") == v["ACCOUNT_ID"]
             and bool(ids[0].get("id")) and bool(ids[0].get("iam_id")), "service ID differs")
    service_id, iam_id = ids[0]["id"], ids[0]["iam_id"]
    groups = get("https://iam.cloud.ibm.com/v2/groups", params={
        "account_id": v["ACCOUNT_ID"], "iam_id": iam_id,
    }) or {}
    _require(not _items(groups, "groups"), "registry service ID inherits access-group permissions")
    # Registry IAM supports a regional namespace boundary, not a repository or digest boundary.
    attributes = {"accountId": v["ACCOUNT_ID"], "serviceName": "container-registry",
                  "region": "us-south", "resourceType": "namespace", "resource": "watch-authenticity-3235996"}

    def policies() -> list[dict[str, Any]]:
        items = _items(get(IAM + "/v1/policies", params={"iam_id": iam_id}) or {}, "policies")
        for policy in items:
            resources = policy.get("resources", [])
            observed = resources[0].get("attributes", []) if len(resources) == 1 else []
            _require(policy.get("type") == "access" and len(observed) == len(attributes)
                     and {item.get("name"): item.get("value") for item in observed} == attributes
                     and [role.get("role_id") for role in policy.get("roles", [])]
                     == ["crn:v1:bluemix:public:iam::::serviceRole:Reader"],
                     "registry service ID has unexpected IAM permissions")
        _require(len(items) <= 1, "duplicate registry Reader policies")
        return items

    if not policies():
        cli("iam", "service-policy-create", service_id, "--roles", "Reader", "--service-name",
            "container-registry", "--region", "us-south", "--resource-type", "namespace",
            "--resource", "watch-authenticity-3235996", "--force", "--output", "json")
        _require(len(policies()) == 1, "registry Reader policy was not persisted")
    keys = _items(get(IAM + "/v1/apikeys", params={
        "account_id": v["ACCOUNT_ID"], "iam_id": iam_id, "pagesize": "100",
    }) or {}, "apikeys")
    _require(len(keys) <= 1 and all(key.get("name") == SERVICE for key in keys),
             "unexpected registry API keys")
    if secret is None:
        _require(not keys, "orphan registry API key needs explicit revocation before retry")
        key_file.touch(mode=0o600)
        cli("iam", "service-api-key-create", SERVICE, service_id, "--file", str(key_file),
            "--action-if-leaked", "disable", "--force", "--output", "json")
        _require(key_file.stat().st_mode & 0o777 == 0o600, "registry key file permissions differ")
        cli("ce", "registry", "create", "--name", SECRET, "--server", "us.icr.io", "--username",
            "iamapikey", "--password-from-json-file", str(key_file), "--output", "json")
        key_file.unlink()
        secret = get(ce + f"/secrets/{SECRET}")
        keys = _items(get(IAM + "/v1/apikeys", params={
            "account_id": v["ACCOUNT_ID"], "iam_id": iam_id, "pagesize": "100",
        }) or {}, "apikeys")
    _require(secret is not None and secret.get("name") == SECRET and secret.get("format") == "registry"
             and secret.get("project_id") == v["CE_PROJECT_ID"], "registry secret identity differs")
    assert secret is not None
    data = secret.get("data", {})
    _require(data.get("server") == "us.icr.io" and data.get("username") == "iamapikey"
             and isinstance(data.get("password"), str) and bool(data["password"]), "registry secret data differs")
    details = get(IAM + "/v1/apikeys/details", extra_headers={"IAM-ApiKey": data["password"]})
    _require(details is not None and details.get("iam_id") == iam_id
             and details.get("account_id") == v["ACCOUNT_ID"] and details.get("name") == SERVICE
             and len(keys) == 1 and details.get("id") == keys[0].get("id")
             and bool(details.get("id")),
             "registry secret is not bound to the dedicated service ID")
    desired = {"image_reference": IMAGE, "image_secret": SECRET,
               "run_service_account": "none", "run_compute_resource_token_enabled": True,
               "run_commands": ["python3"], "run_arguments": ["-m", "infra.ibm_watchdog"]}
    environment = {f"WATCH_AUTH_{name}": v[name] for name in (
        "ACCOUNT_ID", "RESOURCE_GROUP", "RESOURCE_GROUP_ID", "REGION", "COS_BUCKET",
        "COS_ENDPOINT", "COS_SERVICE_INSTANCE_ID", "TRUSTED_PROFILE_ID",
    )}
    if job is not None:
        variables = job.get("run_env_variables", [])
        _require(isinstance(variables, list) and all(isinstance(item, dict)
                 and item.get("name") in environment for item in variables)
                 and len({item["name"] for item in variables}) == len(variables),
                 "unexpected watchdog environment needs explicit reconciliation")

    def job_matches(observed: dict[str, Any] | None) -> bool:
        if observed is None or any(observed.get(k) != value for k, value in desired.items()):
            return False
        variables = observed.get("run_env_variables", [])
        return len(variables) == len(environment) and all(
            item.get("type") == "literal" and item.get("name") in environment
            and item.get("value") == environment[item["name"]] for item in variables
        ) and len({item.get("name") for item in variables}) == len(environment)

    if not job_matches(job):
        args = ["ce", "job", "update" if job else "create", "--name", JOB, "--image", IMAGE,
                "--registry-secret", SECRET, "--service-account", "none", "--trusted-profiles-enabled=true",
                "--command", "python3", "--arg=-m", "--arg", "infra.ibm_watchdog"]
        for name, value in environment.items():
            args.extend(("--env", f"{name}={value}"))
        cli(*args, "--output", "json")
        _require(job_matches(get(ce + f"/jobs/{JOB}")), "watchdog job readback differs")
    if not links:
        cli("iam", "trusted-profile-link-create", v["TRUSTED_PROFILE_ID"], "--name", JOB,
            "--cr-type", "CE", "--link-crn", project_crn, "--link-component-type", "job",
            "--link-component-name", JOB, "--output", "json")
        _require(len(trust_links()) == 1, "trusted-profile link was not persisted")


def main() -> None:
    try:
        _require(len(sys.argv) == 2 or (len(sys.argv) == 3 and sys.argv[2] == "--deploy"),
                 "private temporary key path and optional --deploy are required")
        bootstrap(os.environ, Path(sys.argv[1]), deploy=len(sys.argv) == 3)
    except BootstrapBlocked as error:
        raise SystemExit(f"IBM registry bootstrap blocked: {error}") from None
    except (RoutingProofUnavailable, OSError, ValueError, TypeError, KeyError, AttributeError):
        # Do not print provider responses, keys, encoded secrets or exception payloads.
        raise SystemExit("IBM registry bootstrap blocked; reconcile provider state before retry") from None


if __name__ == "__main__":
    main()
