"""Bind an authenticated Code Engine readback to a Cloud Logs canary query."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import requests


class RoutingProofUnavailable(RuntimeError):
    """A provider link is absent, ambiguous or different from the authorized probe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoutingProofUnavailable(f"IBM routing verification blocked: {message} (PENDENTE E2E).")


@dataclass(frozen=True)
class RoutingProbe:
    account_id: str
    region: str
    project_id: str
    project_name: str
    job_name: str
    job_run_name: str
    service_account: str
    trusted_profile_id: str
    image_digest: str
    logs_crn: str
    nonce: str

    def __post_init__(self) -> None:
        for value in (self.account_id, self.region, self.project_name, self.job_name,
                      self.job_run_name, self.service_account, self.trusted_profile_id):
            _require(bool(re.fullmatch(r"[A-Za-z0-9_-]+", value)), "invalid probe identity")
        try:
            _require(str(UUID(self.project_id)) == self.project_id, "invalid project GUID")
            _require(str(UUID(self.nonce, version=4)) == self.nonce, "nonce must be a UUID4")
        except (ValueError, AttributeError):
            raise RoutingProofUnavailable("invalid project GUID or nonce (PENDENTE E2E)") from None
        _require(bool(re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", self.image_digest)),
                 "probe image must be digest-pinned")
        parts = self.logs_crn.split(":")
        _require(len(parts) == 10 and parts[:5] == ["crn", "v1", "bluemix", "public", "logs"]
                 and parts[6] == f"a/{self.account_id}" and parts[8:] == ["", ""],
                 "Logs destination differs from the authorized account")
        _require(bool(re.fullmatch(r"[a-z]{2}-[a-z]+", parts[5])), "invalid Logs region")
        try:
            _require(str(UUID(parts[7])) == parts[7], "invalid Logs instance GUID")
        except ValueError:
            raise RoutingProofUnavailable("invalid Logs instance GUID (PENDENTE E2E)") from None

    @property
    def marker(self) -> str:
        return f"watch-auth-routing-probe {self.nonce}"

    @property
    def logs_url(self) -> str:
        parts = self.logs_crn.split(":")
        return f"https://{parts[7]}.api.{parts[5]}.logs.cloud.ibm.com"


def _mapping(value: object, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"missing or malformed {label} provider readback")
    assert isinstance(value, dict)
    return value


def _env_value(resource: dict[str, Any], name: str) -> str | None:
    variables = resource.get("run_env_variables")
    _require(isinstance(variables, list), "missing provider environment inventory")
    assert isinstance(variables, list)
    matches = [item for item in variables if isinstance(item, dict) and item.get("name") == name]
    _require(len(matches) == 1 and matches[0].get("type") == "literal"
             and isinstance(matches[0].get("value"), str), f"missing or ambiguous {name} binding")
    return matches[0]["value"]


def _verify_iam(evidence: dict[str, Any], probe: RoutingProbe, project_crn: str) -> None:
    # Official IAM Identity V1 TrustedProfile, ProfileClaimRuleList and ProfileLinkList:
    # https://cloud.ibm.com/apidocs/iam-identity-token-api#get-profile
    profile = _mapping(evidence.get("trusted_profile"), "IAM trusted profile")
    _require(profile.get("id") == probe.trusted_profile_id
             and profile.get("account_id") == probe.account_id,
             "IAM trusted profile identity or account differs")
    _require(isinstance(profile.get("iam_id"), str) and bool(profile["iam_id"]),
             "IAM trusted profile has no IAM identity")
    parts = str(profile.get("crn", "")).split(":")
    _require(len(parts) == 10
             and parts[:5] == ["crn", "v1", "bluemix", "public", "iam-identity"]
             and parts[6:] == [f"a/{probe.account_id}", "", "profile", probe.trusted_profile_id],
             "IAM trusted profile CRN differs")
    rules = _mapping(evidence.get("trusted_profile_rules"), "IAM trust rules").get("rules")
    links = _mapping(evidence.get("trusted_profile_links"), "IAM trust links").get("links")
    _require(isinstance(rules, list) and isinstance(links, list), "IAM trust inventory is malformed")
    assert isinstance(rules, list) and isinstance(links, list)
    compute_types = {"VSI", "PVS", "BMS", "IKS_SA", "ROKS_SA", "CE"}
    for rule in rules:
        _require(isinstance(rule, dict) and isinstance(rule.get("type"), str)
                 and rule["type"] in {"Profile-SAML", "Profile-CR"},
                 "IAM trust rule type is absent or unsupported")
        if rule["type"] == "Profile-CR":
            _require(isinstance(rule.get("cr_type"), str) and rule["cr_type"] in compute_types,
                     "IAM compute trust rule is malformed")
            # A direct, exact CE link is sufficient; do not guess rule-claim semantics.
            _require(rule["cr_type"] != "CE",
                     "additional CE claim-rule trust is not verified; require one exact direct CE link")
    for item in links:
        _require(isinstance(item, dict) and isinstance(item.get("cr_type"), str)
                 and item["cr_type"] in compute_types,
                 "IAM trust link type is absent or unsupported")
    ce_links = [item for item in links if item["cr_type"] == "CE"]
    _require(len(ce_links) == 1, "IAM requires exactly one direct CE trust link")
    item = ce_links[0]
    _require(isinstance(item.get("id"), str) and bool(item["id"])
             and item.get("is_cross_account", False) is False,
             "IAM CE trust link identity or account scope is ambiguous")
    link = _mapping(item.get("link"), "IAM CE trust link")
    _require(link.get("crn") == project_crn and link.get("component_type") == "job"
             and link.get("component_name") == probe.job_name,
             "IAM CE trust link does not bind the expected project and job")


def _verify_source(evidence: object, probe: RoutingProbe, now: datetime) -> datetime:
    _require(now.tzinfo is not None and now.utcoffset() is not None, "now must be timezone-aware")
    evidence = _mapping(evidence, "routing")
    project = _mapping(evidence.get("project"), "project")
    job = _mapping(evidence.get("job"), "job")
    run = _mapping(evidence.get("job_run"), "job run")
    # Native CE v2 response fields: https://cloud.ibm.com/apidocs/code-engine/v2
    _require(project.get("id") == probe.project_id and project.get("name") == probe.project_name,
             "project identity differs")
    crn = f"crn:v1:bluemix:public:codeengine:{probe.region}:a/{probe.account_id}:{probe.project_id}::"
    _require(project.get("crn") == crn, "project account or region differs")
    for resource, name in ((job, probe.job_name), (run, probe.job_run_name)):
        _require(resource.get("name") == name and resource.get("project_id") == probe.project_id
                 and resource.get("region") == probe.region, "job/run/project identity differs")
    _require(job.get("run_compute_resource_token_enabled") is True,
             "job has no enabled Trusted Profile support")
    _require(job.get("run_service_account") == probe.service_account, "job service account differs")
    _require(_env_value(job, "WATCH_AUTH_TRUSTED_PROFILE_ID") == probe.trusted_profile_id,
             "job trusted profile configuration differs")
    _require(job.get("image_reference") == probe.image_digest, "job image differs")
    _require(run.get("job_name") == probe.job_name, "run does not reference the expected job")
    # The run snapshot must attest the same settings; a job can change after submission.
    _require(run.get("run_compute_resource_token_enabled") is True,
             "run has no enabled Trusted Profile support")
    for key, expected in (("run_service_account", probe.service_account),
                          ("image_reference", probe.image_digest)):
        _require(run.get(key) == expected, f"run {key} is absent or differs")
    _require(_env_value(run, "WATCH_AUTH_ROUTING_PROBE_NONCE") == probe.nonce,
             "run nonce differs")
    _require(_env_value(run, "WATCH_AUTH_TRUSTED_PROFILE_ID") == probe.trusted_profile_id,
             "run trusted profile configuration differs")
    _verify_iam(evidence, probe, crn)
    try:
        created_at = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        _require(created_at.tzinfo is not None and now - timedelta(minutes=15) <= created_at <= now,
                 "canary run is stale or future dated")
    except (KeyError, TypeError, ValueError, AttributeError):
        raise RoutingProofUnavailable("invalid provider run creation time (PENDENTE E2E)") from None
    return created_at


def verify_routing_probes(evidence: object = None, *, probe: RoutingProbe | None = None,
                          now: datetime | None = None) -> None:
    """Verify source readbacks and the result of the exact provider-side query."""
    _require(isinstance(probe, RoutingProbe), "an authorized provider-bound canary is required")
    assert probe is not None
    _verify_source(evidence, probe, now or datetime.now(timezone.utc))
    evidence = _mapping(evidence, "routing")
    output = evidence.get("logs_output")
    _require(isinstance(output, str), "Cloud Logs query did not return raw text")
    assert isinstance(output, str)
    _require(len([line for line in output.splitlines() if line.strip()]) == 1,
             "Cloud Logs must return exactly one raw log line")
    occurrences = re.findall(rf"(?<![A-Za-z0-9_-]){re.escape(probe.nonce)}(?![A-Za-z0-9_-])", output)
    _require(len(occurrences) == 1, "Cloud Logs must return exactly one canary nonce occurrence")


def _cli(*arguments: str) -> str:
    try:
        result = subprocess.run(["ibmcloud", *arguments], check=True, capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        # Never include CLI output: oauth-tokens returns the session's bearer token.
        raise RoutingProofUnavailable("IBM CLI readback failed or timed out (PENDENTE E2E)") from None
    _require(not result.stderr.strip(), "IBM CLI returned an ambiguous warning or error")
    return result.stdout


def _json(raw: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            _require(key not in value, "duplicate provider JSON field")
            value[key] = item
        return value
    try:
        return _mapping(json.loads(raw, object_pairs_hook=unique), "JSON")
    except (TypeError, ValueError):
        raise RoutingProofUnavailable("malformed provider JSON (PENDENTE E2E)") from None


def collect_routing_probes(probe: RoutingProbe, *, now: datetime) -> dict[str, Any]:
    """Read authenticated providers directly; never accept local proof files or flags."""
    token = _json(_cli("iam", "oauth-tokens", "--output", "json")).get("iam_token")
    _require(isinstance(token, str) and token.startswith("Bearer ") and len(token) > 7,
             "IBM session has no IAM bearer token")
    base = f"https://api.{probe.region}.codeengine.cloud.ibm.com/v2/projects/{probe.project_id}"
    evidence: dict[str, Any] = {}
    iam = f"https://iam.cloud.ibm.com/v1/profiles/{probe.trusted_profile_id}"
    readbacks = [(label, base + suffix, {"version": "2026-09-04"})
                 for label, suffix in (("project", ""), ("job", f"/jobs/{probe.job_name}"),
                                       ("job_run", f"/job_runs/{probe.job_run_name}"))]
    readbacks.extend((label, iam + suffix, {}) for label, suffix in (
        ("trusted_profile", ""), ("trusted_profile_rules", "/rules"),
        ("trusted_profile_links", "/links")))
    for label, url, params in readbacks:
        try:
            response = requests.get(url, params=params,
                                    headers={"Authorization": token, "Accept": "application/json"},
                                    timeout=20, allow_redirects=False)
            _require(response.status_code == 200, f"IBM {label} readback failed")
            evidence[label] = _json(response.text)
        except requests.RequestException:
            raise RoutingProofUnavailable("IBM provider readback failed (PENDENTE E2E)") from None
    created_at = _verify_source(evidence, probe, now)
    # IBM documents DataPrime fields and logs-raw; no invented SSE/records envelope.
    # https://cloud.ibm.com/docs/codeengine?topic=codeengine-logging
    # https://cloud.ibm.com/docs/cloud-logs?topic=cloud-logs-query-synch
    fields = {"app": "codeengine", "originator": "user", "codeengine.region": probe.region,
              "codeengine.projectGuid": probe.project_id, "codeengine.componentType": "job_run",
              "codeengine.component": probe.job_name, "codeengine.subcomponentType": "job_run",
              "codeengine.subcomponent": probe.job_run_name, "message.message": probe.marker}
    query = "source logs | filter " + " && ".join(f"$d.{key} == '{value}'" for key, value in fields.items())
    evidence["logs_output"] = _cli(
        "logs", "query", "--service-url", probe.logs_url, "--query", query,
        "--syntax", "dataprime", "--start-date", created_at.isoformat(),
        "--end-date", now.isoformat(), "--limit", "2", "--output", "logs-raw",
    )
    verify_routing_probes(evidence, probe=probe, now=now)
    return evidence


def main() -> None:
    try:
        probe = RoutingProbe(
            account_id=os.environ["WATCH_AUTH_ACCOUNT_ID"], region=os.environ["WATCH_AUTH_REGION"],
            project_id=os.environ["WATCH_AUTH_CE_PROJECT_ID"],
            project_name=os.environ["WATCH_AUTH_CE_PROJECT"], job_name="watch-auth-watchdog",
            job_run_name=os.environ["WATCH_AUTH_ROUTING_JOB_RUN"],
            service_account=os.environ["WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT"],
            trusted_profile_id=os.environ["WATCH_AUTH_TRUSTED_PROFILE_ID"],
            image_digest=os.environ["WATCH_AUTH_WATCHDOG_IMAGE_DIGEST"],
            logs_crn=os.environ["WATCH_AUTH_PLATFORM_LOGS_CRN"],
            nonce=os.environ["WATCH_AUTH_ROUTING_PROBE_NONCE"],
        )
        collect_routing_probes(probe, now=datetime.now(timezone.utc))
        # Logs routing does not establish the independent VPC metric delivery path.
        raise RoutingProofUnavailable(
            "IBM Monitoring routing proof is still unavailable: VPC platform metric delivery "
            "must be observed independently (PENDENTE E2E)."
        )
    except KeyError as exc:
        raise SystemExit(f"missing routing probe configuration: {exc.args[0]} (PENDENTE E2E)") from None
    except RoutingProofUnavailable as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
