"""Authenticated canary readbacks; caller-supplied hashes never establish provenance."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from typing import Any, Callable
from uuid import UUID

import requests

from infra.ibm_cos import IntegrityError, load_immutable_json, put_immutable_json
from infra.ibm_routing import RoutingProbe, _cli, _json, collect_routing_probes
from infra.ibm_types import RunSpec, WatchdogCanaryProof
from infra.ibm_vpc import confirm_instance_absent


@dataclass(frozen=True)
class CanaryReadbackClients:
    vpc: Any
    probe: RoutingProbe
    instance_id: str
    http: Any = requests
    cli: Callable[..., str] = _cli


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canary_log_message(
    *, account_id: str, region: str, resource_group_id: str, run_id: UUID,
    instance_sha256: str, nonce: UUID, deleted_at: datetime,
) -> str:
    return "watch-auth-canary " + _canonical({
        "account_id": account_id, "region": region, "resource_group_id": resource_group_id,
        "run_id": str(run_id), "instance_sha256": instance_sha256, "nonce": str(nonce),
        "deleted_at": deleted_at.isoformat(), "state": "deleted",
    }).decode()


def _readbacks(
    proof: WatchdogCanaryProof, clients: CanaryReadbackClients, *, now: datetime,
) -> dict[str, bytes]:
    probe = clients.probe
    job_crn = (f"crn:v1:bluemix:public:codeengine:{probe.region}:"
               f"a/{probe.account_id}:{probe.project_id}:job:{probe.job_name}")
    if (probe.account_id != proof.account_id or probe.region != proof.region
            or probe.nonce != str(proof.canary_nonce) or job_crn != proof.watchdog_job_crn
            or probe.logs_crn != proof.logs_crn or probe.monitoring_crn != proof.monitoring_crn
            or not re.fullmatch(r"[A-Za-z0-9_-]+", clients.instance_id)
            or hashlib.sha256(clients.instance_id.encode()).hexdigest() != proof.canary_instance_sha256):
        raise IntegrityError("canary provider readback identity differs")
    message = canary_log_message(
        account_id=proof.account_id, region=proof.region, resource_group_id=proof.resource_group_id,
        run_id=proof.canary_run_id, instance_sha256=proof.canary_instance_sha256,
        nonce=proof.canary_nonce, deleted_at=proof.canary_deleted_at,
    )
    # Reuse the IAM, CE, destination and routing validators; no local flag substitutes.
    routing = collect_routing_probes(
        probe, now=proof.observed_at, http=clients.http, cli=clients.cli, message=message,
    )
    if routing["logs_output"].strip() != message:
        raise IntegrityError("canary log content differs from deletion identity")
    created_at = datetime.fromisoformat(routing["job_run"]["created_at"].replace("Z", "+00:00"))
    if not created_at <= proof.canary_deleted_at <= proof.observed_at <= now <= proof.expires_at:
        raise IntegrityError("canary provider observation window differs")
    # IBM VPC's documented segmentation: ibm_resource, ibm_scope, ibm_location.
    # https://cloud.ibm.com/docs/vpc?topic=vpc-vpc-monitoring-metrics
    labels = {"ibm_resource": clients.instance_id, "ibm_scope": proof.account_id,
              "ibm_location": proof.region}
    metric = "ibm_is_instance_cpus"
    selector = metric + "{" + ",".join(f"{k}={json.dumps(v)}" for k, v in labels.items()) + "}"
    start = (proof.canary_deleted_at - timedelta(minutes=15)).timestamp()
    end = proof.canary_deleted_at.timestamp()
    token = _json(clients.cli("iam", "oauth-tokens", "--output", "json")).get("iam_token")
    if not isinstance(token, str) or not token.startswith("Bearer ") or len(token) <= 7:
        raise IntegrityError("canary Monitoring authentication is unavailable")
    # IBM documents this Prometheus prefix and IAM/IBMInstanceID authentication.
    # https://cloud.ibm.com/docs/monitoring?topic=monitoring-metrics_api
    try:
        response = clients.http.get(
            f"https://{probe.region}.monitoring.cloud.ibm.com/api/prometheus/api/v1/query_range",
            params={"query": selector, "start": start, "end": end, "step": 60},
            headers={"Authorization": token, "IBMInstanceID": probe.monitoring_crn.split(":")[7],
                     "Accept": "application/json"}, timeout=20, allow_redirects=False,
        )
    except requests.RequestException:
        raise IntegrityError("canary Monitoring readback failed") from None
    if response.status_code != 200:
        raise IntegrityError("canary Monitoring readback failed")
    raw_metric = response.content
    document = _json(raw_metric.decode())
    data = document.get("data")
    if (document.get("status") != "success" or document.get("warnings")
            or not isinstance(data, dict) or data.get("resultType") != "matrix"
            or not isinstance(data.get("result"), list) or len(data["result"]) != 1):
        raise IntegrityError("canary metric is absent or ambiguous")
    series = data["result"][0]
    expected = {"__name__": metric, **labels}
    if (not isinstance(series, dict) or not isinstance(series.get("metric"), dict)
            or any(series["metric"].get(k) != v for k, v in expected.items())
            or not isinstance(series.get("values"), list) or not series["values"]):
        raise IntegrityError("canary metric identity or samples differ")
    previous = float("-inf")
    for sample in series["values"]:
        try:
            if (not isinstance(sample, list) or len(sample) != 2
                    or isinstance(sample[0], bool) or not isinstance(sample[0], (int, float))
                    or not isinstance(sample[1], str)):
                raise ValueError
            timestamp, value = sample[0], float(sample[1])
            if not (start <= timestamp <= end and timestamp > previous
                    and math.isfinite(value) and value > 0):
                raise ValueError
            previous = timestamp
        except (TypeError, ValueError, OverflowError):
            raise IntegrityError("canary metric samples are invalid or outside the window") from None
    expected_endpoint = f"https://{proof.region}.iaas.cloud.ibm.com/v1"
    if getattr(clients.vpc, "service_url", None) != expected_endpoint:
        raise IntegrityError("canary VPC readback region differs")
    confirm_instance_absent(clients.vpc, clients.instance_id, deadline=time.monotonic() + 20)
    return {
        "log": routing["logs_output"].encode(),
        "metric": raw_metric,
        "404": _canonical({"endpoint": expected_endpoint, "instance_id": clients.instance_id,
                           "status": 404, "consecutive_reads": 2}),
    }


def collect_watchdog_canary(
    proof: WatchdogCanaryProof, clients: CanaryReadbackClients, *, now: datetime,
) -> WatchdogCanaryProof:
    """Replace claimed observation hashes only after authenticated provider retrieval."""

    readbacks = _readbacks(proof, clients, now=now)
    values = {field.name: getattr(proof, field.name) for field in fields(proof)}
    for kind, body in readbacks.items():
        values[f"canary_{kind}_sha256"] = hashlib.sha256(body).hexdigest()
    document = {key: value.isoformat() if isinstance(value, datetime)
                else str(value) if isinstance(value, UUID) else value
                for key, value in values.items() if key != "evidence_sha256"}
    values["evidence_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return replace(proof, **values)


def verify_watchdog_canary(
    spec: RunSpec, proof: WatchdogCanaryProof, clients: CanaryReadbackClients | None,
    cos: Any | None, *, now: datetime,
) -> None:
    """Re-fetch all observations before consuming money authorization; persist verified bytes."""

    if clients is None or cos is None:
        raise PermissionError("canary authenticated provider readbacks and COS are required")
    readbacks = _readbacks(proof, clients, now=now)
    for kind, raw in readbacks.items():
        digest = hashlib.sha256(raw).hexdigest()
        if digest != getattr(proof, f"canary_{kind}_sha256"):
            raise IntegrityError("canary observation hash differs from authenticated provider bytes")
        body = _canonical({
            "account_id": proof.account_id, "region": proof.region,
            "resource_group_id": proof.resource_group_id, "run_id": str(proof.canary_run_id),
            "nonce": str(proof.canary_nonce), "instance_sha256": proof.canary_instance_sha256,
            "observed_at": now.isoformat(), "sha256": digest, "raw": raw.decode(),
        })
        key = put_immutable_json(cos, spec, f"{spec.prefix}/evidence/canary/{kind}.json", body)
        if load_immutable_json(cos, spec, key) != json.loads(body):
            raise IntegrityError("canary COS readback differs")
