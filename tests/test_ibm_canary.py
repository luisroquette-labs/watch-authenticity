"""Provider-native canary readbacks, including hostile but self-consistent claims."""

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from infra.ibm_canary import CanaryReadbackClients, canary_log_message, collect_watchdog_canary
from infra.ibm_cos import IntegrityError
from infra.ibm_routing import RoutingProofUnavailable
from tests.conftest import NOW
from tests.test_ibm_cos import FakeCOS
from tests.test_ibm_routing import probe as routing_probe
from tests.test_ibm_routing import provider_evidence

INSTANCE_ID = "canary-vsi-1"


def observation_bytes(values):
    message = canary_log_message(
        account_id=values["account_id"], region=values["region"],
        resource_group_id=values["resource_group_id"], run_id=values["canary_run_id"],
        instance_sha256=values["canary_instance_sha256"], nonce=values["canary_nonce"],
        deleted_at=values["canary_deleted_at"],
    )
    labels = {"__name__": "ibm_is_instance_cpus", "ibm_resource": INSTANCE_ID,
              "ibm_scope": values["account_id"], "ibm_location": values["region"]}
    metric = {"status": "success", "data": {"resultType": "matrix", "result": [{
        "metric": labels, "values": [[values["canary_deleted_at"].timestamp(), "2"]],
    }]}}
    absent = {"endpoint": f"https://{values['region']}.iaas.cloud.ibm.com/v1",
              "instance_id": INSTANCE_ID, "status": 404, "consecutive_reads": 2}
    return {"log": (message + "\n").encode(), "metric": json.dumps(metric).encode(),
            "404": json.dumps(absent, sort_keys=True, separators=(",", ":")).encode()}


def canary_clients(proof):
    from dataclasses import asdict

    probe = replace(routing_probe.__wrapped__(), account_id=proof.account_id, region=proof.region,
                    nonce=str(proof.canary_nonce), logs_crn=proof.logs_crn,
                    monitoring_crn=proof.monitoring_crn)
    evidence = provider_evidence.__wrapped__(probe)
    evidence["job_run"]["created_at"] = (proof.canary_deleted_at - timedelta(minutes=2)).isoformat()
    bodies = observation_bytes(asdict(proof))
    responses = [SimpleNamespace(status_code=200, text=json.dumps(evidence[key])) for key in (
        "project", "job", "job_run", "trusted_profile", "trusted_profile_rules", "trusted_profile_links",
        "logs_instance", "monitoring_instance", "metrics_targets", "metrics_routes", "metrics_settings",
    )]
    responses.append(SimpleNamespace(status_code=200, content=bodies["metric"]))
    http = Mock()
    http.get.side_effect = responses
    cli = Mock(side_effect=lambda *args: json.dumps({"iam_token": "Bearer test-only"})
               if args[0] == "iam" else bodies["log"].decode())
    vpc = Mock()
    absent = RuntimeError("provider absence")
    absent.code = 404
    vpc.get_instance.side_effect = absent
    vpc.service_url = f"https://{proof.region}.iaas.cloud.ibm.com/v1"
    vpc.http_config = {}
    return CanaryReadbackClients(vpc=vpc, probe=probe, instance_id=INSTANCE_ID, http=http, cli=cli)


def _consume(tmp_path, *, proof=None, clients=None, cos=None):
    from infra.ibm_train import consume_authorization
    from tests.test_ibm_train import _decision, _proof, _spec

    spec = _spec()
    proof = proof or _proof(spec)
    decision = _decision(spec)
    return consume_authorization(
        spec, decision, decision.evidence_hashes, watchdog_proof=proof,
        now=NOW, store=tmp_path / "auth", canary_clients=clients or canary_clients(proof),
        cos=cos or FakeCOS(bucket=spec.bucket, prefix=spec.prefix),
    )


def test_authenticated_readbacks_are_required_and_persisted(tmp_path):
    from tests.test_ibm_train import _proof, _spec

    spec, proof = _spec(), _proof()
    clients = canary_clients(proof)
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    run = _consume(tmp_path, proof=proof, clients=clients, cos=cos)
    assert run.spec == spec
    assert clients.vpc.get_instance.call_count == 2
    assert len(cos.objects) == 3
    for body, _metadata in cos.objects.values():
        record = json.loads(body)
        assert record["sha256"] == hashlib.sha256(record["raw"].encode()).hexdigest()
        assert record["nonce"] == str(proof.canary_nonce)
    request = clients.http.get.call_args
    assert request.args[0].endswith("/api/prometheus/api/v1/query_range")
    assert request.kwargs["allow_redirects"] is False
    assert request.kwargs["headers"]["IBMInstanceID"] == proof.monitoring_crn.split(":")[7]


def test_collector_hashes_provider_bytes_instead_of_claimed_hashes():
    from tests.test_ibm_train import _proof

    proof = _proof(canary_log_sha256="a" * 64, canary_metric_sha256="b" * 64,
                   canary_404_sha256="c" * 64)
    observed = collect_watchdog_canary(proof, canary_clients(proof), now=NOW)
    assert observed == _proof()


@pytest.mark.parametrize("kind", ["log", "metric", "404"])
def test_rehashed_fabricated_observation_cannot_authorize(tmp_path, kind):
    from tests.test_ibm_train import _proof

    with pytest.raises(IntegrityError, match="hash differs"):
        _consume(tmp_path, proof=_proof(**{f"canary_{kind}_sha256": "a" * 64}))
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("defect", ["empty", "foreign-instance", "foreign-account", "foreign-region",
                                  "stale", "future", "nan", "wrong-type", "missing-label", "duplicate"])
def test_metric_native_samples_fail_closed(tmp_path, defect):
    from tests.test_ibm_train import _proof

    proof = _proof()
    clients = canary_clients(proof)
    responses = list(clients.http.get.side_effect)
    metric = json.loads(responses[-1].content)
    series = metric["data"]["result"][0]
    if defect == "empty":
        metric["data"]["result"] = []
    elif defect.startswith("foreign"):
        label = {"foreign-instance": "ibm_resource", "foreign-account": "ibm_scope",
                 "foreign-region": "ibm_location"}[defect]
        series["metric"][label] = "foreign"
    elif defect == "missing-label":
        del series["metric"]["ibm_resource"]
    elif defect == "duplicate":
        metric["data"]["result"].append(series)
    elif defect == "wrong-type":
        metric["data"]["resultType"] = "vector"
    elif defect == "nan":
        series["values"][0][1] = "NaN"
    else:
        series["values"][0][0] += -3600 if defect == "stale" else 3600
    responses[-1].content = json.dumps(metric).encode()
    clients.http.get.side_effect = responses
    with pytest.raises(IntegrityError, match="metric"):
        _consume(tmp_path, proof=proof, clients=clients)
    assert not clients.vpc.get_instance.called
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("defect", ["missing-log", "wrong-nonce", "wrong-run", "wrong-instance",
                                  "vpc-403", "vpc-live", "vpc-region", "cos-readback"])
def test_missing_or_divergent_provider_observation_blocks_capability(tmp_path, defect):
    from tests.test_ibm_train import _proof, _spec

    proof, spec = _proof(), _spec()
    clients = canary_clients(proof)
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    if defect in {"missing-log", "wrong-nonce", "wrong-run", "wrong-instance"}:
        raw = observation_bytes(vars(proof))["log"].decode()
        raw = {"missing-log": "", "wrong-nonce": raw.replace(str(proof.canary_nonce), "different"),
               "wrong-run": raw.replace(str(proof.canary_run_id), "different"),
               "wrong-instance": raw.replace(proof.canary_instance_sha256, "different")}[defect]
        clients.cli.side_effect = lambda *args: json.dumps({"iam_token": "Bearer test-only"}) if args[0] == "iam" else raw
    elif defect == "vpc-403":
        denied = RuntimeError("provider denied")
        denied.code = 403
        clients.vpc.get_instance.side_effect = denied
    elif defect == "vpc-live":
        clients.vpc.get_instance.side_effect = None
        clients.vpc.get_instance.return_value = {}
    elif defect == "vpc-region":
        clients.vpc.service_url = "https://eu-de.iaas.cloud.ibm.com/v1"
    else:
        cos.get_object = Mock(side_effect=TimeoutError("COS unavailable"))
    with pytest.raises((IntegrityError, RoutingProofUnavailable, RuntimeError, TimeoutError)):
        _consume(tmp_path, proof=proof, clients=clients, cos=cos)
    assert not tuple(tmp_path.rglob("*.json"))


@pytest.mark.parametrize("outcome", ["two-404", "reappears", "forbidden", "auxiliary-404"])
def test_canary_absence_uses_only_two_direct_vpc_reads(monkeypatch, outcome):
    import time

    from infra.ibm_vpc import confirm_instance_absent
    from tests.test_ibm_train import _proof

    clients = canary_clients(_proof())
    monkeypatch.setattr("infra.ibm_vpc._pause", lambda *_args: None)
    absent = RuntimeError("VPC GET 404")
    absent.code = 404
    forbidden = RuntimeError("VPC GET 403")
    forbidden.code = 403
    clients.vpc.get_instance.side_effect = {
        "two-404": [absent, absent], "reappears": [absent, {}],
        "forbidden": [forbidden], "auxiliary-404": [{}],
    }[outcome]
    if outcome == "two-404":
        confirm_instance_absent(clients.vpc, INSTANCE_ID, deadline=time.monotonic() + 5)
        assert clients.vpc.get_instance.call_count == 2
    else:
        with pytest.raises(RuntimeError):
            confirm_instance_absent(clients.vpc, INSTANCE_ID, deadline=time.monotonic() + 5)
    assert all(call.args == (INSTANCE_ID,) for call in clients.vpc.get_instance.call_args_list)
