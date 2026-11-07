"""IBM-only runtime regressions and a zero-cost complete lifecycle proof."""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _aws_references(source: str) -> list[str]:
    tree = ast.parse(source)
    ibm_names = {"ibm_boto3"}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name.split(".")[0] in {"boto3", "botocore"}:
                    violations.append(name.name)
                if name.name == "ibm_boto3":
                    ibm_names.add(name.asname or name.name)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in {"boto3", "botocore"}:
                violations.append(node.module or "")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "client":
            continue
        service = node.args[0]
        if isinstance(service, ast.Constant) and service.value in {"ec2", "s3"}:
            owner = node.func.value
            if service.value == "ec2" or not (
                isinstance(owner, ast.Name) and owner.id in ibm_names
            ):
                violations.append(ast.unparse(node))
    return violations


def test_runtime_has_no_aws_provider():
    violations = []
    for directory, directories, files in os.walk(ROOT):
        directories[:] = [
            name for name in directories
            if not name.startswith(".") and name not in {"tests", "graphify-out", "__pycache__"}
        ]
        for name in files:
            if name.endswith(".py"):
                path = Path(directory) / name
                violations.extend(
                    f"{path.relative_to(ROOT)}: {item}"
                    for item in _aws_references(path.read_text())
                )
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert not any(line.strip().split("==")[0] == "boto3" for line in requirements)
    assert not violations, "\n".join(violations)
    assert not (ROOT / "infra/aws_train.py").exists()
    assert not (ROOT / "infra/s3_sync.py").exists()


@pytest.mark.parametrize("source", [
    "import boto3 as cloud",
    "from boto3 import client",
    "from botocore.config import Config",
    "cloud.client('ec2')",
    "cloud.client('s3')",
])
def test_aws_regression_detects_imports_aliases_and_clients(source):
    assert _aws_references(source)


def test_aws_regression_preserves_ibm_cos_sdk():
    assert not _aws_references(
        "import ibm_boto3 as cos\ncos.client('s3', token_manager=auth.token_manager)"
    )
    assert "ibm-cos-sdk==2.16.2" in (ROOT / "requirements.txt").read_text().splitlines()


def test_preflight_fails_when_mypy_fails(tmp_path):
    for command, status in (("ruff", 0), ("mypy", 42), ("pytest", 0)):
        executable = tmp_path / command
        executable.write_text(f"#!/bin/sh\necho {command} >> calls\nexit {status}\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", str(ROOT / "scripts/preflight.sh")],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 42
    assert (tmp_path / "calls").read_text().splitlines() == ["ruff", "mypy"]


def test_ibm_only_mocked_e2e_persists_artifacts_and_resource_deletions(monkeypatch, tmp_path):
    import hashlib
    import json
    import socket
    from types import SimpleNamespace
    from unittest.mock import Mock

    import infra.ibm_train as controller
    import infra.ibm_vpc as lifecycle
    import training.worker as worker
    from core.types import ModelClass
    from infra.ibm_cos import load_event_inventory, load_immutable_json
    from infra.ibm_train import IBMClients, authorize_training, launch_training
    from infra.ibm_types import run_spec_transport_sha256
    from tests.conftest import NOW, valid_credit, valid_post_promo_plan, valid_pricing, vpc_instance
    from tests.test_ibm_canary import canary_clients
    from tests.test_ibm_cos import FakeCOS
    from tests.test_ibm_train import _proof, _spec
    from tests.test_ibm_vpc import (
        _client,
        _floating_ip,
        _NotFound,
        _page,
        _Response,
        _tagger,
    )

    def no_network(*_args, **_kwargs):
        pytest.fail("mocked E2E must never contact a provider")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    recovery = tmp_path / "recovery"
    monkeypatch.setattr(lifecycle, "_RECOVERY_DIR", recovery)
    monkeypatch.setattr(controller, "_RECOVERY_DIR", tmp_path / "controller-recovery")
    spec = _spec()
    cos = FakeCOS(bucket=spec.bucket, prefix=spec.prefix)
    cos._watch_auth_spec = spec
    usage = Mock()
    usage.get_resource_usage_account.return_value.get_result.return_value = {
        "resources": [], "count": 0,
    }
    run = authorize_training(
        cos, usage, spec, valid_credit(), valid_pricing(spec=spec), valid_post_promo_plan(),
        watchdog_proof=_proof(spec), canary_clients=canary_clients(_proof(spec)), now=NOW, store=tmp_path / "auth",
    )
    assert Path(run.pending_record_path).is_file()
    client = _client()
    tagger = _tagger()
    instance = vpc_instance()
    live = False
    worker_calls = []

    def load_data(_cos, observed_spec):
        assert _cos is cos and observed_spec == spec
        return "mocked immutable dataset"

    def train(bundle, observed_spec, output_dir):
        assert bundle == "mocked immutable dataset" and observed_spec == spec
        worker_calls.append("train")
        bodies = {
            "checkpoint.pt": b"mocked six-class checkpoint",
            "class-map.json": controller._canonical({item.name: item.value for item in ModelClass}),
            "metrics.json": b'{"epochs_completed":1}',
        }
        artifacts = {}
        for name, body in bodies.items():
            path = output_dir / name
            path.write_bytes(body)
            artifacts[path] = hashlib.sha256(body).hexdigest()
        result = {
            "artifacts": {path.name: digest for path, digest in artifacts.items()},
            "best_epoch": 1, "best_validation_nll": 0.5, "epochs_completed": 1,
            "lineage": {
                "code_sha": spec.code_sha, "image_digest": spec.image_digest,
                "manifest_sha256": spec.manifest_sha256, "splits_sha256": spec.splits_sha256,
                "class_map_sha256": hashlib.sha256(bodies["class-map.json"]).hexdigest(),
            },
        }
        (output_dir / "run-result.json").write_bytes(controller._canonical(result))
        return SimpleNamespace(artifacts=artifacts)

    monkeypatch.setattr(worker, "_require_gpu", lambda: None)
    monkeypatch.setattr(worker, "load_training_data", load_data)
    monkeypatch.setattr(worker, "run_training", train)

    def create(prototype):
        nonlocal live
        assert not live
        assert hashlib.sha256(prototype["user_data"].encode()).hexdigest() == spec.bootstrap_sha256
        # Simulate the authorized bootstrap passing its identity into the worker.
        environment = {
            "RUN_ID": str(spec.run_id), "WATCH_AUTH_REGION": spec.region,
            "WATCH_AUTH_COS_ENDPOINT": spec.cos_endpoint, "WATCH_AUTH_COS_BUCKET": spec.bucket,
            "WATCH_AUTH_COS_PREFIX": spec.prefix,
            "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": spec.cos_service_instance_id,
            "WATCH_AUTH_TRUSTED_PROFILE_ID": spec.trusted_profile_id,
            "WATCH_AUTH_CODE_SHA": spec.code_sha, "WATCH_AUTH_IMAGE_DIGEST": spec.image_digest,
            "WATCH_AUTH_MANIFEST_SHA256": spec.manifest_sha256,
            "WATCH_AUTH_SPLITS_SHA256": spec.splits_sha256,
            "WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256": run_spec_transport_sha256(spec),
        }
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        live = True
        worker.execute_worker(cos)
        return _Response(instance)

    def get_instance(_instance_id):
        assert _instance_id == instance["id"]
        if not live:
            raise _NotFound()
        return _Response(instance)

    def delete_instance(_instance_id):
        nonlocal live
        assert _instance_id == instance["id"]
        assert [event.state for event in load_event_inventory(cos, spec).events][-2:] == [
            "succeeded", "deleting",
        ]
        live = False

    def delete_volume(_volume_id):
        client.get_volume.side_effect = _NotFound()

    def delete_gateway(_gateway_id):
        client.get_public_gateway.side_effect = _NotFound()
        client.get_floating_ip.return_value = _Response(_floating_ip(target=None))

    client.list_instances.side_effect = lambda **_kwargs: _page(instance) if live else _page()
    client.create_instance.side_effect = create
    client.get_instance.side_effect = get_instance
    client.delete_instance.side_effect = delete_instance
    client.delete_volume.side_effect = delete_volume
    client.delete_public_gateway.side_effect = delete_gateway

    terminal = launch_training(IBMClients(client, tagger, cos), run, confirm=True, now=NOW)

    assert worker_calls == ["train"]
    client.create_instance.assert_called_once()
    operations = [entry[0] for entry in client.mock_calls]
    for resource in ("instance", "volume", "subnet", "public_gateway", "floating_ip"):
        getattr(client, f"delete_{resource}").assert_called_once()
        delete_index = operations.index(f"delete_{resource}")
        assert f"get_{resource}" in operations[delete_index + 1:]
    receipt = load_immutable_json(cos, spec, f"{spec.prefix}/receipts/training.json")
    assert receipt["status"] == "succeeded"
    assert len(receipt["artifacts"]) == 4
    for artifact in receipt["artifacts"]:
        body, metadata = cos.objects[artifact["key"]]
        assert hashlib.sha256(body).hexdigest() == metadata["sha256"] == artifact["sha256"]
        assert len(body) == artifact["length"] == int(metadata["length"])
    events = load_event_inventory(cos, spec).events
    assert [event.state for event in events] == [
        "planned", "authorized", "creating", "running", "artifact_sync", "succeeded",
        "deleting", "deleted",
    ]
    assert events[-1] == terminal
    local_receipts = [json.loads(path.read_bytes()) for path in recovery.glob("*-deleted-*.json")]
    assert len(local_receipts) == 5
    deletion_receipts = {
        json.loads(body)["resource_type"]: json.loads(body)
        for key, (body, _) in cos.objects.items() if "/receipts/deletion/" in key
    }
    assert set(deletion_receipts) == {
        "vpc-vsi", "boot-volume", "subnet", "public-gateway", "floating-ip",
    }
    assert {name: receipt["resource_id"] for name, receipt in deletion_receipts.items()} == {
        "vpc-vsi": instance["id"],
        "boot-volume": instance["boot_volume_attachment"]["volume"]["id"],
        "subnet": spec.subnet_id,
        "public-gateway": spec.public_gateway_id,
        "floating-ip": spec.public_gateway_floating_ip_id,
    }
    assert all(
        receipt["provider_state"] == "404" and receipt["run_id"] == str(spec.run_id)
        for receipt in deletion_receipts.values()
    )
    deletion_hashes = {
        resource: hashlib.sha256(controller._canonical(receipt)).hexdigest()
        for resource, receipt in deletion_receipts.items()
    }
    assert terminal.payload_sha256 == controller._event_payload(
        "deleted", {"deletion_receipt_sha256": deletion_hashes},
    )
    assert not live
    # Provider usage is intentionally unreconciled: mocked teardown cannot release the fence.
    assert Path(run.pending_record_path).is_file()
