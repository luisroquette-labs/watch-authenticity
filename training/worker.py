"""Secretless IBM worker that executes Task 6 and publishes immutable results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from infra.ibm_cos import IntegrityError, upload_verified, verify_cos_configuration
from infra.ibm_evidence import run_spec_sha256
from infra.ibm_train import _canonical, _put_immutable, _response_bytes, parse_run_spec
from infra.ibm_types import RunSpec, run_spec_transport_sha256

_REQUIRED_ENV = (
    "RUN_ID",
    "WATCH_AUTH_REGION",
    "WATCH_AUTH_COS_ENDPOINT",
    "WATCH_AUTH_COS_BUCKET",
    "WATCH_AUTH_COS_PREFIX",
    "WATCH_AUTH_COS_SERVICE_INSTANCE_ID",
    "WATCH_AUTH_TRUSTED_PROFILE_ID",
    "WATCH_AUTH_CODE_SHA",
    "WATCH_AUTH_IMAGE_DIGEST",
    "WATCH_AUTH_MANIFEST_SHA256",
    "WATCH_AUTH_SPLITS_SHA256",
    "WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256",
)


def load_training_data(cos: Any, spec: RunSpec) -> Any:
    from training.data import load_training_data as implementation

    return implementation(cos, spec)


def run_training(bundle: Any, spec: RunSpec, output_dir: Path) -> Any:
    from training.run import run_training as implementation

    return implementation(bundle, spec, output_dir)


def _environment() -> dict[str, str]:
    values = {name: os.environ.get(name, "") for name in _REQUIRED_ENV}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("worker environment is incomplete: " + ", ".join(missing))
    return values


def _bootstrap_cos(env: dict[str, str]) -> Any:
    import ibm_boto3
    from ibm_botocore.client import Config
    from ibm_cloud_sdk_core.authenticators import VPCInstanceAuthenticator

    auth = VPCInstanceAuthenticator(iam_profile_id=env["WATCH_AUTH_TRUSTED_PROFILE_ID"])
    client = ibm_boto3.client(
        "s3",
        endpoint_url=env["WATCH_AUTH_COS_ENDPOINT"],
        ibm_service_instance_id=env["WATCH_AUTH_COS_SERVICE_INSTANCE_ID"],
        token_manager=auth.token_manager,
        config=Config(signature_version="oauth"),
    )
    setattr(client, "_watch_auth_bucket", env["WATCH_AUTH_COS_BUCKET"])
    setattr(client, "_watch_auth_prefix", env["WATCH_AUTH_COS_PREFIX"].strip("/"))
    setattr(client, "_watch_auth_configuration_verified", False)
    return client


def _load_spec(cos: Any, env: dict[str, str]) -> RunSpec:
    key = f"{env['WATCH_AUTH_COS_PREFIX']}/run-spec-{env['WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256']}.json"
    body = _response_bytes(cos.get_object(Bucket=env["WATCH_AUTH_COS_BUCKET"], Key=key))
    try:
        document = json.loads(body)
    except json.JSONDecodeError as exc:
        raise IntegrityError("RunSpec transport is not JSON") from exc
    if _canonical(document) != body:
        raise IntegrityError("RunSpec transport is not canonical JSON")
    spec = parse_run_spec(document)
    expected = {
        "RUN_ID": str(spec.run_id),
        "WATCH_AUTH_REGION": spec.region,
        "WATCH_AUTH_COS_ENDPOINT": spec.cos_endpoint,
        "WATCH_AUTH_COS_BUCKET": spec.bucket,
        "WATCH_AUTH_COS_PREFIX": spec.prefix,
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": spec.cos_service_instance_id,
        "WATCH_AUTH_TRUSTED_PROFILE_ID": spec.trusted_profile_id,
        "WATCH_AUTH_CODE_SHA": spec.code_sha,
        "WATCH_AUTH_IMAGE_DIGEST": spec.image_digest,
        "WATCH_AUTH_MANIFEST_SHA256": spec.manifest_sha256,
        "WATCH_AUTH_SPLITS_SHA256": spec.splits_sha256,
        "WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256": run_spec_transport_sha256(spec),
    }
    if any(env[name] != value for name, value in expected.items()):
        raise IntegrityError("RunSpec transport differs from bootstrap identity")
    return spec


def _require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA GPU is required for IBM training")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory
    if "L4" not in name or memory < 22 * 1024**3:
        raise RuntimeError("authorized NVIDIA L4 capacity is unavailable")


def _receipt(spec: RunSpec, status: str, **payload: object) -> bytes:
    return _canonical(
        {
            "run_id": str(spec.run_id),
            "run_spec_sha256": run_spec_sha256(spec),
            "status": status,
            **payload,
        }
    )


def execute_worker(cos: Any | None = None) -> None:
    env = _environment()
    cos = cos or _bootstrap_cos(env)
    spec = _load_spec(cos, env)
    verify_cos_configuration(cos, spec)
    receipt_key = f"{spec.prefix}/receipts/training.json"
    try:
        _require_gpu()
        bundle = load_training_data(cos, spec)
        with tempfile.TemporaryDirectory(prefix="watch-auth-training-") as directory:
            output_dir = Path(directory)
            result = run_training(bundle, spec, output_dir)
            artifacts = dict(result.artifacts)
            result_path = output_dir / "run-result.json"
            artifacts[result_path] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            published: list[dict[str, object]] = []
            for path, digest in sorted(artifacts.items(), key=lambda item: item[0].name):
                ref = upload_verified(
                    cos,
                    path,
                    spec.bucket,
                    f"{spec.prefix}/artifacts/{path.name}",
                    digest,
                )
                published.append(
                    {
                        "key": ref.key,
                        "length": ref.length,
                        "name": path.name,
                        "sha256": ref.sha256,
                    }
                )
        if {item["name"] for item in published} != {
            "checkpoint.pt",
            "class-map.json",
            "metrics.json",
            "run-result.json",
        }:
            raise IntegrityError("worker did not produce the exact artifact set")
        _put_immutable(cos, spec, receipt_key, _receipt(spec, "succeeded", artifacts=published))
    except BaseException as exc:
        try:
            _put_immutable(
                cos,
                spec,
                receipt_key,
                _receipt(spec, "failed", error_type=type(exc).__name__),
            )
        except BaseException:
            pass
        raise


def main() -> None:
    execute_worker()


if __name__ == "__main__":
    main()
