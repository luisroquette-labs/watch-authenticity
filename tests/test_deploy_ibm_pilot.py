"""The IBM pilot deploy keeps credentials secret and cannot drift to another image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def test_deploy_requires_private_password_and_pinned_image(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    password = tmp_path / "password"
    password.write_text("p" * 32)
    password.chmod(0o644)
    env = {
        **os.environ,
        "WATCH_AUTH_CE_PROJECT": "watch-authenticity",
        "WATCH_AUTH_CE_PROJECT_CRN": "crn:v1:project",
        "WATCH_AUTH_PILOT_IMAGE_DIGEST": "us.icr.io/watch-authenticity-3235996/pilot@sha256:" + "a" * 64,
        "WATCH_AUTH_REGISTRY_SECRET": "pull-secret",
        "WATCH_AUTH_COS_BUCKET": "private",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
        "WATCH_AUTH_PILOT_PASSWORD_FILE": str(password),
    }
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert "owner-only" in result.stderr
    password.chmod(0o600)
    result = subprocess.run(
        ["bash", str(script)], env={**env, "WATCH_AUTH_PILOT_IMAGE_DIGEST": "pilot:latest"},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 2
    assert "pinned" in result.stderr
    malformed = subprocess.run(
        ["bash", str(script)],
        env={**env, "WATCH_AUTH_PILOT_IMAGE_DIGEST": "us.icr.io/watch-authenticity-3235996/pilot@sha256:abc"},
        capture_output=True, text=True, timeout=10,
    )
    assert malformed.returncode == 2


@pytest.fixture
def deployment_env(tmp_path: Path) -> dict[str, str]:
    fake = tmp_path / "ibmcloud"
    fake.write_text("""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$CALL_LOG"
case "$*" in
  "ce project select"*) ;;
  "ce secret get --name watch-auth-pilot-gemini"*)
    if [[ -n "${MISSING_PROVIDER:-}" ]]; then exit 1; fi
    if [[ -n "${PROVIDER_SECRET:-}" ]]; then printf '%s' "$PROVIDER_SECRET"
    else printf '%s' '{"format":"generic","data":{"WATCH_AUTH_OPENROUTER_PILOT_KEY":"fixture-openrouter-key-secret-123456"}}'; fi
    ;;
  "ce secret get"*)
    [[ -f "$SECRET_STATE" ]] || exit 1
    printf '{"format":"generic","data":{"WATCH_AUTH_PILOT_PASSWORD":"%s"}}' "$(cat "$SECRET_STATE")"
    ;;
  "ce secret create"*) cp "$WATCH_AUTH_PILOT_PASSWORD_FILE" "$SECRET_STATE" ;;
  "ce app get"*)
    [[ -f "$APP_STATE" ]] || exit 1
    if [[ "$*" == *"--output url"* ]]; then printf 'https://pilot.example\\n';
    else
      revision=$(cat "$REVISION_STATE")
      extra_env=''
      extra_secret=''
      if [[ -f "$MODE_STATE" ]]; then
        extra_env=',{"name":"WATCH_AUTH_PILOT_MODE","value":"gemini"}'
      fi
      if [[ -f "$PROVIDER_STATE" ]]; then
        extra_secret=',{"secretRef":{"name":"watch-auth-pilot-gemini"}}'
      fi
      cat <<JSON
{
  "metadata": {"namespace": "test"},
  "status": {
    "conditions": [{"type": "Ready", "status": "$(if [[ -f "$UNREADY_STATE" ]]; then printf Unknown; else printf True; fi)"}],
    "latestCreatedRevisionName": "$revision",
    "latestReadyRevisionName": "${DRIFT_READY_REVISION:-$revision}",
    "traffic": [{"revisionName": "${DRIFT_TRAFFIC:-$revision}", "percent": 100}]
  },
  "spec": {"template": {
    "metadata": {"annotations": {
      "autoscaling.knative.dev/minScale": "${DRIFT_MIN_SCALE:-0}",
      "autoscaling.knative.dev/maxScale": "1",
      "autoscaling.knative.dev/scale-down-delay": "120s",
      "codeengine.cloud.ibm.com/mount-compute-resource-token": "true"
    }},
    "spec": {
      "containers": [{
        "image": "$(if [[ -f "$OLD_IMAGE_STATE" ]]; then printf old:image; else printf '%s' "${DRIFT_IMAGE:-$WATCH_AUTH_PILOT_IMAGE_DIGEST}"; fi)",
        "resources": {"requests": {"cpu": "2", "memory": "8G", "ephemeral-storage": "4G"}},
        "env": [
          {"name": "WATCH_AUTH_COS_BUCKET", "value": "$WATCH_AUTH_COS_BUCKET"},
          {"name": "WATCH_AUTH_COS_ENDPOINT", "value": "$WATCH_AUTH_COS_ENDPOINT"},
          {"name": "WATCH_AUTH_COS_SERVICE_INSTANCE_ID", "value": "$WATCH_AUTH_COS_SERVICE_INSTANCE_ID"},
          {"name": "WATCH_AUTH_TRUSTED_PROFILE_ID", "value": "$WATCH_AUTH_TRUSTED_PROFILE_ID"}$extra_env
        ],
        "envFrom": [{"secretRef": {"name": "${DRIFT_SECRET:-watch-auth-pilot-basic}"}}$extra_secret]
      }],
      "imagePullSecrets": [{"name": "$WATCH_AUTH_REGISTRY_SECRET"}],
      "serviceAccountName": "test-none",
      "containerConcurrency": 1,
      "timeoutSeconds": 300
    }
  }}
}
JSON
    fi
    ;;
  "ce app create"*|"ce app update"*)
    touch "$APP_STATE"
    rm -f "$UNREADY_STATE" "$OLD_IMAGE_STATE"
    revision=watch-auth-pilot-00001
    while [[ $# -gt 0 ]]; do
      case "$1" in
        --revision-name) revision=$2; shift ;;
        WATCH_AUTH_PILOT_MODE=gemini) touch "$MODE_STATE" ;;
        watch-auth-pilot-gemini) touch "$PROVIDER_STATE" ;;
      esac
      shift
    done
    printf '%s' "$revision" > "$REVISION_STATE"
    ;;
  "iam trusted-profile-links"*)
    if [[ -f "$LINK_STATE" ]]; then
      printf '[{"cr_type":"CE","link":{"crn":"crn:v1:project","component_type":"application","component_name":"watch-auth-pilot"}}]'
    else printf '[]'; fi
    ;;
  "iam trusted-profile-link-create"*) touch "$LINK_STATE" ;;
esac
""")
    fake.chmod(0o755)
    password = tmp_path / "password"
    password.write_text("p" * 32)
    password.chmod(0o600)
    return {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "calls"),
        "SECRET_STATE": str(tmp_path / "secret"),
        "APP_STATE": str(tmp_path / "app"),
        "UNREADY_STATE": str(tmp_path / "unready"),
        "OLD_IMAGE_STATE": str(tmp_path / "old-image"),
        "LINK_STATE": str(tmp_path / "link"),
        "MODE_STATE": str(tmp_path / "mode"),
        "PROVIDER_STATE": str(tmp_path / "provider"),
        "REVISION_STATE": str(tmp_path / "revision"),
        "WATCH_AUTH_PILOT_MODE": "local-vision",
        "WATCH_AUTH_CE_PROJECT": "watch-authenticity",
        "WATCH_AUTH_CE_PROJECT_CRN": "crn:v1:project",
        "WATCH_AUTH_PILOT_IMAGE_DIGEST": "us.icr.io/watch-authenticity-3235996/pilot@sha256:" + "a" * 64,
        "WATCH_AUTH_REGISTRY_SECRET": "pull-secret",
        "WATCH_AUTH_COS_BUCKET": "private",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
        "WATCH_AUTH_PILOT_PASSWORD_FILE": str(password),
    }


def test_deploy_creates_protected_bounded_app_and_reuses_it(
    tmp_path: Path, deployment_env: dict[str, str],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    env = deployment_env
    password = Path(env["WATCH_AUTH_PILOT_PASSWORD_FILE"])
    first = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == "https://pilot.example"
    calls = (tmp_path / "calls").read_text()
    assert "--env-from-secret watch-auth-pilot-basic" in calls
    assert "--min-scale 0 --max-scale 1 --concurrency 1" in calls
    assert "--cpu 2 --memory 8G --ephemeral-storage 4G" in calls
    assert "--trusted-profiles-enabled=true" in calls
    assert calls.count("ce secret create") == 1
    assert calls.count("ce app create") == 1
    assert calls.count("iam trusted-profile-link-create") == 1
    assert calls.index("iam trusted-profile-link-create") < calls.index("ce app create")
    assert "ce app update" not in calls
    assert "p" * 32 not in calls + first.stdout + first.stderr

    again = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode == 0, again.stderr
    calls = (tmp_path / "calls").read_text()
    assert calls.count("ce app create") == 1
    assert calls.count("iam trusted-profile-link-create") == 1

    (tmp_path / "unready").touch()
    recovered = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert recovered.returncode == 0, recovered.stderr
    calls = (tmp_path / "calls").read_text()
    assert calls.count("ce app update") == 1
    assert "--image " + env["WATCH_AUTH_PILOT_IMAGE_DIGEST"] in calls
    assert not (tmp_path / "unready").exists()

    (tmp_path / "old-image").touch()
    upgraded = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert upgraded.returncode == 0, upgraded.stderr
    assert (tmp_path / "calls").read_text().count("ce app update") == 2

    password.write_text("q" * 32)
    mismatch = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True, timeout=10,
    )
    assert mismatch.returncode != 0
    assert "secret differs" in mismatch.stderr
    assert "q" * 32 not in mismatch.stdout + mismatch.stderr
    password.write_text("p" * 32)

    drift = subprocess.run(
        ["bash", str(script)], env={**env, "DRIFT_IMAGE": "wrong:latest"},
        capture_output=True, text=True, timeout=10,
    )
    assert drift.returncode != 0
    assert "app image differs" in drift.stderr

    scale_drift = subprocess.run(
        ["bash", str(script)], env={**env, "DRIFT_MIN_SCALE": "2"},
        capture_output=True, text=True, timeout=10,
    )
    assert scale_drift.returncode != 0
    assert "app readback differs" in scale_drift.stderr

    secret_drift = subprocess.run(
        ["bash", str(script)], env={**env, "DRIFT_SECRET": "wrong-secret"},
        capture_output=True, text=True, timeout=10,
    )
    assert secret_drift.returncode != 0
    assert "app readback differs" in secret_drift.stderr


def test_deploy_activates_gemini_on_existing_pinned_image(
    tmp_path: Path, deployment_env: dict[str, str],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    legacy = subprocess.run(
        ["bash", str(script)], env=deployment_env, capture_output=True, text=True, timeout=10,
    )
    assert legacy.returncode == 0, legacy.stderr
    gemini_env = {**deployment_env, "WATCH_AUTH_PILOT_MODE": "gemini"}
    activated = subprocess.run(
        ["bash", str(script)], env=gemini_env, capture_output=True, text=True, timeout=10,
    )
    assert activated.returncode == 0, activated.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    updates = [call for call in calls if call.startswith("ce app update ")]
    assert len(updates) == 1
    assert "--env WATCH_AUTH_PILOT_MODE=gemini" in updates[0]
    assert "--env-from-secret watch-auth-pilot-gemini" in updates[0]
    assert "--image " + gemini_env["WATCH_AUTH_PILOT_IMAGE_DIGEST"] in updates[0]
    assert "fixture-openrouter-key" not in activated.stdout + activated.stderr + "\n".join(calls)

    repeated = subprocess.run(
        ["bash", str(script)], env=gemini_env, capture_output=True, text=True, timeout=10,
    )
    assert repeated.returncode == 0, repeated.stderr
    assert (tmp_path / "calls").read_text().count("ce app update") == 1

    rollback = subprocess.run(
        ["bash", str(script)], env=deployment_env, capture_output=True, text=True, timeout=10,
    )
    assert rollback.returncode != 0
    assert "refusing to switch" in rollback.stderr
    assert (tmp_path / "calls").read_text().count("ce app update") == 1


@pytest.mark.parametrize(
    "invalid_provider",
    [
        {"MISSING_PROVIDER": "1"},
        {"PROVIDER_SECRET": "not json"},
        {"PROVIDER_SECRET": '{"format":"generic","data":{"WRONG":"fixture-openrouter-key"}}'},
        {"PROVIDER_SECRET": '{"format":"generic","data":{"WATCH_AUTH_OPENROUTER_PILOT_KEY":" "}}'},
        {"PROVIDER_SECRET": '{"format":"generic","data":{"WATCH_AUTH_OPENROUTER_PILOT_KEY":"fixture-openrouter-key","OTHER":"unapproved"}}'},
    ],
)
def test_invalid_provider_secret_fails_before_writes(
    tmp_path: Path, deployment_env: dict[str, str], invalid_provider: dict[str, str],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    env = {**deployment_env, "WATCH_AUTH_PILOT_MODE": "gemini", **invalid_provider}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "dedicated Code Engine provider secret" in result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert calls == [
        "ce project select --name watch-authenticity",
        "ce secret get --name watch-auth-pilot-gemini --output json",
    ]
    assert "fixture-openrouter-key" not in result.stdout + result.stderr


@pytest.mark.parametrize("drift", ["DRIFT_TRAFFIC", "DRIFT_READY_REVISION"])
def test_gemini_requires_the_created_revision_to_serve(
    tmp_path: Path, deployment_env: dict[str, str], drift: str,
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    env = {**deployment_env, "WATCH_AUTH_PILOT_MODE": "gemini", drift: "watch-auth-pilot-older"}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert "requested pilot revision is not ready and serving all traffic" in result.stderr
    calls = (tmp_path / "calls").read_text().splitlines()
    assert sum(call.startswith("ce app create ") for call in calls) == 1
    assert not any(call.endswith("--output url") for call in calls)
    assert "fixture-openrouter-key" not in result.stdout + result.stderr + "\n".join(calls)


def test_gemini_creates_and_verifies_the_first_revision(
    tmp_path: Path, deployment_env: dict[str, str],
) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    env = {**deployment_env, "WATCH_AUTH_PILOT_MODE": "gemini"}
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "https://pilot.example"
    calls = (tmp_path / "calls").read_text().splitlines()
    create = next(call for call in calls if call.startswith("ce app create "))
    assert "--env-from-secret watch-auth-pilot-basic" in create
    assert "--env-from-secret watch-auth-pilot-gemini" in create
    assert "--env WATCH_AUTH_PILOT_MODE=gemini" in create
    assert "--min-scale 0 --max-scale 1 --concurrency 1" in create
    assert calls[-1].endswith("--output url")
    assert "fixture-openrouter-key" not in result.stdout + result.stderr + "\n".join(calls)
