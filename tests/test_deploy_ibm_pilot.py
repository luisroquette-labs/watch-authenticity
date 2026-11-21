"""The IBM pilot deploy keeps credentials secret and cannot drift to another image."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


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


def test_deploy_creates_protected_bounded_app_and_reuses_it(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/deploy_ibm_pilot.sh"
    fake = tmp_path / "ibmcloud"
    fake.write_text("""#!/usr/bin/env bash
set -eu
printf '%s\\n' "$*" >> "$CALL_LOG"
case "$*" in
  "ce project select"*) ;;
  "ce secret get"*)
    [[ -f "$SECRET_STATE" ]] || exit 1
    printf '{"format":"generic","data":{"WATCH_AUTH_PILOT_PASSWORD":"%s"}}' "$(cat "$SECRET_STATE")"
    ;;
  "ce secret create"*) cp "$WATCH_AUTH_PILOT_PASSWORD_FILE" "$SECRET_STATE" ;;
  "ce app get"*)
    [[ -f "$APP_STATE" ]] || exit 1
    if [[ "$*" == *"--output url"* ]]; then printf 'https://pilot.example\\n';
    else printf '{"spec":{"template":{"containers":[{"image":"%s"}]}}}' "${DRIFT_IMAGE:-$WATCH_AUTH_PILOT_IMAGE_DIGEST}"; fi
    ;;
  "ce app create"*) touch "$APP_STATE" ;;
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
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "calls"),
        "SECRET_STATE": str(tmp_path / "secret"),
        "APP_STATE": str(tmp_path / "app"),
        "LINK_STATE": str(tmp_path / "link"),
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
    first = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip().endswith("https://pilot.example")
    calls = (tmp_path / "calls").read_text()
    assert "--env-from-secret watch-auth-pilot-basic" in calls
    assert "--min-scale 0 --max-scale 1 --concurrency 1" in calls
    assert "--cpu 2 --memory 8G --ephemeral-storage 4G" in calls
    assert "--trusted-profiles-enabled=true" in calls
    assert calls.count("ce secret create") == 1
    assert calls.count("ce app create") == 1
    assert calls.count("iam trusted-profile-link-create") == 1
    assert "p" * 32 not in calls + first.stdout + first.stderr

    again = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert again.returncode == 0, again.stderr
    calls = (tmp_path / "calls").read_text()
    assert calls.count("ce app create") == 1
    assert calls.count("iam trusted-profile-link-create") == 1

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
    assert "image readback differs" in drift.stderr
