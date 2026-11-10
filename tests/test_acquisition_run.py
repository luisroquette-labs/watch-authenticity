import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from acquisition.collect import DownloadRateLimited
from acquisition.discovery import DiscoveryPage, DiscoveryRecord
from acquisition.run import run_wikimedia_once
from scrapers.wikimedia import WikimediaAPIError, WikimediaCandidate, WikimediaDiscovery

NOW = datetime(2026, 11, 10, 19, 0, tzinfo=timezone.utc)


def _result(*, cursor=None, next_cursor=None) -> WikimediaDiscovery:
    page = DiscoveryPage(
        source="wikimedia",
        access_record_id="access-wikimedia-1",
        cursor=cursor,
        next_cursor=next_cursor,
        observed_at=NOW,
        records=(
            DiscoveryRecord(
                source="wikimedia",
                source_record_id="123",
                source_version="v1",
                source_url="https://commons.wikimedia.org/wiki/File:Watch.jpg",
                seller_group_id="Author",
                declared_reference="rolex-submariner-126610ln",
                image_ids=("sha1",),
                image_urls=("https://upload.wikimedia.org/watch.jpg",),
            ),
        ),
    )
    candidate = cast(
        WikimediaCandidate,
        SimpleNamespace(
            metadata=SimpleNamespace(source_record_id="123"),
            rights=SimpleNamespace(rights_record_id="rights-123"),
            observed_at=datetime(2020, 1, 2, 3, 4, tzinfo=timezone.utc),
        ),
    )
    return WikimediaDiscovery(page=page, candidates=(candidate,))


def test_run_collects_before_committing_and_repeated_run_is_noop(cos, monkeypatch):
    calls: list[str] = []

    def collect(*_args, **kwargs):
        calls.append("collected_at" if "collected_at" in kwargs else "actual-now")
        return SimpleNamespace(accepted_candidate_ids=("candidate-123",))

    monkeypatch.setattr("acquisition.run.collect_batch", collect)
    result = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=NOW,
        api_client=object(),
        image_client=object(),
        discover=lambda *_args, **_kwargs: _result(),
    )
    writes = cos.write_count
    repeated = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=NOW,
        api_client=object(),
        image_client=object(),
        discover=lambda *_args, **_kwargs: pytest.fail("completed cycle was rediscovered"),
    )

    assert calls == ["actual-now"]
    assert result["state"] == "complete"
    assert result["accepted_candidate_ids"] == ["candidate-123"]
    assert repeated["state"] == "already_complete"
    assert cos.write_count == writes


def test_collection_failure_does_not_advance_checkpoint(cos, monkeypatch):
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("download failed")),
    )

    with pytest.raises(RuntimeError, match="download failed"):
        run_wikimedia_once(
            cos,
            "private",
            "datasets",
            SimpleNamespace(),
            SimpleNamespace(),
            now=NOW,
            api_client=object(),
            image_client=object(),
            discover=lambda *_args, **_kwargs: _result(),
        )

    assert not any("/discovery/" in key for key in cos.bodies)


def test_incomplete_cycle_resumes_next_day_instead_of_restarting(cos, monkeypatch):
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *_args, **_kwargs: SimpleNamespace(accepted_candidate_ids=("candidate-123",)),
    )
    first = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=NOW,
        api_client=object(),
        image_client=object(),
        discover=lambda *_args, **_kwargs: _result(next_cursor="page-2"),
    )

    def second_page(*_args, **kwargs):
        assert kwargs["previous"].next_cursor == "page-2"
        return _result(cursor="page-2")

    second = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=datetime(2026, 11, 11, 19, 0, tzinfo=timezone.utc),
        api_client=object(),
        image_client=object(),
        discover=second_page,
    )

    assert first["state"] == "page_complete"
    assert second["state"] == "complete"
    assert first["cycle"] == second["cycle"] == "2026-11-10"


def test_rate_limit_persists_recoverable_terminal_without_checkpoint(cos):
    def rate_limited(*_args, **_kwargs):
        raise WikimediaAPIError("Wikimedia rate limited; resume later")

    result = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=NOW,
        api_client=object(),
        image_client=object(),
        discover=rate_limited,
    )

    assert result == {
        "cycle": "2026-11-10",
        "source": "wikimedia",
        "state": "rate_limited",
    }
    assert sum("/acquisition-status/" in key for key in cos.bodies) == 1
    assert not any("/discovery/" in key for key in cos.bodies)


def test_image_rate_limit_is_recoverable_and_does_not_commit(cos, monkeypatch):
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DownloadRateLimited("resume later")
        ),
    )

    result = run_wikimedia_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(),
        SimpleNamespace(),
        now=NOW,
        api_client=object(),
        image_client=object(),
        discover=lambda *_args, **_kwargs: _result(),
    )

    assert result["state"] == "rate_limited"
    assert not any("/discovery/" in key for key in cos.bodies)


def test_nonrecoverable_failure_persists_only_error_type(cos):
    def fail(*_args, **_kwargs):
        raise WikimediaAPIError("unsafe payload details")

    with pytest.raises(WikimediaAPIError, match="unsafe payload"):
        run_wikimedia_once(
            cos,
            "private",
            "datasets",
            SimpleNamespace(),
            SimpleNamespace(),
            now=NOW,
            api_client=object(),
            image_client=object(),
            discover=fail,
        )

    bodies = [
        body.decode()
        for key, body in cos.bodies.items()
        if "/acquisition-status/" in key
    ]
    assert len(bodies) == 1
    assert "WikimediaAPIError" in bodies[0]
    assert "unsafe payload details" not in bodies[0]


def test_runtime_artifacts_are_bounded_idempotent_and_secret_safe():
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/deploy_ibm_acquisition.sh").read_text()
    container = (root / "Containerfile.acquisition").read_text()

    assert "@sha256:" in script
    assert "job create" in script and "job update" in script
    assert "sub cron create" in script and "sub cron update" in script
    assert "17 3 * * *" in script and "--maxexecutiontime 600" in script
    assert "secret list" not in script and "API_KEY" not in script
    assert "WATCH_AUTH_REGISTRY_SECRET" in script
    assert "trusted-profile-links" in script and "trusted-profile-link-create" in script
    assert 'CMD ["python", "-m", "acquisition.run"]' in container
    assert "requirements-acquisition.lock" in container
    assert "@sha256:" in container and "USER 10001:10001" in container
    assert container.count("--require-hashes") == 2
    assert "requirements.txt" not in container


def test_deploy_script_creates_and_rechecks_trusted_profile_link(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "ibmcloud"
    fake.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALL_LOG"
case "$*" in
  "ce job get"*) exit 1 ;;
  "ce sub cron get"*) exit 1 ;;
  "iam trusted-profile-links"*)
    if [[ -f "$LINK_STATE" ]]; then
      printf '[{"cr_type":"CE","link":{"crn":"%s","component_type":"job","component_name":"watch-auth-acquisition"}}]' "$WATCH_AUTH_CE_PROJECT_CRN"
    else
      printf '[]'
    fi
    ;;
  "iam trusted-profile-link-create"*) touch "$LINK_STATE" ;;
esac
"""
    )
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(log),
        "LINK_STATE": str(tmp_path / "linked"),
        "WATCH_AUTH_CE_PROJECT": "watch-authenticity",
        "WATCH_AUTH_CE_PROJECT_CRN": "crn:v1:project",
        "WATCH_AUTH_ACQUISITION_IMAGE_DIGEST": "us.icr.io/ns/acquisition@sha256:" + "a" * 64,
        "WATCH_AUTH_REGISTRY_SECRET": "registry-pull",
        "WATCH_AUTH_COS_BUCKET": "private",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
        "WATCH_AUTH_ACCOUNT_ID": "a" * 32,
        "WATCH_AUTH_IAM_READER_POLICY_ID": "11111111-1111-4111-8111-111111111111",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts/deploy_ibm_acquisition.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    calls = log.read_text()
    assert "ce job create" in calls
    assert calls.count("iam trusted-profile-links") == 2
    assert "iam trusted-profile-link-create" in calls
    assert "ce sub cron create" in calls


def test_deploy_fails_closed_when_link_inventory_is_unavailable(tmp_path):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "ibmcloud"
    fake.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALL_LOG"
case "$*" in
  "iam trusted-profile-links"*) exit 9 ;;
esac
"""
    )
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "CALL_LOG": str(log),
        "WATCH_AUTH_CE_PROJECT": "watch-authenticity",
        "WATCH_AUTH_CE_PROJECT_CRN": "crn:v1:project",
        "WATCH_AUTH_ACQUISITION_IMAGE_DIGEST": "us.icr.io/ns/acquisition@sha256:" + "a" * 64,
        "WATCH_AUTH_REGISTRY_SECRET": "registry-pull",
        "WATCH_AUTH_COS_BUCKET": "private",
        "WATCH_AUTH_COS_ENDPOINT": "https://cos.example",
        "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": "cos-1",
        "WATCH_AUTH_TRUSTED_PROFILE_ID": "profile-1",
        "WATCH_AUTH_ACCOUNT_ID": "a" * 32,
        "WATCH_AUTH_IAM_READER_POLICY_ID": "11111111-1111-4111-8111-111111111111",
    }

    result = subprocess.run(
        ["bash", str(root / "scripts/deploy_ibm_acquisition.sh")],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode == 2
    assert "trusted-profile link inventory failed" in result.stderr
    assert "trusted-profile-link-create" not in log.read_text()
