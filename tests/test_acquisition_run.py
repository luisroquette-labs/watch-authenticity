import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from acquisition.collect import DownloadRateLimited, UnsafeImage
from acquisition.discovery import DiscoveryPage, DiscoveryRecord
from acquisition.run import (
    run_metadata_discovery_once,
    run_metadata_discovery_resilient,
    run_public_image_collection_once,
    run_wikimedia_once,
)
from acquisition.types import CollectionAuthorizationRecord
from scrapers.metadata_http import HTTPDeadlineExceeded
from scrapers.replica_sitemap import ReplicaSitemapError
from scrapers.wikimedia import WikimediaAPIError, WikimediaCandidate, WikimediaDiscovery

NOW = datetime(2026, 11, 10, 19, 0, tzinfo=timezone.utc)


def test_acquisition_runtime_import_does_not_require_training_dataset():
    script = """
import importlib.abc
import sys
class DenyDataset(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "dataset" or fullname.startswith("dataset."):
            raise ImportError("training dataset unavailable in acquisition image")
sys.meta_path.insert(0, DenyDataset())
import acquisition.run
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stderr


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


def test_wikimedia_lane_has_independent_checkpoint(cos, monkeypatch):
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *_args, **_kwargs: SimpleNamespace(accepted_candidate_ids=("candidate-123",)),
    )
    run_wikimedia_once(
        cos, "private", "datasets", SimpleNamespace(), SimpleNamespace(),
        now=NOW, api_client=object(), image_client=object(),
        discover=lambda *_args, **_kwargs: _result(next_cursor="q:1:0"),
    )
    lane = run_wikimedia_once(
        cos, "private", "datasets/lanes/speedmaster", SimpleNamespace(), SimpleNamespace(),
        now=NOW, api_client=object(), image_client=object(),
        discover=lambda *_args, **kwargs: _result(
            cursor=kwargs["previous"].next_cursor if kwargs["previous"] else None
        ),
    )
    assert lane["state"] == "complete"
    assert "datasets/lanes/speedmaster/discovery/2026-11-10/wikimedia/children/root.json" in cos.bodies
    assert "datasets/discovery/2026-11-10/wikimedia/children/root.json" in cos.bodies

    replica_lane = run_wikimedia_once(
        cos, "private", "datasets/lanes/speedmaster/replica-research", SimpleNamespace(), SimpleNamespace(),
        now=NOW, api_client=object(), image_client=object(),
        discover=lambda *_args, **kwargs: _result(
            cursor=kwargs["previous"].next_cursor if kwargs["previous"] else None
        ),
    )
    assert replica_lane["state"] == "complete"
    assert "datasets/lanes/speedmaster/replica-research/discovery/2026-11-10/wikimedia/children/root.json" in cos.bodies

    auckland_lane = run_wikimedia_once(
        cos, "private", "datasets/lanes/auckland-oyster/research", SimpleNamespace(), SimpleNamespace(),
        now=NOW, api_client=object(), image_client=object(),
        discover=lambda *_args, **kwargs: _result(
            cursor=kwargs["previous"].next_cursor if kwargs["previous"] else None
        ),
    )
    assert auckland_lane["state"] == "complete"
    assert "datasets/lanes/auckland-oyster/research/discovery/2026-11-10/wikimedia/children/root.json" in cos.bodies


def test_metadata_only_source_commits_without_downloading(cos):
    page = DiscoveryPage(
        source="vsfactoryusa",
        access_record_id="access-vsfactoryusa-1",
        cursor=None,
        next_cursor=None,
        observed_at=NOW,
        records=(
            DiscoveryRecord(
                source="vsfactoryusa",
                source_record_id="126610ln-super-clone",
                source_version="2026-11-09",
                source_url="https://vsfactoryusa.com/126610ln-super-clone/",
                seller_group_id="vsfactoryusa",
                declared_reference="rolex-submariner-126610ln",
                image_ids=("image-1",),
                image_urls=("https://vsfactoryusa.com/images/126610ln-01.webp",),
            ),
        ),
    )

    result = run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "vsfactoryusa",
        now=NOW,
        discover=lambda previous: page,
    )
    writes = cos.write_count
    repeated = run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "vsfactoryusa",
        now=NOW,
        discover=lambda _previous: pytest.fail("completed cycle was rediscovered"),
    )

    assert result["record_count"] == 1
    assert result["state"] == "complete"
    assert repeated["state"] == "already_complete"
    assert cos.write_count == writes


def test_public_collection_downloads_every_discovered_image_before_committing(
    cos, monkeypatch
):
    page = DiscoveryPage(
        source="richfactoryus",
        access_record_id="access-richfactoryus-1",
        cursor=None,
        next_cursor=None,
        observed_at=NOW,
        records=(
            DiscoveryRecord(
                source="richfactoryus",
                source_record_id="qc-sets/rolex-submariner-126610ln-black",
                source_version="v1",
                source_url="https://richfactoryus.com/qc-sets/rolex-submariner-126610ln-black/",
                seller_group_id="richfactoryus",
                declared_reference="rolex-submariner-126610ln",
                image_ids=("image-1", "image-2", "image-3"),
                image_urls=(
                    "https://richfactoryus.com/images/one.webp",
                    "https://richfactoryus.com/images/two.webp",
                    "https://richfactoryus.com/images/three.webp",
                ),
                metadata=(
                    ("classification", "replica_candidate_vendor_declared"),
                    ("factory_claim", "RICH Factory"),
                ),
            ),
        ),
    )
    authorization = CollectionAuthorizationRecord(
        rights_record_id="owner-public-download-20261112-richfactoryus",
        source="richfactoryus",
        acquisition_method="public-sitemap",
        allowed_purposes=("download",),
        authorized_at=NOW,
        authorized_by="project-owner",
        evidence_sha256="a" * 64,
    )
    calls = []

    def collect(records, *_args, **_kwargs):
        normalized = tuple(records)
        calls.append(normalized)
        assert not any("/discovery/" in key for key in cos.bodies)
        return SimpleNamespace(
            accepted_candidate_ids=tuple(record.source_record_id for record in normalized)
        )

    monkeypatch.setattr("acquisition.run.collect_batch", collect)
    result = run_public_image_collection_once(
        cos,
        "private",
        "datasets",
        SimpleNamespace(source="richfactoryus", policy_id="rich-public-v1"),
        SimpleNamespace(),
        authorization,
        now=NOW,
        image_client=object(),
        discover=lambda _previous: page,
    )

    assert len(calls) == 3
    assert [record.image_url for call in calls for record in call] == list(page.records[0].image_urls)
    assert all(record.listing_text == "replica" for call in calls for record in call)
    assert all(record.factory == "RICH Factory" for call in calls for record in call)
    assert result["accepted_count"] == 3
    assert result["rejected_images"] == []
    assert result["rejection_key"] is None
    assert result["state"] == "complete"
    assert any("/public-downloads/discovery/" in key for key in cos.bodies)


def test_public_wave_records_unsafe_image_and_commits_remaining_candidates(cos, monkeypatch):
    from dataclasses import replace

    record = replace(
        _result().page.records[0],
        image_ids=("bad", "good"),
        image_urls=("https://upload.wikimedia.org/bad.png", "https://upload.wikimedia.org/good.jpg"),
    )
    page = replace(_result().page, records=(record,))

    def collect(records, *_args, **_kwargs):
        candidate = records[0]
        if candidate.image_url.endswith("bad.png"):
            raise UnsafeImage("content type does not match image magic")
        return SimpleNamespace(accepted_candidate_ids=("candidate-good",))

    monkeypatch.setattr("acquisition.run.collect_batch", collect)
    result = run_public_image_collection_once(
        cos, "private", "datasets", SimpleNamespace(source="wikimedia"),
        SimpleNamespace(), SimpleNamespace(), now=NOW, image_client=object(),
        discover=lambda _previous: page,
    )

    assert result["state"] == "complete"
    assert result["accepted_candidate_ids"] == ["candidate-good"]
    assert result["rejected_images"] == [{
        "source_record_id": "123#image-1-bad",
        "image_url": "https://upload.wikimedia.org/bad.png",
        "reason": "content type does not match image magic",
    }]
    assert result["rejection_key"] in cos.bodies
    assert cos.bodies[result["rejection_key"]]
    assert any("/public-downloads/discovery/" in key for key in cos.bodies)


def test_public_rejection_survives_status_write_failure_after_cursor_commit(cos, monkeypatch):
    from dataclasses import replace

    record = replace(
        _result().page.records[0], image_ids=("bad",),
        image_urls=("https://upload.wikimedia.org/bad.png",),
    )
    page = replace(_result().page, records=(record,))
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnsafeImage("invalid image")),
    )
    monkeypatch.setattr(
        "acquisition.run._persist_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("status unavailable")),
    )

    with pytest.raises(RuntimeError, match="status unavailable"):
        run_public_image_collection_once(
            cos, "private", "datasets", SimpleNamespace(source="wikimedia"),
            SimpleNamespace(), SimpleNamespace(), now=NOW, image_client=object(),
            discover=lambda _previous: page,
        )

    rejection_keys = [key for key in cos.bodies if "/public-downloads/rejections/" in key]
    assert len(rejection_keys) == 1
    assert b'"source_record_id":"123#image-1-bad"' in cos.bodies[rejection_keys[0]]
    assert any("/public-downloads/discovery/" in key for key in cos.bodies)
    assert run_public_image_collection_once(
        cos, "private", "datasets", SimpleNamespace(source="wikimedia"),
        SimpleNamespace(), SimpleNamespace(), now=NOW, image_client=object(),
        discover=lambda _previous: pytest.fail("completed page should not be rediscovered"),
    )["state"] == "already_complete"


def test_public_image_failure_is_recorded_without_advancing_checkpoint(
    cos, monkeypatch
):
    from acquisition.run import run_public_image_collection_resilient
    from infra.ibm_cos import IntegrityError

    policy = SimpleNamespace(source="richfactoryus")
    access = SimpleNamespace(
        acquisition_method="public-sitemap", requests_per_minute=6
    )
    monkeypatch.setattr(
        "acquisition.run.run_public_image_collection_once",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            IntegrityError("invalid public image")
        ),
    )

    result = run_public_image_collection_resilient(
        cos,
        "private",
        "datasets",
        policy,
        access,
        now=NOW,
        discover=lambda _previous: pytest.fail("stubbed collector must own discovery"),
    )

    assert result["state"] == "failed"
    assert result["error_type"] == "IntegrityError"
    assert not any("/public-downloads/discovery/" in key for key in cos.bodies)
    assert any("/public-downloads/acquisition-status/" in key for key in cos.bodies)


def test_public_permission_failure_does_not_block_later_source(cos, monkeypatch):
    from acquisition.run import run_public_image_collection_resilient

    access = SimpleNamespace(
        acquisition_method="public-sitemap", requests_per_minute=6
    )
    outcomes = iter(
        (
            PermissionError("expired source access"),
            {"source": "superclonerolex", "state": "complete"},
        )
    )

    def collect(*_args, **_kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    failed = run_public_image_collection_resilient(
        cos,
        "private",
        "datasets",
        SimpleNamespace(source="richfactoryus"),
        access,
        now=NOW,
        discover=lambda _previous: pytest.fail("stubbed collector owns discovery"),
    )
    completed = run_public_image_collection_resilient(
        cos,
        "private",
        "datasets",
        SimpleNamespace(source="superclonerolex"),
        access,
        now=NOW,
        discover=lambda _previous: pytest.fail("stubbed collector owns discovery"),
    )

    assert failed["state"] == "failed"
    assert failed["error_type"] == "PermissionError"
    assert completed == {"source": "superclonerolex", "state": "complete"}


def test_recoverable_metadata_failure_does_not_block_later_source(cos):
    failed = run_metadata_discovery_resilient(
        cos,
        "private",
        "datasets",
        "richfactoryus",
        now=NOW,
        discover=lambda _previous: (_ for _ in ()).throw(
            ReplicaSitemapError("exact-reference product images are missing")
        ),
    )
    completed = run_metadata_discovery_resilient(
        cos,
        "private",
        "datasets",
        "superclonerolex",
        now=NOW,
        discover=lambda _previous: DiscoveryPage(
            source="superclonerolex",
            access_record_id="access-superclonerolex-1",
            cursor=None,
            next_cursor=None,
            observed_at=NOW,
            records=(),
        ),
    )

    assert failed == {
        "cycle": "2026-11-10",
        "error_type": "ReplicaSitemapError",
        "source": "richfactoryus",
        "state": "failed",
    }
    assert completed["state"] == "complete"
    assert any(b'"source":"richfactoryus","state":"failed"' in body for body in cos.bodies.values())
    assert any(b'"source":"superclonerolex","state":"complete"' in body for body in cos.bodies.values())


def test_unexpected_metadata_failure_still_fails_job(cos):
    with pytest.raises(RuntimeError, match="programming defect"):
        run_metadata_discovery_resilient(
            cos,
            "private",
            "datasets",
            "richfactoryus",
            now=NOW,
            discover=lambda _previous: (_ for _ in ()).throw(
                RuntimeError("programming defect")
            ),
        )


@pytest.mark.parametrize(
    "failure",
    (
        HTTPDeadlineExceeded("HTTP read deadline exhausted"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
    ),
)
def test_expected_metadata_read_failure_does_not_block_job(cos, failure):
    result = run_metadata_discovery_resilient(
        cos,
        "private",
        "datasets",
        "superclonerolex",
        now=NOW,
        discover=lambda _previous: (_ for _ in ()).throw(failure),
    )

    assert result["state"] == "failed"
    assert result["error_type"] == type(failure).__name__


def test_metadata_source_resumes_its_older_cycle_after_another_source_runs(cos):
    first_page = DiscoveryPage(
        source="vsfactoryusa",
        access_record_id="access-vsfactoryusa-1",
        cursor=None,
        next_cursor="page-2",
        observed_at=NOW,
        records=(
            DiscoveryRecord(
                source="vsfactoryusa",
                source_record_id="first",
                source_version="v1",
                source_url="https://vsfactoryusa.com/first/",
                seller_group_id="vsfactoryusa",
                declared_reference="rolex-submariner-126610ln",
                image_ids=("first",),
                image_urls=("https://vsfactoryusa.com/images/first.webp",),
            ),
        ),
    )
    run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "vsfactoryusa",
        now=NOW,
        discover=lambda _previous: first_page,
    )
    next_day = datetime(2026, 11, 11, 19, 0, tzinfo=timezone.utc)
    other_page = DiscoveryPage(
        source="other",
        access_record_id="access-other-1",
        cursor=None,
        next_cursor=None,
        observed_at=next_day,
        records=(
            DiscoveryRecord(
                source="other",
                source_record_id="other",
                source_version="v1",
                source_url="https://other.example/item/",
                seller_group_id="other",
                declared_reference="watch",
                image_ids=("other",),
                image_urls=("https://other.example/image.webp",),
            ),
        ),
    )
    run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "other",
        now=next_day,
        discover=lambda _previous: other_page,
    )

    def resume(previous):
        assert previous is not None and previous.next_cursor == "page-2"
        return DiscoveryPage(
            source="vsfactoryusa",
            access_record_id="access-vsfactoryusa-1",
            cursor="page-2",
            next_cursor=None,
            observed_at=next_day,
            records=(),
        )

    result = run_metadata_discovery_once(
        cos,
        "private",
        "datasets",
        "vsfactoryusa",
        now=next_day,
        discover=resume,
    )

    assert result["cycle"] == "2026-11-10"
    assert result["state"] == "complete"


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
    assert "job create" in script and "job update" not in script
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


def test_research_census_source_is_accepted_by_runtime_cli():
    result = subprocess.run(
        [sys.executable, "-m", "acquisition.run", "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0
    assert "research-census" in result.stdout


@pytest.mark.parametrize("existing_cron", [False, True])
def test_deploy_script_creates_and_rechecks_trusted_profile_link(tmp_path, existing_cron):
    root = Path(__file__).resolve().parents[1]
    fake = tmp_path / "ibmcloud"
    fake.write_text(
        """#!/usr/bin/env bash
set -eu
printf '%s\n' "$*" >> "$CALL_LOG"
case "$*" in
  "ce job get"*)
    if [[ ! -f "$JOB_STATE" ]] || ! grep -qx "$5" "$JOB_STATE"; then exit 1; fi
    args='["-m","acquisition.run"]'
    if [[ "$5" == watch-auth-supercloneluxury-* ]]; then args='["-m","acquisition.run","--source","supercloneluxury"]'; fi
    if [[ "$5" == watch-auth-cbp-* ]]; then args='["-m","acquisition.run","--source","cbp"]'; fi
    if [[ "$5" == watch-auth-goldsmiths-* ]]; then args='["-m","acquisition.run","--source","goldsmiths"]'; fi
    if [[ "$5" == watch-auth-bobswatches-* ]]; then args='["-m","acquisition.run","--source","bobswatches"]'; fi
    if [[ "$5" == watch-auth-speedmaster-* ]]; then args='["-m","acquisition.run","--source","wikimedia-speedmaster"]'; fi
    if [[ "$5" == watch-auth-auckland-* ]]; then args='["-m","acquisition.run","--source","wikimedia-auckland"]'; fi
    if [[ "$5" == watch-auth-daytona-* ]]; then args='["-m","acquisition.run","--source","wikimedia-daytona"]'; fi
    if [[ "$5" == watch-auth-research-census-* ]]; then args='["-m","acquisition.run","--source","research-census"]'; fi
    image="$WATCH_AUTH_ACQUISITION_IMAGE_DIGEST"
    cpu='0.25'
    service_account='project-none'
    if [[ "${DRIFTED_JOB:-0}" == 1 && "$5" == watch-auth-cbp-* ]]; then cpu='1'; fi
    if [[ "${DRIFTED_JOB:-0}" == 2 && "$5" == watch-auth-goldsmiths-* ]]; then image='unapproved:latest'; fi
    if [[ "${DRIFTED_JOB:-0}" == 3 && "$5" == watch-auth-goldsmiths-* ]]; then service_account='privileged-none'; fi
    printf '{"metadata":{"namespace":"project"},"spec":{"arraySpec":"0","maxExecutionTime":600,"retryLimit":0,"mode":"task","template":{"serviceAccountName":"%s","mountComputeResourceToken":true,"imagePullSecrets":[{"name":"%s"}],"containers":[{"image":"%s","command":["python"],"args":%s,"resources":{"requests":{"cpu":"%sm","ephemeral-storage":"400M","memory":"1G"}},"env":[' "$service_account" "$WATCH_AUTH_REGISTRY_SECRET" "$image" "$args" "${cpu/0.25/250}"
    printf '{"name":"WATCH_AUTH_COS_BUCKET","value":"%s"},' "$WATCH_AUTH_COS_BUCKET"
    printf '{"name":"WATCH_AUTH_COS_ENDPOINT","value":"%s"},' "$WATCH_AUTH_COS_ENDPOINT"
    printf '{"name":"WATCH_AUTH_COS_SERVICE_INSTANCE_ID","value":"%s"},' "$WATCH_AUTH_COS_SERVICE_INSTANCE_ID"
    printf '{"name":"WATCH_AUTH_TRUSTED_PROFILE_ID","value":"%s"},' "$WATCH_AUTH_TRUSTED_PROFILE_ID"
    printf '{"name":"WATCH_AUTH_ACCOUNT_ID","value":"%s"},' "$WATCH_AUTH_ACCOUNT_ID"
    printf '{"name":"WATCH_AUTH_IAM_READER_POLICY_ID","value":"%s"},' "$WATCH_AUTH_IAM_READER_POLICY_ID"
    printf '{"name":"WATCH_AUTH_ACQUISITION_PREFIX","value":"datasets"},'
    printf '{"name":"CE_API_BASE_URL","value":"https://api.private.us-south.codeengine.cloud.ibm.com"},'
    printf '{"name":"CE_PROJECT_ID","value":"project-id"},'
    printf '{"name":"CE_REGION","value":"us-south"}]}]}}}'
    ;;
  "ce job create"*) printf '%s\n' "$5" >> "$JOB_STATE" ;;
  "ce sub cron get"*) [[ "$EXISTING_CRON" == 1 ]] ;;
  "ce sub cron create"*|"ce sub cron update"*)
    if [[ "${FAIL_GOLDSMITHS_CRON:-0}" == 1 && "$*" == *watch-auth-goldsmiths-daily* ]]; then exit 2; fi
    ;;
  "iam trusted-profile-links"*)
    printf '['
    first=1
    if [[ -f "$LINK_STATE" ]]; then
      while IFS= read -r job; do
        [[ "$first" -eq 1 ]] || printf ','
        printf '{"cr_type":"CE","link":{"crn":"%s","component_type":"job","component_name":"%s"}}' "$WATCH_AUTH_CE_PROJECT_CRN" "$job"
        first=0
      done < "$LINK_STATE"
    fi
    printf ']'
    ;;
  "iam trusted-profile-link-create"*)
    for ((i=1; i<=$#; i++)); do
      if [[ "${!i}" == '--link-component-name' ]]; then
        j=$((i+1)); printf '%s\n' "${!j}" >> "$LINK_STATE"; break
      fi
    done
    ;;
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
        "JOB_STATE": str(tmp_path / "jobs"),
        "EXISTING_CRON": "1" if existing_cron else "0",
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

    failed = subprocess.run(
        ["bash", str(root / "scripts/deploy_ibm_acquisition.sh")],
        capture_output=True,
        text=True,
        env={**env, "FAIL_GOLDSMITHS_CRON": "1"},
        timeout=10,
    )
    assert failed.returncode == 2
    assert "watch-auth-acquisition-daily" not in log.read_text()
    log.write_text("")
    (tmp_path / "jobs").unlink(missing_ok=True)
    (tmp_path / "linked").unlink(missing_ok=True)

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
    assert calls.count("iam trusted-profile-links") == 18
    assert calls.count("iam trusted-profile-link-create") == 9
    assert "ce job create --name watch-auth-supercloneluxury-" in calls
    assert "--arg=--source --arg=supercloneluxury" in calls
    cron_action = "update" if existing_cron else "create"
    assert f"ce sub cron {cron_action} --name watch-auth-supercloneluxury-daily" in calls
    assert "ce job create --name watch-auth-cbp-" in calls
    assert "--arg=--source --arg=cbp" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-cbp-daily" in calls
    assert "ce job create --name watch-auth-goldsmiths-" in calls
    assert "--arg=--source --arg=goldsmiths" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-goldsmiths-daily" in calls
    assert "ce job create --name watch-auth-bobswatches-" in calls
    assert "--arg=bobswatches" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-bobswatches-daily" in calls
    assert "7 4 * * *" in calls
    assert "ce job create --name watch-auth-speedmaster-" in calls
    assert "--arg=--source --arg=wikimedia-speedmaster" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-speedmaster-daily" in calls
    assert "ce job create --name watch-auth-auckland-" in calls
    assert "--arg=--source --arg=wikimedia-auckland" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-auckland-daily" in calls
    assert "ce job create --name watch-auth-daytona-" in calls
    assert "--arg=--source --arg=wikimedia-daytona" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-daytona-daily" in calls
    assert "37 4 * * *" in calls
    assert "ce job create --name watch-auth-research-census-" in calls
    assert "--arg=--source --arg=research-census" in calls
    assert f"ce sub cron {cron_action} --name watch-auth-research-census-daily" in calls
    assert "27 4 * * *" in calls
    assert calls.index(f"ce sub cron {cron_action} --name watch-auth-goldsmiths-daily") < calls.index(
        f"ce sub cron {cron_action} --name watch-auth-acquisition-daily"
    )
    assert calls.index("--link-component-name watch-auth-cbp-") < calls.index(
        f"ce sub cron {cron_action} --name watch-auth-acquisition-daily"
    )
    assert "ce job update" not in calls
    if existing_cron:
        log.write_text("")
        repeated = subprocess.run(
            ["bash", str(root / "scripts/deploy_ibm_acquisition.sh")],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
        assert repeated.returncode == 0, repeated.stderr
        assert "ce job create" not in log.read_text()
        for drift in ("1", "2", "3"):
            log.write_text("")
            drifted = subprocess.run(
                ["bash", str(root / "scripts/deploy_ibm_acquisition.sh")],
                capture_output=True,
                text=True,
                env={**env, "DRIFTED_JOB": drift},
                timeout=10,
            )
            assert drifted.returncode == 2
            assert "staged job readback differs" in drifted.stderr
            assert "ce sub cron update" not in log.read_text()


def test_deploy_fails_closed_when_job_readback_is_unavailable(tmp_path):
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
    assert "staged job readback differs" in result.stderr
    assert "trusted-profile-link-create" not in log.read_text()
    assert "ce sub cron" not in log.read_text()


def test_goldsmiths_has_its_own_deadline_and_is_absent_from_daily_job():
    source = Path("acquisition/run.py").read_text(encoding="utf-8")

    main = source[source.index("def main"):source.index('if __name__ == "__main__":')]
    dedicated = main.index('if source == "goldsmiths":')
    daily = main.index('image_requests_per_minute =', dedicated)
    precision = main.index('"precisionwatches",', daily)
    watches_of_switzerland = main.index('"watchesofswitzerland",', precision)

    assert dedicated < daily < precision < watches_of_switzerland
    assert '"goldsmiths",' not in main[daily:]


def test_github_fallback_runs_preflight_without_building_images():
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    script = Path("scripts/preflight-ci.sh").read_text(encoding="utf-8")

    assert "runs-on: ubuntu-latest" in workflow
    assert "timeout-minutes: 10" in workflow
    assert "bash scripts/preflight-ci.sh" in workflow
    assert "docker build" not in workflow + script
    assert "next build" not in workflow + script
    assert "ruff check ." in script
    assert "mypy . --config-file pyproject.toml" in script
    assert "pytest tests/" in script
    assert "--ignore=tests/test_training_image_runtime.py" in script


def _planned_row():
    from dataclasses import replace

    from acquisition.types import Angle
    from dataset.angle_review import record_usability_review
    from tests.test_dataset_readiness import _record

    original = _record(1)
    return replace(
        original,
        candidate_id="candidate",
        source_domain="partner",
        seller_group_id="seller",
        usability_review=record_usability_review(
            "candidate", original.image_sha256, Angle.DIAL, True, True,
            "angle-reviewer", NOW,
        ),
    )


@pytest.mark.parametrize(
    "admit,known,expected", [(False, False, 0), (True, False, 1), (True, True, 0)]
)
def test_planned_wave_selects_viable_source_and_persists_reviewed_yield(
    cos, monkeypatch, admit, known, expected
):
    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once
    target = AcquisitionTarget(
        "rolex-submariner-126610ln", "genuine", "dial", 500, 100, "partner", "seller", 0, True, 20, 11, None
    )
    row = _planned_row()
    calls = []
    def collect(*args, **kwargs):
        calls.append(args[3].source)
        kwargs["on_collected"](("candidate",), "2026-11-10", "public-batch", 1)
        return {
            "cycle": "2026-11-10", "source": "partner", "state": "complete",
            "commit_sha256": "a" * 64, "accepted_candidate_ids": ["candidate"],
        }

    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    policy, access, authorization = _planned_source()
    status = run_planned_public_collection_once(
        cos,
        "private",
        "datasets",
        (target,),
        policies={"partner": policy},
        accesses={"partner": access},
        authorizations={"partner": authorization},
        discoveries={"partner": lambda previous: None},
        now=NOW,
        image_client=object(),
        known_entity_ids=frozenset({"watch-1"} if known else ()),
        review=lambda target, ids, batch_id: ({"candidate": "watch-1"}, (row,) if admit else ()),
    )
    assert calls == ["partner"]
    assert status["eligible_yield"] == [1, expected]
    assert status["output"] == "genuine"
    assert status["angle"] == "dial"
    assert status["exploratory"] is True
    assert any("eligible-yields/batches/" in key for key in cos.bodies)


def test_first_planned_wave_can_admit_one_reviewed_watch(cos, monkeypatch):
    from acquisition.planner import plan_acquisition
    from acquisition.run import run_planned_public_collection_once
    from acquisition.types import Angle
    from dataset.readiness import ReadinessReport

    report = ReadinessReport("rolex-submariner-126610ln", False, (),
                             (("genuine/dial", 0, 0),), (Angle.DIAL,))
    targets = plan_acquisition(
        report, (), source_entity_counts={"partner": 0, "z-source": 0},
        seller_entity_counts={"seller": 0, "z-seller": 0},
        eligible_yield_by_source={}, candidate_consumption_by_source={},
    )
    def collect(*args, **kwargs):
        kwargs["on_collected"](("candidate",), "2026-11-10", "first-batch", 1)
        return {"state": "complete"}
    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    policy, access, authorization = _planned_source()
    status = run_planned_public_collection_once(
        cos, "private", "datasets", targets,
        policies={"partner": policy}, accesses={"partner": access},
        authorizations={"partner": authorization},
        discoveries={"partner": lambda previous: None}, now=NOW,
        image_client=object(), known_entity_ids=frozenset(),
        review=lambda target, ids, batch_id: ({"candidate": "watch-1"}, (_planned_row(),)),
    )
    assert status["eligible_yield"] == [1, 1]
    assert status["eligible_source_capacity"] == 1


@pytest.mark.parametrize("matching", [False, True])
def test_planned_wave_binds_admission_to_target_reference(cos, monkeypatch, matching):
    from dataclasses import replace

    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once

    target = AcquisitionTarget(
        "omega-seamaster-21030422001001", "genuine", "dial", 500, 100,
        "partner", "seller", 0, True, 20, 1, None,
    )
    policy, access, authorization = _planned_source()

    def collect(*args, **kwargs):
        kwargs["on_collected"](("candidate",), "2026-11-10", "wrong-reference", 1)
        return {"state": "complete"}

    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    row = _planned_row()
    if matching:
        row = replace(row, reference_id=target.reference_id)

    def run():
        return run_planned_public_collection_once(
            cos, "private", "datasets", (target,), policies={"partner": policy},
            accesses={"partner": access}, authorizations={"partner": authorization},
            discoveries={"partner": lambda previous: None}, now=NOW,
            image_client=object(), known_entity_ids=frozenset(),
            review=lambda target, ids, batch_id: ({"candidate": "watch-1"}, (row,)),
        )
    if matching:
        assert run()["eligible_yield"] == [1, 1]
    else:
        with pytest.raises(ValueError, match="reviewed planned cell"):
            run()


def test_planned_wave_skips_unavailable_sources_without_collection(cos, monkeypatch):
    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once
    from acquisition.types import SourceAccessStatus

    target = AcquisitionTarget(
        "rolex-submariner-126610ln", "genuine", "dial", 500, 100, "partner", "seller", 0, True, 20, 11, None
    )
    monkeypatch.setattr(
        "acquisition.run.run_public_image_collection_once",
        lambda *args, **kwargs: pytest.fail("unavailable source collected"),
    )
    result = run_planned_public_collection_once(
        cos,
        "private",
        "datasets",
        (target,),
        policies={"partner": SimpleNamespace()},
        accesses={"partner": SimpleNamespace(status=SourceAccessStatus.UNAVAILABLE)},
        authorizations={"partner": SimpleNamespace()},
        discoveries={"partner": lambda p: None},
        now=NOW,
        image_client=object(),
        known_entity_ids=frozenset(),
        review=lambda target, ids, batch_id: pytest.fail("unexpected review"),
    )
    assert result == {"state": "no_viable_target"}


def test_bounded_public_wave_counts_images_and_preserves_unconsumed_cursor(cos, monkeypatch):
    from dataclasses import replace

    page = _result().page
    record = replace(
        page.records[0],
        image_ids=("one", "two"),
        image_urls=("https://upload.wikimedia.org/one.jpg", "https://upload.wikimedia.org/two.jpg"),
    )
    page = replace(page, records=(record,))
    monkeypatch.setattr(
        "acquisition.run.collect_batch", lambda *args, **kwargs: pytest.fail("over-limit download")
    )
    with pytest.raises(ValueError, match="bounded candidate wave"):
        run_public_image_collection_once(
            cos,
            "private",
            "datasets",
            SimpleNamespace(source="wikimedia"),
            SimpleNamespace(),
            SimpleNamespace(),
            now=NOW,
            image_client=object(),
            discover=lambda previous: page,
            candidate_limit=1,
        )
    assert not cos.bodies


def test_failed_evidence_review_does_not_advance_collection_cursor(cos, monkeypatch):
    monkeypatch.setattr(
        "acquisition.run.collect_batch",
        lambda *args, **kwargs: SimpleNamespace(accepted_candidate_ids=("candidate",)),
    )

    def review(*args):
        raise RuntimeError("evidence review unavailable")

    with pytest.raises(RuntimeError, match="evidence review unavailable"):
        run_public_image_collection_once(
            cos, "private", "datasets", SimpleNamespace(source="wikimedia"),
            SimpleNamespace(), SimpleNamespace(), now=NOW, image_client=object(),
            discover=lambda previous: _result().page, candidate_limit=20,
            on_collected=review,
        )
    assert not any("/discovery/" in key for key in cos.bodies)


def _planned_source(source="partner", *, may_download=True):
    from acquisition.types import SourceAccessStatus

    policy = SimpleNamespace(source=source, may_download=may_download, may_train=False,
                             requires_rightsholder_permission=False,
                             allowed_methods=("public-sitemap",), required_rights_purposes=("download",))
    access = SimpleNamespace(source=source, status=SourceAccessStatus.ACTIVE,
                             acquisition_method="public-sitemap", verified_at=NOW,
                             expires_at=NOW.replace(year=2027))
    authorization = CollectionAuthorizationRecord(
        rights_record_id="rights-" + source, source=source, acquisition_method="public-sitemap",
        allowed_purposes=("download",), authorized_at=NOW, authorized_by="owner",
        evidence_sha256="a" * 64)
    return policy, access, authorization


@pytest.mark.parametrize("rejection", ["download", "method", "access_source", "rights_source", "purpose"])
def test_planned_wave_denied_first_source_falls_through_without_discovery(cos, monkeypatch, rejection):
    from dataclasses import replace

    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once

    first = AcquisitionTarget("rolex-submariner-126610ln", "genuine", "dial", 500, 100, "denied", "seller", 0, True, 20, 11, None)
    second = replace(first, source="partner")
    denied = _planned_source("denied", may_download=rejection != "download")
    if rejection == "method":
        denied[1].acquisition_method = "unapproved-method"
    elif rejection == "access_source":
        denied[1].source = "different-source"
    elif rejection == "rights_source":
        denied = (denied[0], denied[1], replace(denied[2], source="different-source"))
    elif rejection == "purpose":
        denied[0].required_rights_purposes = ("train",)
    allowed = _planned_source()
    calls = []
    def collect(*args, **kwargs):
        calls.append(args[3].source)
        kwargs["discover"](None)
        kwargs["on_collected"]((), "2026-11-10", "batch-allowed", 0)
        return {"state": "complete"}
    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    run_planned_public_collection_once(
        cos, "private", "datasets", (first, second),
        policies={"denied": denied[0], "partner": allowed[0]},
        accesses={"denied": denied[1], "partner": allowed[1]},
        authorizations={"denied": denied[2], "partner": allowed[2]},
        discoveries={"denied": lambda p: pytest.fail("unauthorized discovery"),
                     "partner": lambda p: SimpleNamespace(records=())},
        now=NOW, image_client=object(), known_entity_ids=frozenset(),
        review=lambda target, ids, batch_id: ({}, ()))
    assert calls == ["partner"]


def test_review_result_reused_after_cursor_write_failure(cos, monkeypatch):
    from dataclasses import replace

    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once

    policy, access, authorization = _planned_source()
    target = AcquisitionTarget("rolex-submariner-126610ln", "genuine", "dial", 500, 100, "partner", "seller", 0, True, 20, 11, None)
    row = _planned_row()
    from acquisition.discovery import append_discovery_page

    calls = []
    def append(*args, **kwargs):
        calls.append("cursor")
        if len(calls) == 1:
            raise RuntimeError("cursor write unavailable")
        return append_discovery_page(*args, **kwargs)
    monkeypatch.setattr("acquisition.run.append_discovery_page", append)
    monkeypatch.setattr("acquisition.run.collect_batch",
                        lambda *args, **kwargs: SimpleNamespace(accepted_candidate_ids=("candidate",)))
    record = replace(_result().page.records[0], source="partner", seller_group_id="seller",
                     image_ids=tuple(f"image-{i}" for i in range(20)),
                     image_urls=tuple(f"https://upload.wikimedia.org/{i}.jpg" for i in range(20)))
    page = replace(_result().page, source="partner", records=(record,))
    reviews = []
    def review(target, ids, batch_id):
        reviews.append(ids)
        return {"candidate": "watch-1"}, (row,)
    kwargs = dict(policies={"partner": policy}, accesses={"partner": access},
                  authorizations={"partner": authorization}, discoveries={"partner": lambda p: page},
                  now=NOW, image_client=object(), review=review)
    with pytest.raises(RuntimeError, match="cursor write unavailable"):
        run_planned_public_collection_once(cos, "private", "datasets", (target,),
                                           known_entity_ids=frozenset(), **kwargs)
    status = run_planned_public_collection_once(cos, "private", "datasets", (target,),
                                               known_entity_ids=frozenset({"watch-1"}), **kwargs)
    assert len(reviews) == 1
    assert status["eligible_yield"] == [1, 1]
    assert status["baseline_entity_ids"] == []
    assert status["candidate_consumption"] == 20


def test_admission_is_idempotent_when_review_result_write_fails(cos, monkeypatch):
    from acquisition.planner import AcquisitionTarget
    from acquisition.run import run_planned_public_collection_once

    policy, access, authorization = _planned_source()
    target = AcquisitionTarget("rolex-submariner-126610ln", "genuine", "dial", 500, 100, "partner", "seller", 0, True, 20, 11, None)
    row = _planned_row()
    admissions = []
    saved_decisions = {}

    def review(target, ids, batch_id=None):
        # The admission owner persists its decision under the supplied operation key.
        if batch_id is None or batch_id not in saved_decisions:
            admissions.append("watch-1")
            result = ({"candidate": "watch-1"}, (row,))
            if batch_id is not None:
                saved_decisions[batch_id] = result
            return result
        return saved_decisions[batch_id]

    def collect(*args, **kwargs):
        kwargs["on_collected"](("candidate",), "2026-11-10", "stable-batch", 20)
        return {"state": "complete"}

    original_put = cos.put_object_if_absent
    failures = []

    def put(**kwargs):
        if kwargs["Key"].endswith("/review.json") and not failures:
            failures.append(kwargs["Key"])
            raise RuntimeError("review result COS unavailable")
        return original_put(**kwargs)

    monkeypatch.setattr(cos, "put_object_if_absent", put)
    monkeypatch.setattr("acquisition.run.run_public_image_collection_once", collect)
    kwargs = dict(policies={"partner": policy}, accesses={"partner": access},
                  authorizations={"partner": authorization}, discoveries={"partner": lambda p: None},
                  now=NOW, image_client=object(), review=review)
    with pytest.raises(RuntimeError, match="review result COS unavailable"):
        run_planned_public_collection_once(cos, "private", "datasets", (target,),
                                           known_entity_ids=frozenset(), **kwargs)
    status = run_planned_public_collection_once(cos, "private", "datasets", (target,),
                                               known_entity_ids=frozenset({"watch-1"}), **kwargs)
    assert admissions == ["watch-1"]
    assert list(saved_decisions) == ["stable-batch"]
    assert status["eligible_yield"] == [1, 1]
    assert status["baseline_entity_ids"] == []
