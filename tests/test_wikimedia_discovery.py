from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.discovery import append_discovery_page
from acquisition.source_policy import (
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from acquisition.types import EvidenceTier, SourceAccessRecord, SourceAccessStatus
from core.rules import propose_source_label
from scrapers.wikimedia import WikimediaAPIError, discover_wikimedia

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, payload: object, status_code: int = 200):
        self.payload = payload
        self.status_code = status_code

    def json(self):
        return self.payload


def _access() -> SourceAccessRecord:
    return SourceAccessRecord(
        access_record_id="access-wikimedia-test",
        source="wikimedia",
        acquisition_method="official-api",
        status=SourceAccessStatus.ACTIVE,
        evidence_url="https://commons.wikimedia.org/w/api.php",
        evidence_sha256="a" * 64,
        verified_at=NOW - timedelta(minutes=1),
        expires_at=NOW + timedelta(days=1),
        allowed_image_hosts=("upload.wikimedia.org",),
        requests_per_minute=20,
        reason="test fixture",
    )


def _policy():
    return get_source_policy(
        load_source_policies(Path("config/source_policies.json")), "wikimedia"
    )


def _page(
    *,
    license_name: str = "CC BY-SA 4.0",
    license_url: str | None = "https://creativecommons.org/licenses/by-sa/4.0/",
    image_host: str = "upload.wikimedia.org",
):
    extmetadata = {
        "LicenseShortName": {"value": license_name},
        "Artist": {"value": "<b>Alice Example</b>"},
    }
    if license_url is not None:
        extmetadata["LicenseUrl"] = {"value": license_url}
    return {
        "pageid": 123,
        "title": "File:Rolex Submariner 126610LN.jpg",
        "canonicalurl": "https://commons.wikimedia.org/wiki/File:Rolex_Submariner_126610LN.jpg",
        "imageinfo": [{
            "timestamp": "2020-01-02T03:04:05Z",
            "sha1": "abc123",
            "url": f"https://{image_host}/full.jpg",
            "thumburl": f"https://{image_host}/thumb/1280px-watch.jpg",
            "extmetadata": extmetadata,
        }],
    }


def _discover(http, cos, **overrides):
    values = {
        "http_client": http,
        "policy": _policy(),
        "access": _access(),
        "now": NOW,
    }
    values.update(overrides)
    return discover_wikimedia(**values)


def test_commits_allowed_license_as_unverified_candidate(cos):
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_page()]}})

    result = _discover(http, cos)
    assert cos.write_count == 0
    commit = append_discovery_page(
        cos, "private", "datasets/discovery", result.page
    )

    assert commit.next_cursor is None
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.classification == "original_candidate_unverified"
    assert candidate.evidence_tier is EvidenceTier.INCONCLUSIVE
    assert candidate.observed_at.isoformat() == "2020-01-02T03:04:05+00:00"
    assert candidate.metadata.listing_text == "original_candidate_unverified"
    assert propose_source_label(
        candidate.metadata.source, listing_text=candidate.metadata.listing_text
    ) is None
    assert candidate.author == "Alice Example"
    assert candidate.license_name == "CC BY-SA 4.0"
    assert candidate.rights.allowed_purposes == ("download",)
    page = json.loads(cos.bodies[f"datasets/discovery/wikimedia/pages/{commit.page_sha256}.json"])
    assert page["records"][0]["source_url"].startswith("https://commons.wikimedia.org/")
    assert page["records"][0]["image_urls"] == [
        "https://upload.wikimedia.org/thumb/1280px-watch.jpg"
    ]
    assert page["records"][0]["metadata"] == {
        "author": "Alice Example",
        "classification": "original_candidate_unverified",
        "evidence_tier": "inconclusive",
        "image_timestamp": "2020-01-02T03:04:05Z",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    }
    params = http.get.call_args.kwargs["params"]
    assert params["gcmtitle"] == "Category:Rolex Submariner"
    assert params["gcmlimit"] == 6
    assert params["iiurlwidth"] == 1280


def test_real_commons_license_shapes_are_normalized():
    http = MagicMock()
    http.get.return_value = _Response(
        {
            "query": {
                "pages": [
                    _page(
                        license_name="CC BY-SA 3.0",
                        license_url="http://creativecommons.org/licenses/by-sa/3.0/",
                    ),
                    {
                        **_page(license_name="Public domain", license_url=None),
                        "pageid": 124,
                        "title": "File:Public domain Submariner 126610LN.jpg",
                        "canonicalurl": "https://commons.wikimedia.org/wiki/File:Public_domain_Submariner_126610LN.jpg",
                    },
                ]
            }
        }
    )

    result = _discover(http, None)

    assert [candidate.license_url for candidate in result.candidates] == [
        "https://creativecommons.org/licenses/by-sa/3.0/",
        "https://commons.wikimedia.org/wiki/File:Public_domain_Submariner_126610LN.jpg",
    ]


def test_current_registry_accepts_commons_thumbnail_host():
    http = MagicMock()
    http.get.return_value = _Response(
        {"query": {"pages": [_page(image_host="thumb.wikimedia.org")]}}
    )
    access = load_source_access_registry(Path("config/source_access.json"))["wikimedia"]

    result = _discover(
        http,
        None,
        access=access,
        now=datetime(2026, 11, 10, 21, 24, tzinfo=timezone.utc),
    )

    assert result.page.records[0].image_urls == (
        "https://thumb.wikimedia.org/thumb/1280px-watch.jpg",
    )


@pytest.mark.parametrize("license_name", ["All Rights Reserved", "CC BY-NC 4.0"])
def test_denied_license_is_skipped_without_download_candidate(cos, license_name):
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_page(license_name=license_name)]}})

    result = _discover(http, cos)

    assert result.candidates == ()
    assert result.page.records == ()
    assert cos.write_count == 0


def test_generic_submariner_is_not_mislabeled_as_126610ln(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Rolex Submariner 16613.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Rolex_Submariner_16613.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos)

    assert result.candidates == ()
    assert result.page.records == ()
    assert cos.write_count == 0


def test_rate_limit_preserves_checkpoint(cos):
    http = MagicMock()
    http.get.return_value = _Response({}, status_code=429)
    with pytest.raises(WikimediaAPIError, match="resume later"):
        _discover(http, cos)
    assert cos.write_count == 0


def test_unapproved_image_host_never_commits(cos):
    http = MagicMock()
    http.get.return_value = _Response(
        {"query": {"pages": [_page(image_host="evil.example")]}}
    )
    with pytest.raises(WikimediaAPIError, match="host is not approved"):
        _discover(http, cos)
    assert cos.write_count == 0


def test_inactive_access_refuses_before_network(cos):
    http = MagicMock()
    access = _access()
    access = SourceAccessRecord(**{**access.__dict__, "status": SourceAccessStatus.UNAVAILABLE})
    with pytest.raises(PermissionError):
        _discover(http, cos, access=access)
    http.get.assert_not_called()
    assert cos.write_count == 0
