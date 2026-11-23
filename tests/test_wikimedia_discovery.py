from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from acquisition.discovery import DiscoveryCommit, append_discovery_page
from acquisition.source_policy import (
    authorize_candidate,
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
        "title": "File:Seiko Prospex SPB143J1.png",
        "canonicalurl": "https://commons.wikimedia.org/wiki/File:Seiko_Prospex_SPB143J1.png",
        "imageinfo": [{
            "timestamp": "2020-01-02T03:04:05Z",
            "sha1": "abc123",
            "mime": "image/png",
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

    assert commit.next_cursor == "q:1:0"
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.classification == "brand_family_candidate_unverified"
    assert candidate.evidence_tier is EvidenceTier.INCONCLUSIVE
    assert candidate.observed_at.isoformat() == "2020-01-02T03:04:05+00:00"
    assert candidate.metadata.listing_text == "brand_family_candidate_unverified"
    assert candidate.metadata.declared_reference == "unverified"

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
        "brand_claim_from_text": "seiko",
        "classification": "brand_family_candidate_unverified",
        "evidence_tier": "inconclusive",
        "family_claim_from_text": "prospex",
        "image_timestamp": "2020-01-02T03:04:05Z",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    }
    params = http.get.call_args.kwargs["params"]
    assert params["gsrsearch"] == "Seiko Prospex watch"
    assert params["gsrnamespace"] == 6
    assert params["gsrlimit"] == 6
    assert "mime" in params["iiprop"]
    assert params["iiurlwidth"] == 1280


def test_auckland_museum_cc_by_photos_remain_unverified_research_candidates(cos):
    http = MagicMock()
    pages = []
    for number in (2, 22):
        page = _page(license_name="CC BY 4.0", license_url="https://creativecommons.org/licenses/by/4.0/")
        page["pageid"] = 65962680 + number
        page["title"] = f"File:Watch, wrist (AM 2014.7.70-{number}).jpg"
        page["imageinfo"][0]["sha1"] = f"sha-{number}"
        page["imageinfo"][0]["extmetadata"]["ImageDescription"] = {
            "value": "Rolex Oyster Perpetual, dating from 1953, Auckland Museum"
        }
        pages.append(page)
    http.get.return_value = _Response({"query": {"pages": pages}})

    result = _discover(http, cos, query_range=(8, 9))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "1953 Rolex Oyster Perpetual Auckland Museum"
    assert len(result.candidates) == 2
    assert {candidate.rights.allowed_purposes for candidate in result.candidates} == {("download",)}
    assert all(candidate.evidence_tier is EvidenceTier.INCONCLUSIVE for candidate in result.candidates)
    assert all(candidate.metadata.declared_reference == "unverified" for candidate in result.candidates)
    assert result.page.next_cursor is None


def test_daytona_replica_queries_skip_genuine_search_hits_and_keep_weak_labels(cos):
    http = MagicMock()
    genuine = _page()
    genuine["pageid"] = 101
    genuine["title"] = "File:Rolex Le Mans Daytona 126528LN.jpg"
    genuine["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Rolex_Le_Mans_Daytona_126528LN.jpg"
    genuine["imageinfo"][0]["extmetadata"]["ImageDescription"] = {"value": "Rolex Daytona wristwatch"}
    replica = _page(license_name="Public domain", license_url=None)
    replica["pageid"] = 102
    replica["title"] = "File:Rolex Daytona 126500LN Panda Replica.jpg"
    replica["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Rolex_Daytona_126500LN_Panda_Replica.jpg"
    http.get.return_value = _Response({"query": {"pages": [genuine, replica]}})

    first = _discover(http, cos, query_range=(10, 12))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "replica Rolex Daytona"
    assert len(first.candidates) == 1
    assert first.page.records[0].source_record_id == "q10:102"
    assert first.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert first.candidates[0].rights.allowed_purposes == ("download",)
    assert first.page.next_cursor == "q:11:0"

    fake = _page(license_name="Public domain", license_url=None)
    fake["pageid"] = 103
    fake["title"] = "File:Illegally copied wristwatch which shall look like Rolex Daytona.jpg"
    fake["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Illegally_copied_wristwatch_which_shall_look_like_Rolex_Daytona.jpg"
    fake["imageinfo"][0]["extmetadata"]["ImageDescription"] = {"value": "Fake Rolex Daytona watch"}
    http.get.return_value = _Response({"query": {"pages": [fake]}})
    prior = DiscoveryCommit("wikimedia", _access().access_record_id, None, "0" * 64, "q:11:0", (), "0" * 64)

    second = _discover(http, cos, query_range=(10, 12), previous=prior)

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "fake Rolex Daytona watch"
    assert second.page.records[0].source_record_id == "q11:103"
    assert second.page.next_cursor is None
    assert second.candidates[0].metadata.declared_reference == "unverified"


def test_default_discovery_does_not_enter_auckland_lane(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Fake Rolex Datejust.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Fake_Rolex_Datejust.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})
    prior = DiscoveryCommit("wikimedia", _access().access_record_id, None, "0" * 64, "q:7:0", (), "0" * 64)

    result = _discover(http, cos, previous=prior)

    assert result.page.next_cursor is None
    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "fake Rolex Datejust"


def test_classic_speedmaster_photos_remain_unverified_research_candidates(cos):
    http = MagicMock()
    page = _page(license_name="CC BY 2.0", license_url="https://creativecommons.org/licenses/by/2.0/")
    page["pageid"] = 97460531
    page["title"] = "File:Classic Speedy (24067321725).jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Classic_Speedy_(24067321725).jpg"
    page["imageinfo"][0]["extmetadata"]["ImageDescription"] = {
        "value": "Omega Speedmaster Professional Moonwatch (Ref.: 311.30.42.30.01.005)"
    }
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(9, 10))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Omega Speedmaster Professional Moonwatch"
    assert len(result.candidates) == 1
    assert result.candidates[0].rights.allowed_purposes == ("download",)
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert result.page.next_cursor is None


def test_royal_oak_research_keeps_licensed_photo_unverified(cos):
    http = MagicMock()
    page = _page(license_name="CC BY-SA 4.0")
    page["pageid"] = 124097422
    page["title"] = "File:Audemars Piguet Royal Oak ref. 15202.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Audemars_Piguet_Royal_Oak_ref._15202.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(12, 13))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Audemars Piguet Royal Oak"
    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert result.candidates[0].rights.allowed_purposes == ("download",)
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert result.page.next_cursor is None


def test_counterfeit_category_keeps_mixed_brands_as_unverified_research(cos):
    http = MagicMock()
    montblanc = _page()
    montblanc["title"] = "File:Chronograph watch.jpeg"
    montblanc["imageinfo"][0]["extmetadata"]["ImageDescription"] = {
        "value": "Illegal copy of a Montblanc Timewalker chronograph watch"
    }
    patek = _page(license_name="CC BY 3.0", license_url="https://creativecommons.org/licenses/by/3.0/")
    patek["pageid"] = 124
    patek["title"] = "File:Illegal copy of Patek-Phillipe wristwatch.jpg"
    patek["imageinfo"][0]["sha1"] = "patek-sha"
    http.get.return_value = _Response({"query": {"pages": [montblanc, patek]}})

    result = _discover(http, cos, query_range=(13, 14))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == (
        'incategory:"Illegal copies of watches of quality brands"'
    )
    assert len(result.candidates) == 2
    assert result.page.next_cursor is None
    assert all(candidate.evidence_tier is EvidenceTier.INCONCLUSIVE for candidate in result.candidates)
    assert all(candidate.metadata.declared_reference == "unverified" for candidate in result.candidates)
    assert all(candidate.rights.allowed_purposes == ("download",) for candidate in result.candidates)
    assert all(propose_source_label(candidate.metadata.source, listing_text=candidate.metadata.listing_text) is None
               for candidate in result.candidates)
    assert dict(result.page.records[0].metadata)["brand_claim_from_text"] == "unverified"
    assert dict(result.page.records[0].metadata)["research_category"] == (
        "Illegal copies of watches of quality brands"
    )


def test_broad_counterfeit_category_keeps_cbp_group_unverified(cos):
    http = MagicMock()
    page = _page(license_name="Public domain", license_url=None)
    page["pageid"] = 57419144
    page["title"] = "File:San Juan CBP seizes 20 counterfeit luxury watches.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:San_Juan_CBP_seizes_20_counterfeit_luxury_watches.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(15, 16))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == 'incategory:"Counterfeit watches"'
    assert result.page.records[0].source_record_id == "q15:57419144"
    assert dict(result.page.records[0].metadata)["research_category"] == "Counterfeit watches"
    assert result.candidates[0].classification == "commons_counterfeit_category_research_unverified"
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert result.candidates[0].rights.allowed_purposes == ("download",)
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE


def test_nasa_speedmaster_category_keeps_nasa_photo_unverified(cos):
    http = MagicMock()
    page = _page(license_name="Public domain", license_url=None)
    page["pageid"] = 18986626
    page["title"] = "File:Wally Schirra during Apollo 7 training.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Wally_Schirra_during_Apollo_7_training.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(14, 15))

    assert 'incategory:"PD NASA"' in http.get.call_args.kwargs["params"]["gsrsearch"]
    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert result.candidates[0].rights.allowed_purposes == ("download",)
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert dict(result.page.records[0].metadata)["research_category"].endswith("PD NASA")
    assert dict(result.page.records[0].metadata)["brand_claim_from_text"] == "unverified"
    assert dict(result.page.records[0].metadata)["family_claim_from_text"] == "unverified"
    assert propose_source_label(result.candidates[0].metadata.source, listing_text=result.candidates[0].metadata.listing_text) is None

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
                        "title": "File:Public domain Seiko Prospex SPB143J1.jpg",
                        "canonicalurl": "https://commons.wikimedia.org/wiki/File:Public_domain_Seiko_Prospex_SPB143J1.jpg",
                    },
                ]
            }
        }
    )

    result = _discover(http, None)

    assert [candidate.license_url for candidate in result.candidates] == [
        "https://creativecommons.org/licenses/by-sa/3.0/",
        "https://commons.wikimedia.org/wiki/File:Public_domain_Seiko_Prospex_SPB143J1.jpg",
    ]


@pytest.mark.parametrize("license_url", (
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/zero/1.0/deed.en",
))
def test_cc0_commons_license_allows_training_rights_without_authenticity_label(license_url):
    http = MagicMock()
    page = _page(license_name="CC0", license_url=license_url)
    http.get.return_value = _Response({"query": {"pages": [page]}})

    candidate = _discover(http, None).candidates[0]
    policy = _policy()
    decision = authorize_candidate(policy, candidate.rights, now=NOW)

    assert decision.may_download and decision.may_train
    assert candidate.rights.allowed_purposes == ("download", "train")
    assert candidate.evidence_tier is EvidenceTier.INCONCLUSIVE
    assert candidate.metadata.declared_reference == "unverified"


def _cbp_public_domain_page():
    page = _page(license_name="Public domain", license_url=None)
    page["pageid"] = 57419144
    page["title"] = "File:San Juan CBP seizes 20 counterfeit luxury watches (32231527976).jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:San_Juan_CBP_seizes_20_counterfeit_luxury_watches_(32231527976).jpg"
    page["imageinfo"][0]["sha1"] = "c3088f52d34c3bc28cd3bae0dbdf2329af8b2929"
    page["imageinfo"][0]["timestamp"] = "2017-03-26T14:48:07Z"
    page["templates"] = [{"ns": 10, "title": "Template:PD-USGov-DHS-CBP"}]
    metadata = page["imageinfo"][0]["extmetadata"]
    metadata.update({
        "Artist": {"value": '<a href="https://www.flickr.com/people/54593278@N03">U.S. Customs and Border Protection</a>'},
        "Credit": {"value": '<a href="https://www.flickr.com/photos/cbpphotos/32231527976/">CBP photo</a>'},
        "UsageTerms": {"value": "Public domain"},
        "Copyrighted": {"value": "False"},
        "AttributionRequired": {"value": "false"},
    })
    return page


def test_exact_cbp_public_domain_basis_grants_train_rights_without_authenticity():
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_cbp_public_domain_page()]}})
    candidate = _discover(http, None, query_range=(15, 16)).candidates[0]
    decision = authorize_candidate(_policy(), candidate.rights, now=NOW)

    assert http.get.call_args.kwargs["params"]["tltemplates"] == "Template:PD-USGov-DHS-CBP"
    assert decision.may_download and decision.may_train
    assert candidate.rights.allowed_purposes == ("download", "train")
    assert candidate.evidence_tier is EvidenceTier.INCONCLUSIVE
    assert candidate.metadata.declared_reference == "unverified"
    evidence = json.loads(candidate.rights_evidence)
    assert evidence["public_domain_template"] == "Template:PD-USGov-DHS-CBP"
    assert evidence["flickr_photo_id"] == "32231527976"
    assert evidence["image_sha1"] == "c3088f52d34c3bc28cd3bae0dbdf2329af8b2929"


@pytest.mark.parametrize("change", (
    "template", "artist", "artist_link", "credit", "usage", "copyrighted",
    "attribution", "license_url", "license", "page_id", "source_url",
    "image_sha1", "image_revision", "flickr_photo_id",
))
def test_incomplete_cbp_public_domain_basis_stays_download_only(change):
    page = copy.deepcopy(_cbp_public_domain_page())
    metadata = page["imageinfo"][0]["extmetadata"]
    if change == "template":
        page["templates"] = []
    elif change == "artist":
        metadata["Artist"]["value"] = "Third party"
    elif change == "artist_link":
        metadata["Artist"]["value"] = "U.S. Customs and Border Protection"
    elif change == "credit":
        metadata["Credit"]["value"] = '<a href="https://www.flickr.com/photos/other/32231527976/">CBP photo</a>'
    elif change == "usage":
        metadata["UsageTerms"]["value"] = "All rights reserved"
    elif change == "copyrighted":
        metadata["Copyrighted"]["value"] = "True"
    elif change == "attribution":
        metadata["AttributionRequired"]["value"] = "true"
    elif change == "license_url":
        metadata["LicenseUrl"] = {"value": "https://creativecommons.org/licenses/by/4.0/"}
    elif change == "license":
        metadata["LicenseShortName"]["value"] = "CC BY 4.0"
        metadata["LicenseUrl"] = {"value": "https://creativecommons.org/licenses/by/4.0/"}
    elif change == "page_id":
        page["pageid"] = 123
    elif change == "source_url":
        page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Other.jpg"
    elif change == "image_sha1":
        page["imageinfo"][0]["sha1"] = "other-image"
    elif change == "image_revision":
        page["imageinfo"][0]["timestamp"] = "2017-03-26T14:48:08Z"
    elif change == "flickr_photo_id":
        metadata["Credit"]["value"] = (
            '<a href="https://www.flickr.com/photos/cbpphotos/32231527977/">CBP photo</a>'
        )
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [page]}})
    candidate = _discover(http, None, query_range=(15, 16)).candidates[0]
    assert candidate.rights.allowed_purposes == ("download",)
    assert not authorize_candidate(_policy(), candidate.rights, now=NOW).may_train


def test_copied_cbp_metadata_on_unrelated_watch_cannot_grant_train_rights():
    page = _cbp_public_domain_page()
    page.update(_page(license_name="Public domain", license_url=None))
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [page]}})
    candidate = _discover(http, None).candidates[0]
    assert candidate.rights.allowed_purposes == ("download",)


@pytest.mark.parametrize(
    ("license_name", "license_url"),
    (("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
     ("Public domain", None),
     ("CC0", "https://creativecommons.org/licenses/by-sa/4.0/"),
     ("CC0", "http://creativecommons.org/publicdomain/zero/1.0/"),
     ("CC0", "https://creativecommons.org/publicdomain/zero/1.0/deed.en#alternate"),
     ("CC0", "https://creativecommons.org:80/publicdomain/zero/1.0/#alternate")),
)
def test_other_commons_license_claims_remain_download_only(license_name, license_url):
    http = MagicMock()
    page = _page(license_name=license_name, license_url=license_url)
    http.get.return_value = _Response({"query": {"pages": [page]}})

    candidate = _discover(http, None).candidates[0]
    policy = _policy()
    decision = authorize_candidate(policy, candidate.rights, now=NOW)

    assert decision.may_download and not decision.may_train
    assert candidate.rights.allowed_purposes == ("download",)


def test_cc0_training_grant_identity_includes_literal_license_and_file_revision():
    canonical = _page(
        license_name="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    )
    noncanonical = _page(
        license_name="CC0",
        license_url="http://creativecommons.org/publicdomain/zero/1.0/",
    )
    next_revision = _page(
        license_name="CC0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    )
    next_revision["imageinfo"][0]["sha1"] = "different-revision"

    grants = []
    for page in (canonical, noncanonical, next_revision):
        http = MagicMock()
        http.get.return_value = _Response({"query": {"pages": [page]}})
        grants.append(_discover(http, None).candidates[0].rights)

    assert len({grant.rights_record_id for grant in grants}) == 3
    assert len({grant.evidence_sha256 for grant in grants}) == 3
    assert [grant.allowed_purposes for grant in grants] == [
        ("download", "train"), ("download",), ("download", "train")
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


def test_family_match_does_not_assign_exact_reference(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Seiko Prospex SPB143J1.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Seiko_Prospex_SPB143J1.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos)

    assert len(result.candidates) == 1
    assert result.page.records[0].declared_reference == "unverified"
    assert cos.write_count == 0


def test_search_offset_and_next_family_are_resumable(cos):
    http = MagicMock()
    http.get.return_value = _Response({
        "query": {"pages": [_page()]}, "continue": {"gsroffset": 6, "continue": "gsroffset||"},
    })
    first = _discover(http, cos)
    assert first.page.next_cursor == "q:0:6"
    commit = append_discovery_page(cos, "private", "datasets/discovery", first.page)

    http.get.return_value = _Response({"batchcomplete": True})
    second = _discover(http, cos, previous=commit)
    assert http.get.call_args.kwargs["params"]["gsroffset"] == 6
    assert second.page.next_cursor == "q:1:0"
    assert second.candidates == ()
    commit = append_discovery_page(cos, "private", "datasets/discovery", second.page,
                                   previous=commit)

    cartier = _page()
    cartier["title"] = "File:Cartier Santos wristwatch.jpg"
    cartier["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Cartier_Santos_wristwatch.jpg"
    http.get.return_value = _Response({"query": {"pages": [cartier]}})
    third = _discover(http, cos, previous=commit)
    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Cartier Santos wristwatch"
    assert third.page.records[0].declared_reference == "unverified"
    assert dict(third.page.records[0].metadata)["brand_claim_from_text"] == "cartier"


def test_patek_search_accepts_hyphenated_official_name(cos):
    http = MagicMock()
    http.get.return_value = _Response({"batchcomplete": True})
    commit = None
    for _ in range(3):
        result = _discover(http, cos, previous=commit)
        commit = append_discovery_page(cos, "private", "datasets/discovery", result.page,
                                       previous=commit)
    page = _page()
    page["title"] = "File:Patek-Philippe-Nautilus-5711-1A-010-1.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Patek-Philippe-Nautilus-5711-1A-010-1.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, previous=commit)

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Patek Philippe Nautilus"
    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert dict(result.page.records[0].metadata)["brand_claim_from_text"] == "patek philippe"


def test_speedmaster_search_resumes_after_existing_families_without_authenticity_claim(cos):
    http = MagicMock()
    http.get.return_value = _Response({"batchcomplete": True})
    commit = None
    for _ in range(5):
        result = _discover(http, cos, previous=commit)
        commit = append_discovery_page(cos, "private", "datasets/discovery", result.page,
                                       previous=commit)
    assert commit.next_cursor == "q:5:0"

    page = _page()
    page["title"] = "File:Omega Speedmaster Professional.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Omega_Speedmaster_Professional.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, previous=commit)

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Omega Speedmaster watch"
    assert len(result.candidates) == 1
    assert result.page.next_cursor == "q:6:0"
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert dict(result.page.records[0].metadata)["family_claim_from_text"] == "speedmaster"


def test_speedmaster_lane_starts_directly_and_keeps_original_query_identity(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Omega Speedmaster Professional.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Omega_Speedmaster_Professional.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(5, 7))

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "Omega Speedmaster watch"
    assert result.page.records[0].source_record_id == "q5:123"
    assert result.page.next_cursor == "q:6:0"
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE


def test_speedmaster_lane_rejects_another_family_checkpoint(cos):
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_page()]}})
    first = _discover(http, cos)
    commit = append_discovery_page(cos, "private", "datasets/discovery", first.page)
    with pytest.raises(WikimediaAPIError, match="another query range"):
        _discover(http, cos, query_range=(5, 7), previous=commit)


def test_replica_query_keeps_uploader_claim_separate_from_authenticity(cos):
    http = MagicMock()
    original = _page()
    original["title"] = "File:Omega Speedmaster Professional.jpg"
    original["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Omega_Speedmaster_Professional.jpg"
    fake = _page()
    fake["title"] = "File:Back of a fake replica Omega Speedmaster watch.jpeg"
    fake["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Back_of_a_fake_replica_Omega_Speedmaster_watch.jpeg"
    http.get.side_effect = [
        _Response({"query": {"pages": [original]}}),
        _Response({"query": {"pages": [original, fake]}}),
    ]

    first = _discover(http, cos, query_range=(5, 7))
    commit = append_discovery_page(cos, "private", "datasets/lanes/speedmaster/discovery", first.page)
    second = _discover(http, cos, query_range=(5, 7), previous=commit)

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "replica Omega Speedmaster"
    assert second.page.next_cursor is None
    assert len(second.candidates) == 1
    assert second.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert second.candidates[0].metadata.declared_reference == "unverified"
    assert second.candidates[0].metadata.listing_text == "commons_replica_keyword_research_unverified"
    assert "fake replica Omega Speedmaster" in dict(second.page.records[0].metadata)["research_keyword_context"]
    assert propose_source_label(
        second.candidates[0].metadata.source,
        listing_text=second.candidates[0].metadata.listing_text,
    ) is None


def test_datejust_fake_search_is_research_only(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Fake Rolex Datejust.jpg"
    page["canonicalurl"] = "https://commons.wikimedia.org/wiki/File:Fake_Rolex_Datejust.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})
    prior = DiscoveryCommit("wikimedia", _access().access_record_id, None, "0" * 64, "q:7:0", (), "0" * 64)

    result = _discover(http, cos, previous=prior)

    assert http.get.call_args.kwargs["params"]["gsrsearch"] == "fake Rolex Datejust"
    assert result.page.next_cursor is None
    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.declared_reference == "unverified"
    assert result.candidates[0].evidence_tier is EvidenceTier.INCONCLUSIVE
    assert "Fake Rolex Datejust" in dict(result.page.records[0].metadata)["research_keyword_context"]


def test_negated_replica_word_does_not_become_authenticity_claim(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Not a fake Omega Speedmaster.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})

    result = _discover(http, cos, query_range=(6, 7))

    assert len(result.candidates) == 1
    assert result.candidates[0].metadata.listing_text == "commons_replica_keyword_research_unverified"
    assert "source_authenticity_claim" not in dict(result.page.records[0].metadata)


def test_unsupported_file_mime_is_skipped(cos):
    http = MagicMock()
    page = _page()
    page["imageinfo"][0]["mime"] = "application/pdf"
    http.get.return_value = _Response({"query": {"pages": [page]}})
    result = _discover(http, cos)
    assert result.page.records == ()


def test_invalid_search_cursor_is_rejected_before_network(cos):
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_page()]}})
    first = _discover(http, cos)
    commit = append_discovery_page(cos, "private", "datasets/discovery", first.page)
    object.__setattr__(commit, "next_cursor", "q:999:0")
    http.reset_mock()
    with pytest.raises(WikimediaAPIError, match="cursor is invalid"):
        _discover(http, cos, previous=commit)
    http.get.assert_not_called()


def test_legacy_category_cursor_transitions_without_network(cos):
    http = MagicMock()
    http.get.return_value = _Response({"query": {"pages": [_page()]}})
    first = _discover(http, cos)
    legacy_page = replace(first.page, next_cursor="file|legacy")
    commit = append_discovery_page(cos, "private", "datasets/discovery", legacy_page)
    http.reset_mock()
    transition = _discover(http, cos, previous=commit)
    assert transition.page.cursor == "file|legacy"
    assert transition.page.next_cursor == "q:0:0"
    assert transition.candidates == ()
    http.get.assert_not_called()
    resumed = append_discovery_page(cos, "private", "datasets/discovery", transition.page,
                                    previous=commit)
    assert resumed.next_cursor == "q:0:0"


def test_same_file_in_two_family_queries_has_distinct_discovery_ids(cos):
    http = MagicMock()
    page = _page()
    page["title"] = "File:Seiko Prospex and Cartier Santos.jpg"
    http.get.return_value = _Response({"query": {"pages": [page]}})
    first = _discover(http, cos)
    commit = append_discovery_page(cos, "private", "datasets/discovery", first.page)
    assert first.page.records[0].source_record_id == "q0:123"
    second = _discover(http, cos, previous=commit)
    assert second.page.records[0].source_record_id == "q1:123"
    assert second.page.records[0].image_ids == first.page.records[0].image_ids
    append_discovery_page(cos, "private", "datasets/discovery", second.page,
                          previous=commit)


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
