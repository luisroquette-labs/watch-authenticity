"""Source and rights gates must fail closed before acquisition."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from acquisition.source_policy import (
    authorize_candidate,
    authorize_discovery,
    get_source_policy,
    load_source_access_registry,
    load_source_policies,
)
from acquisition.types import (
    CollectionAuthorizationRecord,
    RightsStatus,
    SourceAccessRecord,
    SourceAccessStatus,
)


def policy_content_hash(entry):
    payload = {key: value for key, value in entry.items() if key != "policy_evidence_sha256"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_registry_loads_every_declared_source_policy(policy_set):
    assert set(policy_set) == {
        "partner-upload",
        "owner-authorized-upload",
        "mercadolivre",
        "reddit",
        "olx",
        "chrono24",
        "watchuseek",
        "reptime",
        "chinese_replica",
        "wikimedia",
        "vsfactoryusa",
        "richfactoryus",
        "superclonerolex",
        "watchesofswitzerland",
        "goldsmiths",
        "precisionwatches",
    }


def test_unknown_source_cannot_download_or_train(policy_set, rights):
    policy = get_source_policy(policy_set, "unknown")
    decision = authorize_candidate(policy, rights, now=rights.effective_at)

    assert not decision.may_download
    assert not decision.may_train
    assert decision.reasons == ("source policy denies download",)
    with pytest.raises(FrozenInstanceError):
        cast(Any, policy).may_download = True


@pytest.mark.parametrize(
    "source",
    ("vsfactoryusa", "richfactoryus", "superclonerolex"),
)
def test_owner_authorized_public_replica_source_downloads_privately_but_cannot_train(
    policy_set, source, rights
):
    policy = get_source_policy(policy_set, source)
    authorization = CollectionAuthorizationRecord(
        rights_record_id=f"owner-public-download-20261112-{source}",
        source=source,
        acquisition_method=policy.allowed_methods[0],
        allowed_purposes=("download",),
        authorized_at=rights.effective_at,
        authorized_by="project-owner",
        evidence_sha256="a" * 64,
    )

    decision = authorize_candidate(policy, authorization, now=rights.effective_at)

    assert decision.may_download
    assert not decision.may_train
    assert decision.reasons == ("collection authorization does not permit training",)


def test_expired_grant_fails_closed(policy_set, expired_rights):
    policy = get_source_policy(policy_set, "partner-upload")
    decision = authorize_candidate(policy, expired_rights, now=expired_rights.expires_at)

    assert decision.reasons == ("rights expired",)
    assert not decision.may_download
    assert not decision.may_train


@pytest.mark.parametrize(
    ("case", "allowed"),
    [
        ("reddit_missing_rightsholder", False),
        ("mercadolivre_scrape", False),
        ("olx_original_claim", False),
        ("partner_export_with_grant", True),
        ("missing_policy_hash", False),
        ("purpose_mismatch", False),
        ("revoked", False),
        ("expires_exactly_now", False),
    ],
)
def test_source_policy_matrix(policy_case, case, allowed):
    decision = policy_case(case)

    assert decision.may_download is allowed
    assert decision.may_train is allowed


def test_rights_are_not_valid_before_their_effective_time(policy_set, rights):
    policy = get_source_policy(policy_set, rights.source)
    decision = authorize_candidate(policy, rights, now=rights.effective_at - timedelta(seconds=1))

    assert decision.reasons == ("rights not effective",)


def test_loader_rejects_duplicate_sources(tmp_path: Path):
    path = tmp_path / "policies.json"
    entry = json.loads(Path("config/source_policies.json").read_text())[0]
    path.write_text(json.dumps([entry, entry]))

    with pytest.raises(ValueError, match="duplicate source"):
        load_source_policies(path)


def test_loader_rejects_duplicate_policy_ids(tmp_path: Path, policy_set):
    first = get_source_policy(policy_set, "partner-upload")
    second = get_source_policy(policy_set, "owner-authorized-upload")
    entries = [
        {
            **first.__dict__,
            "allowed_methods": list(first.allowed_methods),
            "required_rights_purposes": list(first.required_rights_purposes),
        },
        {
            **second.__dict__,
            "policy_id": first.policy_id,
            "allowed_methods": list(second.allowed_methods),
            "required_rights_purposes": list(second.required_rights_purposes),
        },
    ]
    for entry in entries:
        entry["policy_evidence_sha256"] = policy_content_hash(entry)
    path = tmp_path / "policies.json"
    path.write_text(json.dumps(entries))

    with pytest.raises(ValueError, match="duplicate policy_id"):
        load_source_policies(path)


def test_loader_rejects_missing_policy_evidence_hash(tmp_path: Path):
    path = tmp_path / "policies.json"
    path.write_text(
        """[{"policy_id":"missing-hash","source":"partner-upload",
        "allowed_methods":["partner-export"],"may_download":true,"may_train":true,
        "required_rights_purposes":["download","train"],
        "requires_rightsholder_permission":true}]"""
    )

    with pytest.raises(ValueError, match="policy_evidence_sha256"):
        load_source_policies(path)


def test_loader_rejects_policy_content_that_no_longer_matches_hash(tmp_path: Path):
    path = tmp_path / "policies.json"
    entry = json.loads(Path("config/source_policies.json").read_text())[0]
    entry["may_train"] = False
    path.write_text(json.dumps([entry]))

    with pytest.raises(ValueError, match="does not match policy content"):
        load_source_policies(path)


def test_status_and_source_mismatches_fail_closed(policy_set, rights):
    policy = get_source_policy(policy_set, rights.source)

    revoked = authorize_candidate(
        policy,
        replace(rights, status=RightsStatus.REVOKED),
        now=rights.effective_at,
    )
    wrong_source = authorize_candidate(
        policy,
        replace(rights, source="other-source"),
        now=rights.effective_at,
    )

    assert revoked.reasons == ("rights revoked",)
    assert wrong_source.reasons == ("rights source mismatch",)


def test_versioned_access_registry_keeps_unproven_sources_inactive():
    registry = load_source_access_registry(Path("config/source_access.json"))

    assert set(registry) == {
        "partner-upload",
        "owner-authorized-upload",
        "mercadolivre",
        "reddit",
        "olx",
        "chrono24",
        "watchuseek",
        "reptime",
        "chinese_replica",
        "wikimedia",
        "vsfactoryusa",
        "richfactoryus",
        "superclonerolex",
        "watchesofswitzerland",
        "goldsmiths",
        "precisionwatches",
    }
    assert registry["wikimedia"].status is SourceAccessStatus.ACTIVE
    assert registry["vsfactoryusa"].status is SourceAccessStatus.ACTIVE
    assert registry["richfactoryus"].status is SourceAccessStatus.ACTIVE
    assert registry["superclonerolex"].status is SourceAccessStatus.ACTIVE
    assert registry["watchesofswitzerland"].status is SourceAccessStatus.ACTIVE
    assert registry["goldsmiths"].status is SourceAccessStatus.ACTIVE
    assert registry["precisionwatches"].status is SourceAccessStatus.ACTIVE
    assert all(
        record.status is not SourceAccessStatus.ACTIVE
        for source, record in registry.items()
        if source
        not in {
            "wikimedia",
            "vsfactoryusa",
            "richfactoryus",
            "superclonerolex",
            "watchesofswitzerland",
            "goldsmiths",
            "precisionwatches",
        }
    )


def test_discovery_requires_current_bound_access_evidence(policy_set, rights):
    policy = get_source_policy(policy_set, "mercadolivre")
    active = SourceAccessRecord(
        access_record_id="access-ml-1",
        source="mercadolivre",
        acquisition_method="official-api",
        status=SourceAccessStatus.ACTIVE,
        evidence_url="https://developers.mercadolivre.com.br/itens-e-buscas",
        evidence_sha256="a" * 64,
        verified_at=rights.effective_at,
        expires_at=rights.expires_at,
        allowed_image_hosts=("http2.mlstatic.com",),
        requests_per_minute=60,
        reason="official API access verified for discovery",
    )

    assert not authorize_discovery(policy, None, now=rights.effective_at).allowed
    assert authorize_discovery(policy, active, now=rights.effective_at).allowed
    expired = authorize_discovery(policy, active, now=active.expires_at)
    assert expired.reasons == ("source access expired",)

    no_rights = authorize_candidate(policy, None, now=rights.effective_at)
    assert not no_rights.may_download
    assert not no_rights.may_train


def test_active_source_access_requires_real_evidence_and_network_limits(rights):
    with pytest.raises(ValueError, match="active source access"):
        SourceAccessRecord(
            access_record_id="access-invalid",
            source="mercadolivre",
            acquisition_method="official-api",
            status=SourceAccessStatus.ACTIVE,
            evidence_url=None,
            evidence_sha256="a" * 64,
            verified_at=rights.effective_at,
            expires_at=rights.expires_at,
            allowed_image_hosts=(),
            requests_per_minute=None,
            reason="missing proof",
        )


@pytest.mark.parametrize("host", ["*.mlstatic.com", "localhost", "mlstatic.com:443"])
def test_source_access_rejects_non_exact_hostnames(rights, host):
    with pytest.raises(ValueError, match="exact lowercase hostnames"):
        SourceAccessRecord(
            access_record_id="access-invalid-host",
            source="mercadolivre",
            acquisition_method="official-api",
            status=SourceAccessStatus.ACTIVE,
            evidence_url="https://developers.mercadolivre.com.br/itens-e-buscas",
            evidence_sha256="a" * 64,
            verified_at=rights.effective_at,
            expires_at=rights.expires_at,
            allowed_image_hosts=(host,),
            requests_per_minute=60,
            reason="invalid host test",
        )
