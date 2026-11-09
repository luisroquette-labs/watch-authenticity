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
    get_source_policy,
    load_source_policies,
)
from acquisition.types import RightsStatus


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
    }


def test_unknown_source_cannot_download_or_train(policy_set, rights):
    policy = get_source_policy(policy_set, "unknown")
    decision = authorize_candidate(policy, rights, now=rights.effective_at)

    assert not decision.may_download
    assert not decision.may_train
    assert decision.reasons == ("source policy denies download",)
    with pytest.raises(FrozenInstanceError):
        cast(Any, policy).may_download = True


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
