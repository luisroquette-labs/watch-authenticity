import hashlib
import json
import subprocess
import sys
from typing import Any

import pytest

from evaluation.pilot_reports import census, score, score_frozen


def _case(specimen: str, label: str | None = "fake", **fields: Any) -> dict[str, Any]:
    return {
        "specimen_id": specimen,
        "reference": "126610LN",
        "generation": "2020",
        "label": label,
        "label_source": "synthetic",
        "evidence": "independent",
        "partition": "test",
        "cohort": "representative",
        "capture_type": "inspection",
        "split_family": specimen,
        "provenance_clusters": [f"source:{specimen}"],
        "prepared_request_sha256": hashlib.sha256(f"{specimen}:request".encode()).hexdigest(),
        "prepared_media_sha256": hashlib.sha256(f"{specimen}:prepared-media".encode()).hexdigest(),
        "photos": [
            {"sha256": hashlib.sha256(f"{specimen}:{angle}".encode()).hexdigest(), "angle": angle}
            for angle in ("dial", "crown")
        ],
        "rights": {"gemini": True, "gpt": True},
        **fields,
    }


def _manifest(*cases: dict[str, Any]) -> dict[str, Any]:
    return {
        "freeze": {
            "models": {"gemini": "model-version-a", "gpt": "model-version-b"},
            "prompt_sha256": "a" * 64,
            "policy_sha256": "b" * 64,
            "technical_sources_sha256": "c" * 64,
            "image_treatment_sha256": "d" * 64,
        },
        "cases": list(cases),
    }


def _confirmations(*cases: dict[str, Any], **fields: Any) -> dict[str, Any]:
    data = json.dumps({
        case["specimen_id"]: {
            "reference": case["reference"],
            "generation": case["generation"],
            "label": case["label"],
            "photo_sha256s": [photo["sha256"] for photo in case["photos"]],
            "reviewer_id": "fixture-reviewer",
            "verification_method": "specimen_specific_document",
            "evidence_sha256": hashlib.sha256(f"{case['specimen_id']}:evidence".encode()).hexdigest(),
            "physical_identity_sha256": hashlib.sha256(f"{case['specimen_id']}:identity".encode()).hexdigest(),
            **fields,
        }
        for case in cases
    }).encode()
    return {"confirmation_bytes": data, "confirmation_sha256": hashlib.sha256(data).hexdigest()}


def test_declared_scores_without_confirmation_and_cannot_promote_itself() -> None:
    case = _case("seller", evidence="declared", physical_identity_confirmed=True)
    reports = {"seller": "probable_fake"}
    result = score([case], reports)
    assert result["declared"]["agrees_with_claim"] == 1
    assert result["independent"]["accuracy_all_cases"] is None
    assert census(_manifest(case), **_confirmations(case))["cases"][0]["status"] == "research_only"
    with pytest.raises(ValueError, match="frozen test manifest"):
        score([{**case, "evidence": "independent"}], reports)


@pytest.mark.parametrize(("field", "value"), [
    ("reference", "another-reference"),
    ("generation", "previous"),
    ("label", "original"),
    ("reviewer_id", ""),
    ("physical_identity_sha256", ""),
    ("evidence_sha256", "z" * 64),
    ("verification_method", "seller_claim"),
    ("verification_method", "model_analysis"),
])
def test_independent_confirmation_binds_case_and_requires_independent_evidence(
    field: str, value: str,
) -> None:
    case = _case("a")
    trusted = _confirmations(case, **{field: value})
    result = census(_manifest(case), **trusted)
    assert result["cases"] == [
        {"specimen_id": "a", "status": "research_only", "reasons": ["missing_or_mismatched_confirmation"]},
    ]


def test_independent_confirmation_cannot_be_reordered_reassigned_or_tampered() -> None:
    case = _case("a")
    trusted = _confirmations(case)
    for changed in (
        {**case, "photos": list(reversed(case["photos"]))},
        {**case, "specimen_id": "b"},
        {**case, "photos": [{"sha256": "z" * 64, "angle": "dial"}]},
    ):
        assert census(_manifest(changed), **trusted)["cases"][0]["status"] != "eligible"
    for tampered in (
        {"confirmation_bytes": trusted["confirmation_bytes"]},
        {**trusted, "confirmation_bytes": trusted["confirmation_bytes"] + b" "},
        {**trusted, "confirmation_sha256": "x" * 64},
    ):
        with pytest.raises(ValueError, match="trusted SHA-256"):
            census(_manifest(case), **tampered)


@pytest.mark.parametrize(("field", "reason"), [
    ("photos", "cross_partition_photo"),
    ("prepared_request_sha256", "cross_partition_prepared_request"),
    ("split_family", "cross_partition_split_family"),
    ("provenance_clusters", "cross_partition_provenance"),
])
def test_cross_partition_reposts_and_related_specimens_cannot_enter(
    field: str, reason: str,
) -> None:
    first = _case("a", partition="development")
    second = _case("b", **{field: first[field]})
    result = census(_manifest(first, second), **_confirmations(first, second))
    reasons = [reason]
    if field in ("photos", "prepared_request_sha256"):
        reasons.append(reason.removeprefix("cross_partition_") + "_shared_by_independent_specimens")
    assert result["cases"] == [
        {"specimen_id": key, "status": "blocked", "reasons": reasons} for key in ("a", "b")
    ]
    with pytest.raises(ValueError, match="unique specimen"):
        census(_manifest(first, {**first, "partition": "test"}))


def test_census_counts_watches_and_keeps_representative_and_stress_pairs_separate() -> None:
    denied = _case("denied", rights={"gemini": False})
    representative = [_case("a"), _case("b", "original")]
    stress = _case("stress", "original", cohort="stress")
    other_generation = _case("old", generation="previous", cohort="stress")
    control = _case("control", evidence="control", reference="unknown", label=None)
    result = census(
        _manifest(*representative, stress, other_generation, denied, control),
        **_confirmations(*representative, stress, other_generation, denied),
    )
    assert result["paired_test_references"] == {
        "representative": [{"reference": "126610LN", "generation": "2020"}],
        "stress": [],
    }
    assert result["counts"]["eligible"]["classes"] == {"fake": 2, "original": 2}
    assert result["counts"]["eligible"]["angles"] == {"dial": 4, "crown": 4}
    assert result["cases"][-2:] == [
        {"specimen_id": "denied", "status": "blocked",
         "reasons": ["rights_denied:gemini", "rights_missing:gpt"]},
        {"specimen_id": "control", "status": "research_only",
         "reasons": ["control_only", "unknown_reference"]},
    ]


def test_census_refuses_malformed_freeze_digest() -> None:
    manifest = _manifest(_case("a"))
    manifest["freeze"]["image_treatment_sha256"] = "z" * 64
    with pytest.raises(ValueError, match="SHA-256 digests"):
        census(manifest)


def test_cli_rejects_two_decisions_for_one_watch(tmp_path) -> None:
    cases = tmp_path / "cases.json"
    reports = tmp_path / "reports.json"
    cases.write_text(json.dumps([_case("a", evidence="declared")]))
    reports.write_text('{"a":"probable_fake","a":"probable_original"}')
    result = subprocess.run(
        [sys.executable, "evaluation/pilot_reports.py", str(cases), str(reports)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "duplicate key in evaluation JSON" in result.stderr


def test_two_confirmed_ids_cannot_inflate_scoring_with_the_same_photo() -> None:
    first = _case("a")
    second = _case("b", photos=[first["photos"][0]])
    control = _case("control", evidence="control", photos=first["photos"])
    trusted = _confirmations(first, second)
    result = census(_manifest(first, second, control), **trusted)
    assert result["cases"] == [
        {"specimen_id": key, "status": "blocked", "reasons": ["photo_shared_by_independent_specimens"]}
        for key in ("a", "b")
    ] + [{"specimen_id": "control", "status": "research_only", "reasons": ["control_only"]}]
    result = census(_manifest(first, control), **_confirmations(first))
    assert result["cases"][0] == {"specimen_id": "a", "status": "eligible", "reasons": []}


def test_transitive_provenance_cannot_move_into_test() -> None:
    first = _case("a", partition="development", provenance_clusters=["seller:x"])
    bridge = _case("b", provenance_clusters=["seller:x", "factory:y"])
    last = _case("c", provenance_clusters=["factory:y"])
    result = census(_manifest(first, bridge, last), **_confirmations(first, bridge, last))
    assert result["cases"][2] == {
        "specimen_id": "c", "status": "blocked", "reasons": ["cross_partition_component"],
    }


@pytest.mark.parametrize(("field", "kind"), [
    ("physical_identity_sha256", "physical_identity"), ("evidence_sha256", "evidence"),
])
def test_two_verified_ids_cannot_reuse_identity_or_examination_artifact(field: str, kind: str) -> None:
    first, second = _case("a"), _case("b")
    result = census(_manifest(first, second), **_confirmations(first, second, **{field: "f" * 64}))
    assert result["cases"] == [
        {"specimen_id": key, "status": "blocked", "reasons": [f"{kind}_shared_by_independent_specimens"]}
        for key in ("a", "b")
    ]


def _run_receipt(case: dict[str, Any], conclusion: str) -> dict[str, Any]:
    return {
        "photo_sha256s": [photo["sha256"] for photo in case["photos"]],
        "prepared_request_sha256": case["prepared_request_sha256"],
        "gemini_media_sha256": case["prepared_media_sha256"],
        "technical_facts_sha256": "c" * 64,
        "gemini_model": "model-version-a",
        "gemini_linkage": "consistent",
        "gemini_response_id": f"provider-response-{case['specimen_id']}",
        "gemini_response_sha256": "9" * 64,
        "conclusion": conclusion,
    }


def _frozen_inputs(*cases: dict[str, Any]) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    manifest = _manifest(*cases)
    data = json.dumps(manifest).encode()
    params = {
        "manifest_sha256": hashlib.sha256(data).hexdigest(), "cohort": "representative",
        **_confirmations(*cases),
    }
    run = {
        "capture_mode": "live_provider", "route": "gemini_only", "freeze": manifest["freeze"],
        "manifest_sha256": params["manifest_sha256"], "confirmations_sha256": params["confirmation_sha256"],
        "cohort": params["cohort"], "reports": {},
    }
    return data, params, run


def _frozen_score(data: bytes, params: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    raw = json.dumps(run).encode()
    return score_frozen(data, raw, run_sha256=hashlib.sha256(raw).hexdigest(), **params)


def test_frozen_score_includes_every_admitted_test_case_in_its_cohort() -> None:
    cases = [
        _case("correct"), _case("error"), _case("compatible", "original"), _case("missing"),
        _case("denied", rights={"gemini": True, "gpt": False}),
        _case("calibration", partition="calibration"), _case("stress", cohort="stress"),
        _case("seller", evidence="declared"),
    ]
    data, params, run = _frozen_inputs(*cases)
    run["reports"] = {
        case["specimen_id"]: _run_receipt(case, conclusion)
        for case, conclusion in zip(cases[:3], ("probable_fake", "probable_original", "visually_compatible"))
    }
    independent = _frozen_score(data, params, run)["independent"]
    assert (independent["correct"], independent["incorrect"], independent["inconclusive"],
            independent["unavailable"], independent["fake_called_original"],
            independent["accuracy_all_cases"]) == (1, 1, 1, 1, 1, 0.25)
    assert independent["compatible_but_inconclusive"] == 1
    for case in cases[4:]:
        with pytest.raises(ValueError, match="unknown specimen"):
            _frozen_score(data, params, {**run, "reports": {case["specimen_id"]: _run_receipt(case, "probable_fake")}})
    with pytest.raises(ValueError, match="frozen SHA-256"):
        _frozen_score(data + b" ", params, run)


def test_frozen_score_refuses_unpinned_mock_or_mismatched_run() -> None:
    case = _case("a")
    data, params, run = _frozen_inputs(case)
    with pytest.raises(ValueError, match="trusted SHA-256"):
        score_frozen(data, json.dumps(run).encode(), run_sha256="9" * 64, **params)
    run_changes: tuple[dict[str, Any], ...] = (
        {"capture_mode": "simulated"}, {"freeze": {}}, {"manifest_sha256": "9" * 64},
    )
    for change in run_changes:
        with pytest.raises(ValueError, match="does not match the live frozen test"):
            _frozen_score(data, params, {**run, **change})
    with pytest.raises(ValueError, match="does not match the specimen or Gemini capture"):
        _frozen_score(data, params, {**run, "reports": {"a": "probable_fake"}})
    receipt = _run_receipt(case, "probable_fake")
    for change in ({"photo_sha256s": receipt["photo_sha256s"][::-1]},
                   {"gemini_model": "wrong-model"}, {"prepared_request_sha256": "7" * 64}):
        with pytest.raises(ValueError, match="does not match the specimen or Gemini capture"):
            _frozen_score(data, params, {**run, "reports": {"a": {**receipt, **change}}})
    with pytest.raises(ValueError, match="lacks a matching GPT capture"):
        _frozen_score(data, params, {**run, "route": "with_review", "reports": {"a": receipt}})
    reviewed: dict[str, Any] = {
        **run,
        "route": "with_review",
        "reports": {"a": {
            **receipt, "gpt_model": "model-version-b", "gpt_response_id": "provider-gpt-response-a",
            "gpt_media_sha256": case["prepared_media_sha256"], "gpt_technical_facts_sha256": "c" * 64,
            "gpt_request_sha256": "6" * 64, "gpt_response_sha256": "7" * 64,
        }},
    }
    assert _frozen_score(data, params, reviewed)["independent"]["correct"] == 1
    reviewed["reports"]["a"]["gpt_media_sha256"] = "5" * 64
    with pytest.raises(ValueError, match="lacks a matching GPT capture"):
        _frozen_score(data, params, reviewed)


@pytest.mark.parametrize("linkage", ["mixed", "uncertain"])
def test_with_review_preserves_inconclusive_when_photos_cannot_be_linked(linkage: str) -> None:
    case = _case("a")
    data, params, run = _frozen_inputs(case)
    receipt = {**_run_receipt(case, "inconclusive"), "gemini_linkage": linkage,
               "review_status": "unlinked_photos"}
    reviewed = {**run, "route": "with_review", "reports": {"a": receipt}}
    score = _frozen_score(data, params, reviewed)["independent"]
    assert (score["cases"], score["inconclusive"], score["unavailable"]) == (1, 1, 0)
    receipt["review_status"] = "skipped_for_cost"
    with pytest.raises(ValueError, match="explicit no-review"):
        _frozen_score(data, params, reviewed)
    receipt["conclusion"] = "probable_original"
    with pytest.raises(ValueError, match="unlinked photos require an inconclusive"):
        _frozen_score(data, params, reviewed)
