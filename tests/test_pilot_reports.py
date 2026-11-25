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
            "evidence_sha256": "e" * 64,
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
    ("split_family", "cross_partition_split_family"),
    ("provenance_clusters", "cross_partition_provenance"),
])
def test_cross_partition_reposts_and_related_specimens_cannot_enter(
    field: str, reason: str,
) -> None:
    first = _case("a", partition="development")
    second = _case("b", **{field: first[field]})
    result = census(_manifest(first, second), **_confirmations(first, second))
    assert result["cases"] == [
        {"specimen_id": key, "status": "blocked", "reasons": [reason]} for key in ("a", "b")
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


def test_frozen_score_includes_every_admitted_test_case_in_its_cohort() -> None:
    cases = [
        _case("correct"), _case("error"), _case("compatible", "original"), _case("missing"),
        _case("denied", rights={"gemini": True, "gpt": False}),
        _case("calibration", partition="calibration"), _case("stress", cohort="stress"),
        _case("seller", evidence="declared"),
    ]
    data = json.dumps(_manifest(*cases)).encode()
    params = {
        "manifest_sha256": hashlib.sha256(data).hexdigest(), "cohort": "representative",
        **_confirmations(*cases),
    }
    reports = {
        "correct": "probable_fake", "error": "probable_original", "compatible": "visually_compatible",
    }
    independent = score_frozen(data, reports, **params)["independent"]
    assert (independent["correct"], independent["incorrect"], independent["inconclusive"],
            independent["unavailable"], independent["fake_called_original"],
            independent["accuracy_all_cases"]) == (1, 1, 1, 1, 1, 0.25)
    assert independent["compatible_but_inconclusive"] == 1
    for specimen in ("denied", "calibration", "stress", "seller"):
        with pytest.raises(ValueError, match="unknown specimen"):
            score_frozen(data, {**reports, specimen: "probable_fake"}, **params)
    with pytest.raises(ValueError, match="frozen SHA-256"):
        score_frozen(data + b" ", reports, **params)
