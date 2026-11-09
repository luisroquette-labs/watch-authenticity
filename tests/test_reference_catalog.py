from __future__ import annotations

import copy
import json

import pytest

from dataset.readiness import ReadinessReport
from dataset.reference_catalog import (
    ReferenceCatalogError,
    ReferenceNotAdmitted,
    admit_reference,
    catalog_document_sha256,
    load_reference_catalog,
    parse_reference_catalog,
)


@pytest.fixture
def catalog_document() -> dict[str, object]:
    return json.loads(open("config/reference_catalog.json").read())


@pytest.fixture
def catalog(catalog_document):
    return parse_reference_catalog(
        catalog_document, expected_sha256=catalog_document_sha256(catalog_document)
    )


@pytest.fixture
def admitted_catalog_document(catalog_document):
    document = copy.deepcopy(catalog_document)
    entry = document["references"][0]
    entry.update(
        {
            "identity_evidence_sha256": "1" * 64,
            "gold_freeze_sha256": "2" * 64,
            "holdout_freeze_sha256": "3" * 64,
            "status": "admitted",
            "head": {
                "reference_id": "rolex-submariner-126610ln",
                "generation": 1,
                "checkpoint_sha256": "4" * 64,
                "calibration_sha256": "5" * 64,
            },
        }
    )
    return document


@pytest.fixture
def admitted_catalog(admitted_catalog_document):
    return parse_reference_catalog(
        admitted_catalog_document,
        expected_sha256=catalog_document_sha256(admitted_catalog_document),
    )


def test_photo_volume_cannot_bypass_reference_readiness(catalog):
    report = ReadinessReport(
        "rolex-datejust-126334",
        False,
        ("genuine: source one.example exceeds 40% of train entities",),
        (("genuine/dial", 50_000, 10_000),),
    )

    decision = admit_reference("rolex-datejust-126334", report, catalog)

    assert not decision.allowed
    assert "readiness report failed" in decision.reasons


def test_submariner_rows_cannot_train_an_unadmitted_head(catalog):
    with pytest.raises(ReferenceNotAdmitted):
        catalog.head_for("rolex-datejust-126334")


def test_real_catalog_does_not_claim_a_trained_submariner_head(catalog):
    with pytest.raises(ReferenceNotAdmitted):
        catalog.head_for("rolex-submariner-126610ln")


def test_admitted_submariner_returns_only_its_own_head(admitted_catalog):
    report = ReadinessReport("rolex-submariner-126610ln", True, (), ())

    decision = admit_reference("rolex-submariner-126610ln", report, admitted_catalog)
    head = admitted_catalog.head_for("rolex-submariner-126610ln")

    assert decision.allowed
    assert head.reference_id == "rolex-submariner-126610ln"


@pytest.mark.parametrize(
    "case",
    [
        "alias_collision",
        "generation_mismatch",
        "changed_angles",
        "revoked_admission",
        "wrong_reference_checkpoint",
        "catalog_hash_mismatch",
    ],
)
def test_reference_catalog_refuses_incompatible_state(
    case, catalog_document, admitted_catalog_document, admitted_catalog
):
    admitted_cases = {"changed_angles", "revoked_admission", "wrong_reference_checkpoint"}
    document = copy.deepcopy(
        admitted_catalog_document if case in admitted_cases else catalog_document
    )
    expected = catalog_document_sha256(document)
    previous = None
    if case == "alias_collision":
        document["references"][1]["aliases"] = ["rolex-submariner"]
    elif case == "generation_mismatch":
        document["generation"] = 2
    elif case == "changed_angles":
        document["references"][0]["required_angles"] = ["dial"]
        previous = admitted_catalog
    elif case == "revoked_admission":
        document["references"][0]["status"] = "pending"
        document["references"][0]["head"] = None
        previous = admitted_catalog
    elif case == "wrong_reference_checkpoint":
        document["references"][0]["head"]["reference_id"] = "rolex-datejust-126334"
    else:
        expected = "0" * 64
    if case != "catalog_hash_mismatch":
        expected = catalog_document_sha256(document)

    with pytest.raises(ReferenceCatalogError):
        parse_reference_catalog(document, expected_sha256=expected, previous=previous)


def test_versioned_config_loads_from_disk(catalog_document):
    expected = catalog_document_sha256(catalog_document)

    loaded = load_reference_catalog("config/reference_catalog.json", expected_sha256=expected)

    assert loaded.generation == 1
