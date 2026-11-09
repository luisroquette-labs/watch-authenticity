from __future__ import annotations

import copy

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


def test_photo_volume_cannot_bypass_reference_readiness(catalog_fixture):
    report = ReadinessReport(
        "rolex-datejust-126334",
        False,
        ("genuine: source one.example exceeds 40% of train entities",),
        (("genuine/dial", 50_000, 10_000),),
    )

    decision = admit_reference("rolex-datejust-126334", report, catalog_fixture.catalog)

    assert not decision.allowed
    assert "readiness report failed" in decision.reasons


def test_submariner_rows_cannot_train_an_unadmitted_head(catalog_fixture):
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for("rolex-datejust-126334")


def test_real_catalog_does_not_claim_a_trained_submariner_head(catalog_fixture):
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for("rolex-submariner-126610ln")


def test_admitted_submariner_returns_only_its_own_head(catalog_fixture):
    report = ReadinessReport("rolex-submariner-126610ln", True, (), ())

    decision = admit_reference(
        "rolex-submariner-126610ln", report, catalog_fixture.admitted_catalog
    )
    head = catalog_fixture.admitted_catalog.head_for("rolex-submariner-126610ln")

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
def test_reference_catalog_refuses_incompatible_state(case, catalog_fixture):
    with pytest.raises(ReferenceCatalogError):
        catalog_fixture.load(case)


def test_later_reference_requires_an_admitted_baseline(catalog_fixture):
    document = copy.deepcopy(catalog_fixture.document)
    _admit_datejust(document)

    with pytest.raises(ReferenceCatalogError, match="admitted baseline"):
        parse_reference_catalog(
            document, expected_sha256=catalog_document_sha256(document)
        )


def test_references_cannot_reuse_slices_or_heads(catalog_fixture):
    document = copy.deepcopy(catalog_fixture.admitted_document)
    _admit_datejust(document)
    document["references"][1]["gold_freeze_sha256"] = "2" * 64

    with pytest.raises(ReferenceCatalogError, match="reuses"):
        parse_reference_catalog(
            document, expected_sha256=catalog_document_sha256(document)
        )


def test_alias_must_keep_the_exact_identity(catalog_fixture):
    document = copy.deepcopy(catalog_fixture.document)
    document["references"][0]["aliases"] = ["tudor-black-bay-126610ln"]

    with pytest.raises(ReferenceCatalogError, match="brand, family and reference"):
        parse_reference_catalog(
            document, expected_sha256=catalog_document_sha256(document)
        )


def test_versioned_config_loads_from_disk(catalog_fixture):
    expected = catalog_document_sha256(catalog_fixture.document)

    loaded = load_reference_catalog("config/reference_catalog.json", expected_sha256=expected)

    assert loaded.generation == 1


def _admit_datejust(document):
    document["references"][1].update(
        {
            "identity_evidence_sha256": "6" * 64,
            "gold_freeze_sha256": "7" * 64,
            "holdout_freeze_sha256": "8" * 64,
            "status": "admitted",
            "head": {
                "reference_id": "rolex-datejust-126334",
                "generation": 1,
                "checkpoint_sha256": "9" * 64,
                "calibration_sha256": "a" * 64,
            },
        }
    )
