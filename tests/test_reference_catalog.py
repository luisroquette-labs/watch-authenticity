from __future__ import annotations

import copy

import pytest

from acquisition.types import Angle
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
        catalog_fixture.catalog.entry_for("rolex-datejust-126334").required_angles,
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


def test_patek_reference_is_research_only_without_admitted_head(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("patek-nautilus-58111g001")

    assert entry.brand == "patek"
    assert entry.status == "pending"
    assert tuple(angle.value for angle in entry.required_angles) == (
        "dial", "crown", "clasp", "caseback"
    )
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_patek_5740_is_pending_without_training_admission(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("patek-nautilus-57401g001")
    assert entry.brand == "patek"
    assert entry.family == "nautilus"
    assert entry.status == "pending"
    assert entry.head is None
    assert entry.gold_freeze_sha256 is None
    assert entry.holdout_freeze_sha256 is None
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_cartier_santos_knowledge_is_catalogued_but_not_released(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("cartier-santos-wssa0061")

    assert entry.brand == "cartier"
    assert entry.family == "santos"
    assert entry.status == "pending"
    assert tuple(angle.value for angle in entry.required_angles) == (
        "dial", "crown", "clasp", "caseback"
    )
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_omega_reference_is_research_only_without_admitted_head(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("omega-seamaster-21030422001001")

    assert entry.brand == "omega"
    assert entry.status == "pending"
    assert tuple(angle.value for angle in entry.required_angles) == (
        "dial", "crown", "clasp", "caseback"
    )
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_omega_007_variant_is_catalogued_but_not_released(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("omega-seamaster-21090422001001")

    assert entry.brand == "omega"
    assert entry.reference == "21090422001001"
    assert entry.status == "pending"
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_seiko_spb453j1_is_research_only_without_admitted_head(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("seiko-prospex-spb453j1")

    assert entry.brand == "seiko"
    assert entry.status == "pending"
    assert tuple(angle.value for angle in entry.required_angles) == (
        "dial", "crown", "clasp", "caseback"
    )
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_rm88_target_model_is_research_only_without_admitted_head(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("richardmille-rm88-smiley")

    assert entry.brand == "richardmille"
    assert entry.status == "pending"
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_ap_26331_is_not_admitted_for_authenticity(catalog_fixture):
    entry = catalog_fixture.catalog.entry_for("audemarspiguet-royaloak-26331stoo1220st02")

    assert entry.brand == "audemarspiguet"
    assert entry.status == "pending"
    with pytest.raises(ReferenceNotAdmitted):
        catalog_fixture.catalog.head_for(entry.reference_id)


def test_admitted_submariner_returns_only_its_own_head(catalog_fixture):
    report = ReadinessReport("rolex-submariner-126610ln", True, (), (), tuple(Angle))

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


def test_later_reference_can_be_admitted_without_baseline_head(catalog_fixture):
    document = copy.deepcopy(catalog_fixture.document)
    _admit_datejust(document)

    catalog = parse_reference_catalog(document, expected_sha256=catalog_document_sha256(document))
    entry = catalog.entry_for("rolex-datejust-126334")
    report = ReadinessReport(entry.reference_id, True, (), (), entry.required_angles)

    assert admit_reference(entry.reference_id, report, catalog).allowed


def test_admission_rejects_readiness_for_different_views(catalog_fixture):
    document = copy.deepcopy(catalog_fixture.document)
    _admit_datejust(document)
    catalog = parse_reference_catalog(document, expected_sha256=catalog_document_sha256(document))
    report = ReadinessReport("rolex-datejust-126334", True, (), (), (Angle.DIAL,))

    assert "readiness report uses different required angles" in admit_reference(
        report.reference_id, report, catalog
    ).reasons


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
