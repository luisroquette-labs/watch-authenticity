from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from acquisition.types import Angle
from reference_intelligence import (
    AssessmentStatus,
    CheckStatus,
    SourceClass,
    TechnicalClaim,
    VisualObservation,
    assess_reference,
    load_knowledge,
)

REFERENCE = "rolex-submariner-126610ln"
KNOWLEDGE = Path("config/references/rolex-submariner-126610ln.json")
FRESH_AFTER = date(2026, 1, 1)


def test_official_126610ln_knowledge_is_versioned_and_cited() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)

    assert len(claims) == 10
    assert {claim.source_class for claim in claims} == {
        SourceClass.OFFICIAL_BRAND,
        SourceClass.FORUM,
    }
    official = tuple(claim for claim in claims if claim.source_class is SourceClass.OFFICIAL_BRAND)
    assert len(official) == 7
    assert all(claim.source_url.startswith("https://www.rolex.com/") for claim in official)
    assert all(claim.citation and claim.observed_at <= date.today() for claim in claims)


def test_datejust_126334_knowledge_does_not_universalize_variant_features() -> None:
    reference = "rolex-datejust-126334"
    claims = load_knowledge(Path("config/references/rolex-datejust-126334.json"), reference)

    assert len(claims) == 6
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "bezel_type", "date_magnifier"
    }
    assert not {"dial_color", "bracelet_type"} & {claim.field for claim in claims}
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_daytona_126506_knowledge_identifies_visible_design_without_authentication() -> None:
    reference = "rolex-daytona-126506"
    claims = load_knowledge(Path(f"config/references/{reference}.json"), reference)

    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "bezel_type", "caseback_type"
    }
    for field in ("case_material", "calibre", "bezel_material", "caseback_coating"):
        assert not next(claim for claim in claims if claim.field == field).applicable_angles
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_patek_5811_knowledge_is_cited_without_authenticity_coverage() -> None:
    reference = "patek-nautilus-58111g001"
    claims = load_knowledge(Path("config/references/patek-nautilus-58111g001.json"), reference)

    assert len(claims) == 9
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url == "https://www.patek.com/en/collection/nautilus/5811-1g-001" for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "dial_texture", "date_position", "caseback_type", "clasp_type"
    }
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_patek_5740_perpetual_calendar_stays_research_only() -> None:
    reference = "patek-nautilus-57401g001"
    claims = load_knowledge(Path(f"config/references/{reference}.json"), reference)

    assert len(claims) == 9
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url == "https://www.patek.com/en/collection/nautilus/5740-1g-001" for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "dial_texture", "caseback_type", "clasp_type"
    }
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_ap_26331_official_claims_remain_research_only(catalog_fixture) -> None:
    reference = "audemarspiguet-royaloak-26331stoo1220st02"
    claims = load_knowledge(Path(f"config/references/{reference}.json"), reference)
    entry = catalog_fixture.catalog.entry_for(reference)

    assert entry.status == "pending"
    assert len(claims) == 5
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "dial_texture"
    }
    assert assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER).status is AssessmentStatus.INCONCLUSIVE


def test_ap_official_claim_rejects_impersonated_domain() -> None:
    with pytest.raises(ValueError, match="approved domain"):
        TechnicalClaim(
            reference_id="audemarspiguet-royaloak-26331stoo1220st02",
            field="dial_color",
            value="black",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://audemarspiguet.com.evil.example/watch",
            citation="False official claim",
            observed_at=date.today(),
            applicable_angles=(Angle.DIAL,),
        )


def test_patek_official_claim_rejects_impersonated_domain() -> None:
    with pytest.raises(ValueError, match="approved domain"):
        TechnicalClaim(
            reference_id="patek-nautilus-58111g001",
            field="dial_color",
            value="blue",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://patek.com.evil.example/watch",
            citation="False official claim",
            observed_at=date.today(),
            applicable_angles=(Angle.DIAL,),
        )


def test_omega_seamaster_knowledge_cites_official_sources_without_authenticity_coverage() -> None:
    reference = "omega-seamaster-21030422001001"
    claims = load_knowledge(Path("config/references/omega-seamaster-21030422001001.json"), reference)

    assert len(claims) == 13
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url.startswith("https://www.omegawatches.com/") for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "dial_texture", "date_position", "bezel_color", "bezel_scale_color",
        "helium_escape_valve_shape", "caseback_type", "clasp_type",
    }
    assert not next(claim for claim in claims if claim.field == "water_resistance").applicable_angles
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_speedmaster_001_official_facts_do_not_authenticate_or_conflate_sapphire_variant() -> None:
    reference = "omega-speedmaster-31030425001001"
    claims = load_knowledge(Path(f"config/references/{reference}.json"), reference)
    assert len(claims) == 12
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url.startswith("https://www.omegawatches.com/") for claim in claims)
    assert next(claim.value for claim in claims if claim.field == "crystal_material").startswith("hesalite")
    assert next(claim.value for claim in claims if claim.field == "movement_type") == "manual-winding chronograph"
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    assessment = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert assessment.status is AssessmentStatus.INCONCLUSIVE
    assert not assessment.authenticates_watch


def test_omega_007_official_identity_is_distinct_from_standard_seamaster() -> None:
    edition = "omega-seamaster-21090422001001"
    claims = load_knowledge(Path(f"config/references/{edition}.json"), edition)
    standard = load_knowledge(
        Path("config/references/omega-seamaster-21030422001001.json"),
        "omega-seamaster-21030422001001",
    )

    assert len(claims) == 12
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert {claim.value for claim in claims if claim.field == "reference_designation"} == {
        "210.90.42.20.01.001"
    }
    assert next(claim.value for claim in claims if claim.field == "dial_color") != next(
        claim.value for claim in standard if claim.field == "dial_color"
    )
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    assert not next(claim for claim in claims if claim.field == "bezel_material").applicable_angles
    assert not next(claim for claim in claims if claim.field == "bracelet_material").applicable_angles
    result = assess_reference(edition, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_omega_official_claim_rejects_impersonated_domain() -> None:
    with pytest.raises(ValueError, match="approved domain"):
        TechnicalClaim(
            reference_id="omega-seamaster-21030422001001",
            field="dial_color",
            value="black",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://omegawatches.com.evil.example/watch",
            citation="False official claim",
            observed_at=date.today(),
            applicable_angles=(Angle.DIAL,),
        )


def test_cartier_wssa0061_knowledge_separates_visible_facts_from_physical_specs() -> None:
    reference = "cartier-santos-wssa0061"
    claims = load_knowledge(Path("config/references/cartier-santos-wssa0061.json"), reference)

    assert len(claims) == 12
    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url.startswith("https://www.cartier.com/") for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "dial_color", "hand_shape", "crown_shape", "crown_stone_color",
    }
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    assert not next(claim for claim in claims if claim.field == "crown_stone_material").applicable_angles
    assert not next(claim for claim in claims if claim.field == "hand_material").applicable_angles
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_cartier_official_claim_rejects_impersonated_domain() -> None:
    with pytest.raises(ValueError, match="approved domain"):
        TechnicalClaim(
            reference_id="cartier-santos-wssa0061",
            field="dial_color",
            value="green",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://cartier.com.evil.example/watch",
            citation="False official claim",
            observed_at=date.today(),
            applicable_angles=(Angle.DIAL,),
        )


def test_seiko_spb453j1_official_knowledge_is_research_only() -> None:
    reference = "seiko-prospex-spb453j1"
    claims = load_knowledge(Path("config/references/seiko-prospex-spb453j1.json"), reference)

    assert all(claim.source_class is SourceClass.OFFICIAL_BRAND for claim in claims)
    assert all(claim.source_url.startswith("https://www.seikowatches.com/") for claim in claims)
    assert {claim.field for claim in claims if claim.applicable_angles} == {
        "clasp_type", "dial_color", "bezel_color", "date_position",
    }
    assert not next(claim for claim in claims if claim.field == "calibre").applicable_angles
    assert not next(claim for claim in claims if claim.field == "water_resistance").applicable_angles
    result = assess_reference(reference, claims, (), minimum_source_date=FRESH_AFTER)
    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert not result.authenticates_watch


def test_seiko_official_claim_rejects_impersonated_domain() -> None:
    with pytest.raises(ValueError, match="approved domain"):
        TechnicalClaim(
            reference_id="seiko-prospex-spb453j1",
            field="calibre",
            value="6R55",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://seikowatches.com.evil.example/watch",
            citation="False official claim",
            observed_at=date.today(),
        )


def test_all_visible_facts_must_match_to_be_consistent_but_never_authenticate() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (
            VisualObservation("bezel_color", "black", Angle.DIAL, True),
            VisualObservation("dial_color", " BLACK ", Angle.DIAL, True),
            VisualObservation("date_magnifier", "Cyclops", Angle.CYCLOPS, True),
        ),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.CONSISTENT
    assert not result.authenticates_watch
    assert next(check for check in result.checks if check.field == "dial_color").status is CheckStatus.MATCHED


def test_one_match_with_an_unobserved_visible_fact_is_inconclusive() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("dial_color", "black", Angle.DIAL, True),),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE


def test_non_visual_fact_does_not_block_complete_visual_consistency() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (
            VisualObservation("bezel_color", "black", Angle.DIAL, True),
            VisualObservation("dial_color", "black", Angle.DIAL, True),
            VisualObservation("date_magnifier", "Cyclops", Angle.CYCLOPS, True),
            VisualObservation("calibre", "3235", Angle.CASEBACK, True),
        ),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.CONSISTENT


def test_visible_conflict_is_reported_with_source() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("bezel_color", "green", Angle.DIAL, True),),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.CONFLICTING
    conflict = next(check for check in result.checks if check.status is CheckStatus.CONFLICTING)
    assert conflict.field == "bezel_color"
    assert conflict.source_url.startswith("https://www.rolex.com/")
    assert conflict.source_observed_at == date(2026, 11, 12)


def test_missing_visible_evidence_is_inconclusive() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE, claims, (), minimum_source_date=FRESH_AFTER
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert all(check.status is CheckStatus.UNOBSERVABLE for check in result.checks)


def test_forum_claim_is_counted_only_as_a_lead() -> None:
    official = load_knowledge(KNOWLEDGE, REFERENCE)[3]
    forum = replace(
        official,
        value="blue",
        source_class=SourceClass.FORUM,
        source_url="https://example.com/forum/thread-1",
        citation="Forum post",
    )
    result = assess_reference(
        REFERENCE,
        (forum,),
        (),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert result.lead_count == 1
    assert not result.checks


def test_expert_claim_corroborates_but_cannot_replace_official_fact() -> None:
    official = load_knowledge(KNOWLEDGE, REFERENCE)[3]
    expert = replace(
        official,
        source_class=SourceClass.NAMED_EXPERT,
        source_url="https://example.com/watchmaker/report-1",
        citation="Named watchmaker report",
    )
    result = assess_reference(
        REFERENCE,
        (expert,),
        (),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert result.corroboration_count == 1
    assert result.lead_count == 0


def test_conflicting_official_claims_fail_closed() -> None:
    official = load_knowledge(KNOWLEDGE, REFERENCE)[3]
    conflict = replace(official, value="blue", citation="Conflicting official snapshot")
    result = assess_reference(
        REFERENCE,
        (official, conflict),
        (VisualObservation("dial_color", "black", Angle.DIAL, True),),
        minimum_source_date=FRESH_AFTER,
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert result.source_conflicts == ("dial_color",)


def test_claim_rejects_future_provenance() -> None:
    with pytest.raises(ValueError, match="not in the future"):
        TechnicalClaim(
            reference_id=REFERENCE,
            field="dial_color",
            value="black",
            source_class=SourceClass.OFFICIAL_BRAND,
            source_url="https://www.rolex.com/example",
            citation="Dial",
            observed_at=date.today() + timedelta(days=1),
            applicable_angles=(Angle.DIAL,),
        )


def test_stale_official_fact_is_inconclusive() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("dial_color", "black", Angle.DIAL, True),),
        minimum_source_date=date(2026, 11, 13),
    )

    assert result.status is AssessmentStatus.INCONCLUSIVE
    assert "dial_color" in result.stale_fields


def test_non_visual_specification_cannot_be_proved_by_a_photo() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("calibre", "3235", Angle.CASEBACK, True),),
        minimum_source_date=FRESH_AFTER,
    )

    calibre = next(check for check in result.checks if check.field == "calibre")
    assert calibre.status is CheckStatus.UNOBSERVABLE
    assert result.status is AssessmentStatus.INCONCLUSIVE


def test_knowledge_refuses_another_reference() -> None:
    with pytest.raises(ValueError, match="another exact reference"):
        load_knowledge(KNOWLEDGE, "rolex-submariner-116610ln")


def test_official_claim_rejects_spoofed_domain() -> None:
    official = load_knowledge(KNOWLEDGE, REFERENCE)[0]

    with pytest.raises(ValueError, match="approved domain"):
        replace(official, source_url="https://rolex.com.seller.example/spec")


@pytest.mark.parametrize("field", ["authenticity", "unknown_detail"])
def test_hallucinated_observation_field_is_rejected(field: str) -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)

    with pytest.raises(ValueError, match="unknown technical fields"):
        assess_reference(
            REFERENCE,
            claims,
            (VisualObservation(field, "genuine", Angle.DIAL, True),),
            minimum_source_date=FRESH_AFTER,
        )


def test_diameter_and_material_are_not_visually_provable_without_measurement() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (
            VisualObservation("diameter_mm", "41", Angle.DIAL, True),
            VisualObservation("case_material", "Oystersteel", Angle.CROWN, True),
        ),
        minimum_source_date=FRESH_AFTER,
    )

    checks = {check.field: check for check in result.checks}
    assert checks["diameter_mm"].status is CheckStatus.UNOBSERVABLE
    assert checks["case_material"].status is CheckStatus.UNOBSERVABLE
