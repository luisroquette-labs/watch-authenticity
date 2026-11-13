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
