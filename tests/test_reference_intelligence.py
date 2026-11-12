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


def test_official_126610ln_knowledge_is_versioned_and_cited() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)

    assert len(claims) == 7
    assert {claim.source_class for claim in claims} == {SourceClass.OFFICIAL_BRAND}
    assert all(claim.source_url.startswith("https://www.rolex.com/") for claim in claims)
    assert all(claim.citation and claim.observed_at <= date.today() for claim in claims)


def test_matching_fact_is_consistent_but_never_authenticates() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("dial_color", " BLACK ", Angle.DIAL, True),),
    )

    assert result.status is AssessmentStatus.CONSISTENT
    assert not result.authenticates_watch
    assert next(check for check in result.checks if check.field == "dial_color").status is CheckStatus.MATCHED


def test_visible_conflict_is_reported_with_source() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(
        REFERENCE,
        claims,
        (VisualObservation("bezel_color", "green", Angle.DIAL, True),),
    )

    assert result.status is AssessmentStatus.CONFLICTING
    conflict = next(check for check in result.checks if check.status is CheckStatus.CONFLICTING)
    assert conflict.field == "bezel_color"
    assert conflict.source_url.startswith("https://www.rolex.com/")
    assert conflict.source_observed_at == date(2026, 11, 12)


def test_missing_visible_evidence_is_inconclusive() -> None:
    claims = load_knowledge(KNOWLEDGE, REFERENCE)
    result = assess_reference(REFERENCE, claims, ())

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
        (VisualObservation("dial_color", "blue", Angle.DIAL, True),),
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
        (VisualObservation("dial_color", "black", Angle.DIAL, True),),
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
    )

    calibre = next(check for check in result.checks if check.field == "calibre")
    assert calibre.status is CheckStatus.UNOBSERVABLE
    assert result.status is AssessmentStatus.INCONCLUSIVE


def test_knowledge_refuses_another_reference() -> None:
    with pytest.raises(ValueError, match="another exact reference"):
        load_knowledge(KNOWLEDGE, "rolex-submariner-116610ln")
