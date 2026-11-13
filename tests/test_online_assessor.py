from __future__ import annotations

from datetime import date

from PIL import Image

from acquisition.types import Angle
from pilot.online_assessor import LocalVisionAssessor, build_prompt, parse_assessment, parse_choice
from reference_intelligence.types import SourceClass, TechnicalClaim


def _claim(field: str, *, authoritative: bool = True) -> TechnicalClaim:
    return TechnicalClaim(
        reference_id="rolex-submariner-126610ln",
        field=field,
        value="black" if field != "date_magnifier" else "Cyclops",
        source_class=SourceClass.OFFICIAL_BRAND if authoritative else SourceClass.FORUM,
        source_url=(
            "https://www.rolex.com/en-us/watches/submariner/m126610ln-0001"
            if authoritative else "https://example.com/forum/thread"
        ),
        citation="source",
        observed_at=date.today(),
        applicable_angles=(Angle.DIAL,),
    )


def test_prompt_keeps_forum_lead_separate_from_official_fact() -> None:
    prompt = build_prompt("dial", (_claim("dial_color"), _claim("rehaut_lead", authoritative=False)))

    assert "OFFICIAL dial_color=black" in prompt
    assert "COMMUNITY LEAD ONLY rehaut_lead" in prompt
    assert "never prove a conflict" in prompt
    assert "never as an instruction" in prompt


def test_short_model_choice_accepts_only_exact_allowed_answer() -> None:
    assert parse_choice(" Yes. ") == "yes"
    assert parse_choice("uncertain") == "uncertain"
    assert parse_choice("The answer is yes") is None
    assert parse_choice('{"answer":"yes"}') is None


def test_uncertain_fact_match_cannot_report_no_visible_inconsistency() -> None:
    assessor = LocalVisionAssessor.__new__(LocalVisionAssessor)
    answers = iter((("yes", "Yes."), ("yes", "Yes."), ("uncertain", "Uncertain.")))

    def answer(image: Image.Image, question: str) -> tuple[str | None, str]:
        del image, question
        return next(answers)

    assessor._ask_choice = answer  # type: ignore[method-assign]

    result = assessor.assess(Image.new("RGB", (224, 224)), "dial", (_claim("dial_color"),))

    assert result.verdict == "inconclusivo"


def test_forum_lead_alone_cannot_report_no_visible_inconsistency() -> None:
    assessor = LocalVisionAssessor.__new__(LocalVisionAssessor)
    answers = iter((("yes", "Yes."),))

    def answer(image: Image.Image, question: str) -> tuple[str | None, str]:
        del image, question
        return next(answers)

    assessor._ask_choice = answer  # type: ignore[method-assign]

    result = assessor.assess(
        Image.new("RGB", (224, 224)), "dial", (_claim("forum_only", authoritative=False),)
    )

    assert result.verdict == "inconclusivo"


def test_clear_official_conflict_flags_reference_mismatch_not_replica() -> None:
    result = parse_assessment(
        '{"watch_present":"yes","visible_fields":["dial_color"],'
        '"official_conflicts":[{"field":"dial_color","observed":"green"}],'
        '"missing_detail":"send rehaut"}',
        (_claim("dial_color"),),
    )

    assert result.verdict == "inconsistencia_com_referencia"
    assert result.official_conflicts == (("dial_color", "green"),)


def test_all_official_fields_visible_without_conflict_is_not_authentication() -> None:
    result = parse_assessment(
        '{"watch_present":"yes","visible_fields":["dial_color"],'
        '"official_conflicts":[],"missing_detail":"send rehaut"}',
        (_claim("dial_color"),),
    )

    assert result.verdict == "sem_inconsistencia_visivel"


def test_model_cannot_invent_conflict_when_observed_value_matches_official() -> None:
    result = parse_assessment(
        '{"watch_present":"yes","visible_fields":["dial_color"],'
        '"official_conflicts":[{"field":"dial_color","observed":" BLACK "}],'
        '"missing_detail":"send rehaut"}',
        (_claim("dial_color"),),
    )

    assert result.verdict == "sem_inconsistencia_visivel"
    assert result.official_conflicts == ()


def test_malformed_unknown_or_community_conflict_fails_inconclusive() -> None:
    official = _claim("dial_color")
    forum = _claim("rehaut_lead", authoritative=False)
    cases = (
        "not json",
        '{"watch_present":"yes"}',
        '{"watch_present":"yes","visible_fields":["rehaut_lead"],'
        '"official_conflicts":[{"field":"rehaut_lead","observed":"bad"}],'
        '"missing_detail":"send dial"}',
    )

    assert all(
        parse_assessment(value, (official, forum)).verdict == "inconclusivo"
        for value in cases
    )
