from acquisition.planner import plan_acquisition
from acquisition.types import Angle
from dataset.readiness import ReadinessReport
from evaluation.evaluate import CollectionWave


def _report(*cells):
    return ReadinessReport("rolex-submariner-126610ln", False, (), tuple(cells), tuple(Angle))


def _plan(report, waves=(), yields=None, consumption=None):
    return plan_acquisition(
        report,
        waves,
        source_entity_counts={
            "dominant.example": 40,
            "partner-b.example": 30,
            "partner-c.example": 30,
        },
        eligible_yield_by_source=yields or {},
        candidate_consumption_by_source=consumption if consumption is not None else {name: counts[0] for name, counts in (yields or {}).items()},
        seller_entity_counts={**{f"seller-{index}": 10 for index in range(10)}, "seller-new": 0},
    )


def test_missing_entity_and_angle_deficits_rank_before_raw_volume():
    targets = _plan(
        _report(
            ("genuine/dial", 0, 99),
            ("fake-4/caseback", 490, 0),
        )
    )

    assert targets[0].output == "fake-4"
    assert targets[0].angle == "caseback"
    assert targets[0].missing_entities == 100


def test_plan_uses_one_low_share_source_and_seller_for_every_target():
    targets = _plan(
        _report(("genuine/dial", 0, 0), ("fake-1/crown", 0, 0)),
        yields={"partner-b.example": (20, 20)},
    )

    assert {target.source for target in targets} == {"partner-b.example"}
    assert {target.seller_group_id for target in targets} == {"seller-new"}
    planned = sum(target.planned_new_entities for target in targets)
    assert planned == 11
    assert (30 + planned) / (100 + planned) <= 0.4
    assert planned / (100 + planned) <= 0.1


def test_two_saturated_waves_stop_same_distribution():
    waves = (
        CollectionWave(0.20, 0.8, 1.5),
        CollectionWave(0.21, 0.7, 1.2),
    )

    assert _plan(_report(("genuine/dial", 0, 0)), waves) == ()


def test_ranking_is_deterministic_when_cells_and_counts_change_order():
    report = _report(("fake-2/clasp", 10, 10), ("genuine/dial", 10, 10))
    forward = _plan(report)
    reverse = _plan(ReadinessReport(report.reference_id, False, (), tuple(reversed(report.cell_counts)), report.required_angles))

    assert forward == reverse


def test_reposts_with_zero_independent_gain_lose_to_useful_source():
    targets = _plan(
        _report(("genuine/dial", 1000, 1), ("fake-4/caseback", 490, 0)),
        yields={"partner-b.example": (1000, 0), "partner-c.example": (20, 8)},
    )
    assert targets[0].source == "partner-c.example"
    assert targets[0].angle == "caseback"
    assert targets[0].eligible_yield == (20, 8)
    assert not targets[0].exploratory
    assert targets[0].candidate_limit == 20
    assert targets[0].planned_new_entities == 8


def test_unknown_source_gets_one_bounded_unknown_wave():
    targets = _plan(_report(("genuine/dial", 0, 0), ("fake-1/crown", 0, 0)))
    assert len(targets) == 1
    assert targets[0].exploratory
    assert targets[0].candidate_limit == 20
    assert targets[0].eligible_yield is None
    assert targets[0].planned_new_entities == 0


def test_first_exploration_with_no_admitted_entities_is_bounded():
    targets = plan_acquisition(
        _report(("genuine/dial", 0, 0)),
        (),
        source_entity_counts={"source-a": 0, "source-b": 0},
        seller_entity_counts={"seller-a": 0, "seller-b": 0},
        eligible_yield_by_source={},
        candidate_consumption_by_source={},
    )

    assert len(targets) == 1
    assert targets[0].source == "source-a"
    assert targets[0].seller_group_id == "seller-a"
    assert targets[0].exploratory
    assert targets[0].candidate_limit == 20
    assert targets[0].eligible_source_capacity == 0
    assert targets[0].planned_new_entities == 0


def test_measured_zero_yield_is_not_retried():
    assert (
        _plan(
            _report(("genuine/dial", 0, 0)),
            yields={
                "partner-b.example": (20, 0),
                "partner-c.example": (20, 0),
            },
        )
        == ()
    )


def test_empty_exploration_is_not_repeated():
    assert (
        _plan(
            _report(("genuine/dial", 0, 0)),
            yields={
                "partner-b.example": (0, 0),
                "partner-c.example": (0, 0),
            },
        )
        == ()
    )


def test_twenty_reposts_of_one_watch_lose_to_twenty_distinct_watches():
    targets = _plan(_report(("genuine/dial", 0, 0)),
                    yields={"partner-b.example": (1, 1), "partner-c.example": (20, 20)},
                    consumption={"partner-b.example": 20, "partner-c.example": 20})
    assert targets[0].source == "partner-c.example"


def test_low_nonzero_yield_still_plans_positive_entity_gain():
    targets = _plan(_report(("genuine/dial", 0, 0)),
                    yields={"partner-b.example": (100, 1), "partner-c.example": (20, 0)},
                    consumption={"partner-b.example": 200, "partner-c.example": 20})
    assert targets[0].planned_new_entities == 1
