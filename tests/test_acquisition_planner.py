from acquisition.planner import plan_acquisition
from dataset.readiness import ReadinessReport
from evaluation.evaluate import CollectionWave


def _report(*cells):
    return ReadinessReport("rolex-submariner-126610ln", False, (), tuple(cells))


def _plan(report, waves=()):
    return plan_acquisition(
        report,
        waves,
        source_entity_counts={"dominant.example": 40, "partner-b.example": 30, "partner-c.example": 30},
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
    targets = _plan(_report(("genuine/dial", 0, 0), ("fake-1/crown", 0, 0)))

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
    reverse = _plan(ReadinessReport(report.reference_id, False, (), tuple(reversed(report.cell_counts))))

    assert forward == reverse
