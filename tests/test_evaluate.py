from __future__ import annotations

from evaluation.evaluate import (
    CollectionWave,
    LearningCurvePoint,
    SourceEmbedding,
    collection_decision,
    evaluate_source_probe,
    learning_curve_report,
)


def _biased_embeddings() -> list[SourceEmbedding]:
    rows: list[SourceEmbedding] = []
    for index in range(20):
        rows.append(SourceEmbedding((float(index) / 100, 0.0), "dealer.example"))
        rows.append(SourceEmbedding((10.0 + float(index) / 100, 0.0), "replica.example"))
    return rows


def test_source_signal_above_permutation_gate_blocks_release():
    report = evaluate_source_probe(_biased_embeddings(), permutations=200, seed=42)

    assert report.real_ci_low > report.shuffled_ci_high + 0.10
    assert report.blocked
    assert report.driving_sources == ("dealer.example", "replica.example")


def test_source_probe_is_deterministic():
    first = evaluate_source_probe(_biased_embeddings(), permutations=50, seed=42)
    second = evaluate_source_probe(_biased_embeddings(), permutations=50, seed=42)

    assert first == second


def test_two_saturated_collection_waves_stop_same_distribution():
    decision = collection_decision(
        [
            CollectionWave(0.20, 0.8, 1.5),
            CollectionWave(0.21, 0.7, 1.2),
        ]
    )

    assert decision.stop_same_distribution
    assert not decision.continue_collection


def test_named_weak_slice_keeps_collection_open():
    decision = collection_decision(
        [
            CollectionWave(0.20, 0.8, 1.5),
            CollectionWave(0.21, 0.7, 1.2, weak_slice="grade-4/caseback"),
        ]
    )

    assert not decision.stop_same_distribution
    assert decision.continue_collection


def test_learning_curve_requires_all_four_entity_waves():
    points = [
        LearningCurvePoint(fraction, entities, 0.8, 0.6, 0.04, 0.05)
        for fraction, entities in ((0.25, 25), (0.5, 50), (0.75, 75), (1.0, 100))
    ]

    report = learning_curve_report(list(reversed(points)))

    assert report.points == tuple(points)

