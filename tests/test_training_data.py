from __future__ import annotations

import hashlib
from dataclasses import replace
from types import MappingProxyType

import pytest

from tests.conftest import valid_run_spec
from tests.test_training_run import _seed, _valid_payloads
from training.data import (
    TrainingDataRefused,
    entity_wave_asset_ids,
    load_training_data,
    training_wave_bundle,
)


@pytest.fixture
def training_fixture():
    manifest, splits, images = _valid_payloads(valid_run_spec())
    cos, spec = _seed(manifest, splits, images)
    return cos, spec, manifest, splits, images


def test_training_refuses_tier_c_rows(training_fixture):
    cos, spec, manifest, splits, images = training_fixture
    manifest["records"][0]["evidence_tier"] = "c"
    cos, spec = _seed(manifest, splits, images)

    with pytest.raises(TrainingDataRefused, match="Tier A or B"):
        load_training_data(cos, spec)


def test_training_refuses_frozen_source_with_copied_official_media(training_fixture):
    from tests.test_training_run import _canonical

    _cos, _spec, manifest, splits, images = training_fixture
    manifest["source_records"][0]["original_sha256"] = (
        "f6552812a866a41d8285ef2b5997aa66b9370414ea26bd5e3cb55faf0f112c98"
    )
    manifest["freeze"]["asset_raw_sha256s"] = sorted(
        [record["asset_id"], record["original_sha256"]]
        for record in manifest["source_records"]
    )
    manifest["freeze"]["manifest_sha256"] = hashlib.sha256(
        _canonical(manifest["source_records"])
    ).hexdigest()
    cos, spec = _seed(manifest, splits, images, refresh_freeze=False)

    with pytest.raises(TrainingDataRefused, match="reuses official brand media"):
        load_training_data(cos, spec)


def test_entity_waves_count_entities_instead_of_duplicate_photos(training_fixture):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)

    waves = entity_wave_asset_ids(bundle)

    assert tuple(waves) == (0.25, 0.5, 0.75, 1.0)
    assert set(waves[0.25]) <= set(waves[0.5]) <= set(waves[0.75]) <= set(waves[1.0])
    assert set(waves[1.0]) == set(bundle.splits["train"])

    final_bundle = training_wave_bundle(bundle, waves[1.0])
    assert set(final_bundle.splits["train"]) == set(bundle.splits["train"])


def test_entity_wave_never_splits_one_watch_across_angles(training_fixture):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)
    genuine = [record for record in bundle.records if record.label.value == "genuine"][:3]
    records = (
        replace(genuine[0], asset_id="a-dial", subject_id="a", angle="dial"),
        replace(genuine[1], asset_id="z-dial", subject_id="z", angle="dial"),
        replace(genuine[2], asset_id="z-caseback", subject_id="z", angle="caseback"),
    )
    split_ids = tuple(record.asset_id for record in records)
    small = replace(
        bundle,
        records=records,
        splits=MappingProxyType({**bundle.splits, "train": split_ids}),
        split_by_asset=MappingProxyType({asset_id: "train" for asset_id in split_ids}),
    )

    first_wave = set(entity_wave_asset_ids(small)[0.25])

    assert {"z-dial", "z-caseback"} <= first_wave



@pytest.mark.parametrize("case", ["missing-digest", "wrong-digest", "wrong-angle", "hidden", "tampered", "legacy-source"])
def test_loader_refuses_unreviewed_or_mismatched_angle(training_fixture, case):
    import hashlib

    from acquisition.types import Angle
    from dataset.angle_review import record_usability_review, usability_review_json
    from tests.conftest import NOW
    from tests.test_training_run import _canonical

    _cos, _spec, manifest, splits, images = training_fixture
    if case == "missing-digest":
        del manifest["records"][0]["usability_review_sha256"]
    elif case == "wrong-digest":
        manifest["records"][0]["usability_review_sha256"] = "0" * 64
    else:
        source = manifest["source_records"][0]
        if case == "legacy-source":
            del source["usability_review"]
        elif case in {"wrong-angle", "hidden"}:
            review = record_usability_review(
                source["candidate_id"], source["image_sha256"],
                Angle.CLASP if case == "wrong-angle" else Angle(source["angle"]),
                case != "hidden", case != "hidden", "angle-reviewer", NOW,
            )
            source["usability_review"] = usability_review_json(review)
            manifest["records"][0]["usability_review_sha256"] = review.review_sha256
        else:
            source["usability_review"]["reviewer_id"] = "changed"
        manifest["freeze"]["manifest_sha256"] = hashlib.sha256(_canonical(manifest["source_records"])).hexdigest()
    cos, spec = _seed(manifest, splits, images, refresh_freeze=False)
    with pytest.raises(TrainingDataRefused, match="schema|usability|canonical"):
        load_training_data(cos, spec)


def test_loader_retains_review_digest_and_rejection_exception(training_fixture):
    cos, spec, manifest, _splits, _images = training_fixture
    records = load_training_data(cos, spec).records
    assert records[0].usability_review_sha256 == manifest["records"][0]["usability_review_sha256"]
    assert all(record.usability_review_sha256 is None for record in records if record.label.value == "rejection")


def test_loader_refuses_rejection_subtype_different_from_canonical_source(training_fixture):
    _cos, _spec, manifest, splits, images = training_fixture
    rejection = next(record for record in manifest["records"] if record["rejection_subtype"] == "no-watch")
    rejection["rejection_subtype"] = "non-target-reference"
    cos, spec = _seed(manifest, splits, images, refresh_freeze=False)

    with pytest.raises(TrainingDataRefused, match="projection differs"):
        load_training_data(cos, spec)
