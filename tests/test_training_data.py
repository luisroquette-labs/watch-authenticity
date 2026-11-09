from __future__ import annotations

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


def test_entity_waves_count_entities_instead_of_duplicate_photos(training_fixture):
    cos, spec, _manifest, _splits, _images = training_fixture
    bundle = load_training_data(cos, spec)

    waves = entity_wave_asset_ids(bundle)

    assert tuple(waves) == (0.25, 0.5, 0.75, 1.0)
    assert set(waves[0.25]) <= set(waves[0.5]) <= set(waves[0.75]) <= set(waves[1.0])
    assert set(waves[1.0]) == set(bundle.splits["train"])

    final_bundle = training_wave_bundle(bundle, waves[1.0])
    assert set(final_bundle.splits["train"]) == set(bundle.splits["train"])

