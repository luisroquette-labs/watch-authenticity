"""Smoke test: verify all package imports resolve."""


def test_core_import():
    """Verify core package imports."""
    import core
    assert core is not None


def test_dataset_import():
    """Verify dataset package imports."""
    import dataset
    assert dataset is not None


def test_model_import():
    """Verify model package imports."""
    import model
    assert model is not None


def test_evaluation_import():
    """Verify evaluation package imports."""
    import evaluation
    assert evaluation is not None


def test_smoke_pass():
    """Trivial passing test to confirm pytest harness works."""
    assert True
