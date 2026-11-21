# Keep training image builds outside routine CI

## Context

The routine GitHub check validates Python code and the acquisition path. Training image runtime behavior needs an actual image environment.

## Decision

Run Ruff, mypy, and the regular Python tests in `scripts/preflight-ci.sh`. Exclude `tests/test_training_image_runtime.py` from that check because it requires an image environment.

## Consequence

A green pull request check does not prove the training image runtime. That runtime remains unverified until the image-specific test runs against the built image.
