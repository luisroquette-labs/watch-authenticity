#!/bin/bash
set -e

ruff check .
mypy . --config-file pyproject.toml
# GitHub must not build images; the acquisition diff cannot affect the training image.
pytest tests/ --ignore=tests/test_training_image_runtime.py
