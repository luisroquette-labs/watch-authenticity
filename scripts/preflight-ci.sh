#!/bin/bash
set -e

python3 -m ruff check .
python3 -m mypy . --config-file pyproject.toml
# GitHub must not build images; the acquisition diff cannot affect the training image.
python3 -m pytest tests/ --ignore=tests/test_training_image_runtime.py
