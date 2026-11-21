#!/bin/bash
set -e

python3 scripts/check_comments.py
python3 -m ruff check .
python3 -m mypy . --config-file pyproject.toml
python3 -m pytest tests/ --ignore=tests/test_training_image_runtime.py
