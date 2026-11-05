#!/bin/bash
set -e

if [ -x .venv/bin/ruff ]; then
    PATH="$PWD/.venv/bin:$PATH"
fi

echo "=== Phase 1: Code formatting check (ruff) ==="
ruff check .

echo ""
echo "=== Phase 2: Type checking (mypy) ==="
mypy . --config-file pyproject.toml

echo ""
echo "=== Phase 3: Running test suite (pytest) ==="
pytest tests/

echo ""
echo "✓ Preflight complete."
