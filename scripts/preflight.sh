#!/bin/bash
set -e

if [ -x .venv/bin/ruff ]; then
    PATH="$PWD/.venv/bin:$PATH"
fi

echo "=== Phase 1: Code formatting check (ruff) ==="
if command -v ruff >/dev/null; then ruff check .; else python3 -m ruff check .; fi

echo ""
echo "=== Phase 2: Type checking (mypy) ==="
if command -v mypy >/dev/null; then mypy . --config-file pyproject.toml; else python3 -m mypy . --config-file pyproject.toml; fi

echo ""
echo "=== Phase 3: Running test suite (pytest) ==="
if command -v pytest >/dev/null; then pytest tests/; else python3 -m pytest tests/; fi

echo ""
echo "✓ Preflight complete."
