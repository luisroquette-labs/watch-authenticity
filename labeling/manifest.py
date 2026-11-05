"""Versioned manifest I/O over `core.types.ManifestRecord` (CK-10).

Every scraped photo must pass through here before it is eligible for
training: this module is the single call site into `core.grade_map` for
fake-grade assignment (no scraper or dataset stage may re-derive its own
factory/price -> grade mapping — CK-22), and it persists rows as append-only
newline-delimited JSON so a new dataset version is a new file, never an
in-place rewrite of a prior version's rows (reproducibility/audit trail,
SKILL.md:135).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from core.rules import grade_map
from core.types import Label, ManifestRecord

# Each dataset version is its own file, not a flag on a row — this is what
# makes "append a version" structurally unable to mutate an earlier version.
_MANIFEST_FILENAME_TEMPLATE = "manifest_v{version}.jsonl"


def manifest_path(manifest_dir: Path, version: int) -> Path:
    """Returns the file path for one versioned manifest."""
    return manifest_dir / _MANIFEST_FILENAME_TEMPLATE.format(version=version)


def build_fake_row(
    *,
    path: str,
    source: str,
    url: str,
    phash: str,
    scraped_at: date,
    factory: str | None = None,
    price: float | None = None,
    blur_score: float | None = None,
) -> ManifestRecord:
    """Builds a FAKE manifest row, assigning its grade via `core.grade_map`.

    This is the only place a fake photo's grade is decided (CK-22): callers
    supply the raw `factory`/`price` signal and get back a complete,
    already-graded `ManifestRecord` — no stage downstream re-implements
    the factory/price -> grade mapping itself.
    """
    return ManifestRecord(
        path=path,
        source=source,
        url=url,
        label=Label.FAKE,
        grade=grade_map(factory=factory, price=price),
        phash=phash,
        scraped_at=scraped_at,
        blur_score=blur_score,
    )


def append(record: ManifestRecord, *, manifest_dir: Path, version: int) -> Path:
    """Appends one row to a versioned manifest file, never overwriting it.

    Creates `manifest_dir` and the version's file on first write. Appending
    rather than rewriting is what keeps every row ever recorded for this
    version on the audit trail (CK-10).
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    target = manifest_path(manifest_dir, version)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_json_row(record)) + "\n")
    return target


# The Expected Output names both "write_row" and "append" for this
# operation; `append` is the one implementation, `write_row` its alias.
write_row = append


def read_manifest(manifest_dir: Path, version: int) -> list[ManifestRecord]:
    """Reads every row of one manifest version back into `ManifestRecord`s.

    Returns an empty list for a version that was never written, rather
    than raising — a not-yet-created version is a normal, not an error,
    state for a dataset that versions by file.
    """
    target = manifest_path(manifest_dir, version)
    if not target.exists():
        return []

    lines = target.read_text(encoding="utf-8").splitlines()
    return [_from_json_row(json.loads(line)) for line in lines if line]


def _to_json_row(record: ManifestRecord) -> dict:
    """Serializes a `ManifestRecord` to a JSON-compatible dict."""
    row = asdict(record)
    row["label"] = record.label.value
    row["scraped_at"] = record.scraped_at.isoformat()
    return row


def _from_json_row(row: dict) -> ManifestRecord:
    """Deserializes one JSON row back into a `ManifestRecord`."""
    return ManifestRecord(
        path=row["path"],
        source=row["source"],
        url=row["url"],
        label=Label(row["label"]),
        grade=row["grade"],
        phash=row["phash"],
        scraped_at=date.fromisoformat(row["scraped_at"]),
        blur_score=row.get("blur_score"),
    )
