"""Known seller images copied from official brand media."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@lru_cache(maxsize=1)
def _quarantined_hashes() -> frozenset[str]:
    path = Path(__file__).with_name("official_media_reuse.json")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or set(document) != {"schema_version", "records"}:
        raise ValueError("official media reuse ledger schema is invalid")
    records = document["records"]
    if document["schema_version"] != 1 or not isinstance(records, list):
        raise ValueError("official media reuse ledger version is invalid")
    hashes: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "raw_sha256", "derivative_sha256", "evidence_file"
        }:
            raise ValueError("official media reuse ledger record is invalid")
        if not isinstance(record["evidence_file"], str) or not record["evidence_file"]:
            raise ValueError("official media reuse ledger evidence is missing")
        for name in ("raw_sha256", "derivative_sha256"):
            digest = record[name]
            if digest is None and name == "derivative_sha256":
                continue
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("official media reuse ledger hash is invalid")
            hashes.add(digest)
    return frozenset(hashes)


def is_known_official_media_reuse(raw_sha256: str, derivative_sha256: str) -> bool:
    return raw_sha256 in _quarantined_hashes() or derivative_sha256 in _quarantined_hashes()
