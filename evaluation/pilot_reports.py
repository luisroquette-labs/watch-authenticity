"""Score frozen photo reports per physical specimen, separately from source claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score(cases: list[dict[str, Any]], reports: dict[str, str]) -> dict[str, Any]:
    buckets = {name: {key: 0 for key in ("cases", "correct", "incorrect", "inconclusive",
                                         "unavailable", "fake_called_original", "original_called_fake")}
               for name in ("independent", "declared")}
    seen: set[str] = set()
    details = []
    for case in cases:
        specimen = case["specimen_id"]
        label = case["label"]
        evidence = case["evidence"]
        if (not isinstance(specimen, str) or not specimen.strip() or specimen in seen
            or label not in ("fake", "original") or evidence not in buckets
            or not isinstance(case.get("label_source"), str) or not case["label_source"].strip()):
            raise ValueError("requires unique specimen IDs, label, evidence and provenance")
        if evidence == "independent" and case.get("physical_identity_confirmed") is not True:
            raise ValueError("independent cases require confirmed physical identity")
        seen.add(specimen)
        actual = reports.get(specimen, "unavailable")
        bucket = buckets[evidence]
        bucket["cases"] += 1
        if actual in ("inconclusive", "unavailable"):
            outcome = actual
        elif actual in ("probable_fake", "probable_original"):
            outcome = "correct" if actual == f"probable_{label}" else "incorrect"
            if outcome == "incorrect":
                bucket["fake_called_original" if label == "fake" else "original_called_fake"] += 1
        else:
            raise ValueError("unsupported report conclusion")
        bucket[outcome] += 1
        details.append({"specimen_id": specimen, "evidence": evidence, "label": label,
                        "report": actual, "outcome": outcome})
    if set(reports) - seen:
        raise ValueError("report refers to an unknown specimen")
    independent = buckets["independent"]
    declared = buckets["declared"]
    return {"independent": {**independent, "accuracy_all_cases": (
                independent["correct"] / independent["cases"] if independent["cases"] else None)},
            "declared": {k: v for k, v in declared.items() if k not in {"correct", "incorrect"}}
            | {"agrees_with_claim": declared["correct"], "disagrees_with_claim": declared["incorrect"]},
            "specimens": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, help="Frozen JSON list of specimens and label provenance")
    parser.add_argument("reports", type=Path, help="JSON object mapping specimen IDs to conclusions")
    args = parser.parse_args()
    print(json.dumps(score(json.loads(args.cases.read_bytes()), json.loads(args.reports.read_bytes())),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
