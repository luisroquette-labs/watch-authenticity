"""Admit paired cases and score frozen reports per physical specimen."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, TypeGuard

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARTITIONS = ("development", "calibration", "test")
_COHORTS = ("representative", "stress")
_LABELS = ("fake", "original")
_METHODS = ("physical_examination", "specimen_specific_document", "manufacturer_service")


def _text(value: Any) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _photo_hashes(case: dict[str, Any]) -> list[str]:
    photos = case.get("photos")
    if not isinstance(photos, list) or not photos or any(
        not isinstance(photo, dict) or not _sha256(photo.get("sha256")) for photo in photos
    ):
        return []
    return [photo["sha256"] for photo in photos]


def _read_json(data: bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value = dict(pairs)
        if len(value) != len(pairs):
            raise ValueError("duplicate key in evaluation JSON")
        return value

    return json.loads(data, object_pairs_hook=unique_object)


def _load_confirmations(
    data: bytes | None, trusted_sha256: str | None,
) -> dict[str, dict[str, Any]]:
    """Use an expected digest from the evaluator's trusted channel."""
    if data is None and trusted_sha256 is None:
        return {}
    if data is None or not _sha256(trusted_sha256) or hashlib.sha256(data).hexdigest() != trusted_sha256:
        raise ValueError("confirmation file does not match the trusted SHA-256")

    records = _read_json(data)
    if not isinstance(records, dict) or any(
        not _text(specimen) or not isinstance(record, dict) for specimen, record in records.items()
    ):
        raise ValueError("confirmation file must map specimen IDs to records")
    return records


def _is_confirmed(case: dict[str, Any], record: dict[str, Any]) -> bool:
    reference = case.get("reference")
    photos = _photo_hashes(case)
    return (
        case.get("evidence") == "independent"
        and _text(reference)
        and reference.casefold() != "unknown"
        and case.get("label") in ("fake", "original")
        and bool(photos)
        and len(set(photos)) == len(photos)
        and record.get("reference") == reference
        and record.get("label") == case["label"]
        and record.get("photo_sha256s") == photos
        and record.get("verification_method") in _METHODS
        and _sha256(record.get("evidence_sha256"))
    )


def census(
    manifest: dict[str, Any],
    *,
    confirmation_bytes: bytes | None = None,
    confirmation_sha256: str | None = None,
) -> dict[str, Any]:
    records = _load_confirmations(confirmation_bytes, confirmation_sha256)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    freeze = manifest.get("freeze")
    cases = manifest.get("cases")
    if not isinstance(freeze, dict):
        raise ValueError("manifest requires a freeze object")
    models = freeze.get("models")
    if (
        not isinstance(models, dict) or len(models) != 2
        or not all(_text(provider) and _text(model) for provider, model in models.items())
        or not all(_sha256(freeze.get(key)) for key in (
            "prompt_sha256", "policy_sha256", "technical_sources_sha256", "image_treatment_sha256",
        ))
    ):
        raise ValueError("freeze requires two provider model IDs and four SHA-256 digests")
    if not isinstance(cases, list) or any(
        not isinstance(case, dict) or not _text(case.get("specimen_id")) for case in cases
    ):
        raise ValueError("manifest requires a list of cases with specimen IDs")

    if len({case["specimen_id"] for case in cases}) != len(cases):
        raise ValueError("manifest requires unique specimen IDs")
    partitions_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_keys: dict[str, list[tuple[str, str]]] = {}
    independent_photo_owners: Counter[tuple[str, str]] = Counter()
    for case in cases:
        specimen = case["specimen_id"]
        partition = case.get("partition")
        family = case.get("split_family")
        clusters = case.get("provenance_clusters")
        photos = case.get("photos")
        keys = [
            ("photo", photo["sha256"]) for photo in photos
            if isinstance(photo, dict) and _sha256(photo.get("sha256"))
        ] if isinstance(photos, list) else []
        if _text(family):
            keys.append(("split_family", family))
        if isinstance(clusters, list):
            keys.extend(("provenance", value) for value in clusters if _text(value))
        case_keys[specimen] = keys
        if partition in _PARTITIONS:
            for key in keys:
                partitions_by_key[key].add(partition)
            if _is_confirmed(case, records.get(specimen, {})):
                independent_photo_owners.update(
                    (partition, digest) for kind, digest in keys if kind == "photo"
                )

    details = []
    counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    test_labels: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(lambda: defaultdict(set))
    for case in cases:
        specimen = case["specimen_id"]
        evidence = case.get("evidence")
        reference = case.get("reference")
        label = case.get("label")
        cohort = case.get("cohort")
        generation = case.get("generation")
        photos = _photo_hashes(case)
        clusters = case.get("provenance_clusters")
        rights = case.get("rights")
        angles = {
            photo["angle"] for photo in case["photos"] if _text(photo.get("angle"))
        } if photos else set()
        blocked = []
        gaps = []
        if case.get("partition") not in _PARTITIONS or cohort not in _COHORTS:
            blocked.append("invalid_partition_or_cohort")
        if evidence not in ("independent", "declared", "control"):
            blocked.append("invalid_evidence_tier")
        if not photos or len(photos) != len(set(photos)):
            blocked.append("invalid_or_repeated_photo_sha256")
        elif any(not _text(photo.get("angle")) for photo in case["photos"]):
            gaps.append("missing_angle")
        if evidence == "independent" and any(angle.casefold() == "unknown" for angle in angles):
            gaps.append("unknown_angle")
        if not _text(case.get("capture_type")):
            gaps.append("missing_capture_type")
        if not _text(case.get("split_family")) or not isinstance(clusters, list) or not clusters or not all(
            _text(value) for value in clusters
        ):
            gaps.append("missing_provenance_or_split_family")
        for kind in sorted({
            key[0] for key in case_keys[specimen] if len(partitions_by_key[key]) > 1
        }):
            blocked.append(f"cross_partition_{kind}")
        if evidence == "independent" and case.get("partition") in _PARTITIONS and any(
            independent_photo_owners[(case["partition"], digest)] > 1 for digest in photos
        ):
            blocked.append("photo_shared_by_independent_specimens")
        for provider in sorted(models):
            permission = rights.get(provider) if isinstance(rights, dict) else None
            if permission is not True:
                state = "denied" if permission is False else "missing"
                blocked.append(f"rights_{state}:{provider}")
        if evidence != "independent":
            gaps.append("declared_label" if evidence == "declared" else "control_only")
        elif not _is_confirmed(case, records.get(specimen, {})):
            gaps.append("missing_or_mismatched_confirmation")
        if not _text(reference) or reference.casefold() == "unknown":
            gaps.append("unknown_reference")
        if evidence == "independent" and (not _text(generation) or generation.casefold() == "unknown"):
            gaps.append("unknown_generation")
        status = "blocked" if blocked else "research_only" if gaps else "eligible"
        details.append({"specimen_id": specimen, "status": status, "reasons": blocked + gaps})
        counts[status]["cases"]["total"] += 1
        counts[status]["classes"][label if label in _LABELS else "unknown"] += 1
        counts[status]["references"][reference if _text(reference) else "unknown"] += 1
        counts[status]["angles"].update(angles or {"unknown"})
        counts[status]["gaps"].update(blocked + gaps)
        if status == "eligible" and case["partition"] == "test":
            test_labels[cohort][(reference, generation)].add(label)

    return {
        "cases": details,
        "counts": counts,
        "paired_test_references": {
            cohort: [
                {"reference": reference, "generation": generation}
                for (reference, generation), labels in sorted(test_labels[cohort].items())
                if labels == set(_LABELS)
            ]
            for cohort in sorted(_COHORTS)
        },
    }


def score(
    cases: list[dict[str, Any]],
    reports: dict[str, str],
    *,
    confirmation_bytes: bytes | None = None,
    confirmation_sha256: str | None = None,
) -> dict[str, Any]:
    confirmations = _load_confirmations(confirmation_bytes, confirmation_sha256)
    buckets = {name: {key: 0 for key in ("cases", "correct", "incorrect", "inconclusive",
                                         "compatible_but_inconclusive", "unavailable",
                                         "fake_called_original", "original_called_fake")}
               for name in ("independent", "declared")}
    seen: set[str] = set()
    independent_photos: set[str] = set()
    details = []
    for case in cases:
        specimen = case["specimen_id"]
        label = case["label"]
        evidence = case["evidence"]
        if (not isinstance(specimen, str) or not specimen.strip() or specimen in seen
            or label not in ("fake", "original") or evidence not in buckets
            or not isinstance(case.get("label_source"), str) or not case["label_source"].strip()):
            raise ValueError("requires unique specimen IDs, label, evidence and provenance")
        if evidence == "independent":
            if not _is_confirmed(case, confirmations.get(specimen, {})):
                raise ValueError("independent cases require specimen-specific trusted confirmation")
            photos = _photo_hashes(case)
            if independent_photos.intersection(photos):
                raise ValueError("independent specimens cannot share photos")
            independent_photos.update(photos)
        seen.add(specimen)
        actual = reports.get(specimen, "unavailable")
        bucket = buckets[evidence]
        bucket["cases"] += 1
        if actual == "visually_compatible":
            outcome = "inconclusive"
            bucket["compatible_but_inconclusive"] += 1
        elif actual in ("inconclusive", "unavailable"):
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
    parser = argparse.ArgumentParser(epilog="Use census MANIFEST to check paired admission.")
    args_list = sys.argv[1:]
    if args_list[:1] == ["census"]:
        args_list = args_list[1:]
        parser.add_argument("manifest", type=Path, help="Private paired evaluation manifest JSON")
    else:
        parser.add_argument("cases", type=Path, help="Frozen JSON list of specimens and label provenance")
        parser.add_argument("reports", type=Path, help="JSON object mapping specimen IDs to conclusions")
    parser.add_argument("--confirmations", type=Path, help="Evaluator's separate confirmation JSON")
    parser.add_argument("--confirmations-sha256", help="Expected SHA-256 from the trusted evaluator")
    args = parser.parse_args(args_list)
    confirmation = {
        "confirmation_bytes": args.confirmations.read_bytes() if args.confirmations else None,
        "confirmation_sha256": args.confirmations_sha256,
    }
    if hasattr(args, "manifest"):
        result = census(_read_json(args.manifest.read_bytes()), **confirmation)
    else:
        result = score(_read_json(args.cases.read_bytes()), _read_json(args.reports.read_bytes()),
                       **confirmation)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
