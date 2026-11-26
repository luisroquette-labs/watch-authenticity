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

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import false_reassurance_upper95

_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARTITIONS = ("development", "calibration", "test")
_COHORTS = ("representative", "stress")
_LABELS = ("fake", "original")
_METHODS = ("physical_examination", "specimen_specific_document", "manufacturer_service")
_ROUTES = ("previous_policy", "gemini_only", "with_review")


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
        and _text(case.get("generation"))
        and record.get("generation") == case["generation"]
        and record.get("label") == case["label"]
        and record.get("photo_sha256s") == photos
        and _text(record.get("reviewer_id"))
        and record.get("verification_method") in _METHODS
        and _sha256(record.get("evidence_sha256"))
        and _sha256(record.get("physical_identity_sha256"))
    )


def census(
    manifest: dict[str, Any],
    *,
    confirmation_bytes: bytes | None = None,
    confirmation_sha256: str | None = None,
) -> dict[str, Any]:
    result, _ = _census(manifest, confirmation_bytes=confirmation_bytes,
                        confirmation_sha256=confirmation_sha256)
    return result


def _census(
    manifest: dict[str, Any],
    *,
    confirmation_bytes: bytes | None,
    confirmation_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    records = _load_confirmations(confirmation_bytes, confirmation_sha256)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    freeze = manifest.get("freeze")
    cases = manifest.get("cases")
    if not isinstance(freeze, dict):
        raise ValueError("manifest requires a freeze object")
    models = freeze.get("models")
    if (
        not isinstance(models, dict) or set(models) != {"gemini", "gpt"}
        or not all(_text(model) for model in models.values())
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
    specimens_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
    case_keys: dict[str, list[tuple[str, str]]] = {}
    partitions_by_case = {case["specimen_id"]: case.get("partition") for case in cases}
    independent_key_owners: Counter[tuple[str, str]] = Counter()
    confirmed: set[str] = set()
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
        for kind in ("prepared_request", "prepared_media"):
            digest = case.get(f"{kind}_sha256")
            if _sha256(digest):
                keys.append((kind, digest))
        if isinstance(clusters, list):
            keys.extend(("provenance", value) for value in clusters if _text(value))
        confirmation = records.get(specimen, {})
        if _is_confirmed(case, confirmation):
            confirmed.add(specimen)
            keys.extend((kind, confirmation[field]) for kind, field in (
                ("physical_identity", "physical_identity_sha256"), ("evidence", "evidence_sha256"),
            ))
            independent_key_owners.update(
                key for key in keys if key[0] in (
                    "photo", "prepared_request", "prepared_media", "physical_identity", "evidence",
                )
            )
        case_keys[specimen] = keys
        for key in keys:
            specimens_by_key[key].add(specimen)
            if partition in _PARTITIONS:
                partitions_by_key[key].add(partition)

    cross_partition_members: set[str] = set()
    group_by_specimen: dict[str, str] = {}
    remaining = set(case_keys)
    for specimen in case_keys:
        if specimen not in remaining:
            continue
        component = {specimen}
        frontier = [specimen]
        visited_keys: set[tuple[str, str]] = set()
        while frontier:
            current = frontier.pop()
            for key in case_keys[current]:
                if key not in visited_keys:
                    visited_keys.add(key)
                    unseen = specimens_by_key[key] - component
                    component.update(unseen)
                    frontier.extend(unseen)
        remaining.difference_update(component)
        group = min(component)
        group_by_specimen.update((member, group) for member in component)
        if len({partitions_by_case[member] for member in component
                if partitions_by_case[member] in _PARTITIONS}) > 1:
            cross_partition_members.update(component)

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
        if not _text(case.get("label_source")):
            gaps.append("missing_label_source")
        if not _text(case.get("split_family")) or not isinstance(clusters, list) or not clusters or not all(
            _text(value) for value in clusters
        ):
            gaps.append("missing_provenance_or_split_family")
        direct_leaks = {
            key[0] for key in case_keys[specimen] if len(partitions_by_key[key]) > 1
        }
        blocked.extend(f"cross_partition_{kind}" for kind in sorted(direct_leaks))
        if not direct_leaks and specimen in cross_partition_members:
            blocked.append("cross_partition_component")
        if specimen in confirmed:
            duplicates = {key[0] for key in case_keys[specimen] if independent_key_owners[key] > 1}
            blocked.extend(f"{kind}_shared_by_independent_specimens" for kind in sorted(duplicates))
        for provider in sorted(models):
            permission = rights.get(provider) if isinstance(rights, dict) else None
            if permission is not True:
                state = "denied" if permission is False else "missing"
                blocked.append(f"rights_{state}:{provider}")
        if evidence != "independent":
            gaps.append("declared_label" if evidence == "declared" else "control_only")
        elif specimen not in confirmed:
            gaps.append("missing_or_mismatched_confirmation")
        if evidence == "independent" and not _sha256(case.get("prepared_request_sha256")):
            gaps.append("missing_prepared_request_sha256")
        if evidence == "independent" and not _sha256(case.get("prepared_media_sha256")):
            gaps.append("missing_prepared_media_sha256")
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
        counts[status]["sources"][case["label_source"] if _text(case.get("label_source")) else "unknown"] += 1
        counts[status]["gaps"].update(blocked + gaps)
        if status == "eligible" and case["partition"] == "test":
            test_labels[cohort][(reference, generation)].add(label)

    return ({
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
    }, group_by_specimen)


def score(
    cases: list[dict[str, Any]],
    reports: dict[str, str],
) -> dict[str, Any]:
    if any(case.get("evidence") == "independent" for case in cases):
        raise ValueError("independent cases require a frozen test manifest")
    return _score_cases(cases, reports)


def score_frozen(
    manifest_bytes: bytes,
    run_bytes: bytes,
    *,
    manifest_sha256: str,
    run_sha256: str,
    confirmation_bytes: bytes,
    confirmation_sha256: str,
    cohort: str,
) -> dict[str, Any]:
    if not _sha256(manifest_sha256) or hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("test manifest does not match the frozen SHA-256")
    if cohort not in _COHORTS:
        raise ValueError("unknown test cohort")
    if not _sha256(run_sha256) or hashlib.sha256(run_bytes).hexdigest() != run_sha256:
        raise ValueError("run receipt does not match the trusted SHA-256")
    manifest = _read_json(manifest_bytes)
    admitted, group_by_specimen = _census(manifest, confirmation_bytes=confirmation_bytes,
                                         confirmation_sha256=confirmation_sha256)
    eligible = {case["specimen_id"] for case in admitted["cases"] if case["status"] == "eligible"}
    test_cases = [
        case for case in manifest["cases"]
        if case["partition"] == "test" and case["cohort"] == cohort
        and case["specimen_id"] in eligible
    ]
    run = _read_json(run_bytes)
    if (
        not isinstance(run, dict) or run.get("capture_mode") != "live_provider"
        or run.get("manifest_sha256") != manifest_sha256
        or run.get("confirmations_sha256") != confirmation_sha256
        or run.get("cohort") != cohort or run.get("freeze") != manifest["freeze"]
        or run.get("route") not in _ROUTES or not isinstance(run.get("reports"), dict)
    ):
        raise ValueError("run receipt does not match the live frozen test")
    reports: dict[str, str] = {}
    cases_by_id = {case["specimen_id"]: case for case in test_cases}
    models = manifest["freeze"]["models"]
    gemini_ids: set[str] = set()
    gpt_ids: set[str] = set()
    for specimen, receipt in run["reports"].items():
        case = cases_by_id.get(specimen)
        if case is None:
            raise ValueError("run receipt refers to an unknown specimen")
        if (
            not isinstance(receipt, dict)
            or receipt.get("photo_sha256s") != _photo_hashes(case)
            or receipt.get("prepared_request_sha256") != case["prepared_request_sha256"]
            or receipt.get("gemini_media_sha256") != case["prepared_media_sha256"]
            or receipt.get("technical_facts_sha256") != manifest["freeze"]["technical_sources_sha256"]
            or receipt.get("gemini_model") != models["gemini"]
            or not _sha256(receipt.get("gemini_response_sha256"))
            or not _text(receipt.get("gemini_response_id"))
            or receipt["gemini_response_id"] in gemini_ids
        ):
            raise ValueError("run receipt does not match the specimen or Gemini capture")
        linkage = receipt.get("gemini_linkage")
        if linkage not in ("single_photo", "consistent", "mixed", "uncertain"):
            raise ValueError("run receipt requires the Gemini photo linkage")
        conclusion = receipt.get("conclusion")
        if not isinstance(conclusion, str):
            raise ValueError("run receipt requires a conclusion")
        unlinked = linkage in ("mixed", "uncertain")
        if unlinked and conclusion != "inconclusive":
            raise ValueError("unlinked photos require an inconclusive result")
        if run["route"] == "with_review" and unlinked:
            if receipt.get("review_status") != "unlinked_photos" or any(
                key.startswith("gpt_") for key in receipt
            ):
                raise ValueError("unlinked photos require an explicit no-review receipt")
        elif run["route"] == "with_review":
            if (
                receipt.get("gpt_model") != models["gpt"]
                or receipt.get("gpt_media_sha256") != case["prepared_media_sha256"]
                or receipt.get("gpt_technical_facts_sha256") != manifest["freeze"]["technical_sources_sha256"]
                or not _sha256(receipt.get("gpt_request_sha256"))
                or not _sha256(receipt.get("gpt_response_sha256"))
                or not _text(receipt.get("gpt_response_id"))
                or receipt["gpt_response_id"] in gpt_ids
            ):
                raise ValueError("run receipt lacks a matching GPT capture")
            gpt_ids.add(receipt["gpt_response_id"])
        gemini_ids.add(receipt["gemini_response_id"])
        reports[specimen] = conclusion
    return {
        **_score_cases(test_cases, reports),
        "by_reference_generation": _reference_metrics(test_cases, reports, group_by_specimen),
        "frozen_run": {"route": run["route"], "cohort": cohort,
                       "manifest_sha256": manifest_sha256, "run_sha256": run_sha256},
    }


def _reference_metrics(
    cases: list[dict[str, Any]], reports: dict[str, str], group_by_specimen: dict[str, str],
) -> list[dict[str, Any]]:
    scopes: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        scopes[(case["reference"], case["generation"])].append(case)

    results = []
    for (reference, generation), scoped_cases in sorted(scopes.items()):
        scoped_reports = {case["specimen_id"]: reports[case["specimen_id"]] for case in scoped_cases
                          if case["specimen_id"] in reports}
        metrics = _score_cases(scoped_cases, scoped_reports)["independent"]
        metrics["decision_coverage"] = (metrics["correct"] + metrics["incorrect"]) / len(scoped_cases)
        precision = {}
        for label in _LABELS:
            prediction = f"probable_{label}"
            predicted = [case for case in scoped_cases if scoped_reports.get(case["specimen_id"]) == prediction]
            total = len(predicted)
            correct = sum(case["label"] == label for case in predicted)
            groups = {group_by_specimen[case["specimen_id"]] for case in predicted}
            shared = len(groups) < total
            precision[prediction] = {
                "decisions": total,
                "correct": correct,
                "precision": correct / total if total else None,
                "lower_one_sided_95": 1 - false_reassurance_upper95(total - correct, total)
                if total and not shared else None,
                "shared_provenance": shared,
            }
        label_counts = Counter(case["label"] for case in scoped_cases)
        results.append({
            "reference": reference,
            "generation": generation,
            "specimens_by_label": {label: label_counts[label] for label in _LABELS},
            "metrics": metrics,
            "precision_by_prediction": precision,
        })
    return results


def _score_cases(cases: list[dict[str, Any]], reports: dict[str, str]) -> dict[str, Any]:
    if not isinstance(reports, dict) or not all(_text(specimen) for specimen in reports):
        raise ValueError("reports must map specimen IDs to conclusions")
    buckets = {name: {key: 0 for key in ("cases", "correct", "incorrect", "inconclusive",
                                         "compatible_but_inconclusive", "unavailable",
                                         "fake_called_original", "original_called_fake")}
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
    parser = argparse.ArgumentParser(epilog="Use census MANIFEST or frozen MANIFEST RUN_RECEIPT.")
    args_list = sys.argv[1:]
    mode = args_list[0] if args_list and args_list[0] in {"census", "frozen"} else "declared"
    if mode != "declared":
        args_list = args_list[1:]
        parser.add_argument("manifest", type=Path, help="Private paired evaluation manifest JSON")
    else:
        parser.add_argument("cases", type=Path, help="Frozen JSON list of specimens and label provenance")
    if mode == "frozen":
        parser.add_argument("run_receipt", type=Path, help="Trusted live run receipt JSON")
    elif mode == "declared":
        parser.add_argument("reports", type=Path, help="JSON object mapping specimen IDs to conclusions")
    if mode != "declared":
        parser.add_argument("--confirmations", type=Path, required=mode == "frozen",
                            help="Evaluator's separate confirmation JSON")
        parser.add_argument("--confirmations-sha256", required=mode == "frozen",
                            help="Expected SHA-256 from the trusted evaluator")
    if mode == "frozen":
        parser.add_argument("--manifest-sha256", required=True)
        parser.add_argument("--run-sha256", required=True, help="Expected run SHA-256 from the trusted evaluator")
        parser.add_argument("--cohort", choices=_COHORTS, required=True)
    args = parser.parse_args(args_list)
    if mode == "census":
        result = census(_read_json(args.manifest.read_bytes()),
                        confirmation_bytes=args.confirmations.read_bytes() if args.confirmations else None,
                        confirmation_sha256=args.confirmations_sha256)
    elif mode == "frozen":
        result = score_frozen(args.manifest.read_bytes(), args.run_receipt.read_bytes(),
                              manifest_sha256=args.manifest_sha256, cohort=args.cohort,
                              run_sha256=args.run_sha256,
                              confirmation_bytes=args.confirmations.read_bytes(),
                              confirmation_sha256=args.confirmations_sha256)
    else:
        result = score(_read_json(args.cases.read_bytes()), _read_json(args.reports.read_bytes()))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
