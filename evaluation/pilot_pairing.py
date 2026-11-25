"""Verify pairing before comparing outcomes from three watch report routes."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Callable, cast

from pilot.flash_report import FlashReport
from pilot.flash_research import MODEL


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ReplayInput:
    case_id: str
    photo_sha256s: tuple[str, ...]
    prepared_request_sha256: str
    technical_facts_sha256: str
    gemini_model: str
    gemini_response_sha256: str
    gemini_response_id: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        ids = (self.case_id, self.gemini_model, self.gemini_response_id)
        if any(not isinstance(value, str) or not value.strip() for value in ids):
            raise ValueError("case, model and response IDs are required")
        if not isinstance(self.photo_sha256s, tuple) or not self.photo_sha256s:
            raise ValueError("at least one ordered photo digest is required")
        digests = (*self.photo_sha256s, self.prepared_request_sha256,
                   self.technical_facts_sha256, self.gemini_response_sha256, self.prompt_sha256)
        if any(not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
               for digest in digests):
            raise ValueError("capture digests must be lowercase SHA-256")


def verify_paired_inputs(
    previous_policy: ReplayInput, gemini_only: ReplayInput, with_review: ReplayInput,
) -> dict[str, str | int | bool]:
    if gemini_only != with_review:
        raise ValueError("candidate arms must reuse the same Gemini capture")
    if (previous_policy.case_id != gemini_only.case_id
        or previous_policy.photo_sha256s != gemini_only.photo_sha256s):
        raise ValueError("previous policy must use the same case and ordered photos")
    return {
        "case_id": gemini_only.case_id,
        "photo_count": len(gemini_only.photo_sha256s),
        "baseline_uses_same_gemini_capture": previous_policy == gemini_only,
        "gemini_response_reused": True,
    }


def replay_with_review(
    report_builder: FlashReport,
    case_id: str,
    photos: list[tuple[str, str, bytes]],
    gemini: Callable[[list[dict[str, object]]], tuple[object, dict[str, object]]],
    review: Callable[[dict[str, Any]], object],
) -> dict[str, Any]:
    """Rehearse the three arms with injected responses and one Gemini capture."""
    content, indices = report_builder.prepare(photos)
    facts: dict[str, dict[str, Any]] = {}
    for claims in report_builder.claims.values():
        for claim in claims:
            if claim.source_class.authoritative:
                for angle in claim.applicable_angles:
                    fact = {"reference_id": claim.reference_id, "angle": angle.value,
                            "field": claim.field, "expected": claim.value,
                            "url": claim.source_url, "citation": claim.citation}
                    facts[_digest(fact)] = fact
    prompt = f"{content[0]['text']}\nFatos técnicos oficiais:\n{json.dumps(facts, ensure_ascii=False, sort_keys=True)}"
    content[0] = {"type": "text", "text": prompt}
    answer, metadata = gemini(deepcopy(content))
    model_response_id = metadata.get("model_response_id")
    if metadata.get("model") != MODEL or not isinstance(model_response_id, str):
        raise ValueError("the Gemini response must identify the pinned model and response")
    report = report_builder.build_report(answer, indices)
    checks = cast(list[dict[str, Any]], report["technical_checks"])
    fields: dict[str, dict[str, Any]] = {}
    expected_statuses: dict[str, set[str]] = {}
    candidates = cast(list[dict[str, Any]], report["candidates"])
    for check in checks:
        fact = {"reference_id": candidates[0]["reference_id"],
                **{name: check[name] for name in ("angle", "field", "expected", "url", "citation")}}
        fact_id = _digest(fact)
        if fact_id not in facts:
            raise ValueError("technical check lacks a supplied catalog fact")
        field = {"photo_id": check["photo"], "field": check["field"], "source_fact_id": fact_id}
        claim_id = _digest(field)
        fields[claim_id] = field
        expected_statuses.setdefault(claim_id, set()).add(
            "supported" if check["status"] == "matched" else "contradicted",
        )
    capture = ReplayInput(
        case_id=case_id,
        photo_sha256s=tuple(hashlib.sha256(raw).hexdigest() for _, _, raw in photos),
        prepared_request_sha256=_digest(content),
        technical_facts_sha256=_digest(facts),
        gemini_model=MODEL,
        gemini_response_sha256=_digest(answer),
        gemini_response_id=model_response_id,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    photo_ids = sorted(indices)
    images = [cast(dict[str, str], item["image_url"]) for item in content
              if item["type"] == "image_url"]
    prepared_photos = [
        {"photo_id": photo_id, "prepared_sha256": hashlib.sha256(
            base64.b64decode(image["url"].removeprefix("data:image/jpeg;base64,"), validate=True),
        ).hexdigest()}
        for photo_id, image in zip(photo_ids, images, strict=True)
    ]
    linked = report["linkage"] in ("single_photo", "consistent")
    hypothesis = report["model_hypothesis"] if linked else "inconclusive"
    result = {
        "capture": capture,
        "prepared_photos": prepared_photos,
        "previous_policy": report["conclusion"],
        "previous_policy_reason": report["decision_reason"],
        "gemini_hypothesis": report["model_hypothesis"],
        "linkage": report["linkage"],
        "gemini_only": hypothesis,
        "with_review": "unavailable",
        "review_status": "no_comparable_fields",
    }
    if not linked:
        return {**result, "with_review": "inconclusive", "review_status": "unlinked_photos"}
    if not fields:
        return result
    token = _digest(asdict(capture))
    request = {
        "case_token": token,
        "photo_ids": photo_ids,
        "photos": [
            {"photo_id": photo_id, "image_url": image}
            for photo_id, image in zip(photo_ids, images, strict=True)
        ],
        "facts": facts,
        "review_fields": fields,
    }
    response = review(deepcopy(request))
    if response is None:
        return {**result, "review_status": "unavailable"}
    if not isinstance(response, dict) or response.get("case_token") != token:
        raise ValueError("review response belongs to a different case")
    findings = response.get("findings")
    if not isinstance(findings, list) or len(findings) != len(fields):
        raise ValueError("review must return one finding per requested field")
    findings_by_claim: dict[str, dict[str, Any]] = {}
    disputed_claim_ids: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict) or not isinstance(finding.get("claim_id"), str):
            raise ValueError("review finding requires a claim ID")
        claim_id = finding["claim_id"]
        target = fields.get(claim_id)
        status = finding.get("status")
        reading = finding.get("reading")
        photo_id = finding.get("photo_id")
        if (
            claim_id in findings_by_claim or target is None
            or not isinstance(photo_id, int) or isinstance(photo_id, bool)
            or photo_id != target["photo_id"]
            or finding.get("source_fact_id") != target["source_fact_id"]
            or status not in ("supported", "weak", "contradicted", "not_visible")
            or not isinstance(reading, str) or not reading.strip() or len(reading) > 360
        ):
            raise ValueError("review finding does not match its requested evidence")
        findings_by_claim[claim_id] = {**target, "status": status, "reading": reading.strip()}
        if expected_statuses[claim_id] != {status}:
            disputed_claim_ids.append(claim_id)
    return {
        **result,
        "with_review": "inconclusive" if disputed_claim_ids else hypothesis,
        "review_status": "needs_verification" if disputed_claim_ids else "supported",
        "review_findings": findings_by_claim,
        "disputed_claim_ids": disputed_claim_ids,
    }
