# Reference Intelligence Implementation Plan

**Goal:** Combine visual evidence with current, reference-specific technical facts without allowing text, forums or an LLM to manufacture authenticity.

**Architecture:** A versioned knowledge file stores atomic claims with exact reference, source class, URL, citation and observation date. A pure comparison layer selects authoritative facts, compares only details visible in the submitted angles and emits matched/conflicting/unobservable checks. Technical conflicts may lower confidence; matches never raise genuine confidence. The visual classifier remains the authenticity signal.

## Source hierarchy

1. Official brand documentation: authoritative for specifications, never for the identity of the photographed watch.
2. Official retailer/service material: corroboration when tied to the exact reference.
3. Named specialist or watchmaker: corroboration or diagnostic lead with author/date.
4. Forum, marketplace or seller: lead-only; never accepted as fact or ground truth.

## Task 1: Add technical-claim contracts

- Record exact reference, field/value, source class, URL, citation, observed date and applicable image angles.
- Reject unknown references, malformed URLs, future dates and claims without provenance.
- Keep forum/marketplace claims structurally lead-only.

## Task 2: Add versioned 126610LN knowledge

- Seed only facts verified against Rolex's official product page/brochure.
- Separate visually testable facts from movement, precision and water-resistance facts that ordinary photos cannot prove.
- Preserve source retrieval status; a blocked refresh remains visible instead of silently reusing it as current.

## Task 3: Compare facts with images

- Compare only explicit structured observations produced from guided angles.
- Return matched, conflicting and unobservable checks with citations.
- Treat authoritative conflicts or absence of visible proof as `inconclusive`.
- Never increase genuine confidence from a match; visible conflicts may reduce it.

## Task 4: Integrate into inference

- Resolve reference intelligence after reference/angle identification and before the final explanation.
- Include sources, dates, discrepancies and the smallest missing angle in the response.
- Keep visual scoring, calibration and grade-4 ceiling unchanged.

## Task 5: Verify and roll out

- Test source hierarchy, stale/missing facts, conflicting official claims, forum-only claims and hallucinated fields.
- Run one live refresh against official sources; if the brand blocks automated retrieval, report `PENDENTE E2E` and retain the last verified snapshot with its date.
- Do not use a paid LLM or paid search API without current owner authorization.

## Completion proof

- A forum claim cannot become authoritative.
- A compatible spec cannot raise genuine confidence.
- A visible contradiction lowers the technical confidence modifier and cites its source.
- Missing or conflicting evidence returns `inconclusive`, never a fabricated verdict.
