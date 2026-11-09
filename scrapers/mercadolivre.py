"""Pure Mercado Livre adapter; only pre-fetched official API records are accepted."""

from collections.abc import Iterable, Mapping
from typing import Literal

from acquisition.collect import CandidateMetadata, SourceUnavailable, normalize_candidate_metadata

IngestMethod = Literal["scrape", "official_api"]


def mercadolivre_ingest(
    method: IngestMethod,
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if method == "scrape":
        raise PermissionError("Mercado Livre raw scraping is prohibited; use its official API")
    if method != "official_api":
        raise ValueError(f"unknown Mercado Livre ingestion method: {method!r}")
    if records is None:
        return SourceUnavailable("mercadolivre", "no approved official API feed was supplied")
    return tuple(normalize_candidate_metadata("mercadolivre", record) for record in records)
