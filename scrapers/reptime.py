"""Pure RepTime adapter for pre-fetched, rights-cleared records."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable, normalize_candidate_metadata


def reptime_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if records is None:
        return SourceUnavailable("reptime", "no approved API, feed or partner export was supplied")
    return tuple(normalize_candidate_metadata("reptime", record) for record in records)
