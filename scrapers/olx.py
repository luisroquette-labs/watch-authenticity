"""Pure OLX adapter for pre-fetched, policy-approved records."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable, normalize_candidate_metadata


def olx_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if records is None:
        return SourceUnavailable("olx", "no approved API or partner export was supplied")
    return tuple(normalize_candidate_metadata("olx", record) for record in records)
