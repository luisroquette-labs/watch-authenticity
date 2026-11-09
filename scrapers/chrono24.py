"""Chrono24 stays unavailable until a licensed feed contract is supplied."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def chrono24_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable("chrono24", "no approved API, feed or partner export was supplied")
