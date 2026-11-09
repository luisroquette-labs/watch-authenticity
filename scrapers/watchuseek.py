"""Watchuseek stays unavailable until a licensed feed contract is supplied."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def watchuseek_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable("watchuseek", "no approved API, feed or partner export was supplied")
