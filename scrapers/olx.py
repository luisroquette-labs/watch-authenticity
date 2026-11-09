"""OLX stays unavailable until an approved inventory export is supplied."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def olx_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable("olx", "no approved API or partner export was supplied")
