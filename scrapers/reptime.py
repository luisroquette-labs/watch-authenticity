"""RepTime stays unavailable until a rights-cleared export is supplied."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def reptime_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable("reptime", "no approved API, feed or partner export was supplied")
