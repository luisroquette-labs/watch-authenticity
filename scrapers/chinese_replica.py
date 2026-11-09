"""Pure replica-source adapter with no placeholder endpoint or network path."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def chinese_replica_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable(
        "chinese_replica", "no approved API, feed or partner export was supplied"
    )
