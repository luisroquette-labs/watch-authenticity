"""Pure replica-source adapter with no placeholder endpoint or network path."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable, normalize_candidate_metadata


def chinese_replica_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if records is None:
        return SourceUnavailable(
            "chinese_replica", "no approved API, feed or partner export was supplied"
        )
    return tuple(normalize_candidate_metadata("chinese_replica", record) for record in records)
