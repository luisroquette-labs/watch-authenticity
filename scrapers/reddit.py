"""Pure Reddit adapter; collection still requires written policy authorization."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable, normalize_candidate_metadata


def reddit_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    if records is None:
        return SourceUnavailable("reddit", "no written Reddit and rightsholder-authorized feed supplied")
    return tuple(normalize_candidate_metadata("reddit", record) for record in records)
