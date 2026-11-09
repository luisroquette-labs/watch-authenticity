"""Pure Reddit adapter; collection still requires written policy authorization."""

from collections.abc import Iterable, Mapping

from acquisition.collect import CandidateMetadata, SourceUnavailable


def reddit_ingest(
    records: Iterable[Mapping[str, object]] | None = None,
) -> tuple[CandidateMetadata, ...] | SourceUnavailable:
    del records
    return SourceUnavailable(
        "reddit", "no written Reddit and rightsholder-authorized feed supplied"
    )
