"""S3 photo storage sync (Requirements item 9: training photos live in S3).

A thin boto3 wrapper — no confirm-gate here, unlike `infra/aws_train.py`.
The real-money risk this task guards against (CK-20) is GPU compute time
costing $5-20/round; S3 storage and transfer for a few thousand photos costs
cents and is trivially reversible, so it gets no gate.
"""

from __future__ import annotations

import boto3


def upload_photo(bucket: str, local_path: str, key: str) -> None:
    """Uploads one local photo to `bucket` under `key`."""
    boto3.client("s3").upload_file(local_path, bucket, key)


def download_photo(bucket: str, key: str, local_path: str) -> None:
    """Downloads one photo from `bucket`/`key` to `local_path`."""
    boto3.client("s3").download_file(bucket, key, local_path)
