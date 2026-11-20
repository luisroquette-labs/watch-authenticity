"""Stream pinned Hugging Face pilot assets into private IBM COS without local files."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

PART_BYTES = 16 * 1024 * 1024
BUCKET_PREFIX = "pilot/model-sources"


def _client(endpoint: str, instance_id: str) -> Any:
    import ibm_boto3
    from ibm_botocore.client import Config
    from ibm_botocore.credentials import DefaultTokenManager

    def token() -> dict[str, object]:
        result = subprocess.run(
            ["ibmcloud", "iam", "oauth-tokens", "--output", "json"],
            capture_output=True, text=True, check=True,
        )
        access_token = json.loads(result.stdout)["iam_token"].removeprefix("Bearer ")
        payload = access_token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        remaining = int(claims["exp"]) - int(time.time())
        if remaining < 120:
            raise RuntimeError("IBM CLI session expires before model transfer")
        return {"access_token": access_token, "token_type": "Bearer", "expires_in": remaining}

    return ibm_boto3.client(
        "s3", endpoint_url=endpoint, ibm_service_instance_id=instance_id,
        token_manager=DefaultTokenManager(auth_function=token),
        config=Config(signature_version="oauth"),
    )


def _hashes(spec: dict[str, Any]) -> tuple[Any, Any]:
    size = spec["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("source file size is invalid")
    source = (
        hashlib.sha256() if spec["digest_type"] == "sha256"
        else hashlib.sha1(f"blob {size}\0".encode())
        if spec["digest_type"] == "git-sha1" else None
    )
    if source is None:
        raise ValueError("source digest type is invalid")
    return source, hashlib.sha256()


def _verify_remote(client: Any, bucket: str, key: str, spec: dict[str, Any]) -> str:
    response = client.get_object(Bucket=bucket, Key=key)
    source, sha256 = _hashes(spec)
    count = 0
    body = response["Body"]
    try:
        for chunk in body.iter_chunks(chunk_size=1024 * 1024):
            source.update(chunk)
            sha256.update(chunk)
            count += len(chunk)
    finally:
        body.close()
    if count != spec["size"] or source.hexdigest() != spec["digest"]:
        raise RuntimeError(f"COS model asset failed readback: {key}")
    return sha256.hexdigest()


def _reconcile_uploads(client: Any, bucket: str, key: str) -> None:
    """Abort only expired uploads for this exact asset; fail if another writer is active."""
    result = client.list_multipart_uploads(Bucket=bucket, Prefix=key)
    if result.get("IsTruncated"):
        raise RuntimeError("incomplete COS multipart inventory")
    now = datetime.now(timezone.utc)
    for upload in result.get("Uploads", []):
        if upload["Key"] != key:
            continue
        started = upload["Initiated"]
        if not isinstance(started, datetime) or started.tzinfo is None:
            raise RuntimeError("COS multipart start time is invalid")
        if now - started < timedelta(hours=1):
            raise RuntimeError(f"another model upload is still active: {key}")
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload["UploadId"])


def stage_file(client: Any, bucket: str, model: dict[str, Any], spec: dict[str, Any]) -> str:
    name, revision, path = model["name"], model["revision"], spec["path"]
    if not all(isinstance(value, str) and value and ".." not in value and "\\" not in value
               for value in (name, revision, path)) or path.startswith("/"):
        raise ValueError("unsafe model source path")
    key = f"{BUCKET_PREFIX}/{name}/{revision}/{path}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except client.exceptions.NoSuchKey:
        head = None
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") != "404":
            raise
        head = None
    if head is not None:
        if head.get("ContentLength") == spec["size"] and head.get("Metadata", {}).get("source-digest") == spec["digest"]:
            try:
                return _verify_remote(client, bucket, key, spec)
            except RuntimeError:
                pass  # A new version is safe because main verified COS versioning first.
        if client.get_bucket_versioning(Bucket=bucket).get("Status") != "Enabled":
            raise RuntimeError(f"corrupt model asset cannot be repaired without COS versioning: {key}")

    url = f"https://huggingface.co/{model['repository']}/resolve/{revision}/{path}"
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        if response.headers.get("Content-Length") and int(response.headers["Content-Length"]) != spec["size"]:
            raise RuntimeError(f"source model asset size differs: {path}")
        source, sha256 = _hashes(spec)
        count = 0
        metadata = {"source-digest": str(spec["digest"]), "digest-type": str(spec["digest_type"])}
        if spec["size"] < PART_BYTES:
            body = response.content
            count = len(body)
            source.update(body)
            sha256.update(body)
            if count != spec["size"] or source.hexdigest() != spec["digest"]:
                raise RuntimeError(f"source model asset digest differs: {path}")
            client.put_object(Bucket=bucket, Key=key, Body=body, Metadata=metadata)
        else:
            _reconcile_uploads(client, bucket, key)
            upload = client.create_multipart_upload(Bucket=bucket, Key=key, Metadata=metadata)
            parts: list[dict[str, object]] = []
            buffer = bytearray()
            try:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    source.update(chunk)
                    sha256.update(chunk)
                    count += len(chunk)
                    buffer.extend(chunk)
                    if len(buffer) >= PART_BYTES:
                        part = client.upload_part(
                            Bucket=bucket, Key=key, UploadId=upload["UploadId"],
                            PartNumber=len(parts) + 1, Body=bytes(buffer),
                        )
                        parts.append({"ETag": part["ETag"], "PartNumber": len(parts) + 1})
                        buffer.clear()
                if count != spec["size"] or source.hexdigest() != spec["digest"]:
                    raise RuntimeError(f"source model asset digest differs: {path}")
                if buffer:
                    part = client.upload_part(
                        Bucket=bucket, Key=key, UploadId=upload["UploadId"],
                        PartNumber=len(parts) + 1, Body=bytes(buffer),
                    )
                    parts.append({"ETag": part["ETag"], "PartNumber": len(parts) + 1})
                client.complete_multipart_upload(
                    Bucket=bucket, Key=key, UploadId=upload["UploadId"],
                    MultipartUpload={"Parts": parts},
                )
            except Exception:
                client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload["UploadId"])
                raise
    if _verify_remote(client, bucket, key, spec) != sha256.hexdigest():
        raise RuntimeError(f"COS model asset changed after upload: {key}")
    return sha256.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--model", choices=("smolvlm", "owlv2"), required=True)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "config/pilot_model_sources.json").read_text())
    model = next(item for item in manifest["models"] if item["name"] == args.model)
    client = _client(args.endpoint, args.instance_id)
    from infra.ibm_cos import verify_private_versioned_bucket

    verify_private_versioned_bucket(client, args.bucket)
    for spec in model["files"]:
        digest = stage_file(client, args.bucket, model, spec)
        print(json.dumps({"model": args.model, "path": spec["path"], "sha256": digest}), flush=True)


if __name__ == "__main__":
    main()
