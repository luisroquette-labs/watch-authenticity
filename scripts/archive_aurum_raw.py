"""Validate or upload originals: python -m scripts.archive_aurum_raw MANIFEST [--execute]."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from acquisition.raw_archive import archive_aurum_raw
from infra.ibm_cos import create_oauth_cos_client, verify_private_versioned_bucket

ACCOUNT_GUID = "df26b5a1b07d4825957bb3fee3326ead"
BUCKET = "watch-authenticity-e2e-3235996-20261106"
INSTANCE = "watch-authenticity-cos"


def _ibm_json(*args: str):
    result = subprocess.run(["ibmcloud", *args, "--output", "json"], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def _token():
    value = _ibm_json("iam", "oauth-tokens")["iam_token"]
    if not isinstance(value, str) or not value.startswith("Bearer "):
        raise RuntimeError("IBM CLI did not return a bearer token")
    token = value.removeprefix("Bearer ")
    encoded = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    remaining = int(payload["exp"] - time.time())
    if remaining < 60:
        raise RuntimeError("IBM CLI token has less than 60 seconds remaining")
    return {"access_token": token, "token_type": "Bearer", "expires_in": remaining}


def _client():
    target = _ibm_json("target")
    if target["account"]["guid"] != ACCOUNT_GUID or target["region"]["name"] != "us-south":
        raise RuntimeError("IBM CLI target is not the approved account and region")
    instances = _ibm_json("resource", "service-instance", INSTANCE)
    matches = [item for item in instances if item.get("name") == INSTANCE and item.get("state") == "active"]
    if len(matches) != 1:
        raise RuntimeError("approved COS instance is unavailable or ambiguous")
    authenticator = SimpleNamespace(token_manager=SimpleNamespace(request_token=_token))
    client = create_oauth_cos_client(
        authenticator,
        endpoint_url="https://s3.us-south.cloud-object-storage.appdomain.cloud",
        service_instance_id=matches[0]["id"],
    )
    verify_private_versioned_bucket(client, BUCKET)
    setattr(client, "_watch_auth_bucket", BUCKET)
    setattr(client, "_watch_auth_prefix", "datasets")
    setattr(client, "_watch_auth_configuration_verified", True)
    return client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--execute", action="store_true", help="upload verified originals and receipt to IBM COS")
    args = parser.parse_args()
    result = archive_aurum_raw(
        args.manifest,
        cos_client=_client() if args.execute else None,
        bucket=BUCKET,
        dry_run=not args.execute,
    )
    json.dump(result, sys.stdout, ensure_ascii=False, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
