"""Verify private COS model bytes in ephemeral storage before serving requests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pilot.model_asset_stage import BUCKET_PREFIX, _hashes

MODEL_DIR = Path("/tmp/watch-auth-models")


def download_models(client: Any, bucket: str, manifest: dict[str, Any], root: Path) -> None:
    for model in manifest["models"]:
        for spec in model["files"]:
            path = spec["path"]
            if not isinstance(path, str) or not path or Path(path).name != path:
                raise ValueError("unsafe model asset path")
            key = f"{BUCKET_PREFIX}/{model['name']}/{model['revision']}/{path}"
            destination = root / model["name"] / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            source, sha256 = _hashes(spec)
            count = 0
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                with temporary.open("wb") as output:
                    for chunk in body.iter_chunks(chunk_size=1024 * 1024):
                        if chunk:
                            count += len(chunk)
                            if count > spec["size"]:
                                raise RuntimeError(f"model asset exceeds pinned size: {key}")
                            source.update(chunk)
                            sha256.update(chunk)
                            output.write(chunk)
                if count != spec["size"] or source.hexdigest() != spec["digest"]:
                    raise RuntimeError(f"model asset differs from pinned source: {key}")
                if spec["digest_type"] == "sha256" and sha256.hexdigest() != spec["digest"]:
                    raise RuntimeError(f"model asset SHA256 differs: {key}")
                temporary.replace(destination)
            finally:
                body.close()
                temporary.unlink(missing_ok=True)


def main() -> None:
    from pilot.server import main as serve

    mode = os.environ.get("WATCH_AUTH_PILOT_MODE", "local-vision")
    bind = ["--host", "0.0.0.0", "--port", "8080"]
    if mode == "gemini":
        serve(["--gemini-flash", *bind])
        return
    if mode != "local-vision":
        raise SystemExit("WATCH_AUTH_PILOT_MODE must be local-vision or gemini")

    from ibm_cloud_sdk_core.authenticators import ContainerAuthenticator

    from infra.ibm_cos import create_oauth_cos_client

    bucket = os.environ["WATCH_AUTH_COS_BUCKET"]
    auth = ContainerAuthenticator(iam_profile_id=os.environ["WATCH_AUTH_TRUSTED_PROFILE_ID"])
    client = create_oauth_cos_client(
        auth,
        endpoint_url=os.environ["WATCH_AUTH_COS_ENDPOINT"],
        service_instance_id=os.environ["WATCH_AUTH_COS_SERVICE_INSTANCE_ID"],
    )
    manifest = json.loads((Path(__file__).resolve().parents[1] / "config/pilot_model_sources.json").read_text())
    download_models(client, bucket, manifest, MODEL_DIR)
    os.environ["WATCH_AUTH_DETECTOR_CHECKPOINT"] = str(MODEL_DIR / "owlv2")
    serve(["--checkpoint", str(MODEL_DIR / "smolvlm"), *bind])


if __name__ == "__main__":
    main()
