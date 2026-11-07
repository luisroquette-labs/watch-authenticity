"""Real container contract for the IBM training worker runtime."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker is required")
def test_built_training_image_imports_pinned_ibm_sdks_offline():
    lock_sha = hashlib.sha256((ROOT / "requirements-train.lock").read_bytes()).hexdigest()
    image = f"watch-auth-training:ibm-sdk-{lock_sha[:12]}"
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "--file",
            "Containerfile.train",
            "--tag",
            image,
            ".",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "--network",
            "none",
            image,
            "python3",
            "-c",
            (
                "import importlib.metadata as m; "
                "import ibm_boto3, ibm_botocore, ibm_cloud_sdk_core; "
                "assert m.version('ibm-cos-sdk') == '2.16.2'; "
                "assert m.version('ibm-cloud-sdk-core') == '3.27.2'"
            ),
        ],
        check=True,
    )
