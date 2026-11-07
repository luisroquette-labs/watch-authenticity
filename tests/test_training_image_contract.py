"""Static contracts for the secretless, immutable GPU training runtime."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

from infra.ibm_bootstrap import bootstrap_sha256_for, render_bootstrap
from infra.ibm_vpc import build_instance_prototype
from tests.conftest import safe_user_data, valid_run_spec

ROOT = Path(__file__).resolve().parents[1]
CONTAINERFILE = ROOT / "Containerfile.train"
BOOTSTRAP = ROOT / "infra/bootstrap_host.sh"
LOCK = ROOT / "requirements-train.lock"
AMD64_BASE_DIGEST = "sha256:23debbe74125dc84df96df79cff42079b3b15265c27140714fd27b5aa718faa4"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_training_image_is_amd64_digest_pinned_and_non_root():
    containerfile = _text(CONTAINERFILE)

    assert (
        "FROM --platform=linux/amd64 "
        f"nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@{AMD64_BASE_DIGEST}"
    ) in containerfile
    assert re.search(r"^USER\s+10001:10001$", containerfile, re.MULTILINE)
    assert "HEALTHCHECK" in containerfile


def test_training_image_uses_only_the_hashed_training_lock_without_secrets():
    containerfile = _text(CONTAINERFILE)
    instructions = tuple(
        line.strip()
        for line in containerfile.splitlines()
        if line.strip().startswith(("ARG ", "ENV "))
    )

    assert "COPY requirements.txt" not in containerfile
    assert "COPY requirements-train.lock" in containerfile
    assert re.search(
        r"pip install\s+(?:--[^\s]+\s+)*--require-hashes\s+-r\s+"
        r"/opt/watch-auth/requirements-train\.lock",
        containerfile,
    )
    assert not any(
        re.search(r"(?i)(secret|token|password|api[_-]?key|auth[_-]?key)", line)
        for line in instructions
    )


def test_training_lock_pins_cuda_126_torch_and_hashes_every_requirement():
    lock = _text(LOCK)
    requirement_lines = tuple(
        line
        for line in lock.splitlines()
        if line and not line.startswith((" ", "#", "--"))
    )

    assert "torch-2.10.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl" in lock
    assert "torchvision-0.25.0%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl" in lock
    assert "2a7a569206f07965eff69b28e147676540bb0ba6e1a39410802b6e4708cb8356" in lock
    assert "5831326b6710366b44c0124ce197b416b7b896efa27340c235081cc7f52870e5" in lock
    assert requirement_lines
    assert all(("==" in line or " @ " in line) and "\\" in line for line in requirement_lines)
    assert lock.count("--hash=sha256:") >= len(requirement_lines)


def test_bootstrap_bytes_are_bound_to_run_spec():
    candidate = valid_run_spec()
    digest = bootstrap_sha256_for(candidate)
    spec = replace(valid_run_spec(), bootstrap_sha256=digest)
    bootstrap = render_bootstrap(spec)

    prototype = build_instance_prototype(spec, bootstrap)

    assert prototype["user_data"] == bootstrap
    assert hashlib.sha256(prototype["user_data"].encode()).hexdigest() == digest
    assert digest != hashlib.sha256(safe_user_data().encode()).hexdigest()
    assert spec.image_digest in bootstrap
    assert spec.prefix in bootstrap
    assert "__WATCH_AUTH_" not in bootstrap


def test_bootstrap_is_two_stage_version_pinned_and_secretless_on_disk():
    bootstrap = _text(BOOTSTRAP)

    assert "stage1)" in bootstrap and "stage2)" in bootstrap
    assert "watch-auth-bootstrap-stage2.service" in bootstrap
    assert "systemctl reboot" in bootstrap
    assert re.search(r"nvidia-driver-550=[^\s\"']+", bootstrap)
    assert re.search(r"nvidia-container-toolkit=[^\s\"']+", bootstrap)
    assert re.search(r"squid=[^\s\"']+", bootstrap)
    assert "apt-key" not in bootstrap
    assert "gpg --show-keys --with-colons" in bootstrap
    assert "instance_identity_token" in bootstrap
    assert "IAM_TOKEN" in bootstrap
    assert not re.search(r"(?:TOKEN|PASSWORD|SECRET|API_KEY).*\b(?:tee|install)\b", bootstrap)
    assert not re.search(r'(?:TOKEN|PASSWORD|SECRET|API_KEY).*>{1,2}\s*"?\$\{', bootstrap)


def test_bootstrap_enforces_exact_proxy_and_dual_stack_isolation():
    bootstrap = _text(BOOTSTRAP)
    proxy_acls = tuple(
        line for line in bootstrap.splitlines() if line.startswith("acl exact_destinations ")
    )

    assert proxy_acls == (
        (
            "acl exact_destinations dstdomain iam.cloud.ibm.com "
            "s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud"
        ),
    )
    assert "dstdomain ." not in bootstrap
    assert "--network host" not in bootstrap
    assert "docker network create" in bootstrap
    assert "--ipv6=false" in bootstrap
    assert "DOCKER-USER" in bootstrap
    assert "ensure_jump iptables filter INPUT WATCH_AUTH_INPUT" in bootstrap
    assert "ensure_jump ip6tables filter FORWARD WATCH_AUTH_FORWARD6" in bootstrap
    assert "ensure_jump ip6tables filter INPUT WATCH_AUTH_INPUT6" in bootstrap
    assert "http_port 172.30.0.1:3128" in bootstrap
    assert "http_access deny all" in bootstrap


def test_bootstrap_proves_network_gpu_and_launches_hardened_container():
    bootstrap = _text(BOOTSTRAP)

    for marker in (
        "prove_direct_https_blocked",
        "prove_host_service_blocked",
        "prove_domain_blocked",
        "prove_ipv6_blocked",
        "prove_dns_blocked",
        "prove_metadata_reachable",
        "prove_cos_reachable",
    ):
        assert marker in bootstrap
    assert "--password-stdin" in bootstrap
    assert 'mktemp -d /run/watch-auth-docker.XXXXXX' in bootstrap
    assert 'DOCKER_CONFIG="${docker_config}"' in bootstrap
    assert 'trap \'rm -rf -- "${docker_config}"\' EXIT' in bootstrap
    assert "docker logout" in bootstrap
    assert "If-None-Match: *" in bootstrap
    assert "/bootstrap/bootstrap-proof-${proof_hash}.json" in bootstrap
    assert "/events/" not in bootstrap
    assert 'readonly STAGE2_COMPLETE="${STATE_DIR}/stage2-complete"' in bootstrap
    assert "--gpus all" in bootstrap
    assert "--network watch-auth-isolated" in bootstrap
    assert bootstrap.count("--dns 127.0.0.1") >= 2
    assert "--read-only" in bootstrap
    assert "--cap-drop=ALL" in bootstrap
    assert "--security-opt=no-new-privileges" in bootstrap
    assert "--tmpfs /tmp" in bootstrap
    assert "torch.__version__.startswith(\"2.10.0\")" in bootstrap
    assert "torch.version.cuda == \"12.6\"" in bootstrap
    assert '"L4" in torch.cuda.get_device_name(0)' in bootstrap
    assert "22 * 1024**3" in bootstrap
    assert re.search(
        r'docker run --rm --gpus all --network none "\$\{TRAIN_IMAGE_DIGEST\}" python3',
        bootstrap,
    )
    assert re.search(
        r'container inspect watch-auth-training.*?State\.Running.*?State\.Health\.Status',
        bootstrap,
        re.DOTALL,
    )
    assert 'readonly STATE_DIR="/var/lib/watch-auth"' in bootstrap
    assert 'readonly PROOF_PATH="${STATE_DIR}/bootstrap-proof.json"' in bootstrap
