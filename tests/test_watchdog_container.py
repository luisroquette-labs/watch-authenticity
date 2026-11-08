from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_watchdog_container_is_minimal_pinned_and_non_root() -> None:
    containerfile = (ROOT / "Containerfile.watchdog").read_text()
    lock = (ROOT / "requirements-watchdog.lock").read_text()

    assert containerfile.startswith(
        "FROM --platform=linux/amd64 "
        "registry.access.redhat.com/ubi9/python-312-minimal@sha256:"
    )
    assert "useradd --uid 10001 --gid watch-auth" in containerfile
    assert "HOME=/home/watch-auth" in containerfile
    assert containerfile.count("--require-hashes") == 2
    assert "--no-build-isolation" in containerfile
    assert "pip uninstall -y setuptools wheel" in containerfile
    assert "COPY --chown=10001:10001 infra infra" in containerfile
    assert "USER 10001:10001" in containerfile
    assert 'CMD ["python", "-m", "infra.ibm_watchdog"]' in containerfile
    assert "sha256:" in lock
