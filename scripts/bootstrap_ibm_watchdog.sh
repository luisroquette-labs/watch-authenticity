#!/usr/bin/env bash
set +x
set -euo pipefail
umask 077

# The key exists only while IAM hands it to the private registry secret.
key_dir="$(mktemp -d "${TMPDIR:-/tmp}/watch-auth-registry.XXXXXX")"
trap 'rm -f "$key_dir/key.json"; rmdir "$key_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
python3 -m infra.ibm_registry_bootstrap "$key_dir/key.json"
