"""Render the secretless IBM host bootstrap bound by ``RunSpec``."""

from __future__ import annotations

import hashlib
import shlex
from pathlib import Path

from infra.ibm_types import RunSpec, run_spec_transport_sha256

_TEMPLATE = Path(__file__).with_name("bootstrap_host.sh")


def _render(spec: RunSpec) -> str:
    values = {
        "__WATCH_AUTH_RUN_ID__": str(spec.run_id),
        "__WATCH_AUTH_REGION__": spec.region,
        "__WATCH_AUTH_IMAGE_DIGEST__": spec.image_digest,
        "__WATCH_AUTH_COS_BUCKET__": spec.bucket,
        "__WATCH_AUTH_COS_PREFIX__": spec.prefix,
        "__WATCH_AUTH_COS_SERVICE_INSTANCE_ID__": spec.cos_service_instance_id,
        "__WATCH_AUTH_TRUSTED_PROFILE_ID__": spec.trusted_profile_id,
        "__WATCH_AUTH_CODE_SHA__": spec.code_sha,
        "__WATCH_AUTH_MANIFEST_SHA256__": spec.manifest_sha256,
        "__WATCH_AUTH_SPLITS_SHA256__": spec.splits_sha256,
        "__WATCH_AUTH_COS_ENDPOINT__": spec.cos_endpoint,
        "__WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256__": run_spec_transport_sha256(spec),
    }
    rendered = _TEMPLATE.read_text(encoding="utf-8")
    for token, value in values.items():
        if rendered.count(token) != 1:
            raise ValueError(f"bootstrap template must contain {token} exactly once")
        rendered = rendered.replace(token, shlex.quote(value))
    if "__WATCH_AUTH_" in rendered:
        raise ValueError("bootstrap template contains an unresolved token")
    return rendered


def bootstrap_sha256_for(spec: RunSpec) -> str:
    """Return the digest callers put in a newly constructed ``RunSpec``."""

    return hashlib.sha256(_render(spec).encode()).hexdigest()


def render_bootstrap(spec: RunSpec) -> str:
    """Render user-data and reject a digest not authorized by ``RunSpec``."""

    rendered = _render(spec)
    if hashlib.sha256(rendered.encode()).hexdigest() != spec.bootstrap_sha256:
        raise ValueError("rendered bootstrap differs from RunSpec.bootstrap_sha256")
    return rendered
