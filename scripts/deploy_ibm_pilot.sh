#!/usr/bin/env bash
set -euo pipefail

required=(
  WATCH_AUTH_CE_PROJECT WATCH_AUTH_CE_PROJECT_CRN WATCH_AUTH_PILOT_IMAGE_DIGEST
  WATCH_AUTH_REGISTRY_SECRET WATCH_AUTH_COS_BUCKET WATCH_AUTH_COS_ENDPOINT
  WATCH_AUTH_COS_SERVICE_INSTANCE_ID WATCH_AUTH_TRUSTED_PROFILE_ID
  WATCH_AUTH_PILOT_PASSWORD_FILE
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: ${name}" >&2; exit 2; }
done
[[ "${WATCH_AUTH_PILOT_IMAGE_DIGEST}" =~ ^us\.icr\.io/watch-authenticity-3235996/pilot@sha256:[0-9a-f]{64}$ ]] || {
  echo 'pilot image must be pinned to the project Registry digest' >&2; exit 2;
}
mode=${WATCH_AUTH_PILOT_MODE-local-vision}
case "${mode}" in
  local-vision|gemini) ;;
  *) echo 'WATCH_AUTH_PILOT_MODE must be local-vision or gemini' >&2; exit 2 ;;
esac
python3 - "${WATCH_AUTH_PILOT_PASSWORD_FILE}" <<'PY'
import os, sys
path = sys.argv[1]
stat = os.stat(path)
value = open(path, 'rb').read()
if stat.st_uid != os.getuid() or stat.st_mode & 0o077 or len(value) < 24 or len(value) > 128 or not all(33 <= byte <= 126 for byte in value):
    raise SystemExit('pilot password file must be owner-only, printable, and 24-128 bytes')
PY

ibmcloud ce project select --name "${WATCH_AUTH_CE_PROJECT}" >/dev/null
app=watch-auth-pilot
secret=watch-auth-pilot-basic
provider_secret=watch-auth-pilot-gemini
provider_key=WATCH_AUTH_OPENROUTER_PILOT_KEY
if [[ "${mode}" == gemini ]]; then
  ibmcloud ce secret get --name "${provider_secret}" --output json --decode | (
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
    python3 -m pilot.flash_research --check-code-engine-secret
  )
fi

read_pilot_app() {
  python3 -c '
import json, os, sys

desired_mode, expected_revision = sys.argv[1:]
app = json.load(sys.stdin)
template = app.get("spec", {}).get("template", {})
pod = template.get("spec", template)
annotations = template.get("metadata", {}).get("annotations", {})
containers = pod.get("containers", [])
container = containers[0] if len(containers) == 1 else {}
resources = container.get("resources", {}).get("requests", {})
environment = container.get("env", [])
env = {item.get("name"): item for item in environment}
env_from = container.get("envFrom", [])
generic_secrets = [
    item.get("secretRef", {}).get("name") for item in env_from
    if not item.get("prefix") and not item.get("configMapRef")
    and not item.get("secretRef", {}).get("optional")
]
basic_secret = "watch-auth-pilot-basic"
password_key = "WATCH_AUTH_PILOT_PASSWORD"
provider_secret = "watch-auth-pilot-gemini"
provider_key = "WATCH_AUTH_OPENROUTER_PILOT_KEY"

def secret_key_ref(item):
    return item.get("valueFrom", {}).get("secretKeyRef", {})

def secret_key_bound(key, name):
    item = env.get(key, {})
    ref = secret_key_ref(item)
    return (item.get("value") is None and ref.get("name") == name
            and ref.get("key") == key and not ref.get("optional"))

password_bound = secret_key_bound(password_key, basic_secret) or (
    password_key not in env and basic_secret in generic_secrets
)
mode_entry = env.get("WATCH_AUTH_PILOT_MODE", {"value": "local-vision"})
current_mode = mode_entry.get("value")
provider_refs = [
    item.get("name") for item in environment if secret_key_ref(item).get("name") == provider_secret
]
provider_bound = (
    provider_refs == [provider_key] and secret_key_bound(provider_key, provider_secret)
    if current_mode == "gemini" else not provider_refs and provider_key not in env
)
expected_env = {
    "WATCH_AUTH_COS_BUCKET": os.environ["WATCH_AUTH_COS_BUCKET"],
    "WATCH_AUTH_COS_ENDPOINT": os.environ["WATCH_AUTH_COS_ENDPOINT"],
    "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": os.environ["WATCH_AUTH_COS_SERVICE_INSTANCE_ID"],
    "WATCH_AUTH_TRUSTED_PROFILE_ID": os.environ["WATCH_AUTH_TRUSTED_PROFILE_ID"],
}
if current_mode == "gemini" and desired_mode == "local-vision":
    raise SystemExit("refusing to switch an active Gemini pilot to local-vision")
if (
    len(environment) != len(env)
    or any(env.get(key, {}).get("value") != value for key, value in expected_env.items())
    or not password_bound
    or pod.get("imagePullSecrets") != [{"name": os.environ["WATCH_AUTH_REGISTRY_SECRET"]}]
    or pod.get("serviceAccountName") != app.get("metadata", {}).get("namespace", "") + "-none"
    or annotations.get("codeengine.cloud.ibm.com/mount-compute-resource-token") != "true"
    or pod.get("containerConcurrency") != 1 or pod.get("timeoutSeconds") != 300
    or resources != {"cpu": "2", "memory": "8G", "ephemeral-storage": "4G"}
    or annotations.get("autoscaling.knative.dev/minScale") != "0"
    or annotations.get("autoscaling.knative.dev/maxScale") != "1"
    or annotations.get("autoscaling.knative.dev/scale-down-delay") != "120s"
    or current_mode not in ("gemini", "local-vision")
    or not provider_bound
    or any(item.get("secretRef", {}).get("name") == provider_secret for item in env_from)
    or desired_mode == "gemini" and (
        len(generic_secrets) != len(env_from)
        or any(name != basic_secret for name in generic_secrets)
    )
):
    raise SystemExit("pilot app readback differs from protected bounded configuration")

status = app.get("status", {})
generation = app.get("metadata", {}).get("generation")
observed_generation = type(generation) is int and generation > 0 and generation == status.get("observedGeneration")
created = status.get("latestCreatedRevisionName")
latest_ready = status.get("latestReadyRevisionName")
ready_status = next((item.get("status") for item in status.get("conditions", [])
                     if item.get("type") == "Ready"), None)
traffic = status.get("traffic", [])
active_traffic = [item for item in traffic if item.get("percent", 0) != 0]
revision_settled = (
    observed_generation and bool(created) and created == latest_ready and len(active_traffic) == 1
    and active_traffic[0].get("revisionName") == created and active_traffic[0].get("percent") == 100
)
ready = ready_status == "True"
image_matches = container.get("image") == os.environ["WATCH_AUTH_PILOT_IMAGE_DIGEST"]
if expected_revision:
    if not image_matches:
        raise SystemExit("pilot app image differs from pinned digest")
    if current_mode != desired_mode:
        raise SystemExit("pilot app mode differs from requested mode")
    if not ready or not revision_settled or (expected_revision != "-" and created != expected_revision):
        raise SystemExit("requested pilot revision is not ready and serving all traffic")
else:
    retry_failed_revision = observed_generation and bool(created) and ready_status == "False" and (
        desired_mode == "gemini" or not latest_ready and not active_traffic
    )
    if not revision_settled and not retry_failed_revision:
        raise SystemExit("existing pilot revision is unsettled; wait for its serving mode to be verified")
    print("current" if image_matches and current_mode == desired_mode and ready else "update")
' "${mode}" "${1:-}"
}

action=create
if current=$(ibmcloud ce app get --name "${app}" --output json 2>/dev/null); then
  action=$(printf '%s' "${current}" | read_pilot_app)
fi
if ibmcloud ce secret get --name "${secret}" --output json >/dev/null 2>&1; then
  ibmcloud ce secret get --name "${secret}" --output json --decode | python3 -c '
import hmac, json, sys
secret = json.load(sys.stdin)
with open(sys.argv[1], "rb") as source:
    expected = source.read()
data = secret.get("data", {})
actual = data.get("WATCH_AUTH_PILOT_PASSWORD", "")
if secret.get("format") != "generic" or set(data) != {"WATCH_AUTH_PILOT_PASSWORD"} or not isinstance(actual, str) or not hmac.compare_digest(actual.encode(), expected):
    raise SystemExit("pilot secret differs from approved password file")
' "${WATCH_AUTH_PILOT_PASSWORD_FILE}"
else
  ibmcloud ce secret create --name "${secret}" --format generic \
    --from-file "WATCH_AUTH_PILOT_PASSWORD=${WATCH_AUTH_PILOT_PASSWORD_FILE}" >/dev/null
fi

links=$(ibmcloud iam trusted-profile-links "${WATCH_AUTH_TRUSTED_PROFILE_ID}" --output json)
if ! printf '%s' "${links}" | python3 -c '
import json, sys
links = json.load(sys.stdin)
items = links.get("links", links) if isinstance(links, dict) else links
raise SystemExit(0 if any(x.get("cr_type") == "CE" and x.get("link", {}).get("crn") == sys.argv[1] and x.get("link", {}).get("component_type") == "application" and x.get("link", {}).get("component_name") == "watch-auth-pilot" for x in items) else 1)
' "${WATCH_AUTH_CE_PROJECT_CRN}"; then
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${app}" --cr-type CE --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type application --link-component-name "${app}" --output json >/dev/null
fi

ibmcloud iam trusted-profile-links "${WATCH_AUTH_TRUSTED_PROFILE_ID}" --output json | python3 -c '
import json, sys
links = json.load(sys.stdin)
items = links.get("links", links) if isinstance(links, dict) else links
if not isinstance(items, list) or not any(
    isinstance(x, dict) and x.get("cr_type") == "CE"
    and x.get("link", {}).get("crn") == sys.argv[1]
    and x.get("link", {}).get("component_type") == "application"
    and x.get("link", {}).get("component_name") == "watch-auth-pilot"
    for x in items
):
    raise SystemExit("pilot trusted-profile link readback differs")
' "${WATCH_AUTH_CE_PROJECT_CRN}"

revision=-
if [[ "${action}" != current ]]; then
  revision="${app}-$(date -u +%Y%m%d%H%M%S)-$$"
  revision_args=(--revision-name "${revision}" --wait --wait-timeout 600)
  if [[ "${mode}" == gemini ]]; then
    revision_args+=(--env WATCH_AUTH_PILOT_MODE=gemini --env-from-secret "${provider_secret}:${provider_key}")
  fi
fi
if [[ "${action}" == create ]]; then
  ibmcloud ce app create \
    --name "${app}" --image "${WATCH_AUTH_PILOT_IMAGE_DIGEST}" \
    --registry-secret "${WATCH_AUTH_REGISTRY_SECRET}" \
    --service-account none --trusted-profiles-enabled=true \
    --cpu 2 --memory 8G --ephemeral-storage 4G \
    --min-scale 0 --max-scale 1 --concurrency 1 --scale-down-delay 120 \
    --port 8080 --request-timeout 300 --visibility public \
    --env "WATCH_AUTH_COS_BUCKET=${WATCH_AUTH_COS_BUCKET}" \
    --env "WATCH_AUTH_COS_ENDPOINT=${WATCH_AUTH_COS_ENDPOINT}" \
    --env "WATCH_AUTH_COS_SERVICE_INSTANCE_ID=${WATCH_AUTH_COS_SERVICE_INSTANCE_ID}" \
    --env "WATCH_AUTH_TRUSTED_PROFILE_ID=${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --env-from-secret "${secret}" "${revision_args[@]}"
elif [[ "${action}" == update ]]; then
  ibmcloud ce app update --name "${app}" --image "${WATCH_AUTH_PILOT_IMAGE_DIGEST}" \
    "${revision_args[@]}"
fi

ibmcloud ce app get --name "${app}" --output json | read_pilot_app "${revision}"
ibmcloud ce app get --name "${app}" --output url
