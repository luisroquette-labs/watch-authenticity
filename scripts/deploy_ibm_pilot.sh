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
python3 - "${WATCH_AUTH_PILOT_PASSWORD_FILE}" <<'PY'
import os, sys
path = sys.argv[1]
stat = os.stat(path)
value = open(path, 'rb').read()
if stat.st_uid != os.getuid() or stat.st_mode & 0o077 or len(value) < 24 or len(value) > 128 or not all(33 <= byte <= 126 for byte in value):
    raise SystemExit('pilot password file must be owner-only, printable, and 24-128 bytes')
PY

ibmcloud ce project select --name "${WATCH_AUTH_CE_PROJECT}" >/dev/null
secret=watch-auth-pilot-basic
if ibmcloud ce secret get --name "${secret}" --output json >/dev/null 2>&1; then
  ibmcloud ce secret get --name "${secret}" --output json | python3 -c '
import hmac, json, sys
secret = json.load(sys.stdin)
with open(sys.argv[1], "rb") as source:
    expected = source.read()
actual = secret.get("data", {}).get("WATCH_AUTH_PILOT_PASSWORD", "")
if secret.get("format") != "generic" or not isinstance(actual, str) or not hmac.compare_digest(actual.encode(), expected):
    raise SystemExit("pilot secret differs from approved password file")
' "${WATCH_AUTH_PILOT_PASSWORD_FILE}"
else
  ibmcloud ce secret create --name "${secret}" --format generic \
    --from-file "WATCH_AUTH_PILOT_PASSWORD=${WATCH_AUTH_PILOT_PASSWORD_FILE}" >/dev/null
fi

app=watch-auth-pilot
if ! ibmcloud ce app get --name "${app}" --output json >/dev/null 2>&1; then
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
    --env-from-secret "${secret}"
fi

ibmcloud ce app get --name "${app}" --output json | python3 -c '
import json, os, sys
app = json.load(sys.stdin)
spec = app.get("spec", {})
template = spec.get("template", {})
pod = template.get("spec", template)
annotations = template.get("metadata", {}).get("annotations", {})
containers = pod.get("containers", [])
container = containers[0] if len(containers) == 1 else {}
resources = container.get("resources", {}).get("requests", {})
env = {item.get("name"): item.get("value") for item in container.get("env", []) if isinstance(item, dict)}
secrets = [item.get("secretRef", {}).get("name") for item in container.get("envFrom", []) if isinstance(item, dict)]
secrets += [item.get("valueFrom", {}).get("secretKeyRef", {}).get("name") for item in container.get("env", []) if isinstance(item, dict)]
expected_env = {
    "WATCH_AUTH_COS_BUCKET": os.environ["WATCH_AUTH_COS_BUCKET"],
    "WATCH_AUTH_COS_ENDPOINT": os.environ["WATCH_AUTH_COS_ENDPOINT"],
    "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": os.environ["WATCH_AUTH_COS_SERVICE_INSTANCE_ID"],
    "WATCH_AUTH_TRUSTED_PROFILE_ID": os.environ["WATCH_AUTH_TRUSTED_PROFILE_ID"],
}
valid = (
    container.get("image") == os.environ["WATCH_AUTH_PILOT_IMAGE_DIGEST"]
    and set(expected_env.items()) <= set(env.items())
    and "watch-auth-pilot-basic" in secrets
    and pod.get("imagePullSecrets") == [{"name": os.environ["WATCH_AUTH_REGISTRY_SECRET"]}]
    and pod.get("serviceAccountName") == app.get("metadata", {}).get("namespace", "") + "-none"
    and pod.get("mountComputeResourceToken") is True
    and pod.get("containerConcurrency") == 1
    and pod.get("timeoutSeconds") == 300
    and resources == {"cpu": "2", "memory": "8G", "ephemeral-storage": "4G"}
    and annotations.get("autoscaling.knative.dev/minScale") == "0"
    and annotations.get("autoscaling.knative.dev/maxScale") == "1"
    and annotations.get("autoscaling.knative.dev/scale-down-delay") == "120"
)
if not valid:
    raise SystemExit("pilot app readback differs from protected bounded configuration")
'

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

ibmcloud ce app get --name "${app}" --output url
