#!/usr/bin/env bash
set -euo pipefail

required=(
  WATCH_AUTH_CE_PROJECT WATCH_AUTH_ACQUISITION_IMAGE_DIGEST
  WATCH_AUTH_CE_PROJECT_CRN WATCH_AUTH_REGISTRY_SECRET
  WATCH_AUTH_COS_BUCKET WATCH_AUTH_COS_ENDPOINT WATCH_AUTH_COS_SERVICE_INSTANCE_ID
  WATCH_AUTH_TRUSTED_PROFILE_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: ${name}" >&2; exit 2; }
done
[[ "${WATCH_AUTH_ACQUISITION_IMAGE_DIGEST}" == *@sha256:* ]] || {
  echo "WATCH_AUTH_ACQUISITION_IMAGE_DIGEST must be digest-pinned" >&2
  exit 2
}

ibmcloud ce project select --name "${WATCH_AUTH_CE_PROJECT}" >/dev/null

job_args=(
  --name watch-auth-acquisition
  --image "${WATCH_AUTH_ACQUISITION_IMAGE_DIGEST}"
  --registry-secret "${WATCH_AUTH_REGISTRY_SECRET}"
  --service-account none
  --trusted-profiles-enabled=true
  --command python
  --arg=-m
  --arg acquisition.run
  --array-indexes 0
  --maxexecutiontime 600
  --retrylimit 0
  --cpu 0.25
  --memory 1G
  --env "WATCH_AUTH_COS_BUCKET=${WATCH_AUTH_COS_BUCKET}"
  --env "WATCH_AUTH_COS_ENDPOINT=${WATCH_AUTH_COS_ENDPOINT}"
  --env "WATCH_AUTH_COS_SERVICE_INSTANCE_ID=${WATCH_AUTH_COS_SERVICE_INSTANCE_ID}"
  --env "WATCH_AUTH_TRUSTED_PROFILE_ID=${WATCH_AUTH_TRUSTED_PROFILE_ID}"
  --env WATCH_AUTH_ACQUISITION_PREFIX=datasets
)
if ibmcloud ce job get --name watch-auth-acquisition --output json >/dev/null 2>&1; then
  ibmcloud ce job update "${job_args[@]}"
else
  ibmcloud ce job create "${job_args[@]}"
fi

link_exists() {
  local output
  output="$(ibmcloud iam trusted-profile-links "${WATCH_AUTH_TRUSTED_PROFILE_ID}" --output json)" || return 2
  printf '%s' "${output}" | python3 -c '
import json, sys
project_crn, job = sys.argv[1:]
document = json.load(sys.stdin)
links = document.get("links", [])
if not isinstance(links, list):
    raise SystemExit(2)
raise SystemExit(0 if any(
    isinstance(item, dict)
    and item.get("cr_type") == "CE"
    and isinstance(item.get("link"), dict)
    and item["link"].get("crn") == project_crn
    and item["link"].get("component_type") == "job"
    and item["link"].get("component_name") == job
    for item in links
) else 1)
' "${WATCH_AUTH_CE_PROJECT_CRN}" watch-auth-acquisition
}
if link_exists; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name watch-auth-acquisition \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name watch-auth-acquisition \
    --output json >/dev/null
  link_exists || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

cron_args=(
  --name watch-auth-acquisition-daily
  --destination-type job
  --destination watch-auth-acquisition
  --schedule '17 3 * * *'
  --time-zone UTC
)
if ibmcloud ce sub cron get --name watch-auth-acquisition-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${cron_args[@]}"
else
  ibmcloud ce sub cron create "${cron_args[@]}"
fi
