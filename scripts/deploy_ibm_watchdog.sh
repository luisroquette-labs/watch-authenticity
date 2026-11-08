#!/usr/bin/env bash
set -euo pipefail

required=(
  WATCH_AUTH_ACCOUNT_ID WATCH_AUTH_RESOURCE_GROUP WATCH_AUTH_RESOURCE_GROUP_ID
  WATCH_AUTH_REGION WATCH_AUTH_COS_ENDPOINT WATCH_AUTH_COS_SERVICE_INSTANCE_ID
  WATCH_AUTH_CE_PROJECT WATCH_AUTH_CE_PROJECT_ID
  WATCH_AUTH_ROUTING_JOB_RUN WATCH_AUTH_ROUTING_PROBE_NONCE
  WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT WATCH_AUTH_WATCHDOG_IMAGE_DIGEST
  WATCH_AUTH_TRUSTED_PROFILE_ID
  WATCH_AUTH_COS_BUCKET WATCH_AUTH_PLATFORM_LOGS_CRN WATCH_AUTH_PLATFORM_MONITORING_CRN
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: ${name}" >&2; exit 2; }
done
[[ "${WATCH_AUTH_WATCHDOG_IMAGE_DIGEST}" == *@sha256:* ]] || {
  echo "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST must be digest-pinned" >&2
  exit 2
}

# Verify credentialless platform routing proof through authenticated provider readbacks before mutation.
python3 -m infra.ibm_routing

# Reuse the separately invocable, inert bootstrap after the platform routing gate.
bash "$(dirname "$0")/bootstrap_ibm_watchdog.sh"

cron_args=(
  --name watch-auth-watchdog-cron
  --destination-type job
  --destination watch-auth-watchdog
  --schedule '*/15 * * * *'
)
if ibmcloud ce sub cron get --name watch-auth-watchdog-cron --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${cron_args[@]}"
else
  ibmcloud ce sub cron create "${cron_args[@]}"
fi
