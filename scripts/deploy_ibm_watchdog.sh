#!/usr/bin/env bash
set -euo pipefail

required=(
  WATCH_AUTH_ACCOUNT_ID WATCH_AUTH_RESOURCE_GROUP WATCH_AUTH_RESOURCE_GROUP_ID
  WATCH_AUTH_REGION WATCH_AUTH_COS_ENDPOINT WATCH_AUTH_COS_SERVICE_INSTANCE_ID
  WATCH_AUTH_CE_PROJECT
  WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT WATCH_AUTH_WATCHDOG_IMAGE_DIGEST
  WATCH_AUTH_TRUSTED_PROFILE_ID WATCH_AUTH_PLATFORM_ROUTING_PROOF
  WATCH_AUTH_COS_BUCKET WATCH_AUTH_PLATFORM_LOGS_CRN WATCH_AUTH_PLATFORM_MONITORING_CRN
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: ${name}" >&2; exit 2; }
done
[[ "${WATCH_AUTH_WATCHDOG_IMAGE_DIGEST}" == *@sha256:* ]] || {
  echo "WATCH_AUTH_WATCHDOG_IMAGE_DIGEST must be digest-pinned" >&2
  exit 2
}

command -v ibmcloud >/dev/null || { echo "ibmcloud CLI is required" >&2; exit 2; }

target_json="$(ibmcloud target --output json)"
python3 -c 'import json,sys
target=json.load(sys.stdin); account,group,region=sys.argv[1:]
if target.get("account",{}).get("guid")!=account: raise SystemExit("selected IBM account differs from WATCH_AUTH_ACCOUNT_ID")
if target.get("resource_group",{}).get("name")!=group: raise SystemExit("selected resource group differs from WATCH_AUTH_RESOURCE_GROUP")
if target.get("region",{}).get("name")!=region: raise SystemExit("selected IBM region differs from WATCH_AUTH_REGION")' \
  "$WATCH_AUTH_ACCOUNT_ID" "$WATCH_AUTH_RESOURCE_GROUP" "$WATCH_AUTH_REGION" <<<"$target_json"

project_json="$(ibmcloud ce project current --output json)"
python3 -c 'import json,sys
project=json.load(sys.stdin)
if project.get("name")!=sys.argv[1]: raise SystemExit("selected Code Engine project differs from WATCH_AUTH_CE_PROJECT")' \
  "$WATCH_AUTH_CE_PROJECT" <<<"$project_json"

# Platform integrations and provider-observed credentialless deliveries are proved before mutation.
logs_json="$(ibmcloud resource service-instance --id "$WATCH_AUTH_PLATFORM_LOGS_CRN" --output json)"
monitoring_json="$(ibmcloud resource service-instance --id "$WATCH_AUTH_PLATFORM_MONITORING_CRN" --output json)"
python3 - "$WATCH_AUTH_ACCOUNT_ID" "$WATCH_AUTH_REGION" \
  "$WATCH_AUTH_PLATFORM_LOGS_CRN" "$WATCH_AUTH_PLATFORM_MONITORING_CRN" \
  "$WATCH_AUTH_PLATFORM_ROUTING_PROOF" "$logs_json" "$monitoring_json" <<'PY'
import json, pathlib, sys

account, region, logs_crn, monitoring_crn, proof_path, logs_raw, monitoring_raw = sys.argv[1:]
instances = (json.loads(logs_raw), json.loads(monitoring_raw))
for instance, crn in zip(instances, (logs_crn, monitoring_crn), strict=True):
    if instance.get("guid") not in crn or instance.get("region_id") != region:
        raise SystemExit("observability instance differs from selected account or region")
    if f":a/{account}:" not in crn:
        raise SystemExit("observability CRN differs from selected account")
proof = json.loads(pathlib.Path(proof_path).read_text(encoding="utf-8"))
expected = {
    "account_id": account,
    "credentialless": True,
    "logs_crn": logs_crn,
    "logs_probe": "delivered",
    "monitoring_crn": monitoring_crn,
    "monitoring_probe": "delivered",
    "region": region,
}
if proof != expected:
    raise SystemExit("credentialless platform routing proof is missing or differs")
PY

job_args=(
  --name watch-auth-watchdog
  --image "$WATCH_AUTH_WATCHDOG_IMAGE_DIGEST"
  --service-account "$WATCH_AUTH_WATCHDOG_SERVICE_ACCOUNT"
  --env WATCH_AUTH_ACCOUNT_ID="$WATCH_AUTH_ACCOUNT_ID"
  --env WATCH_AUTH_RESOURCE_GROUP="$WATCH_AUTH_RESOURCE_GROUP"
  --env WATCH_AUTH_RESOURCE_GROUP_ID="$WATCH_AUTH_RESOURCE_GROUP_ID"
  --env WATCH_AUTH_REGION="$WATCH_AUTH_REGION"
  --env WATCH_AUTH_COS_BUCKET="$WATCH_AUTH_COS_BUCKET"
  --env WATCH_AUTH_COS_ENDPOINT="$WATCH_AUTH_COS_ENDPOINT"
  --env WATCH_AUTH_COS_SERVICE_INSTANCE_ID="$WATCH_AUTH_COS_SERVICE_INSTANCE_ID"
  --env WATCH_AUTH_TRUSTED_PROFILE_ID="$WATCH_AUTH_TRUSTED_PROFILE_ID"
  --command python3
  --arg=-m
  --arg infra.ibm_watchdog
)
if ibmcloud ce job get --name watch-auth-watchdog --output json >/dev/null 2>&1; then
  ibmcloud ce job update "${job_args[@]}"
else
  ibmcloud ce job create "${job_args[@]}"
fi

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
