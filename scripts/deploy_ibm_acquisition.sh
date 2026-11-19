#!/usr/bin/env bash
set -euo pipefail

required=(
  WATCH_AUTH_CE_PROJECT WATCH_AUTH_ACQUISITION_IMAGE_DIGEST
  WATCH_AUTH_CE_PROJECT_CRN WATCH_AUTH_REGISTRY_SECRET
  WATCH_AUTH_COS_BUCKET WATCH_AUTH_COS_ENDPOINT WATCH_AUTH_COS_SERVICE_INSTANCE_ID
  WATCH_AUTH_TRUSTED_PROFILE_ID WATCH_AUTH_ACCOUNT_ID WATCH_AUTH_IAM_READER_POLICY_ID
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "missing required environment variable: ${name}" >&2; exit 2; }
done
[[ "${WATCH_AUTH_ACQUISITION_IMAGE_DIGEST}" == *@sha256:* ]] || {
  echo "WATCH_AUTH_ACQUISITION_IMAGE_DIGEST must be digest-pinned" >&2
  exit 2
}

ibmcloud ce project select --name "${WATCH_AUTH_CE_PROJECT}" >/dev/null

# Content-addressed jobs keep active subscriptions on the previous version during staging.
deployment_id="$(printf '%s\0' "$(shasum -a 256 < "$0")" \
  "${WATCH_AUTH_ACQUISITION_IMAGE_DIGEST}" "${WATCH_AUTH_CE_PROJECT_CRN}" \
  "${WATCH_AUTH_REGISTRY_SECRET}" "${WATCH_AUTH_COS_BUCKET}" \
  "${WATCH_AUTH_COS_ENDPOINT}" "${WATCH_AUTH_COS_SERVICE_INSTANCE_ID}" \
  "${WATCH_AUTH_TRUSTED_PROFILE_ID}" "${WATCH_AUTH_ACCOUNT_ID}" \
  "${WATCH_AUTH_IAM_READER_POLICY_ID}" | shasum -a 256 | cut -c1-12)"
acquisition_job="watch-auth-acquisition-${deployment_id}"
luxury_job="watch-auth-supercloneluxury-${deployment_id}"
cbp_job="watch-auth-cbp-${deployment_id}"
goldsmiths_job="watch-auth-goldsmiths-${deployment_id}"

job_args=(
  --name "${acquisition_job}"
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
  --env "WATCH_AUTH_ACCOUNT_ID=${WATCH_AUTH_ACCOUNT_ID}"
  --env "WATCH_AUTH_IAM_READER_POLICY_ID=${WATCH_AUTH_IAM_READER_POLICY_ID}"
  --env WATCH_AUTH_ACQUISITION_PREFIX=datasets
)
if ibmcloud ce job get --name "${acquisition_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${job_args[@]}"
fi

verify_job() {
  local job_name="$1" source_name="$2"
  ibmcloud ce job get --name "${job_name}" --output json | python3 -c '
import json, os, sys
source = sys.argv[1]
try:
    job = json.load(sys.stdin)
except (ValueError, TypeError):
    raise SystemExit(2)
expected_env = {
    "WATCH_AUTH_COS_BUCKET": os.environ["WATCH_AUTH_COS_BUCKET"],
    "WATCH_AUTH_COS_ENDPOINT": os.environ["WATCH_AUTH_COS_ENDPOINT"],
    "WATCH_AUTH_COS_SERVICE_INSTANCE_ID": os.environ["WATCH_AUTH_COS_SERVICE_INSTANCE_ID"],
    "WATCH_AUTH_TRUSTED_PROFILE_ID": os.environ["WATCH_AUTH_TRUSTED_PROFILE_ID"],
    "WATCH_AUTH_ACCOUNT_ID": os.environ["WATCH_AUTH_ACCOUNT_ID"],
    "WATCH_AUTH_IAM_READER_POLICY_ID": os.environ["WATCH_AUTH_IAM_READER_POLICY_ID"],
    "WATCH_AUTH_ACQUISITION_PREFIX": "datasets",
}
variables = job.get("run_env_variables", []) if isinstance(job, dict) else []
observed_env = {item.get("name"): item.get("value") for item in variables if isinstance(item, dict)}
args = ["-m", "acquisition.run"] + (["--source", source] if source else [])
valid = (isinstance(job, dict)
    and job.get("image_reference") == os.environ["WATCH_AUTH_ACQUISITION_IMAGE_DIGEST"]
    and job.get("image_secret") == os.environ["WATCH_AUTH_REGISTRY_SECRET"]
    and job.get("run_service_account") == "none"
    and job.get("run_compute_resource_token_enabled") is True
    and job.get("run_commands") == ["python"]
    and job.get("run_arguments") == args
    and job.get("scale_array_spec") == "0"
    and job.get("scale_cpu_limit") == "0.25"
    and job.get("scale_max_execution_time") == 600
    and job.get("scale_memory_limit") == "1G"
    and job.get("scale_retry_limit") == 0
    and len(variables) == len(expected_env)
    and len(observed_env) == len(expected_env)
    and observed_env == expected_env
    and all(item.get("type") == "literal" for item in variables if isinstance(item, dict)))
raise SystemExit(0 if valid else 2)
' "${source_name}" || { echo "staged job readback differs" >&2; exit 2; }
}
verify_job "${acquisition_job}" ""

link_exists() {
  local output job_name="$1"
  output="$(ibmcloud iam trusted-profile-links "${WATCH_AUTH_TRUSTED_PROFILE_ID}" --output json)" || return 2
  printf '%s' "${output}" | python3 -c '
import json, sys
project_crn, job = sys.argv[1:]
document = json.load(sys.stdin)
links = document if isinstance(document, list) else document.get("links", []) if isinstance(document, dict) else None
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
' "${WATCH_AUTH_CE_PROJECT_CRN}" "${job_name}"
}
if link_exists "${acquisition_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${acquisition_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${acquisition_job}" \
    --output json >/dev/null
  link_exists "${acquisition_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

cron_args=(
  --name watch-auth-acquisition-daily
  --destination-type job
  --destination "${acquisition_job}"
  --schedule '17 3 * * *'
  --time-zone UTC
)

luxury_job_args=("${job_args[@]}")
luxury_job_args[1]="${luxury_job}"
luxury_job_args+=(--arg=--source --arg=supercloneluxury)
if ibmcloud ce job get --name "${luxury_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${luxury_job_args[@]}"
fi
verify_job "${luxury_job}" supercloneluxury

if link_exists "${luxury_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${luxury_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${luxury_job}" \
    --output json >/dev/null
  link_exists "${luxury_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

luxury_cron_args=("${cron_args[@]}")
luxury_cron_args[1]=watch-auth-supercloneluxury-daily
luxury_cron_args[5]="${luxury_job}"
luxury_cron_args[7]='37 3 * * *'

cbp_job_args=("${job_args[@]}")
cbp_job_args[1]="${cbp_job}"
cbp_job_args+=(--arg=--source --arg=cbp)
if ibmcloud ce job get --name "${cbp_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${cbp_job_args[@]}"
fi
verify_job "${cbp_job}" cbp

if link_exists "${cbp_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${cbp_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${cbp_job}" \
    --output json >/dev/null
  link_exists "${cbp_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

cbp_cron_args=("${cron_args[@]}")
cbp_cron_args[1]=watch-auth-cbp-daily
cbp_cron_args[5]="${cbp_job}"
cbp_cron_args[7]='57 3 * * *'

goldsmiths_job_args=("${job_args[@]}")
goldsmiths_job_args[1]="${goldsmiths_job}"
goldsmiths_job_args+=(--arg=--source --arg=goldsmiths)
if ibmcloud ce job get --name "${goldsmiths_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${goldsmiths_job_args[@]}"
fi
verify_job "${goldsmiths_job}" goldsmiths

if link_exists "${goldsmiths_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${goldsmiths_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${goldsmiths_job}" \
    --output json >/dev/null
  link_exists "${goldsmiths_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

goldsmiths_cron_args=("${cron_args[@]}")
goldsmiths_cron_args[1]=watch-auth-goldsmiths-daily
goldsmiths_cron_args[5]="${goldsmiths_job}"
goldsmiths_cron_args[7]='27 3 * * *'

# Activate the dedicated source first; the old daily job remains a safe fallback
# until the main cron points to an image that omits Goldsmiths.
if ibmcloud ce sub cron get --name watch-auth-goldsmiths-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${goldsmiths_cron_args[@]}"
else
  ibmcloud ce sub cron create "${goldsmiths_cron_args[@]}"
fi

if ibmcloud ce sub cron get --name watch-auth-acquisition-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${cron_args[@]}"
else
  ibmcloud ce sub cron create "${cron_args[@]}"
fi

if ibmcloud ce sub cron get --name watch-auth-supercloneluxury-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${luxury_cron_args[@]}"
else
  ibmcloud ce sub cron create "${luxury_cron_args[@]}"
fi

if ibmcloud ce sub cron get --name watch-auth-cbp-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${cbp_cron_args[@]}"
else
  ibmcloud ce sub cron create "${cbp_cron_args[@]}"
fi
