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
bobs_job="watch-auth-bobswatches-${deployment_id}"
speedmaster_job="watch-auth-speedmaster-${deployment_id}"
census_job="watch-auth-research-census-${deployment_id}"

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
spec = job.get("spec", {}) if isinstance(job, dict) else {}
template = spec.get("template", {}) if isinstance(spec, dict) else {}
containers = template.get("containers", []) if isinstance(template, dict) else []
container = containers[0] if isinstance(containers, list) and len(containers) == 1 else {}
resources = container.get("resources", {}).get("requests", {}) if isinstance(container, dict) else {}
variables = container.get("env", []) if isinstance(container, dict) else []
observed_env = {item.get("name"): item.get("value") for item in variables if isinstance(item, dict)}
args = ["-m", "acquisition.run"] + (["--source", source] if source else [])
valid = (isinstance(job, dict)
    and isinstance(spec, dict) and isinstance(template, dict) and isinstance(container, dict)
    and container.get("image") == os.environ["WATCH_AUTH_ACQUISITION_IMAGE_DIGEST"]
    and template.get("imagePullSecrets") == [{"name": os.environ["WATCH_AUTH_REGISTRY_SECRET"]}]
    and isinstance(job.get("metadata"), dict)
    and isinstance(job["metadata"].get("namespace"), str)
    and template.get("serviceAccountName") == job["metadata"]["namespace"] + "-none"
    and template.get("mountComputeResourceToken") is True
    and container.get("command") == ["python"]
    and container.get("args") == args
    and spec.get("arraySpec") == "0"
    and resources == {"cpu": "250m", "ephemeral-storage": "400M", "memory": "1G"}
    and spec.get("maxExecutionTime") == 600
    and spec.get("retryLimit") == 0
    and spec.get("mode") == "task"
    and isinstance(variables, list) and len(variables) == len(observed_env)
    and all(observed_env.get(name) == value for name, value in expected_env.items())
    and set(observed_env) == set(expected_env) | {"CE_API_BASE_URL", "CE_PROJECT_ID", "CE_REGION"})
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

bobs_job_args=("${job_args[@]}")
bobs_job_args[1]="${bobs_job}"
bobs_job_args+=(--arg=--source --arg=bobswatches)
if ibmcloud ce job get --name "${bobs_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${bobs_job_args[@]}"
fi
verify_job "${bobs_job}" bobswatches

if link_exists "${bobs_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${bobs_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${bobs_job}" \
    --output json >/dev/null
  link_exists "${bobs_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

bobs_cron_args=("${cron_args[@]}")
bobs_cron_args[1]=watch-auth-bobswatches-daily
bobs_cron_args[5]="${bobs_job}"
bobs_cron_args[7]='7 4 * * *'

speedmaster_job_args=("${job_args[@]}")
speedmaster_job_args[1]="${speedmaster_job}"
speedmaster_job_args+=(--arg=--source --arg=wikimedia-speedmaster)
if ibmcloud ce job get --name "${speedmaster_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${speedmaster_job_args[@]}"
fi
verify_job "${speedmaster_job}" wikimedia-speedmaster

if link_exists "${speedmaster_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${speedmaster_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${speedmaster_job}" \
    --output json >/dev/null
  link_exists "${speedmaster_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

speedmaster_cron_args=("${cron_args[@]}")
speedmaster_cron_args[1]=watch-auth-speedmaster-daily
speedmaster_cron_args[5]="${speedmaster_job}"
speedmaster_cron_args[7]='47 3 * * *'

census_job_args=("${job_args[@]}")
census_job_args[1]="${census_job}"
census_job_args+=(--arg=--source --arg=research-census)
if ibmcloud ce job get --name "${census_job}" --output json >/dev/null 2>&1; then
  :
else
  ibmcloud ce job create "${census_job_args[@]}"
fi
verify_job "${census_job}" research-census

if link_exists "${census_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${census_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${census_job}" \
    --output json >/dev/null
  link_exists "${census_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

census_cron_args=("${cron_args[@]}")
census_cron_args[1]=watch-auth-research-census-daily
census_cron_args[5]="${census_job}"
census_cron_args[7]='27 4 * * *'

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

if ibmcloud ce sub cron get --name watch-auth-bobswatches-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${bobs_cron_args[@]}"
else
  ibmcloud ce sub cron create "${bobs_cron_args[@]}"
fi

if ibmcloud ce sub cron get --name watch-auth-speedmaster-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${speedmaster_cron_args[@]}"
else
  ibmcloud ce sub cron create "${speedmaster_cron_args[@]}"
fi

if ibmcloud ce sub cron get --name watch-auth-research-census-daily --output json >/dev/null 2>&1; then
  ibmcloud ce sub cron update "${census_cron_args[@]}"
else
  ibmcloud ce sub cron create "${census_cron_args[@]}"
fi
