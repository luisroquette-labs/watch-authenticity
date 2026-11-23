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

resource_exists() {
  local kind="$1" name="$2" output
  local -a command=(job)
  if [[ "${kind}" == cron ]]; then command=(sub cron); fi
  if output="$(ibmcloud ce "${command[@]}" get --name "${name}" --output json 2>&1)"; then
    return 0
  fi
  case "${output}" in
    *"Not found"*|*"not found"*|*"does not exist"*|*"Does not exist"*) return 1 ;;
  esac
  printf 'Code Engine %s readback failed for %s: %s\n' "${kind}" "${name}" "${output}" >&2
  return 2
}

ensure_job() {
  local name="$1" state
  shift
  if resource_exists job "${name}"; then return 0; else state=$?; fi
  [[ "${state}" -eq 1 ]] || return 2
  ibmcloud ce job create "$@"
}

ensure_cron() {
  local name="$1" state
  shift
  if resource_exists cron "${name}"; then
    ibmcloud ce sub cron update "$@"
    return
  else
    state=$?
  fi
  [[ "${state}" -eq 1 ]] || return 2
  ibmcloud ce sub cron create "$@"
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
rm_replica_job="watch-auth-rm-replica-${deployment_id}"
cbp_job="watch-auth-cbp-${deployment_id}"
goldsmiths_job="watch-auth-goldsmiths-${deployment_id}"
bobs_job="watch-auth-bobswatches-${deployment_id}"
swe_job="watch-auth-swisswatchexpo-${deployment_id}"
speedmaster_job="watch-auth-speedmaster-${deployment_id}"
auckland_job="watch-auth-auckland-${deployment_id}"
daytona_job="watch-auth-daytona-${deployment_id}"
royal_oak_job="watch-auth-royal-oak-${deployment_id}"
counterfeit_category_job="watch-auth-counterfeit-category-${deployment_id}"
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
ensure_job "${acquisition_job}" "${job_args[@]}"

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
ensure_job "${luxury_job}" "${luxury_job_args[@]}"
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

rm_replica_job_args=("${job_args[@]}")
rm_replica_job_args[1]="${rm_replica_job}"
rm_replica_job_args+=(--arg=--source --arg=richardmillereplica)
ensure_job "${rm_replica_job}" "${rm_replica_job_args[@]}"
verify_job "${rm_replica_job}" richardmillereplica

if link_exists "${rm_replica_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${rm_replica_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${rm_replica_job}" \
    --output json >/dev/null
  link_exists "${rm_replica_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

rm_replica_cron_args=("${cron_args[@]}")
rm_replica_cron_args[1]=watch-auth-rm-replica-daily
rm_replica_cron_args[5]="${rm_replica_job}"
rm_replica_cron_args[7]='47 5 * * *'

cbp_job_args=("${job_args[@]}")
cbp_job_args[1]="${cbp_job}"
cbp_job_args+=(--arg=--source --arg=cbp)
ensure_job "${cbp_job}" "${cbp_job_args[@]}"
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
ensure_job "${goldsmiths_job}" "${goldsmiths_job_args[@]}"
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
ensure_job "${bobs_job}" "${bobs_job_args[@]}"
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

swe_job_args=("${job_args[@]}")
swe_job_args[1]="${swe_job}"
swe_job_args+=(--arg=--source --arg=swisswatchexpo)
ensure_job "${swe_job}" "${swe_job_args[@]}"
verify_job "${swe_job}" swisswatchexpo

if link_exists "${swe_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || {
    echo "trusted-profile link inventory failed" >&2
    exit 2
  }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${swe_job}" \
    --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job \
    --link-component-name "${swe_job}" \
    --output json >/dev/null
  link_exists "${swe_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

swe_cron_args=("${cron_args[@]}")
swe_cron_args[1]=watch-auth-swisswatchexpo-daily
swe_cron_args[5]="${swe_job}"
swe_cron_args[7]='17 5 * * *'

speedmaster_job_args=("${job_args[@]}")
speedmaster_job_args[1]="${speedmaster_job}"
speedmaster_job_args+=(--arg=--source --arg=wikimedia-speedmaster)
ensure_job "${speedmaster_job}" "${speedmaster_job_args[@]}"
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

auckland_job_args=("${job_args[@]}")
auckland_job_args[1]="${auckland_job}"
auckland_job_args+=(--arg=--source --arg=wikimedia-auckland)
ensure_job "${auckland_job}" "${auckland_job_args[@]}"
verify_job "${auckland_job}" wikimedia-auckland

if link_exists "${auckland_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || { echo "trusted-profile link inventory failed" >&2; exit 2; }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${auckland_job}" --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job --link-component-name "${auckland_job}" \
    --output json >/dev/null
  link_exists "${auckland_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

auckland_cron_args=("${cron_args[@]}")
auckland_cron_args[1]=watch-auth-auckland-daily
auckland_cron_args[5]="${auckland_job}"
auckland_cron_args[7]='17 4 * * *'

daytona_job_args=("${job_args[@]}")
daytona_job_args[1]="${daytona_job}"
daytona_job_args+=(--arg=--source --arg=wikimedia-daytona)
ensure_job "${daytona_job}" "${daytona_job_args[@]}"
verify_job "${daytona_job}" wikimedia-daytona

if link_exists "${daytona_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || { echo "trusted-profile link inventory failed" >&2; exit 2; }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${daytona_job}" --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job --link-component-name "${daytona_job}" \
    --output json >/dev/null
  link_exists "${daytona_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

daytona_cron_args=("${cron_args[@]}")
daytona_cron_args[1]=watch-auth-daytona-daily
daytona_cron_args[5]="${daytona_job}"
daytona_cron_args[7]='37 4 * * *'

royal_oak_job_args=("${job_args[@]}")
royal_oak_job_args[1]="${royal_oak_job}"
royal_oak_job_args+=(--arg=--source --arg=wikimedia-royal-oak)
ensure_job "${royal_oak_job}" "${royal_oak_job_args[@]}"
verify_job "${royal_oak_job}" wikimedia-royal-oak

if link_exists "${royal_oak_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || { echo "trusted-profile link inventory failed" >&2; exit 2; }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${royal_oak_job}" --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job --link-component-name "${royal_oak_job}" \
    --output json >/dev/null
  link_exists "${royal_oak_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

royal_oak_cron_args=("${cron_args[@]}")
royal_oak_cron_args[1]=watch-auth-royal-oak-daily
royal_oak_cron_args[5]="${royal_oak_job}"
royal_oak_cron_args[7]='57 4 * * *'

counterfeit_category_job_args=("${job_args[@]}")
counterfeit_category_job_args[1]="${counterfeit_category_job}"
counterfeit_category_job_args+=(--arg=--source --arg=wikimedia-counterfeit-category)
ensure_job "${counterfeit_category_job}" "${counterfeit_category_job_args[@]}"
verify_job "${counterfeit_category_job}" wikimedia-counterfeit-category

if link_exists "${counterfeit_category_job}"; then
  :
else
  link_state=$?
  [[ "${link_state}" -eq 1 ]] || { echo "trusted-profile link inventory failed" >&2; exit 2; }
  ibmcloud iam trusted-profile-link-create "${WATCH_AUTH_TRUSTED_PROFILE_ID}" \
    --name "${counterfeit_category_job}" --cr-type CE \
    --link-crn "${WATCH_AUTH_CE_PROJECT_CRN}" \
    --link-component-type job --link-component-name "${counterfeit_category_job}" \
    --output json >/dev/null
  link_exists "${counterfeit_category_job}" || { echo "trusted-profile link readback failed" >&2; exit 2; }
fi

counterfeit_category_cron_args=("${cron_args[@]}")
counterfeit_category_cron_args[1]=watch-auth-counterfeit-category-daily
counterfeit_category_cron_args[5]="${counterfeit_category_job}"
counterfeit_category_cron_args[7]='7 5 * * *'

census_job_args=("${job_args[@]}")
census_job_args[1]="${census_job}"
census_job_args+=(--arg=--source --arg=research-census)
ensure_job "${census_job}" "${census_job_args[@]}"
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
ensure_cron watch-auth-goldsmiths-daily "${goldsmiths_cron_args[@]}"

ensure_cron watch-auth-acquisition-daily "${cron_args[@]}"

ensure_cron watch-auth-supercloneluxury-daily "${luxury_cron_args[@]}"

ensure_cron watch-auth-rm-replica-daily "${rm_replica_cron_args[@]}"

ensure_cron watch-auth-cbp-daily "${cbp_cron_args[@]}"

ensure_cron watch-auth-bobswatches-daily "${bobs_cron_args[@]}"

ensure_cron watch-auth-swisswatchexpo-daily "${swe_cron_args[@]}"

ensure_cron watch-auth-speedmaster-daily "${speedmaster_cron_args[@]}"

ensure_cron watch-auth-auckland-daily "${auckland_cron_args[@]}"

ensure_cron watch-auth-daytona-daily "${daytona_cron_args[@]}"

ensure_cron watch-auth-royal-oak-daily "${royal_oak_cron_args[@]}"

ensure_cron watch-auth-counterfeit-category-daily "${counterfeit_category_cron_args[@]}"

ensure_cron watch-auth-research-census-daily "${census_cron_args[@]}"
