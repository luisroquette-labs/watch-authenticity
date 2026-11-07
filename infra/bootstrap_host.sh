#!/usr/bin/env bash
set -euo pipefail
umask 077

readonly RUN_ID=__WATCH_AUTH_RUN_ID__
readonly IBM_REGION=__WATCH_AUTH_REGION__
readonly TRAIN_IMAGE_DIGEST=__WATCH_AUTH_IMAGE_DIGEST__
readonly COS_BUCKET=__WATCH_AUTH_COS_BUCKET__
readonly COS_RUN_PREFIX=__WATCH_AUTH_COS_PREFIX__
readonly COS_SERVICE_INSTANCE_ID=__WATCH_AUTH_COS_SERVICE_INSTANCE_ID__
readonly TRUSTED_PROFILE_ID=__WATCH_AUTH_TRUSTED_PROFILE_ID__
readonly CODE_SHA=__WATCH_AUTH_CODE_SHA__
readonly MANIFEST_SHA256=__WATCH_AUTH_MANIFEST_SHA256__
readonly SPLITS_SHA256=__WATCH_AUTH_SPLITS_SHA256__
readonly COS_ENDPOINT=__WATCH_AUTH_COS_ENDPOINT__
readonly RUN_SPEC_TRANSPORT_SHA256=__WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256__
readonly NVIDIA_KEY_FINGERPRINT="C95B321B61E88C1809C4F759DDCAE044F796ECB0"
readonly NVIDIA_DRIVER_PACKAGE="nvidia-driver-550=550.163.01-0ubuntu0.22.04.2"
readonly NVIDIA_TOOLKIT_PACKAGE="nvidia-container-toolkit=1.20.1-1"
readonly SQUID_PACKAGE="squid=5.9-0ubuntu0.22.04.8"
readonly DOCKER_PACKAGE="docker.io=29.1.3-0ubuntu3~22.04.2"
readonly BRIDGE_NAME="watch-auth-isolated"
readonly BRIDGE_INTERFACE="watch-auth0"
readonly BRIDGE_ADDRESS="172.30.0.1"
readonly METADATA_ADDRESS="169.254.169.254"
readonly STATE_DIR="/var/lib/watch-auth"
readonly PROOF_PATH="${STATE_DIR}/bootstrap-proof.json"
readonly STAGE2_COMPLETE="${STATE_DIR}/stage2-complete"
readonly INSTALLED_SCRIPT="/usr/local/libexec/watch-auth-bootstrap"
readonly STAGE2_UNIT="watch-auth-bootstrap-stage2.service"

require_root() {
    [[ ${EUID} -eq 0 ]] || { echo "bootstrap must run as root" >&2; exit 1; }
}

verify_nvidia_repository_key() {
    local key_file fingerprint
    key_file=$(mktemp)
    curl --fail --silent --show-error --location \
        https://nvidia.github.io/libnvidia-container/gpgkey -o "${key_file}"
    fingerprint=$(gpg --show-keys --with-colons "${key_file}" | awk -F: '$1 == "fpr" {print $10; exit}')
    [[ ${fingerprint} == "${NVIDIA_KEY_FINGERPRINT}" ]] || {
        rm -f "${key_file}"
        echo "unexpected NVIDIA repository key" >&2
        exit 1
    }
    gpg --batch --yes --dearmor \
        -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg "${key_file}"
    rm -f "${key_file}"
}

write_runtime_files() {
    install -d -m 0700 "${STATE_DIR}" /etc/watch-auth /usr/local/libexec
    install -m 0755 "${BASH_SOURCE[0]}" "${INSTALLED_SCRIPT}"
    cat >/etc/systemd/system/${STAGE2_UNIT} <<EOF
[Unit]
Description=Watch authenticity GPU bootstrap stage 2
After=cloud-final.service docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=${INSTALLED_SCRIPT} stage2
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF
    chmod 0600 /etc/systemd/system/${STAGE2_UNIT}
}

stage1() {
    if [[ -e ${STATE_DIR}/stage1-complete ]]; then
        systemctl enable --now "${STAGE2_UNIT}"
        return
    fi
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends ca-certificates curl gnupg iptables jq
    verify_nvidia_repository_key
    curl --fail --silent --show-error --location \
        https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        >/etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        "${NVIDIA_DRIVER_PACKAGE}" \
        "${NVIDIA_TOOLKIT_PACKAGE}" \
        "${SQUID_PACKAGE}" \
        "${DOCKER_PACKAGE}"
    nvidia-ctk runtime configure --runtime=docker
    write_runtime_files
    systemctl daemon-reload
    systemctl enable "${STAGE2_UNIT}"
    touch "${STATE_DIR}/stage1-complete"
    systemctl reboot
}

require_exact_package() {
    local package=${1%%=*} expected=${1#*=} actual
    actual=$(dpkg-query -W -f='${Version}' "${package}")
    [[ ${actual} == "${expected}" ]] || {
        echo "${package} version mismatch" >&2
        exit 1
    }
}

configure_proxy() {
    cat >/etc/squid/squid.conf <<EOF
visible_hostname watch-auth-proxy
http_port 172.30.0.1:3128
acl SSL_ports port 443
acl CONNECT method CONNECT
acl exact_destinations dstdomain iam.cloud.ibm.com s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud
http_access allow CONNECT SSL_ports exact_destinations
http_access deny all
cache deny all
access_log stdio:/var/log/squid/access.log logformat=squid
strip_query_terms on
via off
forwarded_for delete
EOF
    systemctl restart squid
}

ensure_chain() {
    local family=$1 table=$2 chain=$3
    "${family}" -t "${table}" -N "${chain}" 2>/dev/null || true
    "${family}" -t "${table}" -F "${chain}"
}

ensure_jump() {
    local family=$1 table=$2 parent=$3 child=$4
    "${family}" -t "${table}" -C "${parent}" -j "${child}" 2>/dev/null \
        || "${family}" -t "${table}" -I "${parent}" 1 -j "${child}"
}

configure_network() {
    systemctl enable --now docker
    if ! docker network inspect "${BRIDGE_NAME}" >/dev/null 2>&1; then
        docker network create \
            --driver bridge \
            --subnet 172.30.0.0/24 \
            --gateway "${BRIDGE_ADDRESS}" \
            --ipv6=false \
            --opt com.docker.network.bridge.name="${BRIDGE_INTERFACE}" \
            "${BRIDGE_NAME}" >/dev/null
    fi
    [[ $(docker network inspect -f '{{.EnableIPv6}}' "${BRIDGE_NAME}") == false ]]
    [[ $(docker network inspect -f '{{index .Options "com.docker.network.bridge.name"}}' "${BRIDGE_NAME}") == "${BRIDGE_INTERFACE}" ]]

    ensure_chain iptables filter WATCH_AUTH_FORWARD
    ensure_jump iptables filter DOCKER-USER WATCH_AUTH_FORWARD
    iptables -A WATCH_AUTH_FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    iptables -A WATCH_AUTH_FORWARD -i "${BRIDGE_INTERFACE}" -d "${METADATA_ADDRESS}" -p tcp --dport 80 -j ACCEPT
    iptables -A WATCH_AUTH_FORWARD -i "${BRIDGE_INTERFACE}" -j REJECT

    ensure_chain iptables filter WATCH_AUTH_INPUT
    ensure_jump iptables filter INPUT WATCH_AUTH_INPUT
    iptables -A WATCH_AUTH_INPUT -i "${BRIDGE_INTERFACE}" -d "${BRIDGE_ADDRESS}" -p tcp --dport 3128 -j ACCEPT
    iptables -A WATCH_AUTH_INPUT -i "${BRIDGE_INTERFACE}" -j REJECT

    ensure_chain ip6tables filter WATCH_AUTH_FORWARD6
    ensure_jump ip6tables filter FORWARD WATCH_AUTH_FORWARD6
    ip6tables -A WATCH_AUTH_FORWARD6 -i "${BRIDGE_INTERFACE}" -j REJECT
    ensure_chain ip6tables filter WATCH_AUTH_INPUT6
    ensure_jump ip6tables filter INPUT WATCH_AUTH_INPUT6
    ip6tables -A WATCH_AUTH_INPUT6 -i "${BRIDGE_INTERFACE}" -j REJECT
}

container_curl() {
    docker run --rm \
        --network "${BRIDGE_NAME}" \
        --add-host=host.docker.internal:host-gateway \
        "${TRAIN_IMAGE_DIGEST}" curl "$@"
}

prove_direct_https_blocked() {
    ! container_curl --noproxy '*' --fail --connect-timeout 5 https://1.1.1.1/
}

prove_host_service_blocked() {
    ! container_curl --noproxy '*' --fail --connect-timeout 5 http://host.docker.internal:22/
}

prove_domain_blocked() {
    ! docker run --rm --network "${BRIDGE_NAME}" \
        --add-host=host.docker.internal:host-gateway \
        --env HTTPS_PROXY=http://host.docker.internal:3128 \
        "${TRAIN_IMAGE_DIGEST}" curl --fail --connect-timeout 5 https://example.com/
}

prove_ipv6_blocked() {
    ! container_curl --noproxy '*' --fail --connect-timeout 5 'https://[2606:4700:4700::1111]/'
}

prove_dns_blocked() {
    ! docker run --rm \
        --network "${BRIDGE_NAME}" \
        --dns 127.0.0.1 \
        "${TRAIN_IMAGE_DIGEST}" python3 -c \
            'import socket; socket.getaddrinfo("example.com", 443)'
}

prove_metadata_reachable() {
    container_curl --noproxy '*' --fail --connect-timeout 5 \
        --header 'Metadata-Flavor: ibm' \
        "http://${METADATA_ADDRESS}/metadata/v1/instance?version=2022-03-01" >/dev/null
}

prove_cos_reachable() {
    local status
    status=$(docker run --rm --network "${BRIDGE_NAME}" \
        --add-host=host.docker.internal:host-gateway \
        --env HTTPS_PROXY=http://host.docker.internal:3128 \
        "${TRAIN_IMAGE_DIGEST}" curl --silent --show-error --connect-timeout 5 \
        --output /dev/null --write-out '%{http_code}' \
        "https://s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud/")
    [[ ${status} != 000 ]]
}

obtain_iam_token() {
    local instance_identity_token
    instance_identity_token=$(curl --fail --silent --show-error --request PUT \
        --header 'Metadata-Flavor: ibm' \
        --header 'Content-Type: application/json' \
        --data '{"expires_in":3600}' \
        "http://${METADATA_ADDRESS}/instance_identity/v1/token?version=2022-03-01" \
        | jq -er '.access_token')
    IAM_TOKEN=$(curl --config <(printf 'data-urlencode = "cr_token=%s"\n' "${instance_identity_token}") \
        --fail --silent --show-error --request POST \
        --header 'Content-Type: application/x-www-form-urlencoded' \
        --data-urlencode 'grant_type=urn:ibm:params:oauth:grant-type:cr-token' \
        --data-urlencode "profile_id=${TRUSTED_PROFILE_ID}" \
        https://iam.cloud.ibm.com/identity/token | jq -er '.access_token')
    unset instance_identity_token
}

cos_curl() {
    curl --config <(printf 'header = "Authorization: Bearer %s"\n' "${IAM_TOKEN}") \
        --header "ibm-service-instance-id: ${COS_SERVICE_INSTANCE_ID}" "$@"
}

upload_proof() {
    local proof_hash=$1 existing_file status proof_url
    existing_file=$(mktemp /run/watch-auth-existing-proof.XXXXXX)
    proof_url="https://s3.direct.${IBM_REGION}.cloud-object-storage.appdomain.cloud/${COS_BUCKET}/${COS_RUN_PREFIX}/bootstrap/bootstrap-proof-${proof_hash}.json"
    status=$(cos_curl --silent --show-error --request PUT \
        --header 'If-None-Match: *' \
        --header 'Content-Type: application/json' \
        --upload-file "${PROOF_PATH}" \
        --output /dev/null --write-out '%{http_code}' "${proof_url}")
    if [[ ${status} == 412 ]]; then
        cos_curl --fail --silent --show-error "${proof_url}" --output "${existing_file}"
        cmp --silent "${PROOF_PATH}" "${existing_file}"
    else
        [[ ${status} == 200 ]]
    fi
    rm -f "${existing_file}"
}

verify_gpu() {
    local gpu_csv gpu_name driver_version memory_mib
    gpu_csv=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits)
    IFS=',' read -r gpu_name driver_version memory_mib <<<"${gpu_csv}"
    [[ ${gpu_name} == *L4* ]]
    [[ ${driver_version%%.*} -ge 550 ]]
    [[ ${memory_mib// /} -ge 22528 ]]
    docker run --rm --gpus all --network none "${TRAIN_IMAGE_DIGEST}" python3 -c \
        'import torch; assert torch.__version__.startswith("2.10.0"); assert torch.version.cuda == "12.6"; assert torch.cuda.is_available(); assert "L4" in torch.cuda.get_device_name(0); assert torch.cuda.get_device_properties(0).total_memory >= 22 * 1024**3'
}

pull_training_image() (
    local docker_config icr_host
    docker_config=$(mktemp -d /run/watch-auth-docker.XXXXXX)
    trap 'rm -rf -- "${docker_config}"' EXIT
    icr_host=${TRAIN_IMAGE_DIGEST%%/*}
    printf '%s' "${IAM_TOKEN}" \
        | DOCKER_CONFIG="${docker_config}" docker login "${icr_host}" \
            --username iamtoken --password-stdin >/dev/null
    DOCKER_CONFIG="${docker_config}" docker pull "${TRAIN_IMAGE_DIGEST}" >/dev/null
    DOCKER_CONFIG="${docker_config}" docker logout "${icr_host}" >/dev/null
)

launch_training_container() {
    if docker container inspect watch-auth-training >/dev/null 2>&1; then
        [[ $(docker inspect -f '{{index .Config.Labels "watch-auth.image-digest"}}' watch-auth-training) == "${TRAIN_IMAGE_DIGEST}" ]]
        if [[ $(docker inspect -f '{{.State.Running}}' watch-auth-training) == false ]]; then
            [[ $(docker inspect -f '{{.State.ExitCode}}' watch-auth-training) == 0 ]]
            return
        fi
    fi
    docker run --detach --name watch-auth-training \
        --label "watch-auth.image-digest=${TRAIN_IMAGE_DIGEST}" \
        --gpus all \
        --network watch-auth-isolated \
        --dns 127.0.0.1 \
        --read-only \
        --cap-drop=ALL \
        --security-opt=no-new-privileges \
        --tmpfs /tmp \
        --env HTTPS_PROXY=http://host.docker.internal:3128 \
        --env NO_PROXY=169.254.169.254 \
        --env "RUN_ID=${RUN_ID}" \
        --env "RUN_SPEC_COS_URI=cos://${COS_BUCKET}/${COS_RUN_PREFIX}/run-spec.json" \
        --env "WATCH_AUTH_REGION=${IBM_REGION}" \
        --env "WATCH_AUTH_COS_ENDPOINT=${COS_ENDPOINT}" \
        --env "WATCH_AUTH_COS_BUCKET=${COS_BUCKET}" \
        --env "WATCH_AUTH_COS_PREFIX=${COS_RUN_PREFIX}" \
        --env "WATCH_AUTH_COS_SERVICE_INSTANCE_ID=${COS_SERVICE_INSTANCE_ID}" \
        --env "WATCH_AUTH_TRUSTED_PROFILE_ID=${TRUSTED_PROFILE_ID}" \
        --env "WATCH_AUTH_CODE_SHA=${CODE_SHA}" \
        --env "WATCH_AUTH_IMAGE_DIGEST=${TRAIN_IMAGE_DIGEST}" \
        --env "WATCH_AUTH_MANIFEST_SHA256=${MANIFEST_SHA256}" \
        --env "WATCH_AUTH_SPLITS_SHA256=${SPLITS_SHA256}" \
        --env "WATCH_AUTH_RUN_SPEC_TRANSPORT_SHA256=${RUN_SPEC_TRANSPORT_SHA256}" \
        --add-host=host.docker.internal:host-gateway \
        "${TRAIN_IMAGE_DIGEST}" >/dev/null
    for _ in $(seq 1 20); do
        [[ $(docker inspect -f '{{.State.Health.Status}}' watch-auth-training) == healthy ]] && return
        if [[ $(docker inspect -f '{{.State.Running}}' watch-auth-training) == false ]]; then
            [[ $(docker inspect -f '{{.State.ExitCode}}' watch-auth-training) == 0 ]]
            return
        fi
        sleep 3
    done
    return 1
}

stage2() {
    if [[ -e ${STAGE2_COMPLETE} && -s ${PROOF_PATH} ]] \
        && docker inspect -f '{{.State.Health.Status}}' watch-auth-training 2>/dev/null | grep -qx healthy; then
        exit 0
    fi
    cloud-init status --wait --long | grep -q 'status: done'
    require_exact_package "${NVIDIA_DRIVER_PACKAGE}"
    require_exact_package "${NVIDIA_TOOLKIT_PACKAGE}"
    require_exact_package "${SQUID_PACKAGE}"
    require_exact_package "${DOCKER_PACKAGE}"
    configure_network
    configure_proxy
    obtain_iam_token
    pull_training_image
    prove_direct_https_blocked
    prove_host_service_blocked
    prove_domain_blocked
    prove_ipv6_blocked
    prove_dns_blocked
    prove_metadata_reachable
    prove_cos_reachable
    verify_gpu
    launch_training_container
    install -d -m 0700 "${STATE_DIR}"
    jq -cnS \
        --arg run_id "${RUN_ID}" \
        --arg image_digest "${TRAIN_IMAGE_DIGEST}" \
        --arg gpu "$(nvidia-smi --query-gpu=name --format=csv,noheader)" \
        '{run_id:$run_id,image_digest:$image_digest,gpu:$gpu,network_proof:"passed",worker_launch:"accepted"}' \
        >"${PROOF_PATH}"
    chmod 0600 "${PROOF_PATH}"
    local proof_hash
    proof_hash=$(sha256sum "${PROOF_PATH}" | awk '{print $1}')
    upload_proof "${proof_hash}"
    touch "${STAGE2_COMPLETE}"
    unset IAM_TOKEN
}

require_root
IAM_TOKEN=""
case ${1:-stage1} in
    stage1) stage1 ;;
    stage2) stage2 ;;
    *) echo "usage: $0 [stage1|stage2]" >&2; exit 64 ;;
esac
