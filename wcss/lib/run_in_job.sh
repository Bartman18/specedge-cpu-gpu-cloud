#!/usr/bin/env bash
set -euo pipefail

# Runs a single reproduction experiment inside a SLURM allocation.
# Usage: run_in_job.sh <run_meta.json>

RUN_META="${1:?run_meta.json required}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

wcss_check_project

METHOD=$("${WCSS_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['method'])" "${RUN_META}")
CONFIG=$("${WCSS_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['config_path'])" "${RUN_META}")
RUN_ID=$("${WCSS_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['run_id'])" "${RUN_META}")

cd "${WCSS_SPECEDGE_ROOT}"

export HF_HOME="${WCSS_HF_HOME}"
export XDG_CACHE_HOME="${WCSS_REPRO_ROOT}/xdg_cache"
mkdir -p "${XDG_CACHE_HOME}"

echo "=== SpecEdge reproduction run: ${RUN_ID} ==="
echo "Method:  ${METHOD}"
echo "Config:  ${CONFIG}"
echo "Node:    $(hostname)"
echo "GPUs:    $(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
nvidia-smi || true

wait_for_port() {
    local port="$1"
    local retries=120
    local i
    for ((i = 1; i <= retries; i++)); do
        if "${WCSS_PYTHON}" - <<PY
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(("127.0.0.1", ${port}))
    sys.exit(0)
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
        then
            echo "gRPC port ${port} accepting connections (attempt ${i})"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: gRPC server did not start on port ${port}" >&2
    return 1
}

# Port 8000 opens before the inference child finishes model load / KV prefill.
# Clients must wait for the inference loop or Validate RPCs time out (5s).
wait_for_server_inference() {
    local log_file="$1"
    local retries=360
    local i
    for ((i = 1; i <= retries; i++)); do
        if [[ -f "${log_file}" ]] && grep -q "Starting inference loop" "${log_file}"; then
            echo "Server inference loop ready (attempt ${i})"
            return 0
        fi
        if (( i % 6 == 0 )); then
            echo "Waiting for server inference loop... attempt ${i}/${retries}"
        fi
        sleep 10
    done
    echo "ERROR: server inference loop did not start within timeout" >&2
    if [[ -f "${log_file}" ]]; then
        echo "Last 40 lines of ${log_file}:" >&2
        tail -40 "${log_file}" >&2 || true
    fi
    return 1
}

shutdown_server() {
    local pid="$1"
    if ! kill -0 "${pid}" 2>/dev/null; then
        return 0
    fi
    echo "Stopping batch server (PID ${pid})..."
    kill -TERM "${pid}" 2>/dev/null || true
    local i
    for ((i = 1; i <= 120; i++)); do
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "Server stopped after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "WARN: server did not exit after SIGTERM, sending SIGKILL" >&2
    kill -KILL "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
}

RESULT_DIR=$("${WCSS_PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1]))['result_dir'])" "${RUN_META}")
mkdir -p "${RESULT_DIR}"

{
    echo "run_id=${RUN_ID}"
    echo "method=${METHOD}"
    echo "hostname=$(hostname)"
    echo "slurm_job_id=${SLURM_JOB_ID:-local}"
    echo "start_time=$(date -Iseconds)"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
} > "${RESULT_DIR}/job_env.txt"

case "${METHOD}" in
    server_only)
        ./script/server_only.sh -f "${CONFIG}"
        ;;
    specedge)
        SERVER_LOG="${RESULT_DIR}/raw/run/server.log"
        mkdir -p "${RESULT_DIR}/raw/run"

        ./script/batch_server.sh -f "${CONFIG}" &
        SERVER_PID=$!
        trap 'shutdown_server "${SERVER_PID}"' EXIT

        wait_for_port 8000
        wait_for_server_inference "${SERVER_LOG}"

        "${WCSS_PYTHON}" "${SCRIPT_DIR}/client_local.py" --config "${CONFIG}"
        CLIENT_RC=$?

        shutdown_server "${SERVER_PID}"
        trap - EXIT

        if [[ ${CLIENT_RC} -ne 0 ]]; then
            echo "ERROR: edge clients failed with code ${CLIENT_RC}" >&2
            exit "${CLIENT_RC}"
        fi

        if [[ ! -s "${RESULT_DIR}/raw/run/server.jsonl" ]]; then
            echo "ERROR: server.jsonl is empty after specedge run" >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: unknown method ${METHOD}" >&2
        exit 1
        ;;
esac

echo "end_time=$(date -Iseconds)" >> "${RESULT_DIR}/job_env.txt"
echo "=== Run ${RUN_ID} completed ==="
