#!/usr/bin/env bash
# Launch the full 3-tier cascade on one node: M2 cloud target (gRPC, cuda:0) +
# M0(CPU)->M1(cuda:1) bench client. Mirrors the specedge branch of
# wcss/lib/run_in_job.sh but for our cloud pipeline.
#
# Usage: script/cloud3.sh [-f] (env overrides below)
set -euo pipefail

cd "$(dirname "$0")/.." || { echo "Failed to cd to project root"; exit 1; }

if [ ! -d .venv ]; then
    echo "You need to create a virtual environment first." >&2
    exit 1
fi
source .venv/bin/activate || { echo "Failed to activate venv"; exit 1; }
export PYTHONPATH="src:${PYTHONPATH:-}"

# --- configuration (override via environment) ---
TARGET_MODEL="${CLOUD3_TARGET_MODEL:-Qwen/Qwen3-14B}"
VERIFY_MODEL="${CLOUD3_VERIFY_MODEL:-Qwen/Qwen3-1.7B}"
DRAFT_MODEL="${CLOUD3_DRAFT_MODEL:-Qwen/Qwen3-0.6B}"
SERVER_DEVICE="${CLOUD3_SERVER_DEVICE:-cuda:0}"
VERIFY_DEVICE="${CLOUD3_VERIFY_DEVICE:-cuda:1}"
DTYPE="${CLOUD3_DTYPE:-fp16}"
PORT="${CLOUD3_PORT:-8000}"
DATASET="${CLOUD3_DATASET:-specbench}"
SAMPLE_REQ_CNT="${CLOUD3_SAMPLE_REQ_CNT:-8}"
MAX_REQUEST_NUM="${CLOUD3_MAX_REQUEST_NUM:--1}"
MAX_NEW_TOKENS="${CLOUD3_MAX_NEW_TOKENS:-256}"
GAMMA1="${CLOUD3_GAMMA1:-16}"
TEMPERATURE="${CLOUD3_TEMPERATURE:-0.0}"
RESULT_PATH="${CLOUD3_RESULT_PATH:-result/cloud3}"
EXP_NAME="${CLOUD3_EXP_NAME:-run}"

RESULT_DIR="${RESULT_PATH}/${EXP_NAME}"
mkdir -p "${RESULT_DIR}"
SERVER_LOG="${RESULT_DIR}/server.out"

echo "=== cloud3 run ==="
echo "target(M2)=${TARGET_MODEL}@${SERVER_DEVICE}  verify(M1)=${VERIFY_MODEL}@${VERIFY_DEVICE}  draft(M0)=${DRAFT_MODEL}@cpu"
echo "dataset=${DATASET}  result_dir=${RESULT_DIR}"
nvidia-smi -L 2>/dev/null || true

# --- start M2 cloud target in the background ---
python -O src/script/cloud_server.py \
    --target-model "${TARGET_MODEL}" \
    --device "${SERVER_DEVICE}" \
    --dtype "${DTYPE}" \
    --port "${PORT}" \
    --temperature "${TEMPERATURE}" \
    --result-path "${RESULT_PATH}" \
    --exp-name "${EXP_NAME}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

shutdown_server() {
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "Stopping M2 server (PID ${SERVER_PID})..."
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${SERVER_PID}" 2>/dev/null || return 0
            sleep 1
        done
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap shutdown_server EXIT

echo "Waiting for M2 server on port ${PORT} (model load may take minutes)..."
for i in $(seq 1 360); do
    if grep -q "listening on port" "${SERVER_LOG}" 2>/dev/null; then
        echo "M2 server ready (attempt ${i})"
        break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: M2 server exited during startup. Last 40 lines:" >&2
        tail -40 "${SERVER_LOG}" >&2 || true
        exit 1
    fi
    sleep 5
done

# --- run the cascade bench client ---
python src/script/bench_cloud_pipeline.py \
    --draft-model "${DRAFT_MODEL}" \
    --verify-model "${VERIFY_MODEL}" \
    --draft-device cpu \
    --verify-device "${VERIFY_DEVICE}" \
    --verify-dtype "${DTYPE}" \
    --cloud-host "127.0.0.1:${PORT}" \
    --dataset "${DATASET}" \
    --sample-req-cnt "${SAMPLE_REQ_CNT}" \
    --max-request-num "${MAX_REQUEST_NUM}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --gamma1 "${GAMMA1}" \
    --temperature "${TEMPERATURE}" \
    --result-path "${RESULT_PATH}" \
    --exp-name "${EXP_NAME}"
CLIENT_RC=$?

shutdown_server
trap - EXIT

if [ "${CLIENT_RC}" -ne 0 ]; then
    echo "ERROR: cascade client failed (rc=${CLIENT_RC})" >&2
    exit "${CLIENT_RC}"
fi

if [ ! -s "${RESULT_DIR}/server.jsonl" ]; then
    echo "WARN: server.jsonl is empty (metrics aggregation needs it)" >&2
fi

echo "=== cloud3 run complete -> ${RESULT_DIR} ==="
echo "Metrics: PYTHONPATH=src python src/metric/specedge.py -d ${RESULT_DIR} -s overall --gpu H100_94"
