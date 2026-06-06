#!/usr/bin/env bash
# PubMedQA baseline: vLLM base + judge (text-only), same models as train_pubmedqa.sh.
# Data: datasets_my/PubMedQA/data/{test|context/test} — not Medsets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-32B}"
CONCURRENCY="${CONCURRENCY:-8}"
LIMIT="${LIMIT:-}"
USE_CONTEXT="${USE_CONTEXT:-0}"
SPLIT="${SPLIT:-test}"

BASE_GPU="${BASE_GPU:-0}"
BASE_PORT="${BASE_PORT:-8000}"
JUDGE_GPU="${JUDGE_GPU:-1}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
START_BASE_VLLM="${START_BASE_VLLM:-1}"
START_JUDGE_VLLM="${START_JUDGE_VLLM:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"

mkdir -p "$REPO_DIR/logs"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG="$REPO_DIR/logs/baseline_pubmedqa_${TIMESTAMP}.log"

if [ "$USE_CONTEXT" = "1" ] || [ "$USE_CONTEXT" = "true" ]; then
    _PQA_DATA_DESC="datasets_my/PubMedQA/data/context/test"
else
    _PQA_DATA_DESC="datasets_my/PubMedQA/data/${SPLIT}"
fi

wait_for_vllm() {
    local port="$1"
    local label="$2"
    local max_wait="${3:-$VLLM_STARTUP_TIMEOUT}"
    local pid="${4:-}"
    local logfile="${5:-}"

    for i in $(seq 1 "$max_wait"); do
        if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "[vllm:${label}] ready after ${i}s"
            return 0
        fi

        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo "[vllm:${label}] ERROR: process exited before becoming ready (pid=${pid})"
            if [ -n "$logfile" ] && [ -f "$logfile" ]; then
                echo "[vllm:${label}] Last logs:"
                tail -n 60 "$logfile" || true
            fi
            return 1
        fi

        if [ $((i % 30)) -eq 0 ]; then
            echo "[vllm:${label}] waiting... ${i}s/${max_wait}s"
        fi
        sleep 1
    done
    echo "[vllm:${label}] ERROR: not responding after ${max_wait}s"
    if [ -n "$logfile" ] && [ -f "$logfile" ]; then
        echo "[vllm:${label}] Last logs:"
        tail -n 60 "$logfile" || true
    fi
    return 1
}

start_vllm() {
    local gpu="$1"
    local port="$2"
    local model="$3"
    local label="$4"

    if curl -s "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
        echo "[vllm:${label}] already running on port ${port} — skipping"
        return 0
    fi

    local logfile="$REPO_DIR/logs/vllm_${label}_${TIMESTAMP}.log"
    echo "[vllm:${label}] starting ${model} on GPU ${gpu}, port ${port} (gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION})"

    CUDA_VISIBLE_DEVICES="$gpu" \
        VLLM_USE_DEEP_GEMM="$VLLM_USE_DEEP_GEMM" \
        VLLM_DEEP_GEMM_WARMUP="$VLLM_DEEP_GEMM_WARMUP" \
        VLLM_ATTENTION_BACKEND=FLASH_ATTN \
        nohup python -m vllm.entrypoints.openai.api_server \
            --model "$model" \
            --port "$port" \
            --dtype auto \
            --max-model-len "$MAX_MODEL_LEN" \
            --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
            --no-enable-log-requests \
            --chat-template-content-format string \
            --gdn-prefill-backend triton \
        > "$logfile" 2>&1 &
    local vllm_pid=$!
    echo "[vllm:${label}] PID=${vllm_pid}"

    wait_for_vllm "$port" "$label" "$VLLM_STARTUP_TIMEOUT" "$vllm_pid" "$logfile"
}

{
    echo "========================================"
    echo "  AnyMAC-Improved baseline eval (PubMedQA)"
    echo "  Started      : $(date)"
    echo "  Base model   : $MODEL (GPU $BASE_GPU, port $BASE_PORT)"
    echo "  Judge model  : $JUDGE_MODEL (GPU $JUDGE_GPU, port $JUDGE_PORT)"
    echo "  Concurrency  : $CONCURRENCY"
    echo "  Limit        : ${LIMIT:-all}"
    echo "  Data         : ${_PQA_DATA_DESC} (PubMedQA CSVs; not Medsets)"
    echo "  vLLM startup : base=$START_BASE_VLLM | judge=$START_JUDGE_VLLM | timeout=${VLLM_STARTUP_TIMEOUT}s"
    echo "  vLLM DeepGEMM: use=$VLLM_USE_DEEP_GEMM | warmup=$VLLM_DEEP_GEMM_WARMUP"
    echo "  vLLM max-model-len: $MAX_MODEL_LEN | gpu-memory-utilization: $VLLM_GPU_MEMORY_UTILIZATION | gen max_tokens: $LLM_MAX_TOKENS"
    echo "  Log          : $LOG"
    echo "========================================"

    if [ "$START_BASE_VLLM" = "1" ]; then
        start_vllm "$BASE_GPU" "$BASE_PORT" "$MODEL" "base"
        export BASE_URL="http://localhost:${BASE_PORT}"
    else
        export BASE_URL="${BASE_URL:-http://localhost:${BASE_PORT}}"
        echo "[vllm:base] startup disabled; using BASE_URL=$BASE_URL"
    fi

    if [ "$START_JUDGE_VLLM" = "1" ]; then
        start_vllm "$JUDGE_GPU" "$JUDGE_PORT" "$JUDGE_MODEL" "judge"
        export JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}"
    else
        export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:${JUDGE_PORT}}"
        echo "[vllm:judge] startup disabled; using JUDGE_BASE_URL=$JUDGE_BASE_URL"
    fi

    EXTRA_ARGS=()
    if [ -n "$LIMIT" ]; then
        EXTRA_ARGS+=(--limit "$LIMIT")
    fi
    if [ "$USE_CONTEXT" = "1" ] || [ "$USE_CONTEXT" = "true" ]; then
        EXTRA_ARGS+=(--use_context)
    else
        EXTRA_ARGS+=(--split "$SPLIT")
    fi

    python experiments/baseline_pubmedqa.py \
        --model "$MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --max_tokens "$LLM_MAX_TOKENS" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "[run] Finished: $(date)"
    echo "========================================"
} 2>&1 | tee "$LOG"

echo ""
echo "Full log saved to: $LOG"
echo "Results directory: $REPO_DIR/result/pubmedqa_baseline/"
