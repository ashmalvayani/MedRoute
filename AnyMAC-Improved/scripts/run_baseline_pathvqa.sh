#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"
export MEDSETS_REQUIRED="${MEDSETS_REQUIRED:-1}"

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3.6-27B}"
CONCURRENCY="${CONCURRENCY:-8}"
LIMIT="${LIMIT:-}"
SEED="${SEED:-99}"
TRAIN_JSON_PATH="${TRAIN_JSON_PATH:-}"
TEST_JSON_PATH="${TEST_JSON_PATH:-}"

BASE_GPU="${BASE_GPU:-0}"
BASE_PORT="${BASE_PORT:-8000}"
JUDGE_GPU="${JUDGE_GPU:-1}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
START_BASE_VLLM="${START_BASE_VLLM:-1}"
START_JUDGE_VLLM="${START_JUDGE_VLLM:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-1024}"
HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}"
MEDSETS_EXTRACT_DIR="${MEDSETS_EXTRACT_DIR:-$HOME/.cache/medroute/medsets_extracted}"
DEFAULT_LOCAL_MEDIA_PATH="$MEDSETS_EXTRACT_DIR"
if [ ! -d "$DEFAULT_LOCAL_MEDIA_PATH" ]; then
    DEFAULT_LOCAL_MEDIA_PATH="$HF_CACHE_DIR"
fi
DEFAULT_LOCAL_MEDIA_PATH="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())" "$DEFAULT_LOCAL_MEDIA_PATH")"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$DEFAULT_LOCAL_MEDIA_PATH}"
ALLOWED_LOCAL_MEDIA_PATH="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())" "$ALLOWED_LOCAL_MEDIA_PATH")"

mkdir -p "$REPO_DIR/logs"
TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
LOG="$REPO_DIR/logs/baseline_pathvqa_${TIMESTAMP}.log"

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
            --chat-template-content-format openai \
            --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
            --gdn-prefill-backend triton \
        > "$logfile" 2>&1 &
    local vllm_pid=$!
    echo "[vllm:${label}] PID=${vllm_pid}"

    wait_for_vllm "$port" "$label" "$VLLM_STARTUP_TIMEOUT" "$vllm_pid" "$logfile"
}

{
    echo "========================================"
    echo "  AnyMAC-Improved baseline eval (PathVQA)"
    echo "  Started      : $(date)"
    echo "  Base model   : $MODEL (GPU $BASE_GPU, port $BASE_PORT)"
    echo "  Judge model  : $JUDGE_MODEL (GPU $JUDGE_GPU, port $JUDGE_PORT)"
    echo "  Concurrency  : $CONCURRENCY"
    echo "  Limit        : ${LIMIT:-all}"
    echo "  Seed         : $SEED"
    echo "  train_json   : ${TRAIN_JSON_PATH:-(unset; Medsets PathVQA pathvqa_train.csv)}"
    echo "  test_json    : ${TEST_JSON_PATH:-(unset)}"
    echo "  Media path   : $ALLOWED_LOCAL_MEDIA_PATH"
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
    if [ -n "$TRAIN_JSON_PATH" ]; then
        EXTRA_ARGS+=(--train_json_path "$TRAIN_JSON_PATH")
    fi
    if [ -n "$TEST_JSON_PATH" ]; then
        EXTRA_ARGS+=(--test_json_path "$TEST_JSON_PATH")
    fi

    python experiments/baseline_pathvqa.py \
        --model "$MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --concurrency "$CONCURRENCY" \
        --seed "$SEED" \
        --max_tokens "$LLM_MAX_TOKENS" \
        "${EXTRA_ARGS[@]}"

    echo ""
    echo "[run] Finished: $(date)"
    echo "========================================"
} 2>&1 | tee "$LOG"

echo ""
echo "Full log saved to: $LOG"
echo "Results directory: $REPO_DIR/result/pathvqa_baseline/"
