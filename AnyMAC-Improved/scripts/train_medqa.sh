#!/usr/bin/env bash
# MedQA training: dynamic pool + dynamic prompts, same VLM IDs as PubMedQA / vision (seed=99)
#
# Usage:
#   bash scripts/train_medqa.sh
set -eo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Activate venv
VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"

# ---- Config (locked hyperparams from previous experiments) ----
MODEL="Qwen/Qwen3-8B"
JUDGE_MODEL="Qwen/Qwen3-32B"
DOMAIN="medqa"
TRAIN_NUM=300
NUM_TRACES=16
MAX_ROUTING=3
LR="3e-5"
TEMPERATURE=1.0
DECAY_FACTOR=0.98
BATCH_SIZE=8
EPOCHS=1
TRAIN_PARALLELISM=256
EVAL_PARALLELISM=256
RESULT_DIR="result/medqa_train"
ENTROPY_BETA=0.05
EVAL_LIMIT=1500
EVAL_TEMPS="0.7"
SEED=99
TRAIN_GPU="${TRAIN_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
TRAIN_PORT="${TRAIN_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
START_TRAIN_VLLM="${START_TRAIN_VLLM:-1}"
START_JUDGE_VLLM="${START_JUDGE_VLLM:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.88}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"

mkdir -p "$RESULT_DIR" logs
TIMESTAMP_TRAIN=$(date +%Y-%m-%d-%H-%M-%S)

# ---- Helper: extract accuracy from eval dir ----
get_accuracy() {
    local eval_dir="$1"
    python3 -c "
import json, glob
files = [f for f in glob.glob('${eval_dir}/${DOMAIN}_*.json') if 'details' not in f and 'rejudge' not in f]
if not files: print('N/A N/A')
else:
    data = json.load(open(files[0]))
    items = [r for r in data if 'Index' in r]; n = len(items)
    r = sum(1 for x in items if x.get('Regex_solved',False))/n if n else 0
    j = sum(1 for x in items if x.get('Judge_solved',False))/n if n else 0
    print(f'{r:.4f} {j:.4f}')
" 2>/dev/null || echo "N/A N/A"
}

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

    local logfile="$REPO_DIR/logs/vllm_${label}_${TIMESTAMP_TRAIN}.log"
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

echo "============================================"
echo "  Dynamic Specialist Pool Training"
echo "  Model: $MODEL | Judge: $JUDGE_MODEL"
echo "  Config: train=$TRAIN_NUM, traces=$NUM_TRACES, mr=$MAX_ROUTING"
echo "          lr=$LR, GS_tau=$TEMPERATURE, decay=$DECAY_FACTOR, batch=$BATCH_SIZE"
echo "  Epochs: $EPOCHS | Entropy beta: $ENTROPY_BETA"
echo "  Improvements: dynamic_pool + dynamic_prompts"
echo "  Eval temps: $EVAL_TEMPS"
echo "  vLLM max-model-len: $MAX_MODEL_LEN | gpu-memory-utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  vLLM startup timeout: ${VLLM_STARTUP_TIMEOUT}s"
echo "  vLLM DeepGEMM: use=$VLLM_USE_DEEP_GEMM | warmup=$VLLM_DEEP_GEMM_WARMUP"
echo "  vLLM startup: train=$START_TRAIN_VLLM | judge=$START_JUDGE_VLLM"
echo "  Result dir: $RESULT_DIR"
echo "  Baseline     : bash scripts/run_baseline_medqa.sh"
echo "============================================"

if [ "$START_TRAIN_VLLM" = "1" ]; then
    start_vllm "$TRAIN_GPU" "$TRAIN_PORT" "$MODEL" "train"
    export BASE_URL="http://localhost:${TRAIN_PORT}"
else
    export BASE_URL="${BASE_URL:-http://localhost:${TRAIN_PORT}}"
    echo "[vllm:train] startup disabled; using BASE_URL=$BASE_URL"
fi

if [ "$START_JUDGE_VLLM" = "1" ]; then
    start_vllm "$JUDGE_GPU" "$JUDGE_PORT" "$JUDGE_MODEL" "judge"
    export JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}"
else
    export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:${JUDGE_PORT}}"
    echo "[vllm:judge] startup disabled; using JUDGE_BASE_URL=$JUDGE_BASE_URL"
fi

# ---- Check for existing checkpoints ----
CKPT_DIR=$(ls -dt "$RESULT_DIR"/20*/ 2>/dev/null | head -1 || true)
EXISTING_EPOCHS=0
if [ -n "$CKPT_DIR" ]; then
    EXISTING_EPOCHS=$(ls "$CKPT_DIR"/*_epoch*.pth 2>/dev/null | wc -l)
fi

# ---- Training ----
if [ "$EXISTING_EPOCHS" -ge "$EPOCHS" ]; then
    echo ""
    echo "Already have $EXISTING_EPOCHS epoch checkpoints — skipping training"
else
    echo ""
    echo "Training from scratch for $EPOCHS epochs..."
    python experiments/run_medqa.py \
        --llm_name "$MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --epochs "$EPOCHS" \
        --train_num "$TRAIN_NUM" \
        --max_routing "$MAX_ROUTING" \
        --num_traces "$NUM_TRACES" \
        --trace_parallelism "$TRAIN_PARALLELISM" \
        --batch_size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --decay_factor "$DECAY_FACTOR" \
        --lr "$LR" \
        --entropy_beta "$ENTROPY_BETA" \
        --seed "$SEED" \
        --dynamic_prompts --dynamic_pool \
        --result_dir "$RESULT_DIR"
    echo "Training complete"
fi

# ---- Refresh checkpoint dir ----
CKPT_DIR=$(ls -dt "$RESULT_DIR"/20*/ 2>/dev/null | head -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoints found after training"
    exit 1
fi

# ---- Eval each epoch checkpoint ----
echo ""
echo "============================================"
echo "  Eval all epoch checkpoints at temperature $EVAL_TEMPS"
echo "============================================"

for epoch in $(seq 1 "$EPOCHS"); do
    CKPT=$(ls "$CKPT_DIR"/*_epoch${epoch}.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then
        echo "WARNING: No epoch $epoch checkpoint found — skipping"
        continue
    fi
    echo ""
    echo "Epoch $epoch checkpoint: $CKPT"
    eval_pids=()

    for tau in $EVAL_TEMPS; do
        EDIR="${RESULT_DIR}/eval_epoch${epoch}_temp_${tau}"

        if ls "$EDIR"/${DOMAIN}_*.json >/dev/null 2>&1; then
            echo "  [epoch=$epoch, tau=$tau] Eval exists — skipping"
        else
            mkdir -p "$EDIR"
            CUDA_VISIBLE_DEVICES="" python experiments/run_medqa.py \
                --llm_name "$MODEL" \
                --judge_model "$JUDGE_MODEL" \
                --model_path "$CKPT" \
                --epochs 0 --train_num 0 \
                --max_routing "$MAX_ROUTING" \
                --num_traces "$NUM_TRACES" \
                --trace_parallelism "$EVAL_PARALLELISM" \
                --entropy_beta "$ENTROPY_BETA" \
                --eval_temperature "$tau" \
                --eval_limit "$EVAL_LIMIT" \
                --seed "$SEED" \
                --dynamic_prompts --dynamic_pool \
                --result_dir "$EDIR" &
            eval_pid=$!
            eval_pids+=("$eval_pid")
            echo "  [epoch=$epoch, tau=$tau] Eval launched (PID $eval_pid)"
        fi
    done
    if [ "${#eval_pids[@]}" -gt 0 ]; then
        wait "${eval_pids[@]}"
    fi
done

echo ""
echo "All evals done!"

# ---- Results ----
echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
printf "%-8s %-12s %-10s %-10s\n" "Epoch" "Eval Temp" "Regex" "Judge"
printf "%-8s %-12s %-10s %-10s\n" "------" "----------" "--------" "--------"

for epoch in $(seq 1 "$EPOCHS"); do
    for tau in $EVAL_TEMPS; do
        EDIR="${RESULT_DIR}/eval_epoch${epoch}_temp_${tau}"
        if [ -d "$EDIR" ]; then
            read -r regex judge <<< "$(get_accuracy "$EDIR")"
            printf "%-8s %-12s %-10s %-10s\n" "$epoch" "$tau" "$regex" "$judge"
        fi
    done
done

echo "============================================"
echo "Done!"
