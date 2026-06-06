#!/usr/bin/env bash
set -eo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"

MODEL="Qwen/Qwen3.5-9B"
JUDGE_MODEL="Qwen/Qwen3.6-27B"
CAPTION_MODEL="Qwen/Qwen3.6-27B"
VISION_ENCODER="google/siglip-so400m-patch14-384"
DOMAIN="pathvqa"
TRAIN_NUM=300
TRAIN_SAMPLES=300
NUM_TRACES=16
MAX_ROUTING=3
LR="1e-5"
TEMPERATURE=0.7
DECAY_FACTOR=0.98
BATCH_SIZE=8
EPOCHS=1
TRACE_PARALLELISM=128
RESULT_DIR="result/pathvqa_img_embed"
ENTROPY_BETA=0.05
EVAL_LIMIT=2000
EVAL_TEMPS="0.5"
EVAL_PARALLELISM=128
SEED=99
TRAIN_JSON_PATH="${TRAIN_JSON_PATH:-}"
TEST_JSON_PATH="${TEST_JSON_PATH:-}"
export MEDSETS_REQUIRED="${MEDSETS_REQUIRED:-1}"
TRAIN_GPU="${TRAIN_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
TRAIN_PORT="${TRAIN_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8001}"
START_TRAIN_VLLM="${START_TRAIN_VLLM:-1}"
START_JUDGE_VLLM="${START_JUDGE_VLLM:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
HF_CACHE_DIR="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}}"
MEDSETS_EXTRACT_DIR="${MEDSETS_EXTRACT_DIR:-$HOME/.cache/medroute/medsets_extracted}"
DEFAULT_LOCAL_MEDIA_PATH="$MEDSETS_EXTRACT_DIR"
if [ ! -d "$DEFAULT_LOCAL_MEDIA_PATH" ]; then
    DEFAULT_LOCAL_MEDIA_PATH="$HF_CACHE_DIR"
fi
DEFAULT_LOCAL_MEDIA_PATH="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())" "$DEFAULT_LOCAL_MEDIA_PATH")"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-$DEFAULT_LOCAL_MEDIA_PATH}"
ALLOWED_LOCAL_MEDIA_PATH="$(python -c "import pathlib, sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())" "$ALLOWED_LOCAL_MEDIA_PATH")"
RESUME_GRADIENT_PATH="${RESUME_GRADIENT_PATH:-}"

mkdir -p "$RESULT_DIR" logs

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

    local logfile="$REPO_DIR/logs/vllm_${label}_$(date +%Y-%m-%d-%H-%M-%S).log"
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

DATA_ARGS=()
if [ -n "$TRAIN_JSON_PATH" ]; then
    DATA_ARGS+=(--train_json_path "$TRAIN_JSON_PATH")
fi
if [ -n "$TEST_JSON_PATH" ]; then
    DATA_ARGS+=(--test_json_path "$TEST_JSON_PATH")
fi

TRAIN_ARGS=("${DATA_ARGS[@]}")
if [ -n "$RESUME_GRADIENT_PATH" ]; then
    TRAIN_ARGS+=(--resume_gradient_path "$RESUME_GRADIENT_PATH")
fi

echo "============================================"
echo "  PathVQA Image Embedding Routing Training"
echo "  Agent Model: $MODEL (VLM, also DM)"
echo "  Pool/Prompt: $JUDGE_MODEL"
echo "  Vision Encoder: $VISION_ENCODER (routing embeddings)"
echo "  Config: train=$TRAIN_NUM, samples=$TRAIN_SAMPLES, traces=$NUM_TRACES, mr=$MAX_ROUTING"
echo "          lr=$LR, GS_tau=$TEMPERATURE, decay=$DECAY_FACTOR, batch=$BATCH_SIZE"
echo "  Epochs: $EPOCHS | Entropy beta: $ENTROPY_BETA"
echo "  Eval temps: $EVAL_TEMPS | Eval limit: $EVAL_LIMIT | Eval parallelism: $EVAL_PARALLELISM"
echo "  vLLM max-model-len: $MAX_MODEL_LEN | gpu-memory-utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "  vLLM startup timeout: ${VLLM_STARTUP_TIMEOUT}s"
echo "  vLLM DeepGEMM: use=$VLLM_USE_DEEP_GEMM | warmup=$VLLM_DEEP_GEMM_WARMUP"
echo "  vLLM startup: train=$START_TRAIN_VLLM | judge=$START_JUDGE_VLLM"
echo "  Allowed local media path: $ALLOWED_LOCAL_MEDIA_PATH"
echo "  Medsets root: ${MEDSETS_ROOT:-auto hf cache} (override with MEDSETS_ROOT)"
echo "  path override: ${TRAIN_JSON_PATH:-none} | ${TEST_JSON_PATH:-none} (optional Medsets-style CSV)"
echo "  Result dir: $RESULT_DIR"
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

CKPT_DIR=$(ls -dt "$RESULT_DIR"/20*/ 2>/dev/null | head -1 || true)
EXISTING_EPOCHS=0
if [ -n "$CKPT_DIR" ]; then
    EXISTING_EPOCHS=$(ls "$CKPT_DIR"/*_epoch*.pth 2>/dev/null | wc -l)
fi

if [ "$EXISTING_EPOCHS" -ge "$EPOCHS" ]; then
    echo ""
    echo "Already have $EXISTING_EPOCHS epoch checkpoints — skipping training"
else
    echo ""
    echo "Training from scratch for $EPOCHS epochs..."
    python experiments/run_pathvqa.py \
        --domain "$DOMAIN" \
        --llm_name "$MODEL" \
        --judge_model "$JUDGE_MODEL" \
        --caption_model "$CAPTION_MODEL" \
        --epochs "$EPOCHS" \
        --train_num "$TRAIN_NUM" \
        --train_samples "$TRAIN_SAMPLES" \
        --max_routing "$MAX_ROUTING" \
        --num_traces "$NUM_TRACES" \
        --trace_parallelism "$TRACE_PARALLELISM" \
        --batch_size "$BATCH_SIZE" \
        --temperature "$TEMPERATURE" \
        --decay_factor "$DECAY_FACTOR" \
        --lr "$LR" \
        --entropy_beta "$ENTROPY_BETA" \
        --seed "$SEED" \
        --dynamic_prompts --dynamic_pool \
        --use_image_embeddings --vision_encoder "$VISION_ENCODER" \
        --result_dir "$RESULT_DIR" \
        "${TRAIN_ARGS[@]}"
    echo "Training complete"
fi

CKPT_DIR=$(ls -dt "$RESULT_DIR"/20*/ 2>/dev/null | head -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoints found after training"
    exit 1
fi

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
            python experiments/run_pathvqa.py \
                --domain "$DOMAIN" \
                --llm_name "$MODEL" \
                --judge_model "$JUDGE_MODEL" \
                --caption_model "$CAPTION_MODEL" \
                --model_path "$CKPT" \
                --epochs 0 --train_num 0 \
                --train_samples "$TRAIN_SAMPLES" \
                --max_routing "$MAX_ROUTING" \
                --num_traces "$NUM_TRACES" \
                --trace_parallelism "$EVAL_PARALLELISM" \
                --entropy_beta "$ENTROPY_BETA" \
                --eval_temperature "$tau" \
                --eval_limit "$EVAL_LIMIT" \
                --seed "$SEED" \
                --dynamic_prompts --dynamic_pool \
                --use_image_embeddings --vision_encoder "$VISION_ENCODER" \
                --result_dir "$EDIR" \
                "${DATA_ARGS[@]}" &
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

echo ""
echo "============================================"
echo "  RESULTS"
echo "============================================"
printf "%-8s %-12s %-10s %-10s\n" "Epoch" "Eval Temp" "Overlap" "Judge"
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
