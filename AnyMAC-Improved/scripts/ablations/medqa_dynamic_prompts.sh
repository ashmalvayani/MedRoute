#!/usr/bin/env bash
# MedQA ablation: dynamic prompts only, fixed 60-specialist pool
# 1. Train 1 epoch on 300 samples with entropy regularization
# 2. Eval the checkpoint at 5 different eval_temperatures in parallel
# 3. Print results table
#
# Usage:
#   bash scripts/train_improved_medqa.sh                          # defaults
#   bash scripts/train_improved_medqa.sh --baseline               # no improvements
#   bash scripts/train_improved_medqa.sh --train_num 100          # override
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"
export BASE_URL="${BASE_URL:-http://localhost:8000}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8001}"

# ---- Config ----
MODEL="Qwen/Qwen3-8B"
JUDGE_MODEL="Qwen/Qwen3-32B"
DOMAIN="medqa"
TRAIN_NUM=300
NUM_TRACES=16
MAX_ROUTING=3
LR="3e-5"
TEMPERATURE=1.0          # Gumbel-softmax tau (training)
DECAY_FACTOR=0.98
BATCH_SIZE=8
EPOCHS=1
TRAIN_PARALLELISM=256
EVAL_PARALLELISM=256
RESULT_DIR="result/dynamic_prompts_v2"
ENTROPY_BETA=0.05
EVAL_LIMIT=1500

# Eval temperature (single value, no sweep)
EVAL_TEMPS="0.7"

# Improvement flags
DYNAMIC_PROMPTS="--dynamic_prompts"
DYNAMIC_POOL="--dynamic_pool"
PARTIAL_CREDIT=""
STRUCTURED_HINTS=""
REWARD_ALPHA=0.4
PROMPT_MODEL=""  # if set, uses this model on BASE_URL for dynamic prompts instead of judge

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --judge_model) JUDGE_MODEL="$2"; shift 2;;
        --domain) DOMAIN="$2"; shift 2;;
        --train_num) TRAIN_NUM="$2"; shift 2;;
        --num_traces) NUM_TRACES="$2"; shift 2;;
        --max_routing) MAX_ROUTING="$2"; shift 2;;
        --lr) LR="$2"; shift 2;;
        --temperature) TEMPERATURE="$2"; shift 2;;
        --decay_factor) DECAY_FACTOR="$2"; shift 2;;
        --batch_size) BATCH_SIZE="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --train_parallelism) TRAIN_PARALLELISM="$2"; shift 2;;
        --eval_parallelism) EVAL_PARALLELISM="$2"; shift 2;;
        --result_dir) RESULT_DIR="$2"; shift 2;;
        --reward_alpha) REWARD_ALPHA="$2"; shift 2;;
        --entropy_beta) ENTROPY_BETA="$2"; shift 2;;
        --eval_limit) EVAL_LIMIT="$2"; shift 2;;
        --eval_temps) EVAL_TEMPS="$2"; shift 2;;
        --no-dynamic) DYNAMIC_PROMPTS=""; shift;;
        --no-partial) PARTIAL_CREDIT=""; shift;;
        --no-structured) STRUCTURED_HINTS=""; shift;;
        --dynamic) DYNAMIC_PROMPTS="--dynamic_prompts"; shift;;
        --partial) PARTIAL_CREDIT="--partial_credit"; shift;;
        --structured) STRUCTURED_HINTS="--structured_hints"; shift;;
        --baseline) DYNAMIC_PROMPTS=""; PARTIAL_CREDIT=""; STRUCTURED_HINTS=""; shift;;
        --prompt_model) PROMPT_MODEL="$2"; shift 2;;
        --all-improvements) DYNAMIC_PROMPTS="--dynamic_prompts"; PARTIAL_CREDIT="--partial_credit"; STRUCTURED_HINTS="--structured_hints"; shift;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

mkdir -p "$RESULT_DIR" logs

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

# ---- Print config ----
IMPROVEMENTS=""
[ -n "$DYNAMIC_PROMPTS" ] && IMPROVEMENTS+="dynamic_prompts "
[ -n "$PROMPT_MODEL" ] && IMPROVEMENTS+="prompt_model=$PROMPT_MODEL "
[ -n "$PARTIAL_CREDIT" ] && IMPROVEMENTS+="partial_credit(α=$REWARD_ALPHA) "
[ -n "$STRUCTURED_HINTS" ] && IMPROVEMENTS+="structured_hints "
[ -z "$IMPROVEMENTS" ] && IMPROVEMENTS="NONE (baseline)"

# Build prompt_model flag
PROMPT_MODEL_FLAG=""
[ -n "$PROMPT_MODEL" ] && PROMPT_MODEL_FLAG="--prompt_model $PROMPT_MODEL"

echo "============================================"
echo "  AnyMAC-Improved Router Training"
echo "  Model: $MODEL | Judge: $JUDGE_MODEL"
echo "  Config: train=$TRAIN_NUM, traces=$NUM_TRACES, mr=$MAX_ROUTING"
echo "          lr=$LR, GS_tau=$TEMPERATURE, decay=$DECAY_FACTOR, batch=$BATCH_SIZE"
echo "  Epochs: $EPOCHS | Entropy beta: $ENTROPY_BETA"
echo "  Improvements: $IMPROVEMENTS"
echo "  Eval temps: $EVAL_TEMPS"
echo "  Eval limit: $EVAL_LIMIT questions"
echo "  Result dir: $RESULT_DIR"
echo "============================================"

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
        --reward_alpha "$REWARD_ALPHA" \
        --entropy_beta "$ENTROPY_BETA" \
        $DYNAMIC_PROMPTS $PARTIAL_CREDIT $STRUCTURED_HINTS $PROMPT_MODEL_FLAG \
        --result_dir "$RESULT_DIR"
    echo "Training complete"
fi

# ---- Refresh checkpoint dir ----
CKPT_DIR=$(ls -dt "$RESULT_DIR"/20*/ 2>/dev/null | head -1 || true)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoints found after training"
    exit 1
fi

# Use epoch 1 checkpoint
CKPT=$(ls "$CKPT_DIR"/*_epoch1.pth 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "ERROR: No epoch 1 checkpoint found"
    exit 1
fi
echo ""
echo "Using checkpoint: $CKPT"

# ---- Eval temperature sweep (all in parallel) ----
echo ""
echo "============================================"
echo "  Eval Temperature Sweep"
echo "  Temps: $EVAL_TEMPS"
echo "============================================"

EVAL_PIDS=()
EVAL_TEMP_LIST=()

for tau in $EVAL_TEMPS; do
    EDIR="${RESULT_DIR}/eval_temp_${tau}"
    EVAL_TEMP_LIST+=("$tau")

    if ls "$EDIR"/${DOMAIN}_*.json >/dev/null 2>&1; then
        echo "  [tau=$tau] Eval exists — skipping"
        EVAL_PIDS+=("skip")
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
            $DYNAMIC_PROMPTS $STRUCTURED_HINTS $PROMPT_MODEL_FLAG \
            --result_dir "$EDIR" &
        EVAL_PIDS+=("$!")
        echo "  [tau=$tau] Eval launched (PID $!)"
    fi
done

echo ""
echo "Waiting for all evals to complete..."
for pid in "${EVAL_PIDS[@]}"; do
    [ "$pid" = "skip" ] && continue
    wait "$pid" 2>/dev/null
done
echo "All evals done!"

# ---- Results ----
echo ""
echo "============================================"
echo "  RESULTS: Eval Temperature Sweep"
echo "============================================"
printf "%-12s %-10s %-10s\n" "Eval Temp" "Regex" "Judge"
printf "%-12s %-10s %-10s\n" "----------" "--------" "--------"

BEST_TAU=""
BEST_JUDGE=0

for tau in $EVAL_TEMPS; do
    EDIR="${RESULT_DIR}/eval_temp_${tau}"
    if [ -d "$EDIR" ]; then
        read -r regex judge <<< "$(get_accuracy "$EDIR")"
        printf "%-12s %-10s %-10s\n" "$tau" "$regex" "$judge"

        if [ "$judge" != "N/A" ]; then
            better=$(python3 -c "print(1 if $judge > $BEST_JUDGE else 0)" 2>/dev/null || echo "0")
            if [ "$better" = "1" ]; then
                BEST_JUDGE="$judge"
                BEST_TAU="$tau"
            fi
        fi
    fi
done

echo "============================================"
echo ""
echo "BEST: eval_temperature=$BEST_TAU (judge=$BEST_JUDGE)"
echo "Checkpoint: $CKPT"
echo "$BEST_TAU" > "${RESULT_DIR}/best_eval_temperature.txt"
echo "$CKPT" > "${RESULT_DIR}/best_checkpoint.txt"
echo ""
echo "Done!"
