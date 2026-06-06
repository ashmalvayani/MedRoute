#!/usr/bin/env bash
# PubMedQA ablation: dynamic pool + dynamic prompts, 32B consistent (all Qwen3-32B)
#
# Same locked hyperparams as MedQA 32B consistent experiment.
#
# Requirements:
#   - Qwen3-32B hosted on vLLM at port 8001 (or set BASE_URL/JUDGE_BASE_URL)
#   - No need for 8B server
#
# Usage:
#   bash scripts/train_pubmedqa_dynamic_pool_32b.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

# Activate venv
VENV_DIR="${REPO_DIR}/venv"
source "$VENV_DIR/bin/activate"

export API_KEY="${API_KEY:-EMPTY}"
# Both point to 32B server
export BASE_URL="${BASE_URL:-http://localhost:8001}"
export JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8001}"

# ---- Config (same locked hyperparams, but 32B for everything) ----
MODEL="Qwen/Qwen3-32B"
JUDGE_MODEL="Qwen/Qwen3-32B"
DOMAIN="pubmedqa"
TRAIN_NUM=300          # Sample from PQA-A (211k artificial)
NUM_TRACES=16
MAX_ROUTING=3
LR="3e-5"
TEMPERATURE=1.0
DECAY_FACTOR=0.98
BATCH_SIZE=8
EPOCHS=1
TRAIN_PARALLELISM=128
EVAL_PARALLELISM=128
RESULT_DIR="result/pubmedqa_dynamic_pool_32b"
ENTROPY_BETA=0.05
EVAL_LIMIT=1100
EVAL_TEMPS="0.7"

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

echo "============================================"
echo "  PubMedQA Dynamic Pool Training (32B Consistent)"
echo "  Base Model: $MODEL | Judge: $JUDGE_MODEL"
echo "  Config: train=$TRAIN_NUM, traces=$NUM_TRACES, mr=$MAX_ROUTING"
echo "          lr=$LR, GS_tau=$TEMPERATURE, decay=$DECAY_FACTOR, batch=$BATCH_SIZE"
echo "  Epochs: $EPOCHS | Entropy beta: $ENTROPY_BETA"
echo "  Improvements: dynamic_pool + dynamic_prompts"
echo "  Eval temps: $EVAL_TEMPS"
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
    python experiments/run_pubmedqa.py \
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
        --train_split train \
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

    for tau in $EVAL_TEMPS; do
        EDIR="${RESULT_DIR}/eval_epoch${epoch}_temp_${tau}"

        if ls "$EDIR"/${DOMAIN}_*.json >/dev/null 2>&1; then
            echo "  [epoch=$epoch, tau=$tau] Eval exists — skipping"
        else
            mkdir -p "$EDIR"
            CUDA_VISIBLE_DEVICES="" python experiments/run_pubmedqa.py \
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
                --dynamic_prompts --dynamic_pool \
                --result_dir "$EDIR" &
            echo "  [epoch=$epoch, tau=$tau] Eval launched (PID $!)"
        fi
    done
    wait
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
