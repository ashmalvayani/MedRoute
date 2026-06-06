"""
Signal-aware training sample screening for MedQA.

Runs a small number of rollouts (default 4) on a large candidate pool (default 1000)
from the training split (MedQADataset('dev')), then keeps only "mixed signal" questions:
those where at least one rollout is correct AND at least one is wrong.

All-correct and all-wrong questions produce zero gradient (advantage = 0 for all rollouts
under grouped REINFORCE), so they are useless for training. This script identifies the
learnable subset upfront so training wastes no compute on inert samples.

Output JSON (--output_path) contains:
  - metadata: run config
  - mixed_questions: list of {dataset_index, question, ground_truth, num_correct, rollouts}
  - stats: counts of all_correct / all_wrong / mixed from the candidate pool

Usage:
  python experiments/screen_medqa_samples.py \
      --llm_name Qwen/Qwen3-8B \
      --judge_model Qwen/Qwen3-32B \
      --screen_num 1000 \
      --screen_rollouts 4 \
      --trace_parallelism 128 \
      --output_path result/medqa_mixed_samples.json

  # With a pre-trained router checkpoint (recommended: use best Phase 2.1 ckpt):
  python experiments/screen_medqa_samples.py \
      --model_path result/ablation_phase2_1_train_num/.../epoch1.pth \
      ...
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from datetime import datetime

import aiohttp
import numpy as np
import torch
from tqdm import tqdm

from GDesigner.graph.graph import Graph
from GDesigner.prompt.medqa_prompt_set import ROLES, SPECIALISTS
from GDesigner.utils.const import GDesigner_ROOT
from datasets_my.medqa_dataset import MedQADataset


# ---------------------------------------------------------------------------
# Judge — same config as train_medqa.py
# ---------------------------------------------------------------------------
JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', os.getenv('BASE_URL', 'http://localhost:8001'))
JUDGE_API_KEY  = os.getenv('JUDGE_API_KEY',  os.getenv('API_KEY',  'EMPTY'))

JUDGE_SYSTEM_PROMPT = (
    "You are an answer extractor. Read the model's response and extract the final answer "
    "option the model chose. Focus ONLY on the model's conclusion, not on any options listed "
    "in the question. Reply with ONLY the single letter."
)


async def judge_answer(session, judge_model, question, true_answer, model_response):
    prompt = (
        f"Question:\n{question}\n\n"
        f"Model response:\n{model_response}\n\n"
        f"What is the model's final answer? Focus on the model's conclusion only. "
        f"Reply with ONLY the single letter."
    )
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 4,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{JUDGE_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                return False
            r = await resp.json()
            extracted = r['choices'][0]['message']['content'].strip().upper()
            return len(extracted) > 0 and extracted[0] == true_answer.strip().upper()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main screening logic
# ---------------------------------------------------------------------------

async def screen_all(graph, dataset, candidate_records, candidate_indices, args):
    """Run screen_rollouts rollouts on every candidate, return per-question results."""

    screen_rollouts  = args.screen_rollouts
    trace_parallelism = args.trace_parallelism
    max_routing      = args.max_routing
    temperature      = args.temperature
    judge_model      = args.judge_model

    sem = asyncio.Semaphore(trace_parallelism)

    async def _one_trace(input_dict):
        async with sem:
            with torch.no_grad():
                return await asyncio.wait_for(
                    graph.arun_next_agent_prediction(
                        input=input_dict,
                        max_routing=max_routing,
                        temperature=temperature,
                        available_roles=ROLES,
                        agent_group_type="AnalyzeAgent",
                        max_context=2048,
                    ),
                    timeout=1800,
                )

    async def _screen_one(i_rec, record, idx, judge_session):
        input_dict  = dataset.record_to_input(record)
        true_answer = dataset.record_to_target_answer(record)

        tasks   = [asyncio.create_task(_one_trace(input_dict)) for _ in range(screen_rollouts)]
        rollouts = []
        try:
            for fut in asyncio.as_completed(tasks):
                result = await fut
                if result is None:
                    continue
                answer_list = result.get("answers", [""])
                answer_str  = answer_list[0] if answer_list else ""
                routing_len = result.get("routing_count", max_routing)

                if judge_model and judge_session:
                    correct = await judge_answer(
                        judge_session, judge_model,
                        input_dict["task"], true_answer, answer_str
                    )
                else:
                    correct = bool(dataset.record_to_target_check(
                        true_answer, answer_str, input_dict["task"]
                    ))

                agent_sels = result["routing_results"].get("agent_selections", [])
                routing_trace = [
                    SPECIALISTS[s] if s < len(SPECIALISTS) else "DecisionMaker"
                    for s in agent_sels
                ]
                rollouts.append({
                    "correct": correct,
                    "final_answer": answer_str,
                    "routing_length": routing_len,
                    "routing_trace": routing_trace,
                })
        finally:
            await asyncio.gather(*tasks, return_exceptions=True)

        num_correct = sum(1 for r in rollouts if r["correct"])
        num_total   = len(rollouts)

        now = datetime.now().strftime("%H:%M:%S")
        status = (
            "MIXED   " if 0 < num_correct < num_total else
            "ALL-OK  " if num_correct == num_total else
            "ALL-WRONG"
        )
        print(f"  [{now}] Q{i_rec+1}/{len(candidate_records)} idx={idx:5d} "
              f"{num_correct}/{num_total} correct  [{status}]  {true_answer}")

        return {
            "dataset_index": int(idx),
            "question":      input_dict["task"],
            "ground_truth":  true_answer,
            "num_correct":   num_correct,
            "num_total":     num_total,
            "rollouts":      rollouts,
        }

    judge_timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=judge_timeout) as judge_session:
        js = judge_session if judge_model else None
        jobs = [
            asyncio.create_task(_screen_one(i, rec, idx, js))
            for i, (rec, idx) in enumerate(zip(candidate_records, candidate_indices))
        ]
        return await asyncio.gather(*jobs)


def main():
    p = argparse.ArgumentParser(description="Screen MedQA training samples for mixed signal")

    p.add_argument("--llm_name",        type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--judge_model",     type=str, default=None,
                   help="Judge LLM for correctness (e.g. Qwen/Qwen3-32B). If unset uses regex.")
    p.add_argument("--model_path",      type=str, default=None,
                   help="Pre-trained router checkpoint. If unset uses untrained router.")

    p.add_argument("--screen_num",      type=int, default=1000,
                   help="Number of candidate questions to screen from training split.")
    p.add_argument("--screen_rollouts", type=int, default=4,
                   help="Rollouts per question for screening (default 4).")
    p.add_argument("--trace_parallelism", type=int, default=128,
                   help="Max concurrent rollouts (vLLM handles batching, push this high).")

    p.add_argument("--max_routing",     type=int,   default=3)
    p.add_argument("--temperature",     type=float, default=0.7)
    p.add_argument("--seed",            type=int,   default=42)

    p.add_argument("--output_path",     type=str,
                   default="result/medqa_mixed_samples.json",
                   help="Where to write the filtered mixed-signal question list.")

    # Graph construction (must match training config)
    p.add_argument("--decision_method",     type=str, default="FinalRefer")
    p.add_argument("--optimized_spatial",   action="store_true")
    p.add_argument("--optimized_temporal",  action="store_true")

    args = p.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Output path
    output_path = Path(f"{GDesigner_ROOT}/{args.output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dataset
    print(f"Loading MedQA training split ('dev')...")
    dataset = MedQADataset('dev')
    print(f"  Total training questions: {len(dataset)}")

    # Sample candidate pool
    pool_size = min(args.screen_num, len(dataset))
    candidate_indices = random.sample(range(len(dataset)), pool_size)
    candidate_records = [dataset[i] for i in candidate_indices]
    print(f"  Sampled {pool_size} candidates for screening")

    # Build graph
    print(f"Building graph (llm={args.llm_name})...")
    graph = Graph(
        domain="medqa",
        llm_name=args.llm_name,
        agent_names=["AnalyzeAgent"],
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        use_transformer=True,
        max_routing=args.max_routing,
        available_roles=SPECIALISTS,
    )

    if args.model_path:
        print(f"  Loading router checkpoint: {args.model_path}")
        graph = Graph.load_model(args.model_path)

    graph.set_eval()
    graph.to_device(torch.device("cpu"))   # vLLM owns GPU; router runs on CPU
    print("  Graph ready.")

    # Run screening
    print(f"\nScreening {pool_size} questions with {args.screen_rollouts} rollouts each "
          f"(parallelism={args.trace_parallelism})...")
    t0 = time.time()
    all_results = asyncio.run(screen_all(graph, dataset, candidate_records, candidate_indices, args))
    elapsed = time.time() - t0
    print(f"\nScreening done in {elapsed:.0f}s ({elapsed/pool_size:.1f}s/question)")

    # Partition results
    all_correct_qs = [r for r in all_results if r["num_correct"] == r["num_total"]]
    all_wrong_qs   = [r for r in all_results if r["num_correct"] == 0]
    mixed_qs       = [r for r in all_results if 0 < r["num_correct"] < r["num_total"]]

    total = len(all_results)
    print(f"\n--- Screening results ---")
    print(f"  Total screened : {total}")
    print(f"  All-correct    : {len(all_correct_qs)}  ({len(all_correct_qs)/total*100:.1f}%)  — zero gradient")
    print(f"  All-wrong      : {len(all_wrong_qs)}  ({len(all_wrong_qs)/total*100:.1f}%)  — zero gradient")
    print(f"  Mixed (signal) : {len(mixed_qs)}  ({len(mixed_qs)/total*100:.1f}%)  — actual gradient ✓")

    # Breakdown of mixed by correctness distribution
    print(f"\nMixed breakdown (correct/total):")
    from collections import Counter
    dist = Counter(f"{r['num_correct']}/{r['num_total']}" for r in mixed_qs)
    for k, v in sorted(dist.items()):
        print(f"  {k:6s} : {v}")

    # Save output
    output = {
        "metadata": {
            "timestamp":        time.strftime("%Y-%m-%d-%H-%M-%S"),
            "llm_name":         args.llm_name,
            "judge_model":      args.judge_model,
            "model_path":       args.model_path,
            "screen_num":       pool_size,
            "screen_rollouts":  args.screen_rollouts,
            "max_routing":      args.max_routing,
            "temperature":      args.temperature,
            "seed":             args.seed,
            "elapsed_seconds":  round(elapsed, 1),
        },
        "stats": {
            "total_screened":  total,
            "all_correct":     len(all_correct_qs),
            "all_wrong":       len(all_wrong_qs),
            "mixed":           len(mixed_qs),
            "mixed_pct":       round(len(mixed_qs) / total * 100, 1),
        },
        "mixed_questions":    mixed_qs,
        "all_correct_indices": [r["dataset_index"] for r in all_correct_qs],
        "all_wrong_indices":   [r["dataset_index"] for r in all_wrong_qs],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(mixed_qs)} mixed-signal questions → {output_path}")
    print(f"Use these for signal-aware training: every sample will produce a gradient update.")


if __name__ == "__main__":
    main()
