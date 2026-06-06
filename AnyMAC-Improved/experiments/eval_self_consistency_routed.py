"""
Self-consistency evaluation for MedRoute pipeline.
Runs N traces per question through the trained router, extracts answer letters
via regex, and reports majority-vote accuracy.

Saves each completed trace to a JSONL checkpoint file for resume support.
Stdout is kept clean (only tqdm + summary).

Usage:
    cd AnyMAC-Improved
    source venv/bin/activate
    CUDA_VISIBLE_DEVICES="" python experiments/eval_self_consistency_routed.py \
        --model_path <checkpoint.pth> \
        --num_rollouts 3 \
        --max_routing 3 \
        --temperature 0.7 \
        --parallelism 64
"""

from __future__ import annotations

import sys, os

# Suppress verbose model loading / specialist prompt output
import logging
logging.disable(logging.WARNING)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import re
import time
import copy
import threading
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import random
from tqdm import tqdm

from GDesigner.graph.graph import Graph
from datasets_my.medqa_dataset import MedQADataset
from GDesigner.prompt.medqa_prompt_set import SPECIALISTS


def extract_answer_letter(response: str) -> str:
    """Extract a single answer letter (A-E) from a model response using regex."""
    text = response.replace('assistant', 'Option').strip()
    patterns = [
        r'answer\s+is\s+(?:Option\s+)?(?:\**)([A-E])(?:\**)',
        r'(?:correct|best)\s+(?:answer|option)\s*(?:is|:)\s*(?:\**)([A-E])(?:\**)',
        r'\bOption\s+([A-E])\b',
        r'\(([A-E])\)',
        r'(?:^|\n)\s*\**([A-E])\**\s*[\.\)\:]',
        r'(?:^|[\s,;])\**([A-E])\**[\.\)\:]',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    if text and text[0].upper() in 'ABCDE':
        if len(text) == 1 or not text[1].isalpha():
            return text[0].upper()
    return ""


_THREAD_LOCAL = threading.local()


def _get_thread_devnull():
    """Per-thread /dev/null that stays open for the thread's lifetime."""
    dn = getattr(_THREAD_LOCAL, "devnull", None)
    if dn is None or dn.closed:
        dn = open(os.devnull, 'w')
        _THREAD_LOCAL.devnull = dn
    return dn


def get_thread_graph(base_graph: Graph) -> Graph:
    g = getattr(_THREAD_LOCAL, "graph", None)
    if g is None:
        g = copy.deepcopy(base_graph)
        g.set_eval()
        # Inherit routing settings from base_graph
        g.cos_scaling = base_graph.cos_scaling
        if hasattr(base_graph, 'eval_temperature'):
            g.eval_temperature = base_graph.eval_temperature
        if hasattr(base_graph, 'eval_deterministic'):
            g.eval_deterministic = base_graph.eval_deterministic
        _THREAD_LOCAL.graph = g
    return g


def run_one_trace(base_graph, dataset, record, args):
    """Run a single trace through the routing pipeline. Returns (letter, response, routing_trace)."""
    g = get_thread_graph(base_graph)
    input_dict = dataset.record_to_input(record)

    # Redirect stdout to suppress specialist prompt/response spam
    # Use thread-local devnull that stays open (async tasks may print after try/finally)
    devnull = _get_thread_devnull()
    old_stdout = sys.stdout
    sys.stdout = devnull
    try:
        result = g.run_next_agent_prediction(
            input_dict,
            max_routing=args.max_routing,
            temperature=args.temperature,
            available_roles=SPECIALISTS,
            agent_group_type="AnalyzeAgent",
            max_context=args.max_context,
        )
    finally:
        sys.stdout = old_stdout

    if result is None:
        return "", "", []
    answers = result.get("answers", [""])
    response = answers[0] if answers else ""
    letter = extract_answer_letter(response)

    # Extract routing trace
    routing_results = result.get("routing_results", {})
    agent_sels = routing_results.get("agent_selections", [])
    routing_trace = []
    for idx in agent_sels:
        if idx < len(SPECIALISTS):
            routing_trace.append(SPECIALISTS[idx])
        else:
            routing_trace.append("DecisionMaker")

    return letter, response, routing_trace


def majority_vote(letters: list[str]) -> str:
    valid = [l for l in letters if l]
    if not valid:
        return ""
    return Counter(valid).most_common(1)[0][0]


def atomic_write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(ckpt_file: Path) -> dict[tuple[int,int], dict]:
    """Load completed traces from JSONL checkpoint. Returns {(qi, ri): trace_dict}."""
    done = {}
    if not ckpt_file.exists():
        return done
    with open(ckpt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done[(obj["qi"], obj["ri"])] = obj
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--num_rollouts", type=int, default=3)
    p.add_argument("--max_routing", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_context", type=int, default=2048)
    p.add_argument("--parallelism", type=int, default=64)
    p.add_argument("--result_dir", type=str, default="result/self_consistency_routed")
    p.add_argument("--eval_temperature", type=float, default=None,
                   help="If set, use softmax sampling at this temperature instead of cos_scaling=1e3 argmax")
    p.add_argument("--dynamic_prompts", action="store_true", default=False)
    p.add_argument("--dynamic_pool", action="store_true", default=False,
                   help="Enable dynamic specialist pool per question.")
    p.add_argument("--judge_model", type=str, default="Qwen/Qwen3-32B")
    p.add_argument("--prompt_model", type=str, default=None,
                   help="Model to generate dynamic prompts (uses BASE_URL). If unset, uses judge_model.")
    args = p.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Include rollout count in result dir so different runs don't overwrite
    result_dir = Path(args.result_dir) / f"rollouts_{args.num_rollouts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    result_file = result_dir / f"self_consistency_routed_{timestamp}.json"
    details_file = result_dir / f"self_consistency_routed_{timestamp}_details.json"

    # Checkpoint file (stable name so resume finds it)
    ckpt_file = result_dir / "checkpoint.jsonl"

    dataset = MedQADataset('test')
    n_questions = len(dataset)
    total_traces = n_questions * args.num_rollouts

    # Load checkpoint — skip already-completed traces
    done = load_checkpoint(ckpt_file)
    all_work = [(qi, ri) for qi in range(n_questions) for ri in range(args.num_rollouts)]
    remaining = [(qi, ri) for qi, ri in all_work if (qi, ri) not in done]

    print(f"Dataset: {n_questions} questions × {args.num_rollouts} rollouts = {total_traces} traces")
    print(f"Checkpoint: {len(done)} done, {len(remaining)} remaining")

    # Enable dynamic prompts if requested
    if args.dynamic_prompts:
        import GDesigner.agents.analyze_agent as aa
        aa.DYNAMIC_PROMPTS_ENABLED = True
        aa.JUDGE_MODEL_FOR_PROMPTS = args.judge_model
        if args.prompt_model:
            aa.PROMPT_MODEL_FOR_PROMPTS = args.prompt_model
            print(f"[improved] Dynamic specialist prompts ENABLED (prompt_model={args.prompt_model} on BASE_URL)")
        else:
            print(f"[improved] Dynamic specialist prompts ENABLED (judge={args.judge_model})")

    # Enable dynamic specialist pool if requested
    if args.dynamic_pool:
        import GDesigner.graph.graph as gg
        gg.DYNAMIC_POOL_ENABLED = True
        gg.DYNAMIC_POOL_JUDGE_MODEL = args.judge_model
        print(f"[improved] Dynamic specialist pool ENABLED (judge={args.judge_model})")

    if not remaining:
        print("All traces already completed — skipping inference, computing results.")
    else:
        print(f"Loading router from: {args.model_path}")
        graph = Graph.load_model(args.model_path)
        graph.to_device(torch.device("cpu"))
        graph.set_eval()
        if args.eval_temperature is not None:
            graph.cos_scaling = 1
            graph.eval_temperature = args.eval_temperature
            graph.eval_deterministic = True
            print(f"Eval routing: softmax sampling at temperature={args.eval_temperature}")
        else:
            graph.cos_scaling = 1e3
            print(f"Eval routing: cos_scaling=1e3 (hard argmax)")

        print(f"Parallelism: {args.parallelism}")

        _lock = threading.Lock()

        def worker(qi, ri):
            record = dataset[qi]
            letter, response, routing_trace = run_one_trace(graph, dataset, record, args)
            return qi, ri, letter, response, routing_trace

        with torch.no_grad():
            with ThreadPoolExecutor(max_workers=args.parallelism) as ex:
                futures = [ex.submit(worker, qi, ri) for qi, ri in remaining]
                with open(ckpt_file, "a") as cf:
                    for fut in tqdm(as_completed(futures), total=len(remaining),
                                    initial=len(done), desc="Traces"):
                        try:
                            qi, ri, letter, response, routing_trace = fut.result()
                            trace = {
                                "qi": qi, "ri": ri,
                                "letter": letter,
                                "response": response,
                                "routing_trace": routing_trace,
                            }
                            with _lock:
                                done[(qi, ri)] = trace
                                cf.write(json.dumps(trace, ensure_ascii=False) + "\n")
                                cf.flush()
                        except Exception as e:
                            tqdm.write(f"Error q={qi} r={ri}: {e}")

    # Build per-question summary + details from all traces (checkpoint + this run)
    results: dict[int, list[dict]] = {qi: [] for qi in range(n_questions)}
    for (qi, ri), trace in done.items():
        results[qi].append(trace)

    total_correct = 0
    output_data = []
    details_data = []

    for qi in range(n_questions):
        record = dataset[qi]
        gt = dataset.record_to_target_answer(record).strip().upper()
        input_dict = dataset.record_to_input(record)
        rollouts = results[qi]
        letters = [r["letter"] for r in rollouts]
        voted = majority_vote(letters)
        is_correct = voted == gt
        if is_correct:
            total_correct += 1
        individual_correct = sum(1 for l in letters if l == gt)

        output_data.append({
            "Index": qi,
            "Ground_truth": gt,
            "Voted_answer": voted,
            "Correct": is_correct,
            "Individual_correct": individual_correct,
            "Individual_total": len(letters),
            "Letter_distribution": dict(Counter(letters)),
        })

        details_data.append({
            "Index": qi,
            "Question": input_dict["task"],
            "Ground_truth": gt,
            "Voted_answer": voted,
            "Correct": is_correct,
            "Rollouts": rollouts,
        })

    accuracy = total_correct / n_questions
    individual_total = sum(d["Individual_total"] for d in output_data)
    individual_correct_total = sum(d["Individual_correct"] for d in output_data)
    individual_acc = individual_correct_total / individual_total if individual_total else 0

    print(f"\n{'='*50}")
    print(f"Self-Consistency (Routed) — {args.num_rollouts} rollouts")
    print(f"{'='*50}")
    print(f"Majority Vote Accuracy: {accuracy:.4f} ({total_correct}/{n_questions})")
    print(f"Individual Trace Accuracy: {individual_acc:.4f} ({individual_correct_total}/{individual_total})")
    print(f"{'='*50}")

    summary = {
        "config": vars(args),
        "majority_vote_accuracy": accuracy,
        "individual_trace_accuracy": individual_acc,
        "total_correct": total_correct,
        "total_questions": n_questions,
        "timestamp": timestamp,
    }
    atomic_write_json(result_file, [summary] + output_data)
    atomic_write_json(details_file, details_data)
    print(f"Results: {result_file}")
    print(f"Details: {details_file}")

    # Clean up checkpoint on successful completion
    if ckpt_file.exists():
        ckpt_file.unlink()
        print("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
