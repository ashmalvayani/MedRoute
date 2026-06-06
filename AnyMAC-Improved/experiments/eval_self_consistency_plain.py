"""
Self-consistency evaluation for plain Qwen3-8B (no routing).
Sends each question N times to the LLM, extracts answer letters via regex,
and reports majority-vote accuracy.

Usage:
    cd AnyMAC-Improved
    source venv/bin/activate
    python experiments/eval_self_consistency_plain.py \
        --num_rollouts 2 \
        --parallelism 512
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import re
import time
import aiohttp
import asyncio
from pathlib import Path
from collections import Counter

import numpy as np
import random
from tqdm import tqdm

from datasets_my.medqa_dataset import MedQADataset

BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')
API_KEY = os.getenv('API_KEY', 'EMPTY')


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


SYSTEM_PROMPT = (
    "You are a medical expert. Read the question carefully and select the best answer. "
    "Think step by step, then state your final answer as 'The answer is [LETTER]'."
)


async def query_llm(session, semaphore, model, question, temperature, pbar):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    async with semaphore:
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    pbar.update(1)
                    return ""
                r = await resp.json()
                pbar.update(1)
                return r['choices'][0]['message']['content'].strip()
        except Exception:
            pbar.update(1)
            return ""


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


async def run_all(args):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    # Include rollout count in result dir so different runs don't overwrite
    result_dir = Path(args.result_dir) / f"rollouts_{args.num_rollouts}"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    result_file = result_dir / f"self_consistency_plain_{timestamp}.json"
    details_file = result_dir / f"self_consistency_plain_{timestamp}_details.json"

    dataset = MedQADataset('test')
    n_questions = len(dataset)
    total_requests = n_questions * args.num_rollouts
    print(f"Dataset: {n_questions} questions × {args.num_rollouts} rollouts = {total_requests} requests")

    # Prepare questions
    questions = []
    for qi in range(n_questions):
        record = dataset[qi]
        input_dict = dataset.record_to_input(record)
        questions.append((qi, input_dict["task"]))

    print(f"Parallelism: {args.parallelism}")

    semaphore = asyncio.Semaphore(args.parallelism)
    timeout = aiohttp.ClientTimeout(total=300)
    connector = aiohttp.TCPConnector(limit=args.parallelism, limit_per_host=args.parallelism)

    pbar = tqdm(total=total_requests, desc="Requests")

    # responses_map[(qi, ri)] = response_text
    responses_map: dict[tuple[int, int], str] = {}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = []
        task_keys = []
        for qi, question in questions:
            for ri in range(args.num_rollouts):
                t = asyncio.create_task(
                    query_llm(session, semaphore, args.llm_name, question, args.temperature, pbar)
                )
                tasks.append(t)
                task_keys.append((qi, ri))

        responses = await asyncio.gather(*tasks)

    pbar.close()

    for (qi, ri), resp in zip(task_keys, responses):
        responses_map[(qi, ri)] = resp

    # Build per-question results + details
    total_correct = 0
    output_data = []
    details_data = []

    for qi in range(n_questions):
        record = dataset[qi]
        gt = dataset.record_to_target_answer(record).strip().upper()
        input_dict = dataset.record_to_input(record)

        rollouts = []
        letters = []
        for ri in range(args.num_rollouts):
            resp = responses_map.get((qi, ri), "")
            letter = extract_answer_letter(resp)
            letters.append(letter)
            rollouts.append({
                "rollout": ri,
                "letter": letter,
                "response": resp,
            })

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
    print(f"Self-Consistency (Plain Qwen3-8B) — {args.num_rollouts} rollouts")
    print(f"{'='*50}")
    print(f"Majority Vote Accuracy: {accuracy:.4f} ({total_correct}/{n_questions})")
    print(f"Individual Response Accuracy: {individual_acc:.4f} ({individual_correct_total}/{individual_total})")
    print(f"{'='*50}")

    summary = {
        "config": vars(args),
        "majority_vote_accuracy": accuracy,
        "individual_response_accuracy": individual_acc,
        "total_correct": total_correct,
        "total_questions": n_questions,
        "timestamp": timestamp,
    }
    atomic_write_json(result_file, [summary] + output_data)
    atomic_write_json(details_file, details_data)
    print(f"Results: {result_file}")
    print(f"Details: {details_file}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm_name", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--num_rollouts", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--parallelism", type=int, default=512)
    p.add_argument("--result_dir", type=str, default="result/self_consistency_plain")
    args = p.parse_args()

    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
