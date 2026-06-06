from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from tqdm import tqdm

from datasets_my.pubmedqa_dataset import PubMedQADataset

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", os.getenv("BASE_URL", "http://localhost:8001"))
API_KEY = os.getenv("API_KEY", "EMPTY")
RESULT_DIR = Path("result/pubmedqa_baseline")

SYSTEM_PROMPT = (
    "You are a medical expert. Read the question and choose the best option among A, B, and C "
    "(yes / no / maybe). State the letter of your chosen option and give a brief explanation."
)

JUDGE_SYSTEM_PROMPT = (
    "You are an answer extractor. Read the model response and extract only the final option "
    "letter chosen (A, B, or C). Reply with ONLY that single letter."
)


async def call_llm(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    base_url: str | None = None,
) -> str:
    url = f"{base_url or BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": 42,
    }
    if "qwen3" in model.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            raise RuntimeError(f"LLM error {resp.status}: {await resp.text()}")
        r = await resp.json()
        return r["choices"][0]["message"]["content"].strip()


async def generate_answer(
    session: aiohttp.ClientSession,
    model: str,
    record: dict,
    dataset: PubMedQADataset,
    semaphore: asyncio.Semaphore,
    idx: int,
    max_tokens: int,
) -> dict:
    async with semaphore:
        true_ans = str(dataset.record_to_target_answer(record)).strip().upper()
        input_dict = dataset.record_to_input(record)
        task = input_dict["task"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        try:
            response = await call_llm(session, model, messages, max_tokens=max_tokens)
        except Exception as e:
            response = f"[ERROR: {e}]"

        return {
            "Index": idx,
            "Question": task,
            "Answer": true_ans,
            "Response": response,
        }


async def phase1_generate(args, records, dataset) -> list[dict]:
    print(f"\n=== Phase 1: Generating answers with {args.model} ===")
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    items: list[dict] = [None] * len(records)  # type: ignore

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            generate_answer(session, args.model, rec, dataset, semaphore, i, args.max_tokens)
            for i, rec in enumerate(records)
        ]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
            item = await coro
            items[item["Index"]] = item
    return items


async def judge_one(
    session: aiohttp.ClientSession,
    judge_model: str,
    item: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        prompt = (
            f"Question:\n{item['Question']}\n\n"
            f"Model response:\n{item['Response']}\n\n"
            "What is the model's final answer? Reply with ONLY the letter A, B, or C."
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            verdict = await call_llm(session, judge_model, messages, max_tokens=8, base_url=JUDGE_BASE_URL)
            extracted = verdict.strip().upper()
            letter = next((c for c in extracted if c in "ABC"), "")
            gt = item["Answer"].strip().upper()[:1]
            judge_correct = letter == gt if gt in "ABC" else False
        except Exception as e:
            verdict = f"[ERROR: {e}]"
            judge_correct = False
        return {**item, "Judge_verdict": verdict, "Judge_correct": judge_correct}


async def phase2_judge(args, items: list[dict]) -> list[dict]:
    print(f"\n=== Phase 2: Judging with {args.judge_model} ===")
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)
    judged: list[dict] = [None] * len(items)  # type: ignore

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [judge_one(session, args.judge_model, item, semaphore) for item in items]
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging"):
            item = await coro
            judged[item["Index"]] = item
    return judged


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _dataset_split(args) -> str:
    if args.use_context:
        return "context/test"
    return args.split


async def main_async(args):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    model_slug = args.model.replace(":", "-").replace("/", "-")

    split = _dataset_split(args)
    dataset = PubMedQADataset(split)

    records = []
    for i, rec in enumerate(dataset):
        if args.limit is not None and i >= args.limit:
            break
        records.append(rec)

    print(f"Dataset: PubMedQA ({split}) — {len(records)} questions")

    answers_file = Path(args.answers_file) if args.answers_file else None
    if answers_file and answers_file.exists():
        print(f"Loading pre-generated answers from {answers_file}")
        with open(answers_file, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        items = await phase1_generate(args, records, dataset)
        answers_file = RESULT_DIR / f"answers_{model_slug}_{timestamp}.json"
        write_json(answers_file, items)
        print(f"Answers saved -> {answers_file}")

    regex_hits = sum(
        dataset.record_to_target_check(it["Answer"], it["Response"], it["Question"]) for it in items
    )
    n = len(items)
    print(f"\nRegex accuracy: {regex_hits / n:.4f} ({regex_hits}/{n})")

    judged = await phase2_judge(args, items)
    judge_hits = sum(1 for it in judged if it["Judge_correct"])
    print(f"Judge accuracy: {judge_hits / n:.4f} ({judge_hits}/{n})")

    output = {
        "summary": {
            "model": args.model,
            "judge_model": args.judge_model,
            "split": split,
            "n_samples": n,
            "regex_correct": regex_hits,
            "regex_accuracy": regex_hits / n,
            "judge_correct": judge_hits,
            "judge_accuracy": judge_hits / n,
        },
        "results": judged,
    }
    out_file = RESULT_DIR / f"baseline_{model_slug}_{timestamp}.json"
    write_json(out_file, output)
    print(f"\nFull results saved -> {out_file}")


def main():
    p = argparse.ArgumentParser(description="PubMedQA baseline evaluation (no routing)")
    p.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    p.add_argument("--judge_model", type=str, default="Qwen/Qwen3.6-27B")
    p.add_argument("--split", type=str, default="test", help="Dataset subfolder under datasets_my/PubMedQA/data/")
    p.add_argument("--use_context", action="store_true", help="Use context/test (same as run_pubmedqa --use_context)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--answers_file", type=str, default=None)
    p.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Max new tokens for base model generation.",
    )
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
