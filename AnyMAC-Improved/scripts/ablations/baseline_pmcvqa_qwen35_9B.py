"""
Baseline: Qwen3.5-9B zero-shot on PMC-VQA test set via vLLM API.

Uses file:// URLs for images (no base64 bloat).
Async parallel requests for high throughput.

Usage:
  BASE_URL=http://localhost:8000 python scripts/baseline_pmcvqa_qwen35_9B.py
"""

import os
import sys
import json
import re
import time
import asyncio
import aiohttp
from pathlib import Path
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =====================================================
# CONFIG
# =====================================================
MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "EMPTY")

CONCURRENCY = 128
MAX_TOKENS = 256
OUTPUT_PATH = "result/baseline_pmcvqa_qwen35_9b.json"

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / ".." / "datasets_my" / "PMC-VQA" / "data"
TEST_CSV = DATA_DIR / "pmcvqa_test.csv"
IMAGE_BASE = DATA_DIR / "PMC-VQA"

SYSTEM_PROMPT = "You are a helpful medical assistant."

# =====================================================
# Load test data
# =====================================================
def load_test_data():
    import pandas as pd
    df = pd.read_csv(
        TEST_CSV,
        header=None,
        names=["question", "A", "B", "C", "D", "correct_answer", "image_rel", "existing_caption"],
    )
    records = []
    for idx, row in df.iterrows():
        image_filename = row["image_rel"].strip().split("/")[-1]
        image_path = IMAGE_BASE / "images_test_clean" / image_filename
        records.append({
            "index": idx,
            "question": row["question"],
            "A": str(row["A"]).strip(),
            "B": str(row["B"]).strip(),
            "C": str(row["C"]).strip(),
            "D": str(row["D"]).strip(),
            "correct_answer": row["correct_answer"].strip(),
            "image_path": str(image_path),
        })
    print(f"Loaded {len(records)} test samples")
    return records


def build_prompt(record):
    return (
        f"Question:\n{record['question']}\n\n"
        f"Options:\n"
        f"A. {record['A']}\n"
        f"B. {record['B']}\n"
        f"C. {record['C']}\n"
        f"D. {record['D']}\n\n"
        f"Choose a single best answer.\n"
        f"IMPORTANT:\n"
        f"- Do NOT include any reasoning or explanation.\n"
        f"- Output ONLY in the format:\n"
        f"Final Answer: <A/B/C/D>"
    )


def extract_answer(text):
    """Extract answer letter from model response."""
    text = text.strip()
    # Try "Final Answer: X" pattern
    m = re.search(r'Final\s+Answer\s*:\s*([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try "answer is X"
    m = re.search(r'answer\s+is\s+(?:Option\s+)?([A-D])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Try standalone letter
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1).upper()
    # First char
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return ""


# =====================================================
# Async vLLM API calls
# =====================================================
async def call_vllm(session, sem, record):
    prompt = build_prompt(record)
    abs_path = str(Path(record["image_path"]).resolve())

    if not os.path.exists(abs_path):
        return record["index"], "", "image_missing"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
            ],
        },
    ]

    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    async with sem:
        try:
            url = f"{BASE_URL}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {API_KEY}"}
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    return record["index"], "", f"api_error_{resp.status}"
                data = await resp.json()
                text = data["choices"][0]["message"]["content"]
                # Strip thinking tokens
                if "<think>" in text:
                    think_end = text.find("</think>")
                    if think_end >= 0:
                        text = text[think_end + len("</think>"):].strip()
                return record["index"], text, None
        except Exception as e:
            return record["index"], "", str(e)


async def run_all(records):
    sem = asyncio.Semaphore(CONCURRENCY)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [call_vllm(session, sem, r) for r in records]
        results = []
        done = 0
        for coro in asyncio.as_completed(tasks):
            idx, text, err = await coro
            done += 1
            if done % 100 == 0 or done == len(records):
                print(f"  {done}/{len(records)} completed")
            results.append((idx, text, err))
        return results


# =====================================================
# Main
# =====================================================
def main():
    records = load_test_data()

    print(f"Running Qwen3.5-9B baseline on {len(records)} test samples")
    print(f"  API: {BASE_URL}, Concurrency: {CONCURRENCY}")

    t0 = time.time()
    raw_results = asyncio.run(run_all(records))
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s ({elapsed/len(records):.2f}s per sample)")

    # Map results back
    result_map = {idx: (text, err) for idx, text, err in raw_results}

    # Score
    output = []
    correct = 0
    total = 0
    errors = 0
    answer_dist = Counter()

    for r in records:
        text, err = result_map.get(r["index"], ("", "missing"))
        pred = extract_answer(text)
        gt = r["correct_answer"].strip().upper()
        is_correct = pred == gt
        if is_correct:
            correct += 1
        total += 1
        answer_dist[pred] += 1
        if err:
            errors += 1

        output.append({
            "index": r["index"],
            "question": r["question"],
            "correct_answer": gt,
            "predicted_answer": pred,
            "raw_response": text[:500],
            "correct": is_correct,
            "error": err,
        })

    accuracy = correct / total * 100 if total else 0
    print(f"\n{'='*50}")
    print(f"  Qwen3.5-9B Zero-Shot Baseline on PMC-VQA")
    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Errors: {errors}")
    print(f"  Answer distribution: {dict(answer_dist.most_common())}")
    print(f"{'='*50}")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    summary = {
        "model": MODEL_ID,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "results": output,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
