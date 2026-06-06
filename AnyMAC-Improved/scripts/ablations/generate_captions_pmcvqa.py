"""
Generate PMC-VQA captions using a VLM via vLLM API.

Generates captions for both train and test splits and saves them
to the caption cache directory. Existing captions are preserved
(only missing ones are generated).

Usage:
  # Start vLLM with Qwen3.6-27B first:
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.6-27B --port 8000 --tensor-parallel-size 4 --enforce-eager \
    --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    --allowed-local-media-path "$(pwd)/datasets_my"

  # Then generate captions:
  BASE_URL=http://localhost:8000 python scripts/generate_captions_pmcvqa.py \
    --model Qwen/Qwen3.6-27B --seed 99
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent / "datasets_my" / "PMC-VQA" / "data"
IMAGE_BASE = DATA_DIR / "PMC-VQA"
CAPTION_DIR = DATA_DIR / "captions"

CONCURRENCY = 256
MAX_TOKENS = 512


def load_split(split: str, seed: int) -> pd.DataFrame:
    """Load split using the same sampling as PMCVQADataset._load_data."""

    if split == "train":
        df = pd.read_csv(DATA_DIR / "train_sample_2000.csv")
        df = df.rename(columns={
            "Figure_path": "image_filename",
            "Question": "question",
        })
        df["image_path"] = df["image_filename"].apply(
            lambda x: str(IMAGE_BASE / "train_sample_2000" / x.strip())
        )
    elif split == "test":
        df = pd.read_csv(
            DATA_DIR / "pmcvqa_test.csv",
            header=None,
            names=["question", "A", "B", "C", "D", "correct_answer",
                   "image_rel", "existing_caption"],
        )
        df["image_filename"] = df["image_rel"].apply(lambda x: x.strip().split("/")[-1])
        df["image_path"] = df["image_rel"].apply(
            lambda x: str(IMAGE_BASE / x.strip())
        )
    else:
        raise ValueError(f"Unknown split: {split}")

    return df


def get_cache_path(split: str, seed: int, model: str) -> Path:
    CAPTION_DIR.mkdir(parents=True, exist_ok=True)
    model_tag = model.replace("/", "-")
    return CAPTION_DIR / f"{split}_seed{seed}_{model_tag}.json"


async def generate_captions(
    records: list,
    model: str,
    base_url: str,
    api_key: str,
) -> dict:
    """Generate captions for a list of (key, image_path, question) tuples."""

    requests_data = []
    valid_keys = []

    for key, image_path, question in records:
        if not os.path.exists(image_path):
            print(f"  WARNING: image not found: {image_path}")
            continue

        prompt_text = (
            "You are a medical imaging expert. Describe this medical image in detail. "
            "Include: imaging modality, anatomical region, key findings, abnormalities, "
            "and any relevant clinical observations.\n\n"
            f"Context question (use this to focus your description): {question}\n\n"
            "Provide a thorough, detailed description of the image:"
        )

        abs_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
                ],
            }
        ]
        requests_data.append({
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": MAX_TOKENS,
            "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        valid_keys.append(key)

    if not requests_data:
        return {}

    sem = asyncio.Semaphore(CONCURRENCY)
    completed = 0

    async def _call_one(session, payload):
        nonlocal completed
        async with sem:
            url = f"{base_url}/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            }
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        print(f"  API error: {err[:200]}")
                        return ""
                    data = await resp.json()
                    text = data["choices"][0]["message"]["content"]
                    # Strip thinking tokens
                    if "<think>" in text:
                        think_end = text.find("</think>")
                        if think_end >= 0:
                            text = text[think_end + len("</think>"):].strip()
                    completed += 1
                    if completed % 50 == 0 or completed == len(requests_data):
                        print(f"  {completed}/{len(requests_data)} done")
                    return text.strip()
            except Exception as e:
                print(f"  Request error: {e}")
                return ""

    print(f"Generating {len(requests_data)} captions via {base_url} "
          f"(model={model}, concurrency={CONCURRENCY})...")
    t0 = time.time()
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_call_one(session, req) for req in requests_data]
        results = await asyncio.gather(*tasks)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / max(len(requests_data), 1):.2f}s per image)")

    return {k: v for k, v in zip(valid_keys, results) if v}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--train_n", type=int, default=300,
                        help="Number of train samples to caption (0 = all)")
    args = parser.parse_args()

    base_url = os.getenv("BASE_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "EMPTY")

    for split in args.splits:
        print(f"\n{'='*50}")
        print(f"  {split.upper()} split — model: {args.model}")
        print(f"{'='*50}")

        df = load_split(split, args.seed)
        if split == "train" and args.train_n and args.train_n < len(df):
            # Must match PMCVQADataset._load_data: df.sample(n=N, random_state=seed)
            df = df.sample(n=args.train_n, random_state=args.seed).reset_index(drop=True)
        print(f"Loaded {len(df)} samples")

        cache_path = get_cache_path(split, args.seed, args.model)
        existing = {}
        if cache_path.exists():
            existing = json.loads(cache_path.read_text())
            print(f"Loaded {len(existing)} existing captions from {cache_path.name}")

        # Find missing
        missing = []
        for _, row in df.iterrows():
            key = row["image_filename"]
            if key not in existing:
                missing.append((key, row["image_path"], row["question"]))

        if not missing:
            print("All captions already cached!")
            continue

        print(f"Need to generate {len(missing)} new captions")
        new_captions = asyncio.run(generate_captions(
            missing, args.model, base_url, api_key
        ))

        existing.update(new_captions)
        cache_path.write_text(json.dumps(existing, indent=2))
        print(f"Saved {len(existing)} total captions to {cache_path.name}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
