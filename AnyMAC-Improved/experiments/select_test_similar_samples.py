"""
Test-distribution-aware training sample selection.

Embeds all training + test questions via a vLLM embedding server (LLM-based embeddings,
not BERT), then selects training samples whose embeddings are closest to the test set.

The intuition: training on questions similar to the test distribution gives the router
routing patterns that transfer directly to evaluation questions.

Steps:
  1. Embed all MedQA training (dev) + test questions via /v1/embeddings
  2. Compute cosine similarity: each test q vs all training qs
  3. For each test question, find top-k nearest training neighbors
  4. Rank training questions by how often they appear as neighbors → most test-representative
  5. Select top-N and write to datasets_my/MedQA/data/test_similar/medqa.csv

Usage:
  python experiments/select_test_similar_samples.py \
      --embed_model Qwen/Qwen3-0.6B \
      --embed_url http://localhost:8002 \
      --top_k 5 \
      --select_n 300 \
      --parallelism 128 \
      --output_split test_similar

  # Also combine with mixed-signal filter (recommended):
  --mixed_signal_json result/medqa_mixed_samples.json
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

import aiohttp
import numpy as np
from tqdm import tqdm

from GDesigner.utils.const import GDesigner_ROOT
from datasets_my.medqa_dataset import MedQADataset


EMBED_BASE_URL = os.getenv('EMBED_BASE_URL', 'http://localhost:8002')
EMBED_API_KEY  = os.getenv('EMBED_API_KEY',  'EMPTY')


# ---------------------------------------------------------------------------
# Async embedding
# ---------------------------------------------------------------------------

async def embed_batch(session: aiohttp.ClientSession, model: str, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of embedding vectors."""
    url = f"{EMBED_BASE_URL}/v1/embeddings"
    payload = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {EMBED_API_KEY}"}
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"Embedding API error {resp.status}: {text[:200]}")
        r = await resp.json()
        # Sort by index to preserve order
        return [item['embedding'] for item in sorted(r['data'], key=lambda x: x['index'])]


async def embed_all(texts: list[str], model: str, batch_size: int, parallelism: int) -> np.ndarray:
    """Embed all texts in parallel batches. Returns (N, dim) float32 array."""
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(parallelism)
    results = [None] * len(batches)

    async def _do_batch(i, batch):
        async with sem:
            return await embed_batch(session, model, batch)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(_do_batch(i, b)) for i, b in enumerate(batches)]
        for i, fut in enumerate(tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Embedding batches")):
            batch_idx = tasks.index(fut) if hasattr(fut, '__index__') else i
            result = await fut
            # find which batch this result belongs to — use task order
        # Re-run sequentially with progress to preserve order
        pass

    # Simpler ordered approach with semaphore
    embeddings_flat = []
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _run(batch):
            async with sem:
                return await embed_batch(session, model, batch)

        sem = asyncio.Semaphore(parallelism)
        pending = [asyncio.create_task(_run(b)) for b in batches]

        for task in tqdm(asyncio.as_completed(pending), total=len(pending), desc="Embedding"):
            await task  # just drain

        # Gather in order
        all_vecs = []
        for task in pending:
            vecs = await task
            all_vecs.extend(vecs)

    return np.array(all_vecs, dtype=np.float32)


async def embed_all_ordered(texts: list[str], model: str, batch_size: int, parallelism: int) -> np.ndarray:
    """Embed all texts preserving order, with parallel batches."""
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    sem = asyncio.Semaphore(parallelism)
    timeout = aiohttp.ClientTimeout(total=300)
    pbar = tqdm(total=len(batches), desc="Embedding batches")

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def _run(batch):
            async with sem:
                result = await embed_batch(session, model, batch)
                pbar.update(1)
                return result

        results = await asyncio.gather(*[_run(b) for b in batches])

    pbar.close()

    # Flatten in order (asyncio.gather preserves order)
    all_vecs = []
    for batch_vecs in results:
        all_vecs.extend(batch_vecs)

    return np.array(all_vecs, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity: (N_a, N_b). a=(N_a, D), b=(N_b, D)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return a_norm @ b_norm.T  # (N_a, N_b)


def main():
    p = argparse.ArgumentParser(description="Select test-similar training samples via LLM embeddings")

    p.add_argument("--embed_model",  type=str, default="Qwen/Qwen3-0.6B")
    p.add_argument("--embed_url",    type=str, default="http://localhost:8002")
    p.add_argument("--top_k",        type=int, default=5,
                   help="Nearest test neighbors per training question to count")
    p.add_argument("--select_n",     type=int, default=300,
                   help="Number of training samples to select")
    p.add_argument("--batch_size",   type=int, default=64,
                   help="Questions per embedding request")
    p.add_argument("--parallelism",  type=int, default=128,
                   help="Concurrent embedding requests")
    p.add_argument("--output_split", type=str, default="test_similar",
                   help="Name of new dataset split folder under datasets_my/MedQA/data/")
    p.add_argument("--mixed_signal_json", type=str, default=None,
                   help="Optional: path to screen_medqa_samples.py output to intersect with mixed-signal filter")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    global EMBED_BASE_URL
    EMBED_BASE_URL = args.embed_url

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MedQADataset('dev')
    test_dataset  = MedQADataset('test')
    print(f"  Train: {len(train_dataset)} questions")
    print(f"  Test:  {len(test_dataset)} questions")

    # Build text lists
    train_texts = [train_dataset.record_to_input(train_dataset[i])["task"] for i in range(len(train_dataset))]
    test_texts  = [test_dataset.record_to_input(test_dataset[i])["task"]  for i in range(len(test_dataset))]

    print(f"\nEmbedding {len(train_texts)} train + {len(test_texts)} test questions")
    print(f"  Model: {args.embed_model} @ {EMBED_BASE_URL}")
    print(f"  Batch: {args.batch_size}, Parallelism: {args.parallelism}")

    t0 = time.time()
    all_texts = train_texts + test_texts
    all_embeddings = asyncio.run(embed_all_ordered(all_texts, args.embed_model, args.batch_size, args.parallelism))

    train_embs = all_embeddings[:len(train_texts)]   # (N_train, D)
    test_embs  = all_embeddings[len(train_texts):]   # (N_test,  D)
    elapsed = time.time() - t0
    print(f"\nEmbedding done in {elapsed:.1f}s — shape: train={train_embs.shape}, test={test_embs.shape}")

    # Compute similarity: for each TRAINING question, how similar is it to the TEST set?
    # sim[i, j] = similarity between train[i] and test[j]
    print("\nComputing cosine similarity matrix...")
    sim = cosine_similarity_matrix(train_embs, test_embs)   # (N_train, N_test)
    print(f"  Similarity matrix: {sim.shape}")

    # Strategy 1: mean similarity to test set (selects most "average-test-like" questions)
    mean_sim = sim.mean(axis=1)   # (N_train,)

    # Strategy 2: top-k neighbor frequency (selects questions that are nearest neighbor for many test qs)
    # For each test question, find top-k training neighbors
    neighbor_counts = np.zeros(len(train_texts), dtype=np.int32)
    for j in range(len(test_texts)):
        top_k_train = np.argpartition(sim[:, j], -args.top_k)[-args.top_k:]
        neighbor_counts[top_k_train] += 1

    print(f"\n  Mean similarity: min={mean_sim.min():.4f} max={mean_sim.max():.4f} avg={mean_sim.mean():.4f}")
    print(f"  Neighbor counts: max={neighbor_counts.max()} — {(neighbor_counts>0).sum()} training qs are top-{args.top_k} neighbor for at least 1 test q")

    # Apply mixed-signal filter if provided
    valid_indices = set(range(len(train_texts)))
    if args.mixed_signal_json:
        print(f"\n  Applying mixed-signal filter from {args.mixed_signal_json}...")
        with open(args.mixed_signal_json) as f:
            mdata = json.load(f)
        mixed_idxs = set(q['dataset_index'] for q in mdata['mixed_questions'])
        valid_indices = valid_indices & mixed_idxs
        print(f"  Remaining after mixed-signal filter: {len(valid_indices)} questions")

    # Score = combined: normalize both scores and average
    mean_sim_norm = (mean_sim - mean_sim.min()) / (mean_sim.max() - mean_sim.min() + 1e-8)
    nbr_norm = neighbor_counts / (neighbor_counts.max() + 1e-8)
    score = 0.5 * mean_sim_norm + 0.5 * nbr_norm

    # Rank by score within valid indices
    valid_list = sorted(valid_indices, key=lambda i: score[i], reverse=True)
    selected_indices = valid_list[:args.select_n]

    print(f"\n  Selected {len(selected_indices)} training questions (top by test similarity)")
    sel_scores = [score[i] for i in selected_indices]
    print(f"  Score range: {min(sel_scores):.4f} – {max(sel_scores):.4f}")

    # Write output CSV
    out_dir = Path(f"{GDesigner_ROOT}/datasets_my/MedQA/data/{args.output_split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "medqa.csv"

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for idx in selected_indices:
            record = train_dataset[idx]
            writer.writerow([
                record['question'], record['A'], record['B'],
                record['C'], record['D'], record['E'],
                record['correct_answer'],
            ])

    print(f"\n  Wrote {len(selected_indices)} questions → {out_csv}")

    # Verify
    verify = MedQADataset(args.output_split)
    print(f"  Verified: loaded back {len(verify)} questions")

    # Save metadata
    meta = {
        "embed_model":      args.embed_model,
        "total_train":      len(train_texts),
        "total_test":       len(test_texts),
        "selected_n":       len(selected_indices),
        "top_k":            args.top_k,
        "mixed_filter":     args.mixed_signal_json is not None,
        "elapsed_seconds":  round(elapsed, 1),
        "selected_indices": selected_indices,
        "embedding_dim":    int(train_embs.shape[1]),
    }
    meta_path = out_dir / "selection_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata → {meta_path}")


if __name__ == "__main__":
    main()
