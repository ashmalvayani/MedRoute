# AnyMAC-Improved

The core MedRoute multi-agent routing implementation. Introduces dynamic specialist prompting and dynamic specialist pools on top of a learned routing transformer trained via GRPO. All advanced features are togglable via CLI flags.

## Key Strategies

### 1. Dynamic Specialist Prompts

**Problem:** With a static role description per specialist (e.g., "You are a cardiologist...") that's the same regardless of the question, all 60 specialists are effectively the same LLM (Qwen3-8B) with generic prompts → routing decisions have minimal impact on answer quality → router has no useful signal to learn from.

**Solution:** Before each specialist consultation, we call a prompt-generation model to produce **question-specific guidance**. The model analyzes the medical question and the specialist's role, then produces targeted instructions like: *"Focus on disulfide bond cleavage in mucin proteins and the role of N-acetylcysteine in COPD management."*

**Key design choices:**
1. **Never say "outside your expertise"** — every specialist can contribute useful reasoning. Frame the question through their lens.
2. **Be specific** — reference actual values, symptoms, and findings from the question. No generic advice.
3. **Specialist names in hints** — receiving specialists see "Cardiologist: ..." instead of "Agent 1: ..." so they can weigh opinions by relevance.
4. **Prompt model flexibility** — `--prompt_model` flag allows using Qwen3-8B (faster, same endpoint) or Qwen3-32B (better quality, separate endpoint) for prompt generation.

### 2. Dynamic Specialist Pool

**Problem:** The fixed pool of 60 generic specialists leads to routing collapse — the router memorizes a few "safe" indices (e.g., 99.1% Neurologist first hop) regardless of the question. Most specialists are never selected.

**Solution:** Instead of routing over 60 fixed specialists, a judge model (Qwen3-32B) generates **5-7 question-specific specialists** per question. The router then selects `max_routing` specialists from this small, curated pool.

**Key design choices:**
1. **Question-specific roles** — e.g., "Pediatric Hematologist" for a sickle cell question, not generic "Hematologist"
2. **Tailored descriptions** — each specialist gets a 2-3 sentence briefing specific to the question
3. **On-the-fly embeddings** — role embeddings computed from generated descriptions, not memorized indices
4. **Panel caching** — same question always gets the same panel (cached by question hash)

**Implementation:**
- `GDesigner/prompt/dynamic_prompt.py` — Async prompt generator + panel generator with in-memory caching
- `GDesigner/agents/analyze_agent.py` — Uses dynamic prompt as system prompt when enabled
- `GDesigner/graph/graph.py` — Dynamic pool integration: on-the-fly embeddings, routing, gradient replay
- Fallback: returns the original static description / fixed pool on any error

---

## Usage

```bash
# Dynamic pool + dynamic prompts, 32B consistent (best config)
bash scripts/train_dynamic_pool_32b.sh

# Dynamic pool + dynamic prompts, 8B base + 32B judge
bash scripts/train_dynamic_pool.sh

# Dynamic prompts only, fixed 60-specialist pool (best SC config)
bash scripts/train_improved_medqa.sh

# Dynamic prompts with Qwen3-8B for prompt generation
bash scripts/train_improved_medqa.sh --prompt_model Qwen/Qwen3-8B

# No improvements (baseline with entropy fix only)
bash scripts/train_improved_medqa.sh --no-dynamic
```

### Locked Hyperparameters

| Parameter | Value |
|-----------|-------|
| train_num | 300 |
| num_traces | 16 |
| max_routing | 3 |
| lr | 3e-5 |
| GS tau | 1.0 |
| LLM temp | 0.0 |
| decay | 0.98 |
| batch_size | 8 |
| entropy_beta | 0.05 |
| eval_temperature | 0.7 |
| epochs | 1-10 |

### Direct Python usage

```bash
# Train + eval with dynamic prompts (32B prompt gen)
python experiments/run_medqa.py \
    --llm_name Qwen/Qwen3-8B \
    --judge_model Qwen/Qwen3-32B \
    --dynamic_prompts \
    --epochs 1 --train_num 300 --lr 3e-5

# Train + eval with dynamic prompts (8B prompt gen)
python experiments/run_medqa.py \
    --llm_name Qwen/Qwen3-8B \
    --judge_model Qwen/Qwen3-32B \
    --dynamic_prompts --prompt_model Qwen/Qwen3-8B \
    --epochs 1 --train_num 300 --lr 3e-5

# Self-consistency eval
python experiments/eval_self_consistency_routed.py \
    --model_path <checkpoint.pth> \
    --dynamic_prompts --judge_model Qwen/Qwen3-32B \
    --num_rollouts 3 --eval_temperature 0.7 --parallelism 256
```

## Key Fixes

1. **Routing collapse** — entropy regularization + eval_temperature replaces cos_scaling=1e3 argmax
2. **Dynamic prompts** — question-specific guidance replaces generic role descriptions
3. **Dynamic specialist pool** — question-specific 5-7 specialist panel replaces fixed 60 specialists
4. **Specialist names in hints** — "Cardiologist: ..." instead of "Agent 1: ..."
5. **Prompt model flexibility** — `--prompt_model` flag to use 8B or 32B for prompt generation

## File Layout

### New files
- `GDesigner/prompt/dynamic_prompt.py` — Dynamic prompt + panel generator
- `GDesigner/reward/__init__.py` + `partial_credit.py` — Partial credit reward (not used in best config)
- `GDesigner/hints/__init__.py` + `structured_hints.py` — Structured hint parser (not used in best config)
- `scripts/train_improved_medqa.sh` — Training script (dynamic prompts)
- `scripts/train_dynamic_pool.sh` — Training script (dynamic pool + prompts, 8B base)
- `scripts/train_dynamic_pool_32b.sh` — Training script (dynamic pool + prompts, 32B consistent)
- `experiments/eval_self_consistency_routed.py` — SC evaluation
- `experiments/smoke_test_dynamic_pool.py` — Panel generation smoke test

### Modified files
- `GDesigner/agents/analyze_agent.py` — Dynamic prompt integration + prompt_model support + dynamic_description
- `GDesigner/graph/graph.py` — Entropy regularization, eval_temperature, dynamic pool, specialist names in hints
- `GDesigner/prompt/medqa_prompt_set.py` — Graceful handling of unknown roles (dynamic pool)
- `experiments/run_medqa.py` — New CLI flags (`--dynamic_prompts`, `--dynamic_pool`, `--prompt_model`)
- `experiments/train_medqa.py` — Dynamic pool gradient replay support

---

## PMC-VQA (Vision)

Medical Visual Question Answering on the PMC-VQA dataset (2,000 test questions, 4-option MCQ). The challenge: the GNN router is text-only, but questions require visual understanding of medical images.

### Pipeline Architecture

- **Agent Model:** Qwen3.5-9B (VLM) — specialists see the actual image + question + options
- **Pool/Prompt/Judge Model:** Qwen3.6-27B (VLM) — generates specialist panels and dynamic prompts, sees the image
- **Router:** GNN with transformer — text-only embeddings (MiniLM), or text + image embeddings (SigLIP)
- **Training:** 300 train samples, 16 traces, lr=1e-5, max_routing=3, 1 epoch, entropy_beta=0.05

### Approaches

**1. Baseline — Zero-shot VLM:** Qwen3.5-9B direct inference (no routing, no specialists).

**2. Caption-as-Proxy for Routing:** The GNN router is text-only, so we generate captions from medical images using a VLM and embed `caption + question` as the router input. Specialists (VLM agents) still see the actual image. Tested with three caption models:
- **9B captions** (Qwen3.5-9B) — default, same model as the agent
- **27B captions** (Qwen3.6-27B) — better quality, richer clinical descriptions
- **122B captions** (Qwen3.5-122B-A10B, MoE with 10B active) — more verbose but less visually precise, hurt routing

**3. Panel Variations:**
- **Detailed panel** — specific sub-specialty titles ("Pediatric Neuroradiologist") → unstable GNN embeddings
- **Simple panel** — broad specialty titles ("Radiologist") → stable embeddings, better accuracy. Made default.

**4. Hint Ablation:**
- **With hints** — specialists see previous specialists' outputs → better accuracy
- **No hints** — specialists reason independently → worse accuracy (60.50%)

**5. DM Chain-of-Thought:** Decision Maker reasons step-by-step before outputting `ANSWER: X` instead of a single letter. Forces explicit reasoning.

**6. Image Embedding Routing:** Replace lossy text captions with SigLIP (google/siglip-so400m-patch14-384) image embeddings (1152-dim) concatenated with MiniLM text embeddings (384-dim) as the GNN router input. Eliminates caption generation and provides richer visual features.

**7. L2 Normalization:** SigLIP embeddings (1152-dim, magnitude 0-50+) were dominating MiniLM text embeddings (384-dim, L2-normalized ~1.0) in the projection layer. L2-normalizing both before concatenation balances text and image modalities.

**8. Higher Entropy (beta=0.15):** Increase entropy regularization to encourage specialist exploration. The router was collapsing to 83% Pulmonologist as first choice.

**9. Min Routing Depth 3:** Force every question to consult 3 specialists before the Decision Maker answers. Eliminates depth-0 shortcuts (58.5% accuracy).

**10. Cross-Attention Fusion:** Instead of concatenating text and image embeddings, project each into the transformer hidden space separately, then use 8-head MultiheadAttention where text queries attend to image keys/values. Learns a more expressive text-image alignment than concatenation.

**11. Caption + Image Fusion:** Combine all three signals — text question embedding (384-dim) + caption embedding (384-dim) + SigLIP image embedding (1152-dim) = 1920-dim input. Hypothesis: captions provide semantic understanding while image embeddings provide raw visual features. Result: the high-dimensional input was harder to learn with 300 training samples.

## Cross-Dataset Comparison

| Configuration | MedQA | PubMedQA |
|---|---|---|
| **Baselines** | | |
| Qwen3-8B zero-shot | 59.50% | 50.00% |
| MAM (Qwen3-8B + 32B rolegen) | 63.24% | 44.10% |
| **MedRoute (1 epoch)** | | |
| Static pool, no dynamic prompts | 61.51% | 41.70% |
| Static pool + dynamic prompts | 67.63% | 53.00% |
| Dynamic pool + prompts (seed=42) | 69.05% | 53.70% |
| Dynamic pool + prompts (seed=99) | **71.48%** | **55.10%** |
| **Improvement over zero-shot** | **+11.98%** | **+5.10%** |
