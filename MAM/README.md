# MAM — Medical Multi-Agent Evaluation Fork

This fork of [MAM (ACL 2025 Findings)](https://arxiv.org/abs/2506.19835) is adapted
to evaluate arbitrary **text LLMs** and **vision-language models (VLMs)** on medical
QA benchmarks using [vLLM](https://github.com/vllm-project/vllm) for inference.
The original paper's README lives in [old_readme.md](old_readme.md).

Key changes vs. upstream:

- **vLLM backend.** The text LLM is loaded **once** as a process-wide singleton
  (instead of HuggingFace `transformers` re-loading on every pipeline call).
  Supports both a **local vLLM engine** and a **remote OpenAI-compatible vLLM
  server** (`MAM_LLM_API_URL`).
- **Two-model vision architecture.** A separate VLM endpoint (`MAM_VLM_API_URL`)
  handles all image-understanding steps (type classification, specialist
  discussions, diagnosis, review), while the text LLM handles text-only steps
  (modality selection, role generation, web search summarisation). This lets you
  pair e.g. Qwen3-8B (text) + Qwen3.5-VL-9B (vision) on separate GPUs.
- **Six evaluation scripts.** Text: MedQA (5-option MCQ), PubMedQA (yes/no/maybe).
  Vision: PMC-VQA (4-option MCQ), PathVQA (yes/no + open-ended), DeepLesion
  (8-class lesion MCQ), BTMRI (4-class brain tumor MCQ).
- **Per-run isolation.** Every evaluation run gets its own
  `runs/<run-name>/` directory: predictions, per-sample traces, history,
  config, and a full tee of stdout. Different models never clobber each other.
- **Per-sample traces.** Each example writes `runs/<run-name>/traces/<idx>.json`
  containing every pipeline step (`step_1_modality_selection` →
  `step_9_memory`), every specialist's full turn, moderator summaries, votes,
  LLM-call counts, and token usage.
- **Concurrency.** `--concurrency N` runs N questions in parallel via a
  `ThreadPoolExecutor`. Great for saturating a remote vLLM server.
- **No hard-coded chat template.** The previous Medichat-specific prompt is
  gone; each model uses its own tokenizer template (local mode) or the
  server's native template (API mode).

---

## Results

### Text Benchmarks (Qwen3-8B + Qwen3-32B role generation)

All evaluations use the full 9-step MAM pipeline with `--concurrency 128`.
The base LLM (Qwen3-8B) handles all pipeline steps except role generation (step 3),
which is routed to Qwen3-32B via `MAM_ROLE_GEN_API_URL`.

| Dataset | Samples | 8B Only | 8B + 32B Role Gen | Delta |
|---------|---------|---------|-------------------|-------|
| MedQA (5-option MCQ) | 1,273 | 59.60% | **63.24%** | +3.64 |
| PubMedQA (yes/no/maybe) | 1,000 | 41.50% | **44.10%** | +2.60 |

Using a larger model (Qwen3-32B) for specialist role generation improves accuracy
on both benchmarks. The 32B model produces more relevant specialist assignments
(e.g., "Pediatric Hematologist" instead of generic "Hematologist"), leading to
higher-quality multi-agent discussions downstream.

---

## 1. Install

Create an isolated venv so vLLM's deps don't leak:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install vllm transformers requests datasets
```

> vLLM pulls torch, CUDA wheels, etc. On CPU-only machines, install only
> `requests transformers datasets` — you'll need to use **API mode**.

---

## 2. Datasets

| Dataset | Modality | Task | Samples | Location |
|---------|----------|------|---------|----------|
| MedQA | Text | 5-option USMLE MCQ | 1,273 | `data/medqa/test.jsonl` |
| PubMedQA | Text | Yes/No/Maybe | 1,000 | `data/pubmedqa/test.jsonl` |
| PMC-VQA | Image | 4-option MCQ | 2,000 | `data/pmcvqa/test.jsonl` |
| PathVQA | Image | Yes/No + Open-ended | 6,719 | `data/pathvqa/test.jsonl` |
| DeepLesion | Image | 8-class lesion MCQ | 4,927 | `data/deeplesion/test.jsonl` |
| BTMRI | Image | 4-class brain tumor MCQ | 1,311 | `data/btmri/test.jsonl` |

Text datasets are checked in. Vision datasets require downloading images
(see `data/*/` for details).

---

## 3. Launch vLLM

### Text-only (single server)

On the GPU box:

```bash
vllm serve Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 9001 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
```

On the eval machine:

```bash
export MAM_LLM_API_URL=http://<llm-server>:9001/v1
```

### Text + Role generation model (two text servers)

Use a larger model for specialist role generation (step 3) while keeping
the base LLM for everything else:

```bash
# GPU 0-3: base text LLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 --enforce-eager --max-model-len 40960

# GPU 4-7: larger model for role generation
vllm serve Qwen/Qwen3-32B --host 0.0.0.0 --port 8001 \
    --tensor-parallel-size 4 --enforce-eager --max-model-len 40960
```

```bash
export MAM_LLM_API_URL=http://localhost:8000/v1
export MAM_ROLE_GEN_API_URL=http://localhost:8001/v1
export MAM_ROLE_GEN_MODEL=Qwen/Qwen3-32B
```

### Text + Vision (two servers)

Serve a text LLM and a VLM on separate GPUs:

```bash
# GPU 1: text LLM
vllm serve Qwen/Qwen3-8B --host 0.0.0.0 --port 9001

# GPU 2: vision-language model
vllm serve Qwen/Qwen3.5-VL-9B --host 0.0.0.0 --port 9001
```

On the eval machine, point at both:

```bash
export MAM_LLM_API_URL=http://<llm-server>:9001/v1
export MAM_VLM_API_URL=http://<vlm-server>:9001/v1
```

The text LLM handles role generation, web search summarisation, etc. The VLM
handles all image-understanding steps. If `MAM_VLM_API_URL` is not set, the
text LLM endpoint is used for everything.

---

## 4. Running Evaluations

Common environment variables for all runs:

```bash
export MAM_LLM_API_URL=http://<llm-server>:9001/v1
export MAM_VLM_API_URL=http://<vlm-server>:9001/v1        # vision datasets only
export MAM_LLM_MAX_TOKENS=2048
export MAM_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
export MAM_VLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
```

### MedQA (text, 5-option MCQ)

```bash
.venv/bin/python eval_medqa.py \
    --run-name MedQA/Qwen3-8B \
    --concurrency 128 \
    --quiet
```

### PubMedQA (text, yes/no/maybe)

```bash
.venv/bin/python eval_pubmedqa.py \
    --run-name PubMedQA/Qwen3-8B \
    --concurrency 128 \
    --quiet
```

### PMC-VQA (image, 4-option MCQ)

```bash
.venv/bin/python eval_pmcvqa.py \
    --run-name PMC-VQA/Qwen35-9B_Qwen3-8B \
    --concurrency 128 \
    --quiet
```

### PathVQA (image, yes/no + open-ended)

Full run (both question types):

```bash
.venv/bin/python eval_pathvqa.py \
    --run-name PathVQA/Qwen35-9B_Qwen3-8B \
    --concurrency 128 \
    --quiet
```

Yes/no only:

```bash
.venv/bin/python eval_pathvqa.py \
    --run-name PathVQA-YN/Qwen35-9B_Qwen3-8B \
    --qtype yesno \
    --concurrency 256 \
    --quiet
```

Open-ended only:

```bash
.venv/bin/python eval_pathvqa.py \
    --run-name PathVQA-Open/Qwen35-9B_Qwen3-8B \
    --qtype open \
    --concurrency 256 \
    --quiet
```

### DeepLesion (image, 8-class lesion MCQ)

```bash
.venv/bin/python eval_deeplesion.py \
    --run-name DeepLesion/Qwen35-9B_Qwen3-8B \
    --concurrency 128 \
    --quiet
```

### BTMRI (image, 4-class brain tumor MCQ)

```bash
.venv/bin/python eval_btmri.py \
    --run-name BTMRI/Qwen35-9B_Qwen3-8B \
    --concurrency 256 \
    --quiet
```

### Common CLI flags

| flag | default | meaning |
|---|---|---|
| `--data` | dataset-specific | input file (one JSON record per line) |
| `--data-dir` | dataset-specific | base directory for image paths (vision scripts) |
| `--run-name` | `<model>__<dataset>__<timestamp>` | output dir name under `--runs-dir` |
| `--runs-dir` | `runs` | parent dir for runs |
| `--step_id` | `9` | run pipeline through step N (1-9); 9 = full pipeline incl. memory |
| `--limit` | `0` | only process the first N examples (0 = all) |
| `--start` | `0` | start index (for sharding or resuming) |
| `--concurrency` | `1` | threads for parallel evaluation |
| `--resume` | off | skip indices already written to `predictions.jsonl` |
| `--quiet` | off | suppress per-example pipeline prints on stdout (still in `eval.log`) |
| `--qtype` | all | PathVQA only: filter to `yesno` or `open` questions |

### Smoke test (any dataset)

Add `--limit 2` to any command above to run on just 2 samples:

```bash
.venv/bin/python eval_medqa.py --run-name smoke --limit 2 --quiet
```

### Resume an interrupted run

```bash
.venv/bin/python eval_medqa.py --run-name MedQA/Qwen3-8B --resume --concurrency 128 --quiet
```

---

## 5. Environment Variables

### Backend selection

| var | default | purpose |
|---|---|---|
| `MAM_LLM_API_URL` | *(unset)* | remote vLLM URL for text LLM (e.g. `http://host:9001/v1`); if unset, loads local vLLM engine |
| `MAM_LLM_MODEL` | `sethuiyer/Medichat-Llama3-8B` (local), auto-detected (API) | HF model id |
| `MAM_VLM_API_URL` | *(unset)* | remote vLLM URL for VLM; if unset, falls back to `MAM_LLM_API_URL` |
| `MAM_VLM_MODEL` | auto-detected | VLM model id |
| `MAM_ROLE_GEN_API_URL` | *(unset)* | remote vLLM URL for role generation model; if unset, uses `MAM_LLM_API_URL` |
| `MAM_ROLE_GEN_MODEL` | auto-detected | role generation model id |

### API-mode only

| var | default | purpose |
|---|---|---|
| `MAM_LLM_API_KEY` | `EMPTY` | bearer token for text LLM server |
| `MAM_VLM_API_KEY` | `EMPTY` | bearer token for VLM server |
| `MAM_LLM_API_TIMEOUT` | `600` | per-request timeout in seconds |
| `MAM_LLM_EXTRA_BODY` | `""` | JSON merged into every text LLM request |
| `MAM_VLM_EXTRA_BODY` | `""` | JSON merged into every VLM request |

### Local-mode only

| var | default | purpose |
|---|---|---|
| `MAM_LLM_GPU_MEM_UTIL` | `0.9` | vLLM `gpu_memory_utilization` |
| `MAM_LLM_MAX_MODEL_LEN` | *(model native)* | vLLM `max_model_len` |
| `MAM_LLM_DTYPE` | `auto` | vLLM dtype |
| `MAM_LLM_TENSOR_PARALLEL` | `1` | vLLM `tensor_parallel_size` |
| `CUDA_VISIBLE_DEVICES` | *(all)* | which GPUs vLLM can see |

### Cross-mode

| var | default | purpose |
|---|---|---|
| `MAM_LLM_MAX_TOKENS` | *(unset)* | floor on `max_tokens` for every call. Needed for thinking models whose 512 default gets eaten by `<think>...</think>` |
| `MAM_HISTORY_DIR` | *(set by eval scripts)* | where `pipeline/memory.py` reads/writes history JSONs. Eval scripts point this at `runs/<run-name>/history/` |

---

## 6. Output Layout

```
runs/<run-name>/
    config.json            # model, URL, step_id, args, aggregate usage/accuracy
    predictions.jsonl      # one compact record per example
    eval.log               # full stdout/stderr tee (all pipeline prints)
    traces/<idx>.json      # per-sample full trace (see below)
    history/               # pipeline/memory.py writes here — isolated per run
        text_history.json
```

### `traces/<idx>.json` (full qualitative trace)

Every pipeline step is labelled top-level under `pipeline`:

- `step_1_modality_selection`
- `step_2_type_classification`
- `step_3_role_generation`
- `step_4_web_search`
- `step_5_load_history`
- `step_6_multi_agent_meeting` — `parsed_roles`, `verdict`, `rounds[]`
  (each round has `discussions[]` with every specialist's `content`, plus
  `summary`, `votes`, `outcome`)
- `step_7_final_diagnosis`
- `step_8_review`
- `step_9_memory`

---

## 7. Answer Evaluation

Each dataset uses rule-based answer extraction (no LLM judge):

| Dataset | Method |
|---------|--------|
| MedQA | Regex extracts letter A-E from model output; exact match against gold |
| PubMedQA | Regex extracts yes/no/maybe; exact match |
| PMC-VQA | Regex extracts letter A-D, with option-text fallback; exact match |
| PathVQA (yes/no) | Regex extracts yes/no; exact match |
| PathVQA (open-ended) | Normalized gold answer containment check in model response |
| DeepLesion | Regex extracts letter A-H, with lesion-type name fallback; exact match |
| BTMRI | Regex extracts letter A-D, with tumor-type name fallback; exact match |

---

## 8. Tips

- **Qwen3 / thinking models.** Always set
  `MAM_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'`
  and bump `MAM_LLM_MAX_TOKENS` to at least `2048`, otherwise answers get
  truncated mid-`<think>`.
- **Saturating the server.** `--concurrency` matching the server's
  `--max-num-seqs` (typically 32-256 for a single GPU) maximises throughput.
  Start at 8 and scale up while watching GPU util.
- **Accuracy parity with the paper** requires `--step_id 9` (full pipeline,
  including the memory write that biases subsequent examples). Use `--step_id 8`
  for a stateless evaluation where each example is independent.
- **Comparing models.** Give each run a distinct `--run-name`; everything is
  isolated so you can `diff` their `config.json` / `predictions.jsonl`.
- **Two-model quality.** Using a text LLM for role generation produces better
  structured output than a VLM, which tends to generate verbose responses that
  break the upstream role-parsing regex. The two-model setup is recommended for
  vision datasets.
