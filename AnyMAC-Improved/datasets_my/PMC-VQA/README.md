# PMC-VQA

Medical visual QA over PubMed Central figures.

Download the dataset and place:

- `data/train_sample_2000.csv`  — 2 000 training rows (subset)
- `data/pmcvqa_test.csv`        — test split

Auxiliary directories (already ignored by git):

- `data/PMC-VQA/`               — raw images
- `data/captions/`              — caption cache (Qwen3.5-9B generated)
- `data/image_embeddings/`      — pre-computed embeddings (`.pt`)

## Download

<add download URL / HF dataset ID here>

## Alternate data sources

The loader also accepts:
- `--train_json_path` / `--test_json_path` CLI flags (custom JSON splits)
- Hugging Face `parthpk/Medsets` cache
- Bundled `datasets_my/PMC-VQA-Test/PMC-VQA-Test.json` (only used as last fallback; requires `PMC_VQA_IMAGE_ROOT` env var)
