"""
PMC-VQA dataset class with integrated caption generation.

Captions are generated via Qwen3.5-9B (vision model) and cached to disk.
The router and text-only agents see: caption + question + options.
This keeps the pipeline text-only while leveraging vision understanding.

Bundled ``datasets_my/PMC-VQA-Test/PMC-VQA-Test.json`` is only used after Hugging Face Medsets and
repo CSVs fail: Medsets places images under ``PMC-VQA/images_test_clean/`` next to ``pmcvqa_test.csv``.
That JSON lists author-local paths unless ``PMC_VQA_IMAGE_ROOT`` / ``PMC_VQA_TEST_IMAGE_ROOT`` /
``TEST_JSON_IMAGE_ROOT`` remaps basenames to your tree.

Cache files: datasets_my/PMC-VQA/data/captions/{split}_{n}samples_seed{seed}.json
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

from datasets_my.medsets_csv import (
    extract_choice,
    format_options,
    load_csv_as_records,
    load_local_vision_csv_records,
    load_medsets_records,
    records_to_dataframe,
    require_existing_images,
)


_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "PMC-VQA" / "data"
_IMAGE_BASE = _DATA_DIR / "PMC-VQA"
_CAPTION_DIR = _DATA_DIR / "captions"
_PACKAGE_TEST_JSON = Path(os.path.dirname(os.path.abspath(__file__))) / "PMC-VQA-Test" / "PMC-VQA-Test.json"


class PMCVQADataset(ABC):
    def __init__(
        self,
        split: str,
        sample_n: int | None = None,
        seed: int = 99,
        train_json_path: str | None = None,
        test_json_path: str | None = None,
        caption_model: str = "Qwen/Qwen3.5-9B",
        skip_caption_gen: bool = False,
        use_image_embeddings: bool = False,
        vision_encoder: str = None,
    ) -> None:
        self._split = split
        self._sample_n = sample_n
        self._seed = seed
        self._train_json_path = train_json_path
        self._test_json_path = test_json_path
        self._caption_model = caption_model
        self._use_image_embeddings = use_image_embeddings
        self._vision_encoder = vision_encoder

        self._total_df = self._load_data(split, sample_n, seed)

        # Load or generate captions (skip if using image embeddings instead)
        if not skip_caption_gen and not use_image_embeddings:
            self._ensure_captions()

        # Pre-compute image embeddings if requested
        if use_image_embeddings:
            self._ensure_image_embeddings()

    def limit(self, n: int):
        """Truncate dataset to first n samples (e.g. for smoke tests).
        Call AFTER init so captions are only generated for what's needed."""
        if n < len(self._total_df):
            self._total_df = self._total_df.iloc[:n].reset_index(drop=True)
        return self

    @staticmethod
    def get_domain() -> str:
        return "pmcvqa"

    @staticmethod
    def _is_packaged_pmcvqa_csv(path: Path) -> bool:
        return path.name in ("train_sample_2000.csv", "pmcvqa_test.csv")

    def _load_json_records(self) -> List[Dict[str, Any]] | None:
        train_path = Path(self._train_json_path) if self._train_json_path else None
        test_path = Path(self._test_json_path) if self._test_json_path else None
        if not train_path and not test_path:
            return None

        if train_path and test_path:
            source = train_path if self._split == "train" else test_path
            if source.suffix.lower() == ".csv":
                if self._is_packaged_pmcvqa_csv(source):
                    return None
                return load_csv_as_records(source, source.parent, "pmcvqa")
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data

        source = test_path or train_path
        assert source is not None
        if source.suffix.lower() == ".csv":
            if self._is_packaged_pmcvqa_csv(source):
                return None
            recs = load_csv_as_records(source, source.parent, "pmcvqa")
            rng = np.random.default_rng(self._seed)
            indices = rng.permutation(len(recs))
            cut = int(0.8 * len(indices))
            picked = indices[:cut] if self._split == "train" else indices[cut:]
            return [recs[i] for i in picked]
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        rng = np.random.default_rng(self._seed)
        indices = rng.permutation(len(data))
        cut = int(0.8 * len(indices))
        picked = indices[:cut] if self._split == "train" else indices[cut:]
        return [data[i] for i in picked]

    @staticmethod
    def _remap_packaged_test_figure_paths(records: List[Dict[str, Any]]) -> None:
        """Paths inside PMC-VQA-Test.json point at another machine; join basename to local tree."""
        root = (
            os.environ.get("PMC_VQA_IMAGE_ROOT")
            or os.environ.get("PMC_VQA_TEST_IMAGE_ROOT")
            or os.environ.get("TEST_JSON_IMAGE_ROOT")
        )
        if not root:
            return
        root_path = Path(root).expanduser().resolve()
        for rec in records:
            fp = rec.get("figure_path")
            if fp is None:
                continue
            rec["figure_path"] = str(root_path / Path(str(fp).strip()).name)

    def _try_packaged_pmcvqa_test_json(
        self,
        split: str,
        sample_n: int | None,
        seed: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame | None:
        if split != "test" or not _PACKAGE_TEST_JSON.is_file():
            return None
        with open(_PACKAGE_TEST_JSON, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        self._remap_packaged_test_figure_paths(raw)
        df = records_to_dataframe(raw, sample_n, seed)
        for col in ["A", "B", "C", "D"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        df = df.reset_index(drop=True)
        df = df.reindex(rng.permutation(df.index)).reset_index(drop=True)
        print(
            f"[PMC-VQA] Loaded {len(df)} samples from {_PACKAGE_TEST_JSON.name} ({split} split)"
            + (f" (sampled {sample_n})" if sample_n else "")
        )
        return df

    def _try_packaged_pmcvqa_csv(
        self,
        split: str,
        sample_n: int | None,
        seed: int,
        rng: np.random.Generator,
    ) -> pd.DataFrame | None:
        if split == "train":
            train_csv = _DATA_DIR / "train_sample_2000.csv"
            if not train_csv.is_file():
                return None
            df = pd.read_csv(train_csv)
            df = df.rename(columns={
                "Figure_path": "image_filename",
                "Question": "question",
                "Answer": "answer_text",
                "Choice A": "A",
                "Choice B": "B",
                "Choice C": "C",
                "Choice D": "D",
                "Answer_label": "correct_answer",
            })
            df["image_path"] = df["image_filename"].apply(
                lambda x: str(_IMAGE_BASE / "train_sample_2000" / x.strip())
            )
            if sample_n and sample_n < len(df):
                df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        elif split == "test":
            test_csv = _DATA_DIR / "pmcvqa_test.csv"
            if not test_csv.is_file():
                return None
            df = pd.read_csv(
                test_csv,
                header=None,
                names=["question", "A", "B", "C", "D", "correct_answer",
                       "image_rel", "existing_caption"],
            )
            df["image_filename"] = df["image_rel"].apply(lambda x: x.strip().split("/")[-1])
            df["image_path"] = df["image_rel"].apply(
                lambda x: str(_IMAGE_BASE / x.strip())
            )
            if sample_n and sample_n < len(df):
                df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

        for col in ["A", "B", "C", "D"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        df = df.reset_index(drop=True)
        df = df.reindex(rng.permutation(df.index)).reset_index(drop=True)
        print(
            f"[PMC-VQA] Loaded {len(df)} samples from repo CSV ({split} split)"
            + (f" (sampled {sample_n})" if sample_n else "")
        )
        return df

    def _load_data(self, split: str, sample_n: int | None, seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)

        json_records = self._load_json_records()
        if json_records is not None:
            df = records_to_dataframe(json_records, sample_n, seed)
            for col in ["A", "B", "C", "D"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            df = df.reset_index(drop=True)
            df = df.reindex(rng.permutation(df.index)).reset_index(drop=True)
            print(
                f"[PMC-VQA] Loaded {len(df)} samples from JSON ({split} split)"
                + (f" (sampled {sample_n})" if sample_n else "")
            )
            return df

        medsets_records = load_medsets_records(("PMC-VQA", "PMCVQA"), split, seed, dataset_name="pmcvqa")
        if medsets_records is not None:
            df = records_to_dataframe(medsets_records, sample_n, seed)
            df = df.reindex(rng.permutation(df.index)).reset_index(drop=True)
            print(f"[PMC-VQA] Loaded {len(df)} samples from Medsets {split} split"
                  + (f" (sampled {sample_n})" if sample_n else ""))
            return df

        df_csv = self._try_packaged_pmcvqa_csv(split, sample_n, seed, rng)
        if df_csv is not None:
            return df_csv

        df_pack_json = self._try_packaged_pmcvqa_test_json(split, sample_n, seed, rng)
        if df_pack_json is not None:
            return df_pack_json

        local = load_local_vision_csv_records(_DATA_DIR, split, "pmcvqa", seed)
        if local is not None:
            df = records_to_dataframe(local, sample_n, seed)
            for col in ["A", "B", "C", "D"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip()
            df = df.reset_index(drop=True)
            df = df.reindex(rng.permutation(df.index)).reset_index(drop=True)
            print(
                f"[PMC-VQA] Loaded {len(df)} samples from local Medsets-style CSV ({split})"
                + (f" (sampled {sample_n})" if sample_n else "")
            )
            return df

        raise FileNotFoundError(
            f"[PMC-VQA] No data for split={split!r}. "
            "Options: set TRAIN_JSON_PATH / TEST_JSON_PATH; "
            "use datasets_my/PMC-VQA-Test/PMC-VQA-Test.json "
            "with PMC_VQA_IMAGE_ROOT (or PMC_VQA_TEST_IMAGE_ROOT) pointing at image files; "
            "cache Medsets (parthpk/Medsets; test images live under PMC-VQA/images_test_clean/); "
            f"or place train_sample_2000.csv / pmcvqa_test.csv under {_DATA_DIR}."
        )

    # ------------------------------------------------------------------
    # Caption management
    # ------------------------------------------------------------------

    @property
    def _caption_cache_path(self) -> Path:
        _CAPTION_DIR.mkdir(parents=True, exist_ok=True)
        # Try model-specific cache first (e.g. from Qwen3.6-27B), fall back to default
        model_tag = self._caption_model.replace("/", "-")
        model_specific = _CAPTION_DIR / f"{self._split}_seed{self._seed}_{model_tag}.json"
        if model_specific.exists():
            return model_specific
        return _CAPTION_DIR / f"{self._split}_seed{self._seed}.json"

    def _ensure_captions(self):
        """Load cached captions or generate them if missing."""
        cache_path = self._caption_cache_path
        captions = {}

        if cache_path.exists():
            captions = json.loads(cache_path.read_text())
            print(f"[PMC-VQA] Loaded {len(captions)} cached captions from {cache_path.name}")

        # Check which images still need captions
        missing = []
        for idx, row in self._total_df.iterrows():
            key = row["image_filename"]
            if key not in captions:
                missing.append((idx, key, row["image_path"], row["question"]))

        if missing:
            print(f"[PMC-VQA] Generating captions for {len(missing)} images...")
            new_captions = self._generate_captions(missing)
            captions.update(new_captions)
            cache_path.write_text(json.dumps(captions, indent=2))
            print(f"[PMC-VQA] Saved {len(captions)} captions to {cache_path.name}")

        # Attach captions to dataframe
        self._total_df["caption"] = self._total_df["image_filename"].map(captions).fillna("")

    def _generate_captions(self, missing: list) -> dict:
        """Generate captions using Qwen3.5-9B via the running vLLM API server.

        Uses the OpenAI-compatible API with base64-encoded images.
        This avoids loading a second model instance and works with the
        already-running vLLM server.
        """
        import asyncio
        import aiohttp

        caption_base_url = os.getenv("BASE_URL", "http://localhost:8000")
        caption_api_key = os.getenv("API_KEY", "EMPTY")

        # Build requests
        requests_data = []
        valid_keys = []
        for idx, key, image_path, question in missing:
            if not os.path.exists(image_path):
                print(f"  WARNING: image not found: {image_path}")
                continue

            prompt_text = (
                f"You are a medical imaging expert. Describe this medical image in detail. "
                f"Include: imaging modality, anatomical region, key findings, abnormalities, "
                f"and any relevant clinical observations.\n\n"
                f"Context question (use this to focus your description): {question}\n\n"
                f"Provide a thorough, detailed description of the image:"
            )

            # Use file:// URL so vLLM reads the image directly (avoids base64 context bloat)
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
                "model": self._caption_model,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 512,
                "top_p": 0.9,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            valid_keys.append(key)

        async def _call_one(session, payload, sem):
            async with sem:
                url = f"{caption_base_url}/v1/chat/completions"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {caption_api_key}",
                }
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        err = await resp.text()
                        print(f"  Caption API error: {err[:200]}")
                        return ""
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]

        async def _generate_all():
            sem = asyncio.Semaphore(16)  # concurrent requests
            timeout = aiohttp.ClientTimeout(total=300)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [_call_one(session, req, sem) for req in requests_data]
                return await asyncio.gather(*tasks)

        print(f"[PMC-VQA] Generating captions for {len(requests_data)} images "
              f"via API ({caption_base_url})...")
        t0 = time.time()
        results = asyncio.run(_generate_all())
        elapsed = time.time() - t0
        print(f"[PMC-VQA] Caption generation done in {elapsed:.1f}s "
              f"({elapsed / max(len(requests_data), 1):.2f}s per image)")

        captions = {}
        for key, text in zip(valid_keys, results):
            text = text.strip()
            # Remove thinking tokens if present
            if "<think>" in text:
                think_end = text.find("</think>")
                if think_end >= 0:
                    text = text[think_end + len("</think>"):].strip()
            captions[key] = text

        return captions

    # ------------------------------------------------------------------
    # Image embedding management
    # ------------------------------------------------------------------

    def _ensure_image_embeddings(self):
        """Pre-compute image embeddings using a frozen vision encoder."""
        from GDesigner.llm.image_embedding import precompute_embeddings, get_embedding_dim

        require_existing_images(self._total_df, "PMC-VQA")
        # Collect unique image paths
        image_paths = []
        for _, row in self._total_df.iterrows():
            img_path = row.get("image_path", "")
            if img_path and os.path.exists(str(img_path)):
                image_paths.append(str(img_path))

        if not image_paths:
            print(f"[PMC-VQA] No images found for embedding pre-computation")
            return

        # Check for cached embeddings on disk
        _IMG_EMB_DIR = _DATA_DIR / "image_embeddings"
        _IMG_EMB_DIR.mkdir(parents=True, exist_ok=True)
        model_tag = (self._vision_encoder or "siglip-so400m").replace("/", "-")
        cache_path = _IMG_EMB_DIR / f"{self._split}_seed{self._seed}_{model_tag}.pt"

        import torch
        cached_embs = {}
        if cache_path.exists():
            cached_embs = torch.load(cache_path, map_location="cpu")
            print(f"[PMC-VQA] Loaded {len(cached_embs)} cached image embeddings from {cache_path.name}")

        # Find missing
        missing_paths = [p for p in image_paths if p not in cached_embs]

        if missing_paths:
            new_embs = precompute_embeddings(missing_paths, model_name=self._vision_encoder)
            for p, emb in new_embs.items():
                cached_embs[p] = torch.tensor(emb)
            torch.save(cached_embs, cache_path)
            print(f"[PMC-VQA] Saved {len(cached_embs)} image embeddings to {cache_path.name}")
        else:
            print(f"[PMC-VQA] All {len(cached_embs)} image embeddings cached")

        # Store embeddings in the dataframe as a column of tensors
        self._image_embeddings = cached_embs
        self._image_embedding_dim = get_embedding_dim(self._vision_encoder)

    # ------------------------------------------------------------------
    # Standard dataset interface
    # ------------------------------------------------------------------

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        record = self._total_df.iloc[index]
        assert isinstance(record, (pd.DataFrame, pd.Series))
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        """Build input dict with caption for routing/pool/prompts and plain for agents.

        - 'task': caption + question + options (router embedding, pool gen, prompt gen)
        - 'task_plain': question + options only (specialist VLM agents see the actual image)
        - 'image_path': path to actual image (for VLM agents)
        """
        caption = record.get("caption", "")

        plain_question = (
            f"Question: {record['question']}\n"
            f"{format_options(record)}"
        )

        if caption:
            task_with_caption = f"Image Description:\n{caption}\n\n{plain_question}"
        else:
            task_with_caption = plain_question

        input_dict = {
            "task": task_with_caption,
            "task_plain": plain_question,
        }
        image_path = record.get("image_path", None)
        if image_path and os.path.exists(str(image_path)):
            input_dict["image_path"] = str(image_path)
        return input_dict

    def record_to_input_with_embeddings(self, record: pd.DataFrame, fusion_mode: str = "concat") -> Dict[str, Any]:
        """Like record_to_input but injects pre-computed image embeddings.

        When use_image_embeddings is enabled:
        - 'image_embedding' contains the pre-computed vision encoder embedding
        - For 'concat'/'cross_attention': 'task' uses plain question (no caption)
        - For 'caption_image': 'task' uses plain question, 'task_caption' has caption+question
        """
        input_dict = self.record_to_input(record)

        if self._use_image_embeddings and hasattr(self, '_image_embeddings'):
            image_path = record.get("image_path", "")
            img_emb = self._image_embeddings.get(str(image_path))
            if img_emb is not None:
                input_dict["image_embedding"] = img_emb
                if fusion_mode == "caption_image":
                    # Keep caption+question as task_caption, use plain question as task
                    input_dict["task_caption"] = input_dict["task"]
                    input_dict["task"] = input_dict["task_plain"]
                else:
                    # concat/cross_attention: plain question only
                    input_dict["task"] = input_dict["task_plain"]
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        import re
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if not answer:
            return answer
        return extract_choice(answer)

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record["correct_answer"]
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)}, record={record}")
        return correct_answer.strip()

    @staticmethod
    def record_to_target_check(ground_truth_answer, predicted_answer, question) -> int:
        """Regex-based letter matching for MCQ (A-D)."""
        predicted_answer = predicted_answer.replace("assistant", "Option").strip()
        gt = ground_truth_answer.strip().upper()

        return 1 if extract_choice(predicted_answer) == gt else 0
