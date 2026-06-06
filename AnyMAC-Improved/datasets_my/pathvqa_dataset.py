import json
import os
import re
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from datasets_my.medsets_csv import (
    _read_pathvqa_csv,
    _resolve_image_path,
    format_options,
    load_medsets_pathvqa_records,
    option_letters,
    records_to_dataframe,
    require_existing_images,
)

_DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "PathVQA" / "data"
_DEFAULT_JSON = _DATA_DIR / "pathvqa.json"


def _norm_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


class PathVQADataset(ABC):
    def __init__(
        self,
        split: str,
        sample_n: int | None = None,
        seed: int = 99,
        train_json_path: str | None = None,
        test_json_path: str | None = None,
        caption_model: str = "Qwen/Qwen3.5-9B",
        skip_caption_gen: bool = True,
        use_image_embeddings: bool = False,
        vision_encoder: str | None = None,
    ) -> None:
        self._split = split
        self._sample_n = sample_n
        self._seed = seed
        self._use_image_embeddings = use_image_embeddings
        self._vision_encoder = vision_encoder
        self._train_json_path = train_json_path
        self._test_json_path = test_json_path

        self._total_df = self._load_data(split, sample_n, seed)
        if use_image_embeddings:
            self._ensure_image_embeddings()

    @staticmethod
    def get_domain() -> str:
        return "pathvqa"

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        return self._total_df.iloc[index]

    def _csv_path_to_records(self, source: Path) -> List[Dict[str, Any]]:
        dataset_dir = source.parent
        df = _read_pathvqa_csv(source)
        records: List[Dict[str, Any]] = []
        for i, row in df.iterrows():
            records.append(
                {
                    "figure_path": _resolve_image_path(row["figure_path"], dataset_dir),
                    "question": str(row["question"]).strip(),
                    "answer": str(row["answer_text"]).strip(),
                    "caption": str(row["caption"]).strip() if pd.notna(row["caption"]) else "",
                    "options": {},
                    "question_id": i,
                }
            )
        return records

    def _load_records(self) -> List[Dict[str, Any]]:
        train_path = Path(self._train_json_path) if self._train_json_path else None
        test_path = Path(self._test_json_path) if self._test_json_path else None

        if train_path and test_path:
            source = train_path if self._split == "train" else test_path
            if source.suffix.lower() == ".csv":
                return self._csv_path_to_records(source)
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            return data

        if not train_path and not test_path:
            medsets = load_medsets_pathvqa_records(self._split, self._seed)
            if medsets is not None:
                return medsets

        source = test_path or train_path or _DEFAULT_JSON
        if source.suffix.lower() == ".csv":
            return self._csv_path_to_records(source)
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]

        rng = np.random.default_rng(self._seed)
        indices = rng.permutation(len(data))
        cut = int(0.8 * len(indices))
        picked = indices[:cut] if self._split == "train" else indices[cut:]
        return [data[i] for i in picked]

    def _load_data(self, split: str, sample_n: int | None, seed: int) -> pd.DataFrame:
        if split not in {"train", "test"}:
            raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

        df = records_to_dataframe(self._load_records(), sample_n, seed, require_mcq_options=False)
        print(f"[PathVQA] Loaded {len(df)} samples from {split} split")
        return df

    def _ensure_image_embeddings(self):
        from GDesigner.llm.image_embedding import get_embedding_dim, precompute_embeddings
        import torch

        require_existing_images(self._total_df, "PathVQA")
        image_paths = [str(p) for p in self._total_df["image_path"].tolist() if os.path.exists(str(p))]
        if not image_paths:
            return

        emb_dir = _DATA_DIR / "image_embeddings"
        emb_dir.mkdir(parents=True, exist_ok=True)
        model_tag = (self._vision_encoder or "siglip-so400m").replace("/", "-")
        cache_path = emb_dir / f"{self._split}_seed{self._seed}_{model_tag}.pt"

        cached_embs = {}
        if cache_path.exists():
            cached_embs = torch.load(cache_path, map_location="cpu")

        missing_paths = [p for p in image_paths if p not in cached_embs]
        if missing_paths:
            new_embs = precompute_embeddings(missing_paths, model_name=self._vision_encoder)
            for p, emb in new_embs.items():
                cached_embs[p] = torch.tensor(emb)
            torch.save(cached_embs, cache_path)

        self._image_embeddings = cached_embs
        self._image_embedding_dim = get_embedding_dim(self._vision_encoder)

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        caption = record.get("caption", "")
        letters = option_letters(record)
        opt_lines = format_options(record).rstrip() if letters else ""
        plain_parts = [f"Question: {record['question']}"]
        if opt_lines:
            plain_parts.append(opt_lines)
        task_plain = "\n".join(plain_parts) + "\n"
        if caption:
            task_with_caption = f"Image Description:\n{caption}\n\n{task_plain}"
        else:
            task_with_caption = task_plain
        input_dict = {"task": task_with_caption, "task_plain": task_plain}
        image_path = record.get("image_path", None)
        if image_path and os.path.exists(str(image_path)):
            input_dict["image_path"] = str(image_path)
        return input_dict

    def record_to_input_with_embeddings(self, record: pd.Series, fusion_mode: str = "concat") -> Dict[str, Any]:
        input_dict = self.record_to_input(record)
        if self._use_image_embeddings and hasattr(self, "_image_embeddings"):
            image_path = record.get("image_path", "")
            img_emb = self._image_embeddings.get(str(image_path))
            if img_emb is not None:
                input_dict["image_embedding"] = img_emb
                if fusion_mode == "caption_image":
                    input_dict["task_caption"] = input_dict["task"]
                    input_dict["task"] = input_dict["task_plain"]
                else:
                    input_dict["task"] = input_dict["task_plain"]
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        return answer.strip()

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        return str(record["correct_answer"]).strip()

    @staticmethod
    def record_to_target_check(ground_truth_answer, predicted_answer, question) -> int:
        predicted_answer = str(predicted_answer).replace("assistant", "").strip()
        gt_n = _norm_answer(str(ground_truth_answer))
        pred_n = _norm_answer(predicted_answer)
        if not gt_n:
            return 0
        if gt_n == pred_n:
            return 1
        if gt_n in pred_n or pred_n in gt_n:
            return 1
        return 0
