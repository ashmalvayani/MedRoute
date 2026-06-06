import glob
import json
import re
import os
from abc import ABC
from typing import Any, Dict, List, Literal, Union

import numpy as np
import pandas as pd


class MedMCQADataset(ABC):
    """4-choice MedMCQA (A–D).

    Loads ``datasets_my/MedMCQA/medmcqa.json`` (list of dicts with question, options, answer_idx)
    and splits into disjoint ``dev`` / ``test`` (and ``val`` same as ``test``) using a fixed shuffle.

    Alternatively, place CSVs under ``datasets_my/MedMCQA/data/<split>/*.csv`` with columns:
    question, A, B, C, D, correct_answer — then those override the JSON path.
    """

    def __init__(
        self,
        split: Union[Literal["dev"], Literal["val"], Literal["test"]],
    ) -> None:
        self._split = split
        base = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base, "MedMCQA", "data", self._split) + "/"
        csv_paths = sorted(glob.glob(data_dir + "*.csv"))
        if csv_paths:
            self._total_df = self._load_from_csv(data_dir)
        else:
            json_path = os.path.join(base, "MedMCQA", "medmcqa.json")
            if not os.path.isfile(json_path):
                raise FileNotFoundError(
                    f"MedMCQA data not found: add {json_path} or CSVs under {data_dir}"
                )
            self._total_df = self._load_from_json(json_path, split)

    @staticmethod
    def get_domain() -> str:
        return "medmcqa"

    @staticmethod
    def _load_from_csv(data_path: str) -> pd.DataFrame:
        rng = np.random.default_rng(888)
        csv_paths = sorted(glob.glob(data_path + "*.csv"))
        print("Number of topics: ", len(csv_paths))
        names = ["question", "A", "B", "C", "D", "correct_answer"]
        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None, names=names, encoding="utf-8")
            total_df = pd.concat([total_df, single_df])
        total_df = total_df.reset_index(drop=True)
        total_df = total_df.reindex(rng.permutation(total_df.index))
        print("Total number of questions: ", len(total_df))
        return total_df

    @staticmethod
    def _load_from_json(json_path: str, split: str) -> pd.DataFrame:
        rng = np.random.default_rng(888)
        with open(json_path, encoding="utf-8") as f:
            raw: List[dict] = json.load(f)
        rows = []
        for item in raw:
            o = item.get("options") or {}
            letter = str(item.get("answer_idx", "")).strip().upper()[:1]
            if letter not in "ABCD":
                continue
            rows.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "A": str(o.get("A", "")).strip(),
                    "B": str(o.get("B", "")).strip(),
                    "C": str(o.get("C", "")).strip(),
                    "D": str(o.get("D", "")).strip(),
                    "correct_answer": letter,
                }
            )
        n = len(rows)
        idx = np.arange(n)
        rng.shuffle(idx)
        dev_frac = 0.15
        n_dev = max(1, int(dev_frac * n))
        if split == "dev":
            sel = idx[:n_dev]
        else:
            # test and val: same held-out portion (MedQA-style naming)
            sel = idx[n_dev:]
        sub = [rows[i] for i in sel]
        total_df = pd.DataFrame(sub)
        print(f"MedMCQA from JSON: split={split} | total_in_file={n} | this_split={len(total_df)}")
        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
        )
        return {"task": demo_question}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos + len("answer is") :].strip(":").strip().strip("Option").strip()
            answer = answer[0]
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record["correct_answer"]
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} of type {type(correct_answer)} record={record}"
        )
        return correct_answer

    @staticmethod
    def record_to_target_check(ground_truth_answer, predicted_answer, question) -> int:
        predicted_answer = predicted_answer.replace("assistant", "Option").strip()
        gt = ground_truth_answer.strip().upper()
        patterns = [
            r"answer\s+is\s+(?:Option\s+)?(?:\**)([A-D])(?:\**)",
            r"(?:correct|best)\s+(?:answer|option)\s*(?:is|:)\s*(?:\**)([A-D])(?:\**)",
            r"\bOption\s+([A-D])\b",
            r"\(([A-D])\)",
            r"(?:^|\n)\s*\**([A-D])\**\s*[\.\)\:]",
            r"(?:^|[\s,;])\**([A-D])\**[\.\)\:]",
        ]
        for pattern in patterns:
            match = re.search(pattern, predicted_answer, re.IGNORECASE)
            if match:
                return 1 if match.group(1).upper() == gt else 0
        if predicted_answer and predicted_answer[0].upper() in "ABCD":
            if len(predicted_answer) == 1 or not predicted_answer[1].isalpha():
                return 1 if predicted_answer[0].upper() == gt else 0
        return 0
