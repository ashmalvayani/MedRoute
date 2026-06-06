"""
PMC-VQA domain (vision MCQ via caption-as-text) for AnyMAC/GDesigner.

Same structure as medqa_domain / pubmedqa_domain but for 4-option MCQ (A-D).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re

_MCQL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
_MCQN_RE = re.compile(r"\b([1-4])\b")


def extract_mcq_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MCQL_RE.search(text)
    if m:
        return m.group(1).upper()
    n = _MCQN_RE.search(text)
    if n:
        return {"1": "A", "2": "B", "3": "C", "4": "D"}.get(n.group(1))
    return None


@dataclass
class PMCVQADomain:
    name: str = "pmcvqa"

    def build_task(self, input_dict: Dict[str, Any]) -> str:
        return str(input_dict.get("task", "")).strip()

    def postprocess_answer(self, text: str) -> str:
        return extract_mcq_letter(text) or ""

    def get_final_answer(self, model_output: str) -> str:
        return self.postprocess_answer(model_output)
