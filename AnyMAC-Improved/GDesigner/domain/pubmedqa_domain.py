"""
MedQA domain (text-only MCQ) for AnyMAC/GDesigner.

This is a small domain wrapper intended to be analogous to the existing MMLU/GSM8K domains.

Expected responsibilities:
- Provide task formatting and answer parsing for evaluation.
- Optionally provide "hints" or "question-type" categories (left minimal for now).

If your Graph expects additional methods, mirror the corresponding methods
from GDesigner/domain/mmlu_domain.py and adapt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re

_MCQL_RE = re.compile(r"\b([ABC])\b", re.IGNORECASE)
_MCQN_RE = re.compile(r"\b([1-3])\b")

def extract_mcq_letter(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MCQL_RE.search(text)
    if m:
        return m.group(1).upper()
    n = _MCQN_RE.search(text)
    if n:
        return {"1":"A","2":"B","3":"C"}.get(n.group(1))
    return None


@dataclass
class PubMedQADomain:
    name: str = "pubmedqa"

    def build_task(self, input_dict: Dict[str, Any]) -> str:
        # Graphs typically use input_dict["task"] as the full prompt payload.
        return str(input_dict.get("task", "")).strip()

    def postprocess_answer(self, text: str) -> str:
        return extract_mcq_letter(text) or ""

    # Some domains expose a "target" extraction helper
    def get_final_answer(self, model_output: str) -> str:
        return self.postprocess_answer(model_output)
