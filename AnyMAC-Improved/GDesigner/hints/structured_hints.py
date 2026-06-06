"""
Structured Hint Passing
~~~~~~~~~~~~~~~~~~~~~~~~

Each specialist's output is parsed into a structured format:
    {specialist, answer_choice, confidence, reasoning_summary}

The next specialist receives a clean structured summary instead of raw text.
This enables explicit agreement/disagreement and better information flow.

Detailed logging is built in — every hint extraction is logged for debugging.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Module-level logger
logger = logging.getLogger("structured_hints")

# Global detailed log storage (flushed per question)
_hint_log: List[Dict[str, Any]] = []


@dataclass
class StructuredHint:
    specialist: str
    answer_choice: str  # A, B, C, D, E or "unclear"
    confidence: str  # "high", "medium", "low"
    reasoning_summary: str
    raw_output: str  # original full text for fallback


def extract_answer_choice(text: str) -> str:
    """Extract the MCQ answer letter from specialist output."""
    # First line is supposed to be just the letter
    first_line = text.strip().split('\n')[0].strip()
    if len(first_line) == 1 and first_line.upper() in 'ABCDE':
        return first_line.upper()

    # Fallback: look for common patterns
    patterns = [
        r'^([A-E])\b',           # starts with letter
        r'answer\s*(?:is|:)\s*([A-E])\b',  # "answer is X" or "answer: X"
        r'\b([A-E])\s*(?:is|\.)\s*(?:the|correct)',  # "X is correct"
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()

    return "unclear"


def extract_confidence(text: str) -> str:
    """Infer confidence from the specialist's language."""
    text_lower = text.lower()

    high_signals = ['clearly', 'definitely', 'certainly', 'confident', 'without doubt',
                    'strongly suggest', 'most likely', 'classic presentation', 'pathognomonic']
    low_signals = ['uncertain', 'unclear', 'difficult to determine', 'could be',
                   'not sure', 'ambiguous', 'might be', 'possibly']

    high_count = sum(1 for s in high_signals if s in text_lower)
    low_count = sum(1 for s in low_signals if s in text_lower)

    if high_count > low_count and high_count >= 1:
        return "high"
    elif low_count > high_count and low_count >= 1:
        return "low"
    return "medium"


def extract_reasoning_summary(text: str, max_words: int = 50) -> str:
    """Extract a concise reasoning summary from the full output."""
    # Skip the first line (which is typically just the answer letter)
    lines = text.strip().split('\n')
    if len(lines) > 1:
        reasoning_text = ' '.join(lines[1:]).strip()
    else:
        reasoning_text = text.strip()

    # Truncate to max_words
    words = reasoning_text.split()
    if len(words) > max_words:
        reasoning_text = ' '.join(words[:max_words]) + '...'

    return reasoning_text


def parse_specialist_output(specialist: str, raw_output: str) -> StructuredHint:
    """
    Parse a specialist's raw output into a structured hint.
    Logs the extraction for debugging.
    """
    answer = extract_answer_choice(raw_output)
    confidence = extract_confidence(raw_output)
    reasoning = extract_reasoning_summary(raw_output)

    hint = StructuredHint(
        specialist=specialist,
        answer_choice=answer,
        confidence=confidence,
        reasoning_summary=reasoning,
        raw_output=raw_output,
    )

    # Log the extraction
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "specialist": specialist,
        "answer_extracted": answer,
        "confidence": confidence,
        "reasoning_summary": reasoning,
        "raw_output_length": len(raw_output),
        "raw_first_line": raw_output.strip().split('\n')[0][:100],
    }
    _hint_log.append(log_entry)
    logger.debug(f"[hint] {specialist}: answer={answer}, confidence={confidence}")

    return hint


def format_structured_hints(hints: List[StructuredHint]) -> str:
    """
    Format structured hints into a clear text summary for the next specialist.
    This replaces the old raw text concatenation.
    """
    if not hints:
        return ""

    parts = ["=== Previous Specialist Opinions ===\n"]

    # Answer tally
    answer_counts: Dict[str, int] = {}
    for h in hints:
        if h.answer_choice != "unclear":
            answer_counts[h.answer_choice] = answer_counts.get(h.answer_choice, 0) + 1

    if answer_counts:
        tally = ", ".join(f"{k}: {v} vote(s)" for k, v in sorted(answer_counts.items(), key=lambda x: -x[1]))
        parts.append(f"Current vote tally: {tally}\n")

    # Individual opinions
    for i, h in enumerate(hints, 1):
        conf_marker = {"high": "[HIGH CONF]", "medium": "[MED CONF]", "low": "[LOW CONF]"}.get(h.confidence, "")
        parts.append(
            f"\nSpecialist {i} ({h.specialist}) {conf_marker}:\n"
            f"  Answer: {h.answer_choice}\n"
            f"  Reasoning: {h.reasoning_summary}\n"
        )

    parts.append("\n=== Use your expertise to evaluate these opinions. You may agree or disagree. ===")
    return "\n".join(parts)


def get_hint_log() -> List[Dict[str, Any]]:
    """Return the current hint extraction log."""
    return list(_hint_log)


def clear_hint_log():
    """Clear the hint log (call between questions)."""
    _hint_log.clear()


def get_hint_log_summary() -> Dict[str, Any]:
    """Return summary stats from the hint log."""
    if not _hint_log:
        return {"total_hints": 0}

    total = len(_hint_log)
    answer_found = sum(1 for h in _hint_log if h['answer_extracted'] != 'unclear')
    conf_dist = {}
    for h in _hint_log:
        c = h['confidence']
        conf_dist[c] = conf_dist.get(c, 0) + 1

    return {
        "total_hints": total,
        "answer_extraction_rate": answer_found / total if total else 0,
        "confidence_distribution": conf_dist,
    }
