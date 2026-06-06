"""
Partial Credit Reward Shaping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of binary 0/1 rewards, we ask the judge to score reasoning quality
on a 1-5 scale. This provides gradient signal even for wrong final answers.

Reward formula:
    reward = alpha * binary_correct + (1 - alpha) * (reasoning_score / 5.0)
    reward *= decay_factor ** routing_length

This means:
- A wrong answer with excellent reasoning (score=5) gets: 0 + 0.6*(5/5) = 0.6
- A right answer with poor reasoning (score=1) gets: 0.4 + 0.6*(1/5) = 0.52
- A right answer with great reasoning (score=5) gets: 0.4 + 0.6*(5/5) = 1.0
- A wrong answer with no reasoning (score=1) gets: 0 + 0.6*(1/5) = 0.12
"""

import os
import aiohttp
from typing import Optional, Dict, Any

JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', os.getenv('BASE_URL', 'http://localhost:8001'))
JUDGE_API_KEY = os.getenv('JUDGE_API_KEY', os.getenv('API_KEY', 'EMPTY'))

REASONING_JUDGE_SYSTEM = (
    "You are a medical reasoning evaluator. You will be given a medical question, "
    "the correct answer, and a model's response. Score the REASONING QUALITY of the "
    "model's response on a scale of 1-5:\n\n"
    "1 = Completely wrong reasoning, no relevant clinical knowledge\n"
    "2 = Some relevant concepts but major errors in logic or knowledge\n"
    "3 = Partially correct reasoning, identifies some key features but misses important ones\n"
    "4 = Mostly correct reasoning with minor gaps, identifies key clinical features\n"
    "5 = Excellent reasoning, correct identification of pathophysiology and clinical features\n\n"
    "Reply with ONLY a single digit (1, 2, 3, 4, or 5). Nothing else."
)

# Default blending weight: 0.4 * binary + 0.6 * reasoning
DEFAULT_ALPHA = 0.4


async def judge_reasoning_quality(
    session: aiohttp.ClientSession,
    judge_model: str,
    question: str,
    true_answer: str,
    model_response: str,
) -> int:
    """
    Ask judge to score reasoning quality 1-5.
    Returns integer score, defaults to 3 on error.
    """
    prompt = (
        f"Question:\n{question}\n\n"
        f"Correct Answer: {true_answer}\n\n"
        f"Model's Response:\n{model_response}\n\n"
        f"Score the reasoning quality (1-5). Reply with ONLY a single digit."
    )
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": REASONING_JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 4,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{JUDGE_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                return 3  # neutral default
            r = await resp.json()
            text = r['choices'][0]['message']['content'].strip()
            # Extract first digit
            for ch in text:
                if ch.isdigit() and 1 <= int(ch) <= 5:
                    return int(ch)
            return 3
    except Exception:
        return 3


def compute_partial_reward(
    is_correct: bool,
    reasoning_score: int,
    routing_length: int,
    decay_factor: float = 0.98,
    alpha: float = DEFAULT_ALPHA,
) -> float:
    """
    Compute blended reward from binary correctness and reasoning quality.

    Args:
        is_correct: Whether the final answer is correct
        reasoning_score: Judge's reasoning quality score (1-5)
        routing_length: Number of routing hops taken
        decay_factor: Reward decay per routing hop
        alpha: Weight for binary correctness (1-alpha for reasoning)

    Returns:
        Float reward value
    """
    binary = float(is_correct)
    reasoning = reasoning_score / 5.0
    raw_reward = alpha * binary + (1.0 - alpha) * reasoning
    return (decay_factor ** routing_length) * raw_reward


def reward_breakdown(
    is_correct: bool,
    reasoning_score: int,
    routing_length: int,
    decay_factor: float = 0.98,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Return detailed reward breakdown for logging."""
    binary = float(is_correct)
    reasoning = reasoning_score / 5.0
    raw_reward = alpha * binary + (1.0 - alpha) * reasoning
    decayed = (decay_factor ** routing_length) * raw_reward
    return {
        "is_correct": is_correct,
        "reasoning_score": reasoning_score,
        "binary_component": alpha * binary,
        "reasoning_component": (1.0 - alpha) * reasoning,
        "raw_reward": raw_reward,
        "decay_factor": decay_factor,
        "routing_length": routing_length,
        "final_reward": decayed,
    }
