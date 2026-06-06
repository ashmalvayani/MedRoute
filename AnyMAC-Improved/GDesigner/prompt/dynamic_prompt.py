"""
Dynamic Specialist Prompt Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before each specialist answers a question, we call the judge model to generate
a question-specific guidance prompt. This replaces the generic role description
with a tailored briefing that tells the specialist exactly what to focus on.

Caching: same (question_hash, specialist) pair → same prompt across rollouts.
"""

import os
import hashlib
import aiohttp
from typing import Dict, Optional

JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', os.getenv('BASE_URL', 'http://localhost:8001'))
JUDGE_API_KEY = os.getenv('JUDGE_API_KEY', os.getenv('API_KEY', 'EMPTY'))

# Override for prompt generation endpoint (e.g., use BASE_URL for 8B instead of JUDGE_BASE_URL for 32B)
PROMPT_BASE_URL: Optional[str] = None  # None = use JUDGE_BASE_URL

# In-memory cache: (question_hash, specialist_role) → dynamic prompt
_prompt_cache: Dict[str, str] = {}

DYNAMIC_PROMPT_SYSTEM = (
    "You are a medical education expert. Given a medical question and a specialist role, "
    "generate a focused briefing for that specialist. Your briefing MUST:\n"
    "1. Identify the KEY clinical clues in this question (specific symptoms, lab values, imaging findings)\n"
    "2. List 2-3 differential diagnoses the specialist should consider and why\n"
    "3. Tell them which specific finding or mechanism distinguishes the correct answer\n"
    "4. Connect the question to the specialist's knowledge — every medical specialist has "
    "relevant training in general medicine, pathophysiology, and clinical reasoning. "
    "Frame the question through their lens.\n\n"
    "IMPORTANT: NEVER say the question is outside the specialist's expertise. "
    "Every specialist can contribute useful reasoning. Always be confident and directive.\n\n"
    "Be SPECIFIC to this question — reference actual values, symptoms, and findings from it. "
    "Do NOT give generic advice. Do NOT answer the question. Keep it under 150 words."
)


def _cache_key(question: str, role: str) -> str:
    """Create a cache key from question + role."""
    q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    return f"{q_hash}_{role}"


async def generate_dynamic_prompt(
    judge_model: str,
    question: str,
    specialist_role: str,
    base_description: str,
    session: Optional[aiohttp.ClientSession] = None,
    image_path: Optional[str] = None,
) -> str:
    """
    Generate a question-specific specialist prompt using the judge model.

    Returns the dynamic prompt string. Falls back to base_description on error.
    """
    key = _cache_key(question, specialist_role)

    # Check cache first
    if key in _prompt_cache:
        return _prompt_cache[key]

    user_prompt = (
        f"Medical Question:\n{question}\n\n"
        f"Specialist Role: {specialist_role}\n\n"
        f"Base Role Description: {base_description}\n\n"
        f"Generate a focused guidance prompt for this {specialist_role} that is specific "
        f"to the medical question above. Tell them what clinical features, mechanisms, "
        f"or diagnostic clues they should focus on. Be very specific to this question."
    )

    # Build user content — include image if available (VLM prompt gen)
    if image_path:
        from pathlib import Path
        abs_path = str(Path(image_path).resolve())
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
        ]
    else:
        user_content = user_prompt

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": DYNAMIC_PROMPT_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 300,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    base_url = PROMPT_BASE_URL or JUDGE_BASE_URL
    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        close_session = True

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                return base_description
            r = await resp.json()
            dynamic_prompt = r['choices'][0]['message']['content'].strip()

            # Combine: base description + dynamic guidance
            combined = (
                f"{base_description}\n\n"
                f"=== Question-Specific Guidance ===\n"
                f"{dynamic_prompt}"
            )

            # Cache it
            _prompt_cache[key] = combined
            return combined
    except Exception:
        return base_description
    finally:
        if close_session:
            await session.close()


def get_cached_prompt(question: str, role: str) -> Optional[str]:
    """Return cached dynamic prompt if available."""
    key = _cache_key(question, role)
    return _prompt_cache.get(key)


# ---------------------------------------------------------------------------
# Dynamic Specialist Pool Generation
# ---------------------------------------------------------------------------

PANEL_SYSTEM = (
    "You are a medical triage expert. Given a medical question, generate a panel of "
    "5-7 specialists best suited to collaboratively answer it.\n\n"
    "For each specialist, provide:\n"
    "1. A role title from common specialties (e.g., 'Radiologist', 'Neurologist', 'Pathologist', "
    "'Cardiologist', 'Surgeon'). Use broad, standard titles — avoid overly specific sub-specialties.\n"
    "2. A 2-3 sentence description of what this specialist should focus on for THIS question\n\n"
    "Rules:\n"
    "- Choose specialists whose expertise DIRECTLY relates to the question's domain\n"
    "- Each specialist should bring a DIFFERENT perspective\n"
    "- Use standard specialty names, not niche sub-specialties\n"
    "- Include at least one generalist who can synthesize (e.g., 'Internal Medicine Physician')\n"
    "- Generate EXACTLY 5 to 7 specialists\n\n"
    "Output EXACTLY in this JSON format (no other text):\n"
    '[{"role": "Specialist Title", "description": "What they should focus on for this question..."}]'
)

# Separate cache for panels
_panel_cache: Dict[str, list] = {}


async def generate_specialist_panel(
    judge_model: str,
    question: str,
    session: Optional[aiohttp.ClientSession] = None,
    image_path: Optional[str] = None,
) -> list:
    """
    Generate a question-specific specialist panel using the judge model.

    Returns a list of dicts: [{"role": str, "description": str}, ...]
    Falls back to None on error (caller should use fixed pool).
    """
    import json as _json

    q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    if q_hash in _panel_cache:
        return _panel_cache[q_hash]

    # Build user content — include image if available (VLM pool gen)
    user_text = f"Medical Question:\n{question}"
    if image_path:
        from pathlib import Path
        abs_path = str(Path(image_path).resolve())
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"file://{abs_path}"}},
        ]
    else:
        user_content = user_text

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": PANEL_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0,
        "max_tokens": 500,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    base_url = PROMPT_BASE_URL or JUDGE_BASE_URL
    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}"}

    close_session = False
    if session is None:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
        close_session = True

    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                print(f"[dynamic_pool] HTTP {resp.status} from {url}")
                return None
            r = await resp.json()
            raw = r['choices'][0]['message']['content'].strip()

            # Parse JSON — handle markdown fencing
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            panel = _json.loads(raw)
            if not isinstance(panel, list) or len(panel) < 2:
                print(f"[dynamic_pool] Bad panel (len={len(panel) if isinstance(panel, list) else 'N/A'}): {str(raw)[:200]}")
                return None

            # Validate structure
            for entry in panel:
                if 'role' not in entry or 'description' not in entry:
                    return None

            _panel_cache[q_hash] = panel
            return panel
    except Exception as e:
        print(f"[dynamic_pool] Exception: {type(e).__name__}: {e}")
        return None
    finally:
        if close_session:
            await session.close()


def get_cached_panel(question: str) -> Optional[list]:
    """Return cached panel if available."""
    q_hash = hashlib.md5(question.encode()).hexdigest()[:12]
    return _panel_cache.get(q_hash)


def clear_cache():
    """Clear all caches (e.g., between epochs)."""
    _prompt_cache.clear()
    _panel_cache.clear()


def cache_stats() -> Dict[str, int]:
    """Return cache statistics."""
    return {"cached_prompts": len(_prompt_cache), "cached_panels": len(_panel_cache)}
