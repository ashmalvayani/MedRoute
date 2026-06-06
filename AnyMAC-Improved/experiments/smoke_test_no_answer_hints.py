"""
Smoke test: re-run 20 wrong questions from v2 with answer letters stripped from hints.
Tests whether the echo-chamber effect (later specialists copying first specialist's answer)
is the main failure mode.

This script monkey-patches the hint construction at runtime — no existing code is modified.

Usage:
    cd AnyMAC-Improved
    CUDA_VISIBLE_DEVICES="" python experiments/smoke_test_no_answer_hints.py
"""

from __future__ import annotations
import sys, os, re, json, logging

logging.disable(logging.WARNING)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np
import random
from collections import Counter

from GDesigner.graph.graph import Graph
from datasets_my.medqa_dataset import MedQADataset
from GDesigner.prompt.medqa_prompt_set import SPECIALISTS
import GDesigner.agents.analyze_agent as aa


def strip_answer_from_output(output: str) -> str:
    """Remove the answer letter line from a specialist's output so the next specialist
    can see reasoning but not the chosen answer."""
    lines = output.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just a letter (A-E) optionally with punctuation
        if re.match(r'^[A-E][.\s:)]?\s*$', stripped):
            continue
        # Skip lines starting with "Answer: X" or "The answer is X"
        if re.match(r'^(the\s+)?answer\s*(is\s*)?:?\s*[A-E]', stripped, re.IGNORECASE):
            continue
        # Strip leading answer letter from lines like "B\n\nThe patient..."
        if stripped and stripped[0] in 'ABCDE' and len(stripped) > 1 and not stripped[1].isalpha():
            # Remove just the first character + any following whitespace/punctuation
            cleaned = re.sub(r'^[A-E][.\s:)]*', '', stripped).strip()
            if cleaned:
                filtered.append(cleaned)
            continue
        filtered.append(line)
    return '\n'.join(filtered)


def main():
    # Config
    MODEL_PATH = "result/dynamic_prompts_v2/2026-04-20-05-34-27/2026-04-20-05-34-27_Qwen-Qwen3-8B_medqa_model_epoch1.pth"
    V2_RESULTS = "result/dynamic_prompts_v2/eval_temp_0.7/medqa_Qwen-Qwen3-8B_2026-04-20-06-13-52.json"
    NUM_TEST = 20
    EVAL_TEMP = 0.7
    JUDGE_MODEL = "Qwen/Qwen3-32B"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Enable dynamic prompts
    aa.DYNAMIC_PROMPTS_ENABLED = True
    aa.JUDGE_MODEL_FOR_PROMPTS = JUDGE_MODEL

    # Get wrong question indices from v2
    with open(V2_RESULTS) as f:
        v2_data = json.load(f)
    wrong_indices = [r['Index'] for r in v2_data if 'Index' in r and not r.get('Regex_solved')]
    test_indices = wrong_indices[:NUM_TEST]
    print(f"Testing {len(test_indices)} wrong questions from v2: {test_indices}")

    # Load model
    print(f"Loading: {MODEL_PATH}")
    graph = Graph.load_model(MODEL_PATH)
    graph.to_device(torch.device("cpu"))
    graph.set_eval()
    graph.cos_scaling = 1
    graph.eval_temperature = EVAL_TEMP
    graph.eval_deterministic = True

    dataset = MedQADataset('test')

    # Monkey-patch: intercept hint construction in graph.run_next_agent_prediction
    # We'll patch at the history_states level — strip answer letters from outputs before hints are built
    original_run = graph.run_next_agent_prediction

    def patched_run(input_dict, **kwargs):
        """Wrapper that strips answer letters from history_states outputs after each agent step."""
        # We need to intercept deeper. Instead, let's patch the agent output post-processing.
        # Simpler approach: run normally but strip answers from the hint text construction
        result = original_run(input_dict, **kwargs)
        return result

    # Actually, the cleanest way: monkey-patch the hint construction section
    # We'll wrap the graph's internal method by patching history_states
    import types

    # Store original _build_hints logic by patching at the output level
    # The key insight: hints are built from history_states[i]["output"]
    # We patch by wrapping run_next_agent_prediction and modifying history_states between agent calls

    # Better approach: just run each question twice and compare
    # Run 1: normal (already have results from v2)
    # Run 2: patch analyze_agent to not include other agents' outputs in user_prompt

    # Simplest monkey-patch: override _process_inputs to strip answers from spatial/temporal info
    from GDesigner.agents.analyze_agent import AnalyzeAgent
    original_process = AnalyzeAgent._process_inputs

    async def patched_process_inputs(self, raw_inputs, spatial_info, temporal_info, **kwargs):
        # Strip answer letters from all spatial/temporal outputs before passing to the agent
        patched_spatial = {}
        for id, info in spatial_info.items():
            patched_info = info.copy()
            patched_info['output'] = strip_answer_from_output(info.get('output', ''))
            patched_spatial[id] = patched_info

        patched_temporal = {}
        for id, info in temporal_info.items():
            patched_info = info.copy()
            patched_info['output'] = strip_answer_from_output(info.get('output', ''))
            patched_temporal[id] = patched_info

        return await original_process(self, raw_inputs, patched_spatial, patched_temporal, **kwargs)

    AnalyzeAgent._process_inputs = patched_process_inputs

    # Run the 20 questions
    def extract_answer(response: str) -> str:
        text = response.replace('assistant', 'Option').strip()
        patterns = [
            r'answer\s+is\s+(?:Option\s+)?(?:\**)([A-E])(?:\**)',
            r'(?:correct|best)\s+(?:answer|option)\s*(?:is|:)\s*(?:\**)([A-E])(?:\**)',
            r'\bOption\s+([A-E])\b',
            r'\(([A-E])\)',
            r'(?:^|\n)\s*\**([A-E])\**\s*[\.)\:]',
            r'(?:^|[\s,;])\**([A-E])\**[\.)\:]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        if text and text[0].upper() in 'ABCDE':
            if len(text) == 1 or not text[1].isalpha():
                return text[0].upper()
        return ""

    results = []
    correct_count = 0

    with torch.no_grad():
        for i, qi in enumerate(test_indices):
            record = dataset[qi]
            gt = dataset.record_to_target_answer(record).strip().upper()
            input_dict = dataset.record_to_input(record)

            # Suppress specialist output spam
            devnull = open(os.devnull, 'w')
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                result = graph.run_next_agent_prediction(
                    input_dict,
                    max_routing=3,
                    temperature=0.7,
                    available_roles=SPECIALISTS,
                    agent_group_type="AnalyzeAgent",
                    max_context=2048,
                )
            finally:
                sys.stdout = old_stdout

            if result is None:
                pred = ""
                response = ""
            else:
                answers = result.get("answers", [""])
                response = answers[0] if answers else ""
                pred = extract_answer(response)

            is_correct = pred == gt
            if is_correct:
                correct_count += 1

            # Get v2 original prediction
            v2_pred = ""
            for r in v2_data:
                if r.get('Index') == qi:
                    v2_pred = r.get('Regex_answer', '')
                    break

            routing = result.get("routing_results", {}).get("agent_selections", []) if result else []
            route_names = [SPECIALISTS[idx] if idx < len(SPECIALISTS) else "DecisionMaker" for idx in routing]

            status = "RECOVERED" if is_correct else f"STILL WRONG (pred={pred})"
            print(f"[{i+1}/{len(test_indices)}] Q{qi}: GT={gt}, v2={v2_pred}, new={pred} → {status}  Route: {' -> '.join(route_names)}")

            results.append({
                "qi": qi, "gt": gt, "v2_pred": v2_pred, "new_pred": pred,
                "recovered": is_correct, "route": route_names,
                "response_snippet": response[:200],
            })

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct_count}/{len(test_indices)} recovered ({correct_count/len(test_indices):.1%})")
    print(f"v2 accuracy on these: 0/{len(test_indices)} (all were wrong)")
    print(f"No-answer-hints accuracy: {correct_count}/{len(test_indices)}")
    print(f"{'='*60}")

    # Save results
    out_path = "result/smoke_test_no_answer_hints.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
