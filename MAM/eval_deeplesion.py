#!/usr/bin/env python3
"""Evaluate MAM on the DeepLesion dataset (image modality, 7-choice MCQ).

DeepLesion is a lesion detection dataset without native questions. We frame
it as an 8-class MCQ: given a CT image, identify the lesion type from
{abdomen, bone, kidney, liver, lung, mediastinum, pelvis, soft tissue}.

Typical two-model usage:

    MAM_LLM_API_URL=http://<text-server>:9001/v1 \
    MAM_VLM_API_URL=http://<vlm-server>:9001/v1 \
    MAM_LLM_MAX_TOKENS=2048 \
    MAM_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
    MAM_VLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
        .venv/bin/python eval_deeplesion.py --run-name Qwen3-DeepLesion --quiet
"""

import argparse
import contextlib
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime


# --------------------------------------------------------------------------
# Lesion type MCQ setup
# --------------------------------------------------------------------------
LESION_OPTIONS = {
    "A": "abdomen",
    "B": "bone",
    "C": "kidney",
    "D": "liver",
    "E": "lung",
    "F": "mediastinum",
    "G": "pelvis",
    "H": "soft tissue",
}
# Reverse map: lesion_type string -> letter
LESION_TO_LETTER = {v: k for k, v in LESION_OPTIONS.items()}

# --------------------------------------------------------------------------
# Answer-letter extraction (A-G)
# --------------------------------------------------------------------------
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_UNCLOSED = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
LETTER_ANY = re.compile(r"\b([A-H])\b")
ANSWER_HINT = re.compile(
    r"(?:final\s+answer|answer(?:\s+is)?|correct(?:\s+answer)?(?:\s+is)?|option)"
    r"\s*[:\-]?\s*\*?\*?\(?\s*([A-H])\b",
    re.IGNORECASE,
)
BRACKETED = re.compile(r"\(\s*([A-H])\s*\)")


def _strip_thinking(text):
    if not text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = THINK_OPEN_UNCLOSED.sub("", text)
    return text.strip()


def format_question(example):
    lines = [
        "Look at the provided CT image and identify the type of lesion shown.",
        "",
        "Question: What anatomical region does this lesion belong to?",
        "",
        "Options:",
    ]
    for k in sorted(LESION_OPTIONS.keys()):
        lines.append(f"{k}. {LESION_OPTIONS[k].capitalize()}")
    lines.append("")
    lines.append(
        "Select the single best answer. Conclude your response with a line in "
        "exactly this format: 'Answer: <letter>' where <letter> is one of A, B, C, D, E, F, G, H."
    )
    return "\n".join(lines)


def extract_letter(text, options=None):
    text = _strip_thinking(text or "")
    if not text:
        return None
    m = ANSWER_HINT.search(text)
    if m:
        return m.group(1).upper()
    m = BRACKETED.search(text)
    if m:
        return m.group(1).upper()
    # Try matching lesion type names in text
    low = text.lower()
    best, best_len = None, 0
    for k, v in LESION_OPTIONS.items():
        if v in low and len(v) > best_len:
            best, best_len = k, len(v)
    if best:
        return best
    m = LETTER_ANY.search(text)
    return m.group(1).upper() if m else None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def already_done_indices(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["index"])
            except Exception:
                continue
    return done


def _safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


class _ThreadRoutingStream:
    def __init__(self, base):
        self.base = base
        self._overrides = {}

    def write(self, data):
        buf = self._overrides.get(threading.get_ident())
        (buf if buf is not None else self.base).write(data)

    def flush(self):
        buf = self._overrides.get(threading.get_ident())
        (buf if buf is not None else self.base).flush()

    @contextlib.contextmanager
    def capture(self):
        buf = io.StringIO()
        tid = threading.get_ident()
        self._overrides[tid] = buf
        try:
            yield buf
        finally:
            self._overrides.pop(tid, None)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", default="data/deeplesion/test.jsonl")
    p.add_argument("--data-dir", default="data/deeplesion",
                    help="Base directory for image paths (default: data/deeplesion).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--step_id", type=int, default=9)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()

    print("[eval] Initializing MedicalAssistant ...", flush=True)
    from model.language_model import MedicalAssistant
    assistant = MedicalAssistant()

    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{_safe_name(assistant.model_name)}__deeplesion__{stamp}"
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    predictions_path = os.path.join(run_dir, "predictions.jsonl")
    log_path = os.path.join(run_dir, "eval.log")
    config_path = os.path.join(run_dir, "config.json")
    history_dir = os.path.join(run_dir, "history")
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    os.environ["MAM_HISTORY_DIR"] = history_dir

    log_fh = open(log_path, "a", encoding="utf-8")
    base_out = _Tee(sys.__stdout__, log_fh)
    base_err = _Tee(sys.__stderr__, log_fh)
    router_out = _ThreadRoutingStream(base_out)
    router_err = _ThreadRoutingStream(base_err)
    sys.stdout = router_out
    sys.stderr = router_err

    print(f"[eval] dataset       = DeepLesion (7-class MCQ)")
    print(f"[eval] run_dir       = {run_dir}")
    print(f"[eval] model         = {assistant.model_name}  (mode={assistant.mode})")
    if assistant.mode == "api":
        print(f"[eval] llm_url       = {assistant.api_url}")
    if assistant.vlm_api_url and assistant.vlm_api_url != assistant.api_url:
        print(f"[eval] vlm_url       = {assistant.vlm_api_url}  (model={assistant.vlm_model_name})")

    cfg = {
        "run_name": args.run_name,
        "dataset": "deeplesion_7class",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": assistant.model_name,
        "vlm_model": assistant.vlm_model_name,
        "mode": assistant.mode,
        "api_url": getattr(assistant, "api_url", None),
        "vlm_api_url": getattr(assistant, "vlm_api_url", None),
        "step_id": args.step_id,
        "data": args.data,
        "start": args.start,
        "limit": args.limit,
        "resume": args.resume,
        "concurrency": args.concurrency,
        "env": {
            k: v for k, v in os.environ.items()
            if k.startswith("MAM_") or k == "CUDA_VISIBLE_DEVICES"
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    from inference import run_pipeline

    with open(args.data, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    data_slice = data[args.start : args.start + args.limit] if args.limit else data[args.start :]
    done = already_done_indices(predictions_path) if args.resume else set()
    print(
        f"[eval] {len(data_slice)} examples (from {args.start}), step_id={args.step_id}, "
        f"resume-skip={len(done)}",
        flush=True,
    )

    correct, total = 0, 0
    t0 = time.time()
    mode = "a" if args.resume else "w"

    def process_one(i, ex):
        idx = args.start + i
        formatted = format_question(ex)
        gold_type = ex["lesion_type"]
        gold_letter = LESION_TO_LETTER.get(gold_type)
        image_path = os.path.join(args.data_dir, ex["image"])
        t_ex = time.time()
        trace = None
        with router_out.capture() as buf, assistant.scope() as usage:
            try:
                trace = run_pipeline(args.step_id, formatted, image_path, sample_id=idx)
            except Exception as e:
                print(f"[eval] idx={idx} ERROR: {e}", flush=True)
        return {
            "i": i, "idx": idx, "ex": ex, "gold": gold_letter,
            "gold_type": gold_type,
            "formatted": formatted, "trace": trace, "usage": dict(usage),
            "stdout": buf.getvalue(), "elapsed": time.time() - t_ex,
        }

    out_lock = threading.Lock()

    def emit_result(r):
        nonlocal correct, total
        ex = r["ex"]
        idx = r["idx"]
        trace = r["trace"] or {}
        usage = r["usage"]
        diagnosis = trace.get("diagnosis") or ""
        pred = extract_letter(diagnosis)
        ok = pred == r["gold"]

        pred_type = LESION_OPTIONS.get(pred, "unknown") if pred else None

        trace_path = os.path.join(traces_dir, f"{idx:05d}.json")
        trace_record = {
            "index": idx, "gold": r["gold"], "gold_type": r["gold_type"],
            "pred": pred, "pred_type": pred_type, "correct": ok,
            "lesion_type": ex["lesion_type"],
            "patient_age": ex.get("patient_age"),
            "patient_gender": ex.get("patient_gender"),
            "image": ex["image"],
            "formatted_prompt": r["formatted"],
            "usage": usage, "elapsed_sec": r["elapsed"], "pipeline": trace,
        }
        with open(trace_path, "w", encoding="utf-8") as tf:
            json.dump(trace_record, tf, ensure_ascii=False, indent=2)

        step6 = trace.get("step_6_multi_agent_meeting") or {}
        rec = {
            "index": idx, "gold": r["gold"], "gold_type": r["gold_type"],
            "pred": pred, "pred_type": pred_type, "correct": ok,
            "modality_type": trace.get("modality_type"),
            "type_name": trace.get("type_name"),
            "specialists": [x.get("name") for x in (step6.get("parsed_roles") or [])],
            "verdict": step6.get("verdict"),
            "rounds": len(step6.get("rounds") or []),
            "llm_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "elapsed_sec": round(r["elapsed"], 2),
            "diagnosis": diagnosis[:500],
            "review_result": trace.get("review_result"),
        }
        with out_lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            total += 1
            correct += int(ok)
            if not args.quiet and r["stdout"]:
                base_out.write(r["stdout"])
            dt_total = time.time() - t0
            print(
                f"[eval] [{total}/{len(data_slice)}] idx={idx} "
                f"gold={r['gold']}({r['gold_type']}) pred={pred}({pred_type}) "
                f"{'OK' if ok else 'X '} acc={correct / total:.3f} "
                f"calls={usage['calls']} tok={usage['prompt_tokens']}/{usage['completion_tokens']} "
                f"ex={r['elapsed']:.1f}s total={dt_total:.0f}s",
                flush=True,
            )

    with open(predictions_path, mode, encoding="utf-8") as out_f:
        pending = [(i, ex) for i, ex in enumerate(data_slice) if (args.start + i) not in done]

        if args.concurrency <= 1:
            for i, ex in pending:
                emit_result(process_one(i, ex))
        else:
            print(f"[eval] parallel execution with {args.concurrency} workers", flush=True)
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futures = [pool.submit(process_one, i, ex) for i, ex in pending]
                for fut in as_completed(futures):
                    try:
                        emit_result(fut.result())
                    except Exception as e:
                        print(f"[eval] worker error: {e}", flush=True)

    cfg["finished_at"] = datetime.now().isoformat(timespec="seconds")
    cfg["processed"] = total
    cfg["correct"] = correct
    cfg["accuracy"] = (correct / total) if total else None
    final_snap = assistant.snapshot()
    cfg["total_llm_calls"] = final_snap["calls"]
    cfg["total_prompt_tokens"] = final_snap["prompt_tokens"]
    cfg["total_completion_tokens"] = final_snap["completion_tokens"]
    if total:
        cfg["avg_llm_calls_per_example"] = round(final_snap["calls"] / total, 2)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if total:
        print(f"\n[eval] Final accuracy: {correct}/{total} = {correct / total:.4f}")
    else:
        print("[eval] No examples processed.")


if __name__ == "__main__":
    sys.exit(main())
