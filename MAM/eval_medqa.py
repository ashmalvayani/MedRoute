#!/usr/bin/env python3
"""Evaluate MAM on the MedQA 5-option USMLE test set (text modality).

Every run is isolated under `runs/<run-name>/`, so running multiple LLMs
against the same dataset never clobbers prior results:

    runs/<run-name>/
        predictions.jsonl     # one JSON object per example
        eval.log              # full stdout/stderr tee
        config.json           # model, URL, step_id, args, timestamps
        history/              # used by pipeline step 5/9 (per-run isolated)

If --run-name is not given, it is derived from the served model id (slashes
replaced with underscores) plus an ISO timestamp.

Typical usage against a remote vLLM server:

    MAM_LLM_API_URL=http://13.219.217.59:8000/v1 \\
        .venv/bin/python eval_medqa.py --limit 5 --quiet

For thinking models (e.g. Qwen3) you will likely want:

    MAM_LLM_API_URL=http://13.219.217.59:8000/v1 \\
    MAM_LLM_MAX_TOKENS=2048 \\
    MAM_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \\
        .venv/bin/python eval_medqa.py --limit 5 --quiet
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
# Answer-letter extraction
# --------------------------------------------------------------------------
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_UNCLOSED = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
LETTER_ANY = re.compile(r"\b([A-E])\b")
ANSWER_HINT = re.compile(
    r"(?:final\s+answer|answer(?:\s+is)?|correct(?:\s+answer)?(?:\s+is)?|option)"
    r"\s*[:\-]?\s*\*?\*?\(?\s*([A-E])\b",
    re.IGNORECASE,
)
BRACKETED = re.compile(r"\(\s*([A-E])\s*\)")


def _strip_thinking(text):
    """Remove <think>...</think> reasoning chunks that Qwen3 etc. emit."""
    if not text:
        return text
    text = THINK_BLOCK.sub("", text)
    # If the response was truncated mid-<think> (no closing tag), drop the tail.
    text = THINK_OPEN_UNCLOSED.sub("", text)
    return text.strip()


def format_question(example):
    """Build a single user-message prompt from a MedQA record."""
    q = example["question"]
    opts = example["options"]
    lines = [q, "", "Options:"]
    for k in sorted(opts.keys()):
        lines.append(f"{k}. {opts[k]}")
    lines.append("")
    lines.append(
        "Select the single best answer. Conclude your response with a line in "
        "exactly this format: 'Answer: <letter>' where <letter> is one of A, B, C, D, E."
    )
    return "\n".join(lines)


def extract_letter(text, options):
    """Recover an A–E letter from a free-form diagnosis string."""
    text = _strip_thinking(text or "")
    if not text:
        return None
    m = ANSWER_HINT.search(text)
    if m:
        return m.group(1).upper()
    m = BRACKETED.search(text)
    if m:
        return m.group(1).upper()
    low = text.lower()
    best, best_len = None, 0
    for k, v in options.items():
        if v and v.lower() in low and len(v) > best_len:
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
    """Write to multiple text streams (stdout + log file)."""

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
    """sys.stdout replacement. If the current thread has a capture override set,
    writes are routed to that override (a StringIO); otherwise they fall through
    to `base`. Used so concurrent workers can capture their own pipeline output
    without interleaving stdout."""

    def __init__(self, base):
        self.base = base
        self._overrides = {}  # tid -> StringIO

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
    p.add_argument("--data", default="data/medqa/test.jsonl")
    p.add_argument(
        "--run-name",
        default=None,
        help="Run identifier. Default: <safe-model-id>__<ISO-timestamp>. "
        "Outputs go to runs/<run-name>/.",
    )
    p.add_argument(
        "--runs-dir",
        default="runs",
        help="Parent directory for runs (default: runs/).",
    )
    p.add_argument(
        "--step_id",
        type=int,
        default=9,
        help="Pipeline step (1-9). Default 9 = full pipeline incl. memory save "
        "(matches the upstream README).",
    )
    p.add_argument("--limit", type=int, default=0, help="Evaluate at most N examples (0 = all).")
    p.add_argument("--start", type=int, default=0, help="Start index (for resuming).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-example pipeline prints.")
    p.add_argument("--resume", action="store_true", help="Skip indices already in predictions.jsonl.")
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Run N MedQA questions in parallel (threads). 1 = sequential. "
        "For API mode this is usually the fastest way to saturate the server.",
    )
    args = p.parse_args()

    # ---- Load the LLM singleton first (also tells us the served model id) ----
    print("[eval] Initializing MedicalAssistant ...", flush=True)
    from model.language_model import MedicalAssistant

    assistant = MedicalAssistant()

    # ---- Resolve the run directory ----
    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{_safe_name(assistant.model_name)}__{stamp}"
    run_dir = os.path.join(args.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    predictions_path = os.path.join(run_dir, "predictions.jsonl")
    log_path = os.path.join(run_dir, "eval.log")
    config_path = os.path.join(run_dir, "config.json")
    history_dir = os.path.join(run_dir, "history")
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(traces_dir, exist_ok=True)

    # Point the pipeline at the per-run history directory.
    os.environ["MAM_HISTORY_DIR"] = history_dir

    # Tee stdout to eval.log so every pipeline print is captured, wrapped in a
    # thread-router so worker threads can capture their own output.
    log_fh = open(log_path, "a", encoding="utf-8")
    base_out = _Tee(sys.__stdout__, log_fh)
    base_err = _Tee(sys.__stderr__, log_fh)
    router_out = _ThreadRoutingStream(base_out)
    router_err = _ThreadRoutingStream(base_err)
    sys.stdout = router_out
    sys.stderr = router_err

    print(f"[eval] run_dir       = {run_dir}")
    print(f"[eval] model         = {assistant.model_name}  (mode={assistant.mode})")
    if assistant.mode == "api":
        print(f"[eval] api_url       = {assistant.api_url}")
    print(f"[eval] predictions   = {predictions_path}")
    print(f"[eval] traces_dir    = {traces_dir}")
    print(f"[eval] history_dir   = {history_dir}")
    print(f"[eval] log_file      = {log_path}")

    # Save run config for reproducibility.
    cfg = {
        "run_name": args.run_name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "model": assistant.model_name,
        "mode": assistant.mode,
        "api_url": getattr(assistant, "api_url", None),
        "step_id": args.step_id,
        "data": args.data,
        "start": args.start,
        "limit": args.limit,
        "resume": args.resume,
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("MAM_") or k == "CUDA_VISIBLE_DEVICES"
        },
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Import pipeline AFTER model + env are ready so its imports see MAM_HISTORY_DIR.
    from inference import run_pipeline

    # ---- Load MedQA test set ----
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
        """Runs in a worker thread. Captures its own stdout, scopes its own counters."""
        idx = args.start + i
        formatted = format_question(ex)
        gold = ex["answer_idx"]
        t_ex = time.time()
        trace = None
        with router_out.capture() as buf, assistant.scope() as usage:
            try:
                trace = run_pipeline(args.step_id, formatted, "", sample_id=idx)
            except Exception as e:
                print(f"[eval] idx={idx} ERROR: {e}", flush=True)
        return {
            "i": i,
            "idx": idx,
            "ex": ex,
            "gold": gold,
            "formatted": formatted,
            "trace": trace,
            "usage": dict(usage),
            "stdout": buf.getvalue(),
            "elapsed": time.time() - t_ex,
        }

    out_lock = threading.Lock()

    def emit_result(r):
        nonlocal correct, total
        ex = r["ex"]
        idx = r["idx"]
        trace = r["trace"] or {}
        usage = r["usage"]
        diagnosis = trace.get("diagnosis") or ""
        pred = extract_letter(diagnosis, ex["options"])
        ok = pred == r["gold"]

        # Trace JSON (no stdout — every step is already captured under step_N_...)
        trace_path = os.path.join(traces_dir, f"{idx:05d}.json")
        trace_record = {
            "index": idx,
            "gold": r["gold"],
            "pred": pred,
            "correct": ok,
            "meta_info": ex.get("meta_info", ""),
            "question": ex["question"],
            "options": ex["options"],
            "formatted_prompt": r["formatted"],
            "usage": usage,
            "elapsed_sec": r["elapsed"],
            "pipeline": trace,
        }
        with open(trace_path, "w", encoding="utf-8") as tf:
            json.dump(trace_record, tf, ensure_ascii=False, indent=2)

        # Compact predictions row + live progress (lock: file write + counters)
        step6 = trace.get("step_6_multi_agent_meeting") or {}
        rec = {
            "index": idx,
            "gold": r["gold"],
            "pred": pred,
            "correct": ok,
            "meta_info": ex.get("meta_info", ""),
            "modality_type": trace.get("modality_type"),
            "type_name": trace.get("type_name"),
            "specialists": [x.get("name") for x in (step6.get("parsed_roles") or [])],
            "verdict": step6.get("verdict"),
            "rounds": len(step6.get("rounds") or []),
            "llm_calls": usage["calls"],
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "elapsed_sec": round(r["elapsed"], 2),
            "diagnosis": diagnosis,
            "review_result": trace.get("review_result"),
            "trace_path": trace_path,
        }
        with out_lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            total += 1
            correct += int(ok)
            # If not --quiet, replay worker's captured stdout into the tee.
            if not args.quiet and r["stdout"]:
                base_out.write(r["stdout"])
            dt_total = time.time() - t0
            print(
                f"[eval] [{total}/{len(data_slice)}] idx={idx} gold={r['gold']} pred={pred} "
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

    # Final tally + update config
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
        cfg["avg_tokens_per_example"] = round(
            (final_snap["prompt_tokens"] + final_snap["completion_tokens"]) / total, 1
        )
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    if total:
        print(f"\n[eval] Final accuracy: {correct}/{total} = {correct / total:.4f}")
    else:
        print("[eval] No examples processed.")


if __name__ == "__main__":
    sys.exit(main())
