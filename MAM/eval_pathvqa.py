#!/usr/bin/env python3
"""Evaluate MAM on the PathVQA test set (image modality).

PathVQA has two question types:
  - Yes/No (50%): binary answer
  - Open-ended (50%): short free-text answer

We report accuracy separately for yes/no and open-ended, plus overall.
For yes/no we extract yes/no from the response. For open-ended we check
whether the gold answer appears in the model response (case-insensitive).

Typical two-model usage:

    MAM_LLM_API_URL=http://<text-server>:9001/v1 \
    MAM_VLM_API_URL=http://<vlm-server>:9001/v1 \
    MAM_LLM_MAX_TOKENS=2048 \
    MAM_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
    MAM_VLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}' \
        .venv/bin/python eval_pathvqa.py --run-name Qwen3-PathVQA --concurrency 32 --quiet
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
# Answer extraction
# --------------------------------------------------------------------------
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
THINK_OPEN_UNCLOSED = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)

ANSWER_HINT = re.compile(
    r"(?:final\s+answer|answer(?:\s+is)?)\s*[:\-]?\s*\*?\*?\(?\s*(.+)",
    re.IGNORECASE,
)
YN_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _strip_thinking(text):
    if not text:
        return text
    text = THINK_BLOCK.sub("", text)
    text = THINK_OPEN_UNCLOSED.sub("", text)
    return text.strip()


def _normalize(s):
    """Lowercase, strip punctuation/articles for open-ended comparison."""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def is_yesno(answer):
    return answer.lower().strip() in ("yes", "no")


def extract_yesno(text):
    text = _strip_thinking(text or "")
    if not text:
        return None
    m = ANSWER_HINT.search(text)
    if m:
        tail = m.group(1).strip()
        ym = YN_PATTERN.search(tail)
        if ym:
            return ym.group(1).lower()
    matches = YN_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()
    return None


def extract_openended(text, gold):
    """Check if the gold answer is contained in the model response."""
    text = _strip_thinking(text or "")
    if not text:
        return None, False
    # Try to grab explicit "Answer: ..." line first
    m = ANSWER_HINT.search(text)
    pred = m.group(1).strip().rstrip(".").strip() if m else text.strip()
    # Containment check (normalized)
    ok = _normalize(gold) in _normalize(text)
    return pred, ok


def format_question_yesno(example):
    q = example["question"]
    return (
        f"Look at the provided medical image and answer the following question.\n\n"
        f"Question: {q}\n\n"
        f"Answer with exactly one of: yes or no. Conclude your response with "
        f"a line in exactly this format: 'Answer: <label>' where <label> is yes or no."
    )


def format_question_open(example):
    q = example["question"]
    return (
        f"Look at the provided medical image and answer the following question.\n\n"
        f"Question: {q}\n\n"
        f"Provide a short, precise answer. Conclude your response with a line "
        f"in exactly this format: 'Answer: <your answer>'."
    )


# --------------------------------------------------------------------------
# Helpers (shared boilerplate)
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
    p.add_argument("--data", default="data/pathvqa/test.jsonl")
    p.add_argument("--data-dir", default="data/pathvqa",
                    help="Base directory for image paths (default: data/pathvqa).")
    p.add_argument("--run-name", default=None)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--step_id", type=int, default=9)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--qtype", choices=["yesno", "open"], default=None,
                    help="Only evaluate questions of this type (default: all).")
    args = p.parse_args()

    print("[eval] Initializing MedicalAssistant ...", flush=True)
    from model.language_model import MedicalAssistant
    assistant = MedicalAssistant()

    if args.run_name is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.run_name = f"{_safe_name(assistant.model_name)}__pathvqa__{stamp}"
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

    print(f"[eval] dataset       = PathVQA")
    print(f"[eval] run_dir       = {run_dir}")
    print(f"[eval] model         = {assistant.model_name}  (mode={assistant.mode})")
    if assistant.mode == "api":
        print(f"[eval] llm_url       = {assistant.api_url}")
    if assistant.vlm_api_url and assistant.vlm_api_url != assistant.api_url:
        print(f"[eval] vlm_url       = {assistant.vlm_api_url}  (model={assistant.vlm_model_name})")

    cfg = {
        "run_name": args.run_name,
        "dataset": "pathvqa",
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
    if args.qtype:
        if args.qtype == "yesno":
            data_slice = [ex for ex in data_slice if is_yesno(ex["answer"])]
        else:
            data_slice = [ex for ex in data_slice if not is_yesno(ex["answer"])]
        print(f"[eval] filtered to qtype={args.qtype}: {len(data_slice)} examples")
    done = already_done_indices(predictions_path) if args.resume else set()
    print(
        f"[eval] {len(data_slice)} examples (from {args.start}), step_id={args.step_id}, "
        f"resume-skip={len(done)}",
        flush=True,
    )

    correct, correct_yn, correct_open, total, total_yn, total_open = 0, 0, 0, 0, 0, 0
    t0 = time.time()
    mode = "a" if args.resume else "w"

    def process_one(i, ex):
        idx = args.start + i
        gold = ex["answer"]
        qtype = "yesno" if is_yesno(gold) else "open"
        formatted = format_question_yesno(ex) if qtype == "yesno" else format_question_open(ex)
        image_path = os.path.join(args.data_dir, ex["image"])
        t_ex = time.time()
        trace = None
        with router_out.capture() as buf, assistant.scope() as usage:
            try:
                trace = run_pipeline(args.step_id, formatted, image_path, sample_id=idx)
            except Exception as e:
                print(f"[eval] idx={idx} ERROR: {e}", flush=True)
        return {
            "i": i, "idx": idx, "ex": ex, "gold": gold, "qtype": qtype,
            "formatted": formatted, "trace": trace, "usage": dict(usage),
            "stdout": buf.getvalue(), "elapsed": time.time() - t_ex,
        }

    out_lock = threading.Lock()

    def emit_result(r):
        nonlocal correct, correct_yn, correct_open, total, total_yn, total_open
        trace = r["trace"] or {}
        usage = r["usage"]
        idx = r["idx"]
        diagnosis = trace.get("diagnosis") or ""
        gold = r["gold"]
        qtype = r["qtype"]

        if qtype == "yesno":
            pred = extract_yesno(diagnosis)
            ok = pred == gold.lower().strip()
        else:
            pred, ok = extract_openended(diagnosis, gold)

        trace_path = os.path.join(traces_dir, f"{idx:05d}.json")
        trace_record = {
            "index": idx, "gold": gold, "pred": pred, "correct": ok,
            "qtype": qtype, "question": r["ex"]["question"],
            "image": r["ex"]["image"],
            "formatted_prompt": r["formatted"],
            "usage": usage, "elapsed_sec": r["elapsed"], "pipeline": trace,
        }
        with open(trace_path, "w", encoding="utf-8") as tf:
            json.dump(trace_record, tf, ensure_ascii=False, indent=2)

        step6 = trace.get("step_6_multi_agent_meeting") or {}
        rec = {
            "index": idx, "gold": gold, "pred": pred, "correct": ok,
            "qtype": qtype,
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
            if qtype == "yesno":
                total_yn += 1
                correct_yn += int(ok)
            else:
                total_open += 1
                correct_open += int(ok)
            if not args.quiet and r["stdout"]:
                base_out.write(r["stdout"])
            dt_total = time.time() - t0
            acc_str = f"acc={correct / total:.3f}"
            if total_yn:
                acc_str += f" yn={correct_yn}/{total_yn}={correct_yn/total_yn:.3f}"
            if total_open:
                acc_str += f" open={correct_open}/{total_open}={correct_open/total_open:.3f}"
            print(
                f"[eval] [{total}/{len(data_slice)}] idx={idx} type={qtype} "
                f"gold={gold!r:.30s} pred={str(pred)!r:.30s} "
                f"{'OK' if ok else 'X '} {acc_str} "
                f"calls={usage['calls']} ex={r['elapsed']:.1f}s total={dt_total:.0f}s",
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
    cfg["accuracy_yesno"] = (correct_yn / total_yn) if total_yn else None
    cfg["accuracy_open"] = (correct_open / total_open) if total_open else None
    cfg["total_yesno"] = total_yn
    cfg["total_open"] = total_open
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
        if total_yn:
            print(f"[eval]   Yes/No:      {correct_yn}/{total_yn} = {correct_yn / total_yn:.4f}")
        if total_open:
            print(f"[eval]   Open-ended:  {correct_open}/{total_open} = {correct_open / total_open:.4f}")
    else:
        print("[eval] No examples processed.")


if __name__ == "__main__":
    sys.exit(main())
