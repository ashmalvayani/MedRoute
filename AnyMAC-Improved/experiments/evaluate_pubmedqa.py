import os
import json
import time
import aiohttp
import asyncio
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple

from tqdm import tqdm
import torch

from GDesigner.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from GDesigner.graph.graph import Graph
from GDesigner.prompt.pubmedqa_prompt_set import SPECIALISTS

# ---------------------------------------------------------------------------
# Judge helper — calls vLLM judge via OpenAI-compatible API
# ---------------------------------------------------------------------------
JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', os.getenv('BASE_URL', 'http://localhost:8001'))
JUDGE_API_KEY = os.getenv('JUDGE_API_KEY', os.getenv('API_KEY', 'EMPTY'))

JUDGE_SYSTEM_PROMPT = (
    "You are an answer extractor. Read the model's response and extract the final answer "
    "option the model chose. Focus ONLY on the model's conclusion, not on any options listed "
    "in the question. Reply with ONLY the single letter."
)

async def _judge_one(session, judge_model, question, true_answer, response_text):
    prompt = (
        f"Question:\n{question}\n\n"
        f"Model response:\n{response_text}\n\n"
        f"What is the model's final answer? Focus on the model's conclusion only. "
        f"Reply with ONLY the single letter."
    )
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
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
                return False
            r = await resp.json()
            extracted = r['choices'][0]['message']['content'].strip().upper()
            return len(extracted) > 0 and extracted[0] == true_answer.strip().upper()
    except Exception:
        return False


# -----------------------------
# Safe JSON helpers
# -----------------------------
def load_result(result_file: Path) -> List[Dict[str, Any]]:
    if not result_file.exists():
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
    with open(result_file, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(result_file: Path, data: Any, compact: bool = False) -> None:
    result_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = result_file.with_suffix(result_file.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        if compact:
            f.write("[\n")
            for i, item in enumerate(data):
                line = json.dumps(item, ensure_ascii=False)
                if i < len(data) - 1:
                    f.write(line + ",\n")
                else:
                    f.write(line + "\n")
            f.write("]\n")
        else:
            json.dump(data, f, indent=4, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, result_file)


def collect_records(dataset, limit_questions: Optional[int]) -> List[Tuple[int, Any]]:
    items = []
    for i, record in enumerate(dataset):
        if limit_questions is not None and i >= limit_questions:
            break
        items.append((i, record))
    return items


# -----------------------------
# Async parallel evaluate for PubMedQA
# -----------------------------
def evaluate(
    graph: Graph,
    dataset,
    limit_questions: Optional[int] = None,
    result_file=None,
    result_dir=None,
    args=None,
) -> float:

    result_dir = Path(result_dir)
    result_file = Path(result_file)

    print(f"Evaluating nap on {dataset.__class__.__name__} split {dataset.split}")

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time

    log_file_path = result_dir / f"{args.domain}_{args.llm_name.replace('/', '-')}_{current_time}.log"

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        print("--- Training Finished ---")
        log_file.write("--- Training Finished ---\n")

        # Save training summary once
        data = load_result(result_file)
        try:
            data.append(
                {
                    "time": current_time,
                    "llm_name": args.llm_name,
                    "domain": args.domain,
                    "Training Cost": Cost.instance().value,
                    "Training Prompt Tokens": PromptTokens.instance().value,
                    "Training Completion Tokens": CompletionTokens.instance().value,
                }
            )
            atomic_write_json(result_file, data)
        except Exception as e:
            print(f"Error saving Training tokens results: {e}")
            log_file.write(f"Error saving Training tokens results: {e}\n")
            log_file.flush()

        print("Testing the model on the test set...")
        log_file.write("Testing the model on the test set...\n")
        log_file.flush()

        graph.set_eval()

        # ---- Collect records ----
        items = collect_records(dataset, limit_questions)
        n_items = len(items)

        # ---- Parallelism: use trace_parallelism (same as training) ----
        eval_parallelism = int(getattr(args, "trace_parallelism", 64))
        print(f"Eval parallelism: {eval_parallelism} concurrent samples")

        data = load_result(result_file)

        # Details file
        details_file = result_file.with_name(result_file.stem + "_details" + result_file.suffix)
        details_data = load_result(details_file)

        total_solved = 0
        total_judge_solved = 0
        total_executed = 0

        judge_model = getattr(args, 'judge_model', None)

        async def _eval_one_sample(i_record, record, sem, judge_session):
            nonlocal total_solved, total_judge_solved, total_executed

            async with sem:
                try:
                    input_dict = dataset.record_to_input(record)
                    task = input_dict["task"]

                    with torch.no_grad():
                        inference_result = await asyncio.wait_for(
                            graph.arun_next_agent_prediction(
                                input=input_dict,
                                max_routing=args.max_routing,
                                temperature=args.temperature,
                                available_roles=SPECIALISTS,
                                agent_group_type="AnalyzeAgent",
                                max_context=args.max_context,
                            ),
                            timeout=1800,
                        )

                    if inference_result is None:
                        return i_record, {
                            "Index": i_record, "Ground_truth": None,
                            "Routing_length": None, "Response": "",
                            "Regex_answer": "", "Regex_solved": False,
                            "Judge_solved": False, "Error": "Timeout",
                        }, None

                    true_answer = dataset.record_to_target_answer(record)
                    predict_answer_list = inference_result.get("answers", [""])
                    predict_answer_str = predict_answer_list[0] if predict_answer_list else ""
                    predict_answer_val = dataset.postprocess_answer(predict_answer_str)
                    routing_length = inference_result.get("routing_count", args.max_routing)

                    is_solved_regex = dataset.record_to_target_check(true_answer, predict_answer_str, input_dict["task"])

                    # LLM judge (async)
                    is_solved_judge = False
                    if judge_model and predict_answer_str and judge_session:
                        is_solved_judge = await _judge_one(
                            judge_session, judge_model, task, true_answer, predict_answer_str
                        )

                    # Extract routing trace
                    routing_results = inference_result.get("routing_results", {})
                    agent_sels = routing_results.get("agent_selections", [])
                    routing_trace = []
                    for idx in agent_sels:
                        if idx < len(SPECIALISTS):
                            routing_trace.append(SPECIALISTS[idx])
                        else:
                            routing_trace.append("DecisionMaker")

                    updated_item = {
                        "Index": i_record,
                        "Ground_truth": true_answer,
                        "Routing_length": routing_length,
                        "Response": predict_answer_str,
                        "Regex_answer": predict_answer_val,
                        "Regex_solved": bool(is_solved_regex),
                        "Judge_solved": bool(is_solved_judge),
                    }
                    detail_item = {
                        "Index": i_record,
                        "Question": task,
                        "Ground_truth": true_answer,
                        "Routing_trace": routing_trace,
                        "Conversations": routing_results.get("conversations", []),
                    }
                    return i_record, updated_item, detail_item

                except Exception as e:
                    err_item = {
                        "Index": i_record,
                        "Ground_truth": None,
                        "Routing_length": None,
                        "Response": "",
                        "Regex_answer": "",
                        "Regex_solved": False,
                        "Judge_solved": False,
                        "Error": repr(e),
                    }
                    return i_record, err_item, None

        async def _run_all_eval():
            nonlocal total_solved, total_judge_solved, total_executed

            sem = asyncio.Semaphore(eval_parallelism)

            judge_session = None
            if judge_model:
                judge_session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=120)
                )

            pbar = tqdm(total=n_items, desc="Evaluating")

            try:
                tasks = [
                    asyncio.create_task(
                        _eval_one_sample(i, r, sem, judge_session)
                    )
                    for (i, r) in items
                ]

                for fut in asyncio.as_completed(tasks):
                    i_record, updated_item, detail_item = await fut

                    total_executed += 1
                    if updated_item.get("Regex_solved"):
                        total_solved += 1
                    if updated_item.get("Judge_solved"):
                        total_judge_solved += 1

                    current_accuracy = total_solved / total_executed if total_executed else 0.0
                    judge_accuracy = total_judge_solved / total_executed if total_executed else 0.0
                    updated_item["Total solved"] = total_solved
                    updated_item["Total judge_solved"] = total_judge_solved
                    updated_item["Total executed"] = total_executed
                    updated_item["Regex_accuracy"] = current_accuracy
                    updated_item["Judge_accuracy"] = judge_accuracy

                    data.append(updated_item)
                    if detail_item:
                        details_data.append(detail_item)

                    if updated_item.get("Error"):
                        log_file.write(f"[ERROR] Index={i_record} err={updated_item['Error']}\n")
                        log_file.flush()

                    pbar.set_postfix(regex=f"{current_accuracy:.3f}", judge=f"{judge_accuracy:.3f}")
                    pbar.update(1)

                    # Flush every 50 samples
                    if total_executed % 50 == 0:
                        atomic_write_json(result_file, data)
                        atomic_write_json(details_file, details_data, compact=True)

            finally:
                pbar.close()
                if judge_session:
                    await judge_session.close()

        asyncio.run(_run_all_eval())

        # Final flush
        atomic_write_json(result_file, data)
        atomic_write_json(details_file, details_data, compact=True)

        # Final accuracy
        final_accuracy = total_solved / total_executed if total_executed else 0.0
        final_judge_accuracy = total_judge_solved / total_executed if total_executed else 0.0
        print(f"Final Regex Accuracy: {final_accuracy:.4f} ({total_solved}/{total_executed} solved)")
        print(f"Final Judge Accuracy: {final_judge_accuracy:.4f} ({total_judge_solved}/{total_executed} solved)")

        # Append summary entry
        data.append(
            {
                "time": current_time,
                "llm_name": args.llm_name,
                "mode": args.mode,
                "domain": args.domain,
                "agent_names": args.agent_names,
                "agent_nums": args.agent_nums,
                "num_rounds": args.num_rounds,
                "decision_method": args.decision_method,
                "regex_accuracy": float(final_accuracy),
                "judge_accuracy": float(final_judge_accuracy),
                "total_regex_solved": total_solved,
                "total_judge_solved": total_judge_solved,
                "total_executed": total_executed,
                "cost": Cost.instance().value,
                "prompt_tokens": PromptTokens.instance().value,
                "completion_tokens": CompletionTokens.instance().value,
            }
        )

        atomic_write_json(result_file, data)
        atomic_write_json(details_file, details_data, compact=True)
        print(f"Results: {result_file}")
        print(f"Details: {details_file}")

        print(f"Cost {Cost.instance().value}")
        print(f"PromptTokens {PromptTokens.instance().value}")
        print(f"CompletionTokens {CompletionTokens.instance().value}")

        return float(final_accuracy)
