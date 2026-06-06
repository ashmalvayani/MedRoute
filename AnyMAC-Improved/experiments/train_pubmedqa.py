"""
Training loop for PubMedQA (text-only MCQ) using AnyMAC's next-agent prediction RL.

Same as train_medqa.py but uses PubMedQA specialist roles and dataset.
PubMedQA has 3 options (A/B/C = yes/no/maybe).

IMPORTANT:
This file assumes your Graph exposes:
- run_next_agent_prediction(...)
- run_next_agent_prediction_grad(...)
- routing_transformer.get_gpt_parameters()
- graph.add_to_optimizer(...), graph.save_model(...), graph.set_eval()
"""

import os
import time
import random
import asyncio
import numpy as np
import torch
import aiohttp
from tqdm import tqdm

from typing import List
from pathlib import Path

from GDesigner.utils.globals import Time
from GDesigner.graph.graph import Graph

# ---------------------------------------------------------------------------
# Judge-based reward: call a judge LLM to determine if a response is correct
# ---------------------------------------------------------------------------
JUDGE_BASE_URL = os.getenv('JUDGE_BASE_URL', os.getenv('BASE_URL', 'http://localhost:8001'))
JUDGE_API_KEY = os.getenv('JUDGE_API_KEY', os.getenv('API_KEY', 'EMPTY'))

JUDGE_SYSTEM_PROMPT = (
    "You are an answer extractor. Read the model's response and extract the final answer "
    "option the model chose. Focus ONLY on the model's conclusion, not on any options listed "
    "in the question. Reply with ONLY the single letter."
)

async def judge_answer(
    session: aiohttp.ClientSession,
    judge_model: str,
    question: str,
    true_answer: str,
    model_response: str,
) -> bool:
    """Ask a judge LLM to extract the answer letter, then compare to true_answer."""
    prompt = (
        f"Question:\n{question}\n\n"
        f"Model response:\n{model_response}\n\n"
        f"What is the model's final answer? Focus on the model's conclusion only. "
        f"Reply with ONLY the single letter."
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    payload = {
        "model": judge_model,
        "messages": messages,
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

from GDesigner.prompt.pubmedqa_prompt_set import ROLES, SPECIALISTS
from GDesigner.reward.partial_credit import (
    judge_reasoning_quality,
    compute_partial_reward,
    reward_breakdown,
)

import json

# Global toggle — set from run_pubmedqa.py
PARTIAL_CREDIT_ENABLED = False
REWARD_ALPHA = 0.4


def to_json_safe(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    return obj


def _write_trace_json(results, epoch, result_dir, model_slug, current_time, args):
    """Write structured JSON with full specialist conversations for each training epoch."""
    questions_data = []
    for i_q, (input_dict, true_answer, q_traces, q_rewards, q_rewards_raw) in enumerate(results):
        if not q_traces:
            continue
        rollouts = []
        for r_idx, (trace_data, reward, raw) in enumerate(zip(q_traces, q_rewards, q_rewards_raw)):
            agent_sels = trace_data.get("agent_selections", [])
            routing_trace = []
            for idx in agent_sels:
                if idx < len(SPECIALISTS):
                    routing_trace.append(SPECIALISTS[idx])
                else:
                    routing_trace.append("DecisionMaker")
            rollout_entry = {
                "rollout_id": r_idx,
                "routing_trace": routing_trace,
                "conversations": trace_data.get("conversations", []),
                "final_answer": trace_data.get("_final_answer", ""),
                "routing_count": trace_data.get("_routing_count", 0),
                "reward": float(reward),
                "correct": bool(raw > 0),
                "reasoning_score": trace_data.get("_reasoning_score", None),
            }
            if "hint_extraction_log" in trace_data:
                rollout_entry["hint_extraction_log"] = trace_data["hint_extraction_log"]
            rollouts.append(rollout_entry)
        num_correct = sum(1 for r in q_rewards_raw if r > 0)
        questions_data.append({
            "index": i_q,
            "question": input_dict.get("task", ""),
            "ground_truth": true_answer,
            "num_correct": num_correct,
            "num_total": len(q_traces),
            "rollouts": rollouts,
        })

    trace_json = {
        "metadata": {
            "model": getattr(args, "llm_name", "unknown"),
            "judge": getattr(args, "judge_model", None),
            "epoch": epoch,
            "train_num": getattr(args, "train_num", 0),
            "num_traces": getattr(args, "num_traces", 0),
            "max_routing": getattr(args, "max_routing", 3),
            "timestamp": current_time,
        },
        "questions": questions_data,
    }
    trace_path = Path(result_dir) / f"traces_{model_slug}_{current_time}_epoch{epoch}.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_json, f, indent=2, ensure_ascii=False)
    print(f"[traces] Wrote {len(questions_data)} questions to {trace_path}")


def train(
    graph: Graph,
    dataset,
    result_dir: str | Path | None = None,
    args=None
) -> None:

    def get_random_samples(dataset, sample_size=80) -> List:
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        return [dataset[idx] for idx in indices]

    result_dir = Path(result_dir or "./result/pubmedqa")
    result_dir.mkdir(parents=True, exist_ok=True)

    random_samples = get_random_samples(dataset, sample_size=getattr(args, "train_num", 80))

    # Router model
    graph.routing_transformer.train()
    optimizer = torch.optim.AdamW(
        graph.routing_transformer.get_gpt_parameters(),
        lr=getattr(args, "lr", 1e-5)
    )
    graph.add_to_optimizer(optimizer)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    model_slug = getattr(args, 'llm_name', 'llm').replace('/', '-')
    log_file_path = result_dir / f"log_train_pubmedqa_{model_slug}_{current_time}.txt"

    required_correct = getattr(args, "required_correct_answers", 1)
    max_routing = getattr(args, "max_routing", 3)
    decay_factor = getattr(args, "decay_factor", 0.98)
    judge_model = getattr(args, "judge_model", None)

    _train_cfg = (
        f"Training config: num_traces={getattr(args, 'num_traces', 4)}, "
        f"trace_parallelism={getattr(args, 'trace_parallelism', 8)}, "
        f"required_correct={required_correct}, max_routing={max_routing}, "
        f"train_num={getattr(args, 'train_num', 80)}, epochs={getattr(args, 'epochs', 1)}, "
        f"reward_mode={'judge('+judge_model+')' if judge_model else 'regex'}"
    )
    print(_train_cfg)

    async def _sample_one_trace_async(input_dict, max_routing, temperature, agent_group_type, max_context):
        return await asyncio.wait_for(
            graph.arun_next_agent_prediction(
                input=input_dict,
                max_routing=max_routing,
                temperature=temperature,
                available_roles=ROLES,
                agent_group_type=agent_group_type,
                max_context=max_context,
            ),
            timeout=1800,
        )


    async def _collect_traces_for_record(record, i_record, num_traces, required_correct, sem, judge_session=None):
        input_dict = dataset.record_to_input(record)

        print(input_dict)
        print(f"\n  Sampling traces for question {i_record+1}/{len(random_samples)}")

        question_traces = []
        question_rewards = []
        question_rewards_raw = []
        found_correct = 0

        async def _run_trace():
            async with sem:
                with torch.no_grad():
                    return await _sample_one_trace_async(
                        input_dict=input_dict,
                        max_routing=max_routing,
                        temperature=getattr(args, "temperature", 0.7),
                        agent_group_type=getattr(args, "agent_group_type", "AnalyzeAgent"),
                        max_context=getattr(args, "max_context", 2048),
                    )

        tasks = [asyncio.create_task(_run_trace()) for _ in range(num_traces)]

        try:
            for fut in asyncio.as_completed(tasks):
                trace_result = await fut
                if trace_result is None:
                    continue

                predict_answer_list = trace_result.get("answers", [""])
                predict_answer_str = predict_answer_list[0] if predict_answer_list else ""

                routing_length = trace_result.get("routing_count", max_routing)
                true_answer = dataset.record_to_target_answer(record)

                # --- Reward: judge-based or regex-based ---
                if judge_model and judge_session:
                    is_solved = await judge_answer(
                        judge_session, judge_model,
                        input_dict["task"], true_answer, predict_answer_str,
                    )
                else:
                    is_solved = dataset.record_to_target_check(true_answer, predict_answer_str, input_dict["task"])

                # --- Partial credit reward shaping ---
                reasoning_score = 3
                if PARTIAL_CREDIT_ENABLED and judge_model and judge_session:
                    reasoning_score = await judge_reasoning_quality(
                        judge_session, judge_model,
                        input_dict["task"], true_answer, predict_answer_str,
                    )
                    reward = compute_partial_reward(
                        is_correct=is_solved,
                        reasoning_score=reasoning_score,
                        routing_length=routing_length,
                        decay_factor=decay_factor,
                        alpha=REWARD_ALPHA,
                    )
                    _breakdown = reward_breakdown(is_solved, reasoning_score, routing_length, decay_factor, REWARD_ALPHA)
                    log_file.write(f"    [partial_credit] {_breakdown}\n")
                else:
                    reward = (decay_factor ** routing_length) * float(is_solved)

                trace_routing = trace_result["routing_results"]
                trace_routing["_final_answer"] = predict_answer_str
                trace_routing["_routing_count"] = routing_length
                trace_routing["_reasoning_score"] = reasoning_score
                question_traces.append(trace_routing)
                question_rewards.append(reward)
                question_rewards_raw.append(float(is_solved))

                if is_solved:
                    found_correct += 1

                agent_logits = trace_result["routing_results"].get("agent_logits", [])
                if len(question_traces) == 1 and len(agent_logits) > 0:
                    log_file.write(f"Routing length: {routing_length}\n")
                    log_file.write(f"Agent selections: {trace_result['routing_results']['agent_selections']}\n")
                    log_file.write(f"Hint selections: {trace_result['routing_results']['hint_selections']}\n")
                    log_file.flush()

                if getattr(args, 'early_stop_rollouts', False) and found_correct >= required_correct:
                    for t in tasks:
                        t.cancel()
                    break
        finally:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

        total_ran = len(question_traces)
        reward_mode_label = f"judge:{judge_model}" if judge_model else "regex"
        from datetime import datetime as _dt
        _now = _dt.now().strftime("%H:%M:%S")
        summary = (
            f"  [{_now}] Q{i_record+1}/{len(random_samples)}: "
            f"{found_correct}/{total_ran} correct out of {num_traces} traces ({reward_mode_label})"
        )
        print(summary)
        log_file.write(summary + "\n")
        for t_idx, (trace_data, raw_reward) in enumerate(zip(question_traces, question_rewards_raw)):
            if raw_reward > 0:
                agents = trace_data.get("agent_selections", [])
                hints = trace_data.get("hint_selections", [])
                log_file.write(f"    T{t_idx+1}: Agent Selection {agents} Hint Selection: {hints}\n")
        log_file.flush()

        true_answer = dataset.record_to_target_answer(record)
        return input_dict, true_answer, question_traces, question_rewards, question_rewards_raw

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"--- Starting Training Run: {current_time} ---\n")
        log_file.write(_train_cfg + "\n")
        log_file.flush()

        print("--- Starting Training ---")

        training_samples = getattr(args, "training_samples", 10**9)

        print(getattr(args, "epochs", 1))
        for epoch in range(1, getattr(args, "epochs", 1) + 1):
            log_file.write(f"Training Epoch {epoch}/{getattr(args, 'epochs', 1)}\n")
            log_file.flush()

            eval_interval = getattr(args, "eval_interval", 100)
            if epoch % eval_interval == 0:
                graph.cos_scaling = 1e3
                num_traces = 1
            else:
                graph.cos_scaling = getattr(args, "cos_scaling", 1.5)
                num_traces = getattr(args, "num_traces", 4)

            all_gradient_inputs = []
            total_epoch_reward = 0.0
            trace_count = 0

            # Resume from saved gradient inputs
            _resume_path = getattr(args, 'resume_gradient_path', None)
            _skip_rollouts = False
            if _resume_path and epoch == 1:
                print(f"\n  Resuming from saved gradient inputs: {_resume_path}")
                log_file.write(f"Resuming from saved gradient inputs: {_resume_path}\n")
                with open(_resume_path) as _f:
                    _saved = json.load(_f)
                all_gradient_inputs = _saved['all_gradient_inputs']
                for gi in all_gradient_inputs:
                    gi['agent_outputs_embeddings'] = [
                        torch.tensor(e) for e in gi['agent_outputs_embeddings']
                    ]
                trace_count = len(all_gradient_inputs)
                avg_epoch_reward = 0.0
                _skip_rollouts = True
                print(f"  Loaded {len(all_gradient_inputs)} gradient inputs, skipping rollout collection.")
                log_file.flush()

            print("\n\nCollecting traces for all training samples...")

            graph.set_eval()
            graph.to_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            trace_parallelism = getattr(args, "trace_parallelism", 8)
            sem = asyncio.Semaphore(trace_parallelism)

            async def _collect_all():
                judge_session = None
                judge_timeout = aiohttp.ClientTimeout(total=120)
                if judge_model:
                    judge_session = aiohttp.ClientSession(timeout=judge_timeout)
                try:
                    jobs = [
                        asyncio.create_task(_collect_traces_for_record(
                            record=record,
                            i_record=i_record,
                            num_traces=num_traces,
                            required_correct=required_correct,
                            sem=sem,
                            judge_session=judge_session,
                        ))
                        for i_record, record in enumerate(random_samples)
                    ]
                    return await asyncio.gather(*jobs)
                finally:
                    if judge_session:
                        await judge_session.close()

            if not _skip_rollouts:
                results = asyncio.run(_collect_all())

                _write_trace_json(results, epoch, result_dir, model_slug, current_time, args)

                for (input_dict, _true_ans, question_traces, question_rewards, question_rewards_raw) in results:
                    if not question_traces:
                        continue

                    if len(question_traces) == 1:
                        advantages = np.array([0.0])
                    else:
                        baseline = float(np.mean(question_rewards))
                        advantages = np.array(question_rewards, dtype=np.float32) - baseline
                        std = float(np.std(advantages))
                        if std > 1e-8:
                            advantages = advantages / std

                    store_traces = not all(abs(a) < 1e-8 for a in advantages)
                    print("printing advantages", advantages, store_traces)

                    if store_traces:
                        for i_trace, trace_data in enumerate(question_traces):
                            agent_selections = trace_data["agent_selections"]
                            hint_selections = trace_data.get("hint_selections", [])
                            agent_logits = trace_data.get("agent_logits", [])
                            hint_logits = trace_data.get("hint_logits", [])
                            agent_outputs_embeddings = trace_data.get("agent_outputs_embeddings", [])

                            gi = {
                                "task": input_dict["task"],
                                "advantage": float(advantages[i_trace]),
                                "agent_selections": agent_selections,
                                "hint_selections": hint_selections,
                                "agent_logits": agent_logits,
                                "hint_logits": hint_logits,
                                "trace_length": len(agent_selections),
                                "agent_outputs_embeddings": agent_outputs_embeddings,
                                "batch_record_idx": -1,
                                "trace_idx_in_record": i_trace,
                            }
                            # Store dynamic pool info for gradient replay
                            if "dynamic_pool" in trace_data:
                                gi["dynamic_pool"] = trace_data["dynamic_pool"]
                            all_gradient_inputs.append(gi)

                    total_epoch_reward += question_rewards_raw[0]
                    trace_count += 1

            avg_epoch_reward = total_epoch_reward / trace_count if trace_count > 0 else 0.0
            log_file.write(f"Sampling complete. Collected {len(all_gradient_inputs)} traces.\n")
            log_file.write(f"Average Accuracy: {avg_epoch_reward:.4f}\n")
            log_file.flush()

            # Dump gradient inputs
            gradients_inputs_path = f"{result_dir}/all_gradients_inputs_pubmedqa_{model_slug}_{current_time}.json"
            with open(gradients_inputs_path, "w") as f:
                json.dump(
                    to_json_safe({
                        "epoch": epoch,
                        "all_gradient_inputs": all_gradient_inputs
                    }),
                    f
                )
                f.write("\n")

            print(f"Sampling complete. Collected {len(all_gradient_inputs)} traces.")
            print(f"Average Reward: {avg_epoch_reward:.4f} Epoch {epoch+1} complete.")

            if epoch % eval_interval != 0 and len(all_gradient_inputs) > 0:
                print("Training network with collected traces...")
                log_file.write("Training network with collected traces...\n")
                log_file.flush()

                torch.set_grad_enabled(True)
                graph.set_eval()

                batch_size = getattr(args, "batch_size", 8)
                reuse_time = getattr(args, "reuse_time", 1)
                sparse_context = getattr(args, "sparse_context", False)

                for reuse_iter in tqdm(range(reuse_time), desc="Training Reused Epochs"):
                    print(f"  Training Reused Epochs {reuse_iter+1}/{args.reuse_time}")
                    random.shuffle(all_gradient_inputs)

                    num_batches = (len(all_gradient_inputs) + batch_size - 1) // batch_size
                    for batch_idx in tqdm(range(num_batches), desc="Training batches"):
                        start = batch_idx * batch_size
                        end = min(start + batch_size, len(all_gradient_inputs))
                        batch = all_gradient_inputs[start:end]

                        optimizer.zero_grad()
                        for gradient_input in batch:
                            graph.run_next_agent_prediction_grad(
                                gradient_input,
                                sparse_context=sparse_context
                            )
                        optimizer.step()

                        if batch_idx % 10 == 0:
                            print(f"    Processed batch {batch_idx+1}/{num_batches}")
                            log_file.write(f"    Processed batch {batch_idx+1}/{num_batches}\n")
                            log_file.flush()


            print(f"Epoch {epoch + 1} complete. Average Accuracy: {avg_epoch_reward:.4f}")
            log_file.write(f"Epoch {epoch + 1} complete. Average Accuracy: {avg_epoch_reward:.4f}\n")
            log_file.write(f"Total Reward: {total_epoch_reward:.4f} Total Count: {trace_count}\n")
            log_file.write("-" * 80 + "\n")
            log_file.flush()

            # Save model
            try:
                save_path = f"{result_dir}/{current_time}"
                os.makedirs(save_path, exist_ok=True)
                print(f"Directory created at: {save_path}")
                graph.save_model(f"{save_path}/{current_time}_{model_slug}_pubmedqa_model_epoch{epoch}.pth")
            except Exception as e:
                print(f"Error saving model: {e}")

            if training_samples < 0:
                break

    # Save final model
    try:
        graph.save_model(f"{result_dir}/{current_time}_{model_slug}_pubmedqa_model.pth")
    except Exception as e:
        print(f"Error saving model: {e}")
