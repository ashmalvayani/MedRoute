"""
AnyMAC-Improved: Entry point for PubMedQA router training and evaluation.

Same improvement pipeline as run_medqa.py but for PubMedQA (yes/no/maybe = A/B/C).

Example:
  python experiments/run_pubmedqa.py \
    --llm_name Qwen/Qwen3-8B \
    --judge_model Qwen/Qwen3-32B \
    --dynamic_prompts --dynamic_pool \
    --epochs 3 --train_num 100 --max_routing 3
"""

from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import argparse
from pathlib import Path

from typing import Union, Literal, List
import random
import numpy as np
import torch
import time

from GDesigner.graph.graph import Graph
from datasets_my.pubmedqa_dataset import PubMedQADataset
from GDesigner.prompt.pubmedqa_prompt_set import ROLES, SPECIALISTS
from GDesigner.utils.const import GDesigner_ROOT
from GDesigner.utils.globals import Time

try:
    from GDesigner.domain.pubmedqa_domain import PubMedQADomain
except Exception:
    PubMedQADomain = None  # type: ignore

from experiments.train_pubmedqa import train
from experiments.evaluate_pubmedqa import evaluate


def _apply_llm_temperature(llm_temperature: float | None):
    """Set the default LLM generation temperature globally before any LLM calls."""
    if llm_temperature is not None:
        from GDesigner.llm.llm import LLM
        LLM.DEFAULT_TEMPERATURE = llm_temperature
        print(f"[config] LLM generation temperature set to {llm_temperature}")


def _apply_improvements(args):
    """Wire up the improvement toggles globally."""
    # 1. Dynamic specialist prompts
    if getattr(args, 'dynamic_prompts', False):
        import GDesigner.agents.analyze_agent as aa
        aa.DYNAMIC_PROMPTS_ENABLED = True
        aa.JUDGE_MODEL_FOR_PROMPTS = args.judge_model
        if getattr(args, 'prompt_model', None):
            aa.PROMPT_MODEL_FOR_PROMPTS = args.prompt_model
            print(f"[improved] Dynamic specialist prompts ENABLED (prompt_model={args.prompt_model} on BASE_URL)")
        else:
            print(f"[improved] Dynamic specialist prompts ENABLED (judge={args.judge_model})")

    # 2. Partial credit reward shaping
    if getattr(args, 'partial_credit', False):
        import experiments.train_pubmedqa as tp
        tp.PARTIAL_CREDIT_ENABLED = True
        tp.REWARD_ALPHA = getattr(args, 'reward_alpha', 0.4)
        print(f"[improved] Partial credit rewards ENABLED (alpha={tp.REWARD_ALPHA})")

    # 3. Structured hint passing
    if getattr(args, 'structured_hints', False):
        import GDesigner.graph.graph as gg
        gg.STRUCTURED_HINTS_ENABLED = True
        print(f"[improved] Structured hint passing ENABLED")

    # 4. Dynamic specialist pool
    if getattr(args, 'dynamic_pool', False):
        import GDesigner.graph.graph as gg
        gg.DYNAMIC_POOL_ENABLED = True
        gg.DYNAMIC_POOL_JUDGE_MODEL = args.judge_model
        print(f"[improved] Dynamic specialist pool ENABLED (judge={args.judge_model})")


def build_graph(args):
    domain = args.domain

    roles = SPECIALISTS
    if getattr(args, 'top_k_specialists', None) is not None:
        roles = SPECIALISTS[:args.top_k_specialists]
        print(f"Using top {args.top_k_specialists} specialists: {roles}")

    graph = Graph(
        domain=domain,
        llm_name=args.llm_name,
        agent_names=args.agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        use_transformer=True,
        max_routing=args.max_routing,
        available_roles=roles,
    )
    return graph


def main():
    p = argparse.ArgumentParser(description="AnyMAC-Improved PubMedQA Router")

    p.add_argument("--result_dir", type=str, default="result/pubmedqa_improved")

    p.add_argument('--mode', type=str, default='FullConnected',
                   choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star', 'Mesh',
                            'FakeFullConnected', 'FakeRandom', 'FakeChain', 'FakeStar', 'FakeMesh', 'FakeAGRandom', 'FakeAGFull'])
    p.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'])
    p.add_argument('--agent_nums', nargs='+', type=int, default=[1])
    p.add_argument('--domain', type=str, default="pubmedqa")

    p.add_argument("--llm_name", type=str, default="qwen3:8b-fp16")
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--finetune_path", type=str, default=None)
    p.add_argument("--decision_method", type=str, default="FinalRefer")

    p.add_argument('--num_rounds', type=int, default=1)

    # Router + training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--train_num", type=int, default=100)
    p.add_argument("--train_split", type=str, default="dev")
    p.add_argument("--training_samples", type=int, default=10**9)
    p.add_argument("--num_traces", type=int, default=8)
    p.add_argument("--required_correct_answers", type=int, default=1)

    # Routing behavior
    p.add_argument("--max_routing", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max_context", type=int, default=2048)
    p.add_argument("--decay_factor", type=float, default=0.98)

    # Optimization tricks
    p.add_argument("--reuse_time", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sparse_context", action="store_true")
    p.add_argument("--cos_scaling", type=float, default=1.5)
    p.add_argument("--eval_interval", type=int, default=100)

    # GDesigner options
    p.add_argument("--optimized_spatial", action="store_true")
    p.add_argument("--optimized_temporal", action="store_true")
    p.add_argument("--agent_group_type", type=str, default="AnalyzeAgent")

    # Parallelism
    p.add_argument("--trace_parallelism", type=int, default=8)

    # Judge model
    p.add_argument("--judge_model", type=str, default=None)

    # Resume
    p.add_argument("--resume_gradient_path", type=str, default=None)
    p.add_argument("--early_stop_rollouts", action="store_true", default=False)
    p.add_argument("--top_k_specialists", type=int, default=None)
    p.add_argument("--zero_shot", action="store_true", default=False)

    # LLM generation temperature
    p.add_argument("--llm_temperature", type=float, default=None)

    # =====================================================
    # AnyMAC-Improved: New flags
    # =====================================================
    p.add_argument("--dynamic_prompts", action="store_true", default=False,
                   help="Enable dynamic specialist prompts generated by judge model per question.")
    p.add_argument("--prompt_model", type=str, default=None,
                   help="Model to generate dynamic prompts (uses BASE_URL). If unset, uses judge_model on JUDGE_BASE_URL.")
    p.add_argument("--partial_credit", action="store_true", default=False,
                   help="Enable partial credit reward shaping (reasoning quality 1-5).")
    p.add_argument("--reward_alpha", type=float, default=0.4,
                   help="Weight for binary correctness in partial credit (1-alpha for reasoning). Default 0.4.")
    p.add_argument("--structured_hints", action="store_true", default=False,
                   help="Enable structured hint passing between specialists.")
    p.add_argument("--dynamic_pool", action="store_true", default=False,
                   help="Enable dynamic specialist pool: judge generates 5-7 question-specific specialists per question.")
    p.add_argument("--entropy_beta", type=float, default=0.05,
                   help="Entropy regularization coefficient for routing. Prevents mode collapse. Default 0.05.")
    p.add_argument("--eval_limit", type=int, default=1100,
                   help="Max number of eval questions. Default 1100 (PubMedQA has 1000 test).")
    p.add_argument("--eval_temperature", type=float, default=0.5,
                   help="Temperature for eval softmax sampling. Lower = more deterministic. Default 0.5.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility. Default 42.")
    p.add_argument("--use_context", action="store_true", default=False,
                   help="Use context-enriched PubMedQA splits (with research abstracts).")
    p.add_argument("--eval_no_context", action="store_true", default=False,
                   help="Also evaluate without context (using original test split).")

    args = p.parse_args()

    # Apply LLM temperature
    _apply_llm_temperature(args.llm_temperature)

    # Apply improvement toggles
    _apply_improvements(args)

    # Print active improvements
    improvements = []
    if args.dynamic_prompts:
        improvements.append("dynamic_prompts")
    if args.partial_credit:
        improvements.append(f"partial_credit(alpha={args.reward_alpha})")
    if args.structured_hints:
        improvements.append("structured_hints")
    if getattr(args, 'dynamic_pool', False):
        improvements.append("dynamic_pool")
    if improvements:
        print(f"[AnyMAC-Improved] Active: {', '.join(improvements)}")
    else:
        print(f"[AnyMAC-Improved] No improvements enabled (baseline mode)")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    result_file = None
    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{GDesigner_ROOT}/{args.result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.domain}_{args.llm_name.replace('/', '-')}_{current_time}.json"

    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode, len(agent_names))
    limit_questions = getattr(args, 'eval_limit', 1100)

    if getattr(args, 'use_context', False):
        train_split = 'context/' + getattr(args, 'train_split', 'dev')
        val_split = 'context/test'
    else:
        train_split = getattr(args, 'train_split', 'dev')
        val_split = 'test'
    dataset_train = PubMedQADataset(train_split)
    dataset_val = PubMedQADataset(val_split)

    print(f"Training Dataset Length: {len(dataset_train)}, Validation Dataset Length: {len(dataset_val)}")

    graph = build_graph(args)
    graph.entropy_beta = getattr(args, 'entropy_beta', 0.05)
    graph.eval_temperature = getattr(args, 'eval_temperature', 0.5)
    print(f"Graph Constructed (entropy_beta={graph.entropy_beta}, eval_temperature={graph.eval_temperature})")

    if args.zero_shot:
        print("Zero-shot evaluation with randomly initialized router.")
        graph.to_device(torch.device("cpu"))
        evaluate(graph=graph, dataset=dataset_val, limit_questions=limit_questions, result_file=result_file, result_dir=result_dir, args=args)
    elif args.model_path:
        print(f"Loading pre-trained router from: {args.model_path}")
        graph = Graph.load_model(args.model_path)
        graph.entropy_beta = getattr(args, 'entropy_beta', 0.05)
        graph.eval_temperature = getattr(args, 'eval_temperature', 0.5)
        print("Skipping training — running evaluation only.")
        graph.to_device(torch.device("cpu"))
        evaluate(graph=graph, dataset=dataset_val, limit_questions=limit_questions, result_file=result_file, result_dir=result_dir, args=args)
    elif args.finetune_path:
        print(f"Loading checkpoint for fine-tuning from: {args.finetune_path}")
        graph = Graph.load_model(args.finetune_path)
        print("Continuing training from checkpoint.")
        train(graph=graph, dataset=dataset_train, result_dir=result_dir, args=args)
        print("Fine-tuning complete. Skipping auto-eval — use phase scripts to evaluate checkpoints.")
    else:
        train(graph=graph, dataset=dataset_train, result_dir=result_dir, args=args)
        print("Training complete. Skipping auto-eval — use phase scripts to evaluate checkpoints.")


def get_kwargs(mode: Union[Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'],
                           Literal['Debate'], Literal['Layered'], Literal['Star'], Literal['Mesh'],
                           Literal['FakeFullConnected'], Literal['FakeRandom'], Literal['FakeChain'],
                           Literal['FakeStar'], Literal['FakeMesh'], Literal['FakeAGRandom'], Literal['FakeAGFull']],
               N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0] * N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i + 1, N):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode == 'FullConnected' or mode == 'FakeFullConnected' or mode == 'FakeAGFull':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode == 'FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode == 'FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]

    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs": node_kwargs}


if __name__ == "__main__":
    main()
