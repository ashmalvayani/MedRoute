import argparse
import subprocess
import os
import sys
from typing import List

# Models supported
OPENAI_MODELS = ["gpt-4o-2024-11-20", "o3-mini", "gpt-4-1106-preview", "gpt-4.1-mini"]
SMALL_LLM_MODELS = [
    "qwen2.5:0.5b-instruct-fp16",
    "llama3.2:1b-instruct-fp16",
    "qwen2.5:1.5b-instruct-fp16",
    # "gemma3:1b-it-fp16", not working
    "qwen3:0.6b-fp16",
    "temp0-qwen2.5-1.5b",
    "deepseek-r1:1.5b-qwen-distill-fp16"  # Default model
]


def run_medqa_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the MedQA benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_medqa.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_pubmedqa_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the PubMedQA benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_pubmedqa.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_pmcvqa_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the PubMedQA benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_pmcvqa.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""

    subprocess.run(cmd, env=env)


def run_btmri_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the BTMRI benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_btmri.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""

    subprocess.run(cmd, env=env)

def run_chestxray_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the ChestXray benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_chestxray.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""

    subprocess.run(cmd, env=env)

def run_deeplesion_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the BTMRI benchmark with the specified LLM model
    {method}/
    """
    cmd = [
        "python", f"experiments/run_deeplesion.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM8K benchmarks with different LLM models")
    parser.add_argument("--llm_name", type=str, default="deepseek-r1:1.5b-qwen-distill-fp16",
                        help="Name of the LLM model to use")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=['medqa', 'pubmedqa', 'pmcvqa', 'btmri', 'chestxray', 'deeplesion'],
                        help="Dataset to use for benchmark")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of iterations for benchmark")
    parser.add_argument("--mode", type=str, default="FullConnected",
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation")
    parser.add_argument("--models", nargs="+", type=str,
                        help="Specific models to run benchmarks with")
    parser.add_argument("--method", type=str, default="GDesigner",
                        choices=['GDesigner', 'NAP'],
                        help="Method to use for benchmark")
    
    args = parser.parse_args()
    
    if args.dataset == "medqa":
        run_medqa_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "pubmedqa":
        run_pubmedqa_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == 'pmcvqa':
        run_pmcvqa_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == 'btmri':
        run_btmri_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == 'chestxray':
        run_chestxray_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == 'deeplesion':
        run_deeplesion_benchmark(args.llm_name, args.mode, args.method)
    