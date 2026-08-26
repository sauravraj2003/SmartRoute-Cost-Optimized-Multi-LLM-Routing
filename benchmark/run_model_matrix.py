"""Run every dataset task against every enabled configured model."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from benchmark.cost_calculator import model_cost
from benchmark.evaluators import evaluate_code
from data.load_humaneval import load_humaneval
from data.load_mbpp import load_mbpp
from gateway.adapters import AnthropicAdapter, LocalAdapter, OpenAIAdapter
from router.config import load_models_config

load_dotenv()


def load_models(path: str, pricing_path: str = "configs/pricing.yaml") -> list[dict]:
    return [m for m in load_models_config(path, pricing_path) if m.get("enabled", True)]


def adapter_for(model):
    provider = model.get("provider", "openai").lower()
    return (AnthropicAdapter if provider == "anthropic" else LocalAdapter if provider == "local" else OpenAIAdapter)(model)


async def evaluate_pair(task: dict, model: dict, semaphore: asyncio.Semaphore) -> dict:
    prompt = task["prompt"]
    started = time.perf_counter()
    try:
        async with semaphore:
            result = await adapter_for(model).generate([{"role": "user", "content": prompt}], temperature=0,
                                                       max_tokens=int(os.getenv("BENCHMARK_MAX_TOKENS", "2048")))
        measured_latency = (time.perf_counter() - started) * 1000
        verification_latency = 0.0
        if not task.get("tests"):
            evaluation = {"success": False, "quality_score": 0.0, "passed_tests": 0, "total_tests": 0,
                          "evaluation_error": "task has no tests; response not scored"}
        else:
            try:
                verification_started = time.perf_counter()
                evaluation = await asyncio.to_thread(
                    evaluate_code, result.text, task["tests"], task.get("entry_point", ""),
                    5.0, prompt if task.get("dataset") == "humaneval" else "")
                verification_latency = (time.perf_counter() - verification_started) * 1000
            except Exception as exc:
                verification_latency = (time.perf_counter() - verification_started) * 1000
                evaluation = {"success": False, "quality_score": 0.0, "passed_tests": 0,
                              "total_tests": 0, "evaluation_error": str(exc)}
        input_tokens = result.input_tokens or max(1, (len(prompt) + 3) // 4)
        output_tokens = result.output_tokens or max(1, (len(result.text) + 3) // 4)
        return {"prompt_id": task["prompt_id"], "dataset": task["dataset"], "prompt": prompt,
                "group_id": task.get("group_id") or (task.get("metadata") or {}).get("group_id", task["prompt_id"]),
                "model_id": model["id"], "response": result.text, **{k: v for k, v in evaluation.items() if k != "sandbox"},
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "usage_estimated": not bool(result.input_tokens and result.output_tokens),
                "generation_latency_ms": result.latency_ms or measured_latency,
                "verification_latency_ms": verification_latency,
                "latency_ms": (result.latency_ms or measured_latency) + verification_latency,
                "cost_usd": model_cost(model, input_tokens, output_tokens),
                "error": evaluation.get("evaluation_error")}
    except Exception as exc:
        return {"prompt_id": task["prompt_id"], "dataset": task["dataset"], "prompt": prompt,
                "group_id": task.get("group_id") or (task.get("metadata") or {}).get("group_id", task["prompt_id"]),
                "model_id": model["id"], "response": "", "success": False, "quality_score": 0.0,
                "passed_tests": 0, "total_tests": 0, "input_tokens": 0, "output_tokens": 0,
                "usage_estimated": False,
                "generation_latency_ms": (time.perf_counter() - started) * 1000,
                "verification_latency_ms": 0.0,
                "latency_ms": (time.perf_counter() - started) * 1000, "cost_usd": 0.0, "error": str(exc)}


async def run_matrix(tasks: list[dict], models: list[dict], limit: int | None = None,
                     concurrency: int = 4) -> pd.DataFrame:
    tasks = tasks[:limit] if limit else tasks
    semaphore = asyncio.Semaphore(concurrency)
    rows = await asyncio.gather(*(evaluate_pair(task, model, semaphore) for task in tasks for model in models))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--data", help="Local JSONL source; defaults to Hugging Face dataset")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    parser.add_argument("--output", default="benchmark/results/matrix.csv")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    loader = load_humaneval if args.dataset == "humaneval" else load_mbpp
    tasks = loader(args.data, args.split, args.limit)
    frame = asyncio.run(run_matrix(tasks, load_models(args.models, args.pricing), concurrency=max(1, args.concurrency)))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
    else:
        frame.to_csv(target, index=False)
    print(f"Wrote {len(frame)} model-task rows to {target}")


if __name__ == "__main__":
    main()
