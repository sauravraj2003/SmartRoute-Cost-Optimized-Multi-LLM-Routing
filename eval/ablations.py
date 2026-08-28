"""Train and compare the planned SmartRoute routing ablations on fixed splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from eval.evaluate_router import evaluate
from features.feature_pipeline import FeaturePipeline
from router.train import train_router


def ablation_summary(rows, columns=("success", "cost_usd", "latency_ms")) -> dict[str, dict[str, float | int]]:
    """Aggregate a dataframe containing an ``ablation`` label per result row."""
    if "ablation" not in rows:
        raise ValueError("Ablation results must include an 'ablation' column")
    summaries = {}
    for name, subset in rows.groupby("ablation"):
        summary = {column: float(subset[column].mean()) for column in columns if column in subset}
        summary["n"] = int(len(subset))
        summaries[str(name)] = summary
    return summaries


def run_ablation_suite(matrix: pd.DataFrame, models: list[dict[str, Any]], split: dict[str, Any],
                       routing: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Fit feature variants on train/validation rows and score only held-out test tasks."""
    task_ids = split.get("task_ids", {})
    train_ids = {str(value) for value in task_ids.get("train", [])}
    validation_ids = {str(value) for value in task_ids.get("validation", [])}
    test_ids = {str(value) for value in task_ids.get("test", [])}
    if not train_ids or not test_ids:
        raise ValueError("split.json must contain non-empty train and test task IDs")
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Training, validation, and test task IDs must be disjoint")
    ids = matrix["prompt_id"].astype(str)
    train_rows = matrix[ids.isin(train_ids)]
    validation_rows = matrix[ids.isin(validation_ids)]
    test_rows = matrix[ids.isin(test_ids)]
    if train_rows.empty or test_rows.empty:
        raise ValueError("The matrix must contain both train and held-out test task rows")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mode_configs = routing.get("modes", {})
    balanced = mode_configs.get("balanced", {})
    max_escalations = int(balanced.get("max_escalations", 0))
    common = {
        "max_cost_usd": balanced.get("max_cost_usd"),
        "latency_budget_ms": routing.get("latency_budget_ms"),
        "escalation_order": routing.get("escalation_order"),
    }
    results: dict[str, Any] = {}

    # Compare heuristic with the learned full-feature policy on identical test tasks.
    full_pipeline = FeaturePipeline()
    full_dir = output / "combined_features"
    full_artifacts = train_router(train_rows, full_dir, pipeline=full_pipeline,
                                  validation_rows=validation_rows)
    results["heuristic_vs_learned"] = evaluate(
        test_rows, models, float(balanced.get("success_threshold", .72)), full_dir,
        max_escalations=max_escalations, pipeline=full_pipeline, **common)
    results["heuristic_vs_learned"]["calibrated_models"] = [
        model_id for model_id, artifact in full_artifacts.items() if artifact["calibrator"] is not None]
    results["combined_features"] = results["heuristic_vs_learned"]["smartroute"]

    feature_variants = {
        "embeddings_only": FeaturePipeline(include_embeddings=True, include_scalar=False,
                                            include_task_type=False),
        "scalar_only": FeaturePipeline(include_embeddings=False, include_scalar=True,
                                        include_task_type=False),
    }
    for name, pipeline in feature_variants.items():
        model_dir = output / name
        train_router(train_rows, model_dir, pipeline=pipeline, validation_rows=validation_rows)
        results[name] = evaluate(test_rows, models, float(balanced.get("success_threshold", .72)),
                                 model_dir, max_escalations=max_escalations, pipeline=pipeline,
                                 **common)

    results["calibrated_vs_uncalibrated"] = {
        "calibrated": evaluate(test_rows, models, float(balanced.get("success_threshold", .72)),
                                full_dir, max_escalations=max_escalations,
                                pipeline=full_pipeline, apply_calibration=True, **common),
        "uncalibrated": evaluate(test_rows, models, float(balanced.get("success_threshold", .72)),
                                  full_dir, max_escalations=max_escalations,
                                  pipeline=full_pipeline, apply_calibration=False, **common),
    }
    results["escalation"] = {
        "disabled": evaluate(test_rows, models, float(balanced.get("success_threshold", .72)),
                              full_dir, max_escalations=0, pipeline=full_pipeline, **common),
        "enabled": evaluate(test_rows, models, float(balanced.get("success_threshold", .72)),
                             full_dir, max_escalations=max_escalations,
                             pipeline=full_pipeline, **common),
    }
    results["policy_modes"] = {}
    for mode_name in ("cost_first", "balanced", "quality_first"):
        config = mode_configs.get(mode_name)
        if config is None:
            continue
        results["policy_modes"][mode_name] = evaluate(
            test_rows, models, float(config.get("success_threshold", .72)), full_dir,
            max_cost_usd=config.get("max_cost_usd"),
            max_escalations=(int(config.get("max_escalations", 0))
                             if config.get("test_based_escalation", True) else 0),
            latency_budget_ms=common["latency_budget_ms"],
            escalation_order=common["escalation_order"], pipeline=full_pipeline)
    results["metadata"] = {
        "n_train_tasks": len(train_ids), "n_validation_tasks": len(validation_ids),
        "n_test_tasks": len(test_ids), "test_task_ids": sorted(test_ids),
        "split_seed": split.get("seed"),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="benchmark/results/matrix.csv")
    parser.add_argument("--model-dir", default="router/models")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    parser.add_argument("--routing-config", default="configs/routing.yaml")
    parser.add_argument("--output", default="eval/ablations.json")
    args = parser.parse_args()

    source = Path(args.input)
    matrix = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    split_file = Path(args.model_dir) / "split.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Training split not found: {split_file}; train the router first")
    split = json.loads(split_file.read_text(encoding="utf-8"))
    from router.config import load_models_config
    models = load_models_config(args.models, args.pricing)
    with Path(args.routing_config).open(encoding="utf-8") as stream:
        routing = yaml.safe_load(stream) or {}
    results = run_ablation_suite(matrix, models, split, routing, Path(args.model_dir) / "ablations")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
