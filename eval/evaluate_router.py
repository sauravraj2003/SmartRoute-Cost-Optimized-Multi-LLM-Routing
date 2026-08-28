"""Evaluate fixed baselines and the learned policy on held-out model-matrix rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml
from features.feature_pipeline import FeaturePipeline

from eval.metrics import summarize_results
from router.baselines import cheapest_model, heuristic_model, strongest_model
from router.escalation import escalation_candidates
from router.policy import estimate_cost, select_model
from router.train import SuccessPredictor
from router.config import load_models_config


def evaluate(rows: pd.DataFrame, models: list[dict], threshold: float = .72,
             model_dir: str | Path = "router/models", max_cost_usd: float | None = None,
             max_escalations: int = 0, latency_budget_ms: int | None = None,
             escalation_order: list[int] | None = None,
             pipeline: FeaturePipeline | None = None,
             apply_calibration: bool = True) -> dict:
    results = {}
    lookup = rows.set_index(["prompt_id", "model_id"])
    task_rows = rows.sort_values("prompt_id").drop_duplicates("prompt_id")

    def run_selector(name, selector):
        selected_records = []
        for row in task_rows.itertuples():
            available = [model for model in models if (row.prompt_id, model["id"]) in lookup.index]
            if not available:
                continue
            model = selector(row, available)
            key = (row.prompt_id, model["id"])
            record = lookup.loc[key]
            selected_records.append({"model": model, "success": bool(record.success),
                                     "cost_usd": float(record.cost_usd), "latency_ms": float(record.latency_ms)})
        results[name] = summarize_results(selected_records)

    run_selector("cheapest", lambda row, available: cheapest_model(
                 available, max(1, (len(row.prompt) + 3) // 4), 512))
    run_selector("strongest", lambda row, available: strongest_model(available))
    run_selector("heuristic", lambda row, available: heuristic_model(available, row.prompt))

    predictor = SuccessPredictor(model_dir, pipeline=pipeline, apply_calibration=apply_calibration)
    routed_records = []
    for row in task_rows.itertuples():
        task_models = [model for model in models if (row.prompt_id, model["id"]) in lookup.index]
        if not task_models:
            continue
        predictions = predictor.predict(row.prompt, [model["id"] for model in task_models])
        try:
            decision = select_model(task_models, predictions, threshold,
                                    input_tokens=max(1, (len(row.prompt) + 3) // 4),
                                    expected_output_tokens=512, max_cost_usd=max_cost_usd,
                                    latency_budget_ms=latency_budget_ms)
        except ValueError:
            continue
        current = decision.model
        attempted = {current["id"]}
        calls_cost, calls_latency = 0.0, 0.0
        first_route_cost = None
        first_route_success = None
        record = None
        escalated = False
        for attempt in range(max_escalations + 1):
            key = (row.prompt_id, current["id"])
            if key not in lookup.index:
                break
            record = lookup.loc[key]
            calls_cost += float(record.cost_usd)
            calls_latency += float(record.latency_ms)
            if attempt == 0:
                first_route_cost = float(record.cost_usd)
                first_route_success = bool(record.success)
            if bool(record.success) or attempt >= max_escalations:
                break
            stronger = escalation_candidates(task_models, current["id"], attempted,
                                              escalation_order=escalation_order)
            if not stronger:
                break
            if max_cost_usd is not None and calls_cost + estimate_cost(
                    stronger[0], max(1, (len(row.prompt) + 3) // 4), 512) > max_cost_usd:
                break
            current = stronger[0]
            attempted.add(current["id"])
            escalated = True
        if record is not None:
            routed_records.append({"model": current, "success": bool(record.success), "cost_usd": calls_cost,
                                  "latency_ms": calls_latency, "escalated": escalated,
                                  "predicted_success": predictions.get(decision.model["id"], 0),
                                  "first_route_success": first_route_success,
                                  "extra_cost_usd": calls_cost - (first_route_cost or 0.0)})
    results["smartroute"] = summarize_results(routed_records)
    reference = results.get("strongest", {})
    reference_cost = reference.get("mean_cost_usd") or 0.0
    reference_latency = reference.get("p95_latency_ms") or 0.0
    results["comparison_vs_strongest"] = {
        name: {
            "success_delta_percentage_points": (metric["success_rate"] - reference["success_rate"]) * 100
            if reference.get("n") and metric.get("n") else None,
            "cost_reduction_percent": (reference_cost - metric["mean_cost_usd"]) / reference_cost * 100
            if reference_cost and metric.get("n") else None,
            "p95_latency_reduction_percent": (reference_latency - metric["p95_latency_ms"]) / reference_latency * 100
            if reference_latency and metric.get("n") else None,
        }
        for name, metric in results.items() if name != "comparison_vs_strongest"
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="benchmark/results/matrix.csv")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--pricing", default="configs/pricing.yaml")
    parser.add_argument("--model-dir", default="router/models")
    parser.add_argument("--routing-config", default="configs/routing.yaml")
    parser.add_argument("--mode", default="balanced", choices=["cost_first", "balanced", "quality_first"])
    parser.add_argument("--output", default="eval/results.json")
    parser.add_argument("--pareto-plot", help="Optional PNG path for the cost/quality Pareto plot")
    args = parser.parse_args()
    source = Path(args.input)
    rows = pd.read_parquet(source) if source.suffix == ".parquet" else pd.read_csv(source)
    split_path = Path(args.model_dir) / "split.json"
    if split_path.exists():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        held_out = {str(value) for value in split.get("task_ids", {}).get("test", [])}
        overlap = held_out.intersection(set(rows.prompt_id.astype(str)))
        if overlap:
            rows = rows[rows.prompt_id.astype(str).isin(held_out)]
    models = load_models_config(args.models, args.pricing)
    with open(args.routing_config, encoding="utf-8") as stream:
        routing = yaml.safe_load(stream)
    mode = routing["modes"][args.mode]
    result = evaluate(rows, models, float(mode["success_threshold"]), args.model_dir,
                      mode.get("max_cost_usd"),
                      int(mode.get("max_escalations", 0)) if mode.get("test_based_escalation", True) else 0,
                      routing.get("latency_budget_ms"), routing.get("escalation_order"))
    points = [{"policy": name, **metrics} for name, metrics in result.items()
              if isinstance(metrics, dict) and metrics.get("n", 0) > 0
              and "mean_cost_usd" in metrics and "success_rate" in metrics]
    from eval.pareto import pareto_frontier
    result["pareto_frontier"] = pareto_frontier(points)
    if args.pareto_plot:
        from eval.pareto import plot_pareto
        plot_pareto(points, args.pareto_plot)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
