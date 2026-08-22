"""Cost and latency constrained model selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RoutingDecision:
    model: dict[str, Any]
    predicted_success: float
    estimated_cost_usd: float
    reason: str


def estimate_cost(model: dict[str, Any], input_tokens: int, output_tokens: int) -> float:
    input_rate = float(model.get("input_cost_per_million_tokens", 0))
    output_rate = float(model.get("output_cost_per_million_tokens", 0))
    threshold = model.get("long_context_threshold_tokens")
    if threshold is not None and input_tokens > int(threshold):
        input_rate *= float(model.get("long_context_input_multiplier", 1))
        output_rate *= float(model.get("long_context_output_multiplier", 1))
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def select_model(models: list[dict[str, Any]], predictions: dict[str, float],
                 success_threshold: float = .72, input_tokens: int = 512,
                 expected_output_tokens: int = 512, max_cost_usd: float | None = None,
                 latency_budget_ms: int | None = None) -> RoutingDecision:
    candidates = [m for m in models if m.get("enabled", True)
                  and (m.get("context_window") is None or
                       input_tokens + expected_output_tokens <= int(m["context_window"]))]
    if not candidates:
        raise ValueError("No enabled model has enough context for the estimated request")
    costs = {m["id"]: estimate_cost(m, input_tokens, expected_output_tokens) for m in candidates}
    eligible = [m for m in candidates if float(predictions.get(m["id"], 0)) >= success_threshold
                and (max_cost_usd is None or costs[m["id"]] <= max_cost_usd)
                and (latency_budget_ms is None or (m.get("typical_latency_ms", {}).get("p95") is not None
                                                   and m["typical_latency_ms"]["p95"] <= latency_budget_ms))]
    fallback_reason = "lowest-cost model meeting success, cost and latency constraints"
    if not eligible:
        eligible = [m for m in candidates if (max_cost_usd is None or costs[m["id"]] <= max_cost_usd)
                    and (latency_budget_ms is None or (m.get("typical_latency_ms", {}).get("p95") is not None
                                                       and m["typical_latency_ms"]["p95"] <= latency_budget_ms))]
        fallback_reason = "no model met the success threshold; selected strongest feasible fallback"
    if not eligible:
        raise ValueError("No enabled model meets the configured cost and latency limits")
    strongest = max(eligible, key=lambda m: (m.get("tier", 0), predictions.get(m["id"], 0)))
    model = min(eligible, key=lambda m: (costs[m["id"]], -predictions.get(m["id"], 0))) if fallback_reason.startswith("lowest") else strongest
    return RoutingDecision(model, float(predictions.get(model["id"], 0)), costs[model["id"]], fallback_reason)


class RoutingPolicy:
    """Config-backed policy wrapper, convenient for integrations and tests."""
    def __init__(self, models: list[dict[str, Any]], threshold: float = .72,
                 max_cost_usd: float | None = None, latency_budget_ms: int | None = None) -> None:
        self.models, self.threshold = models, threshold
        self.max_cost_usd, self.latency_budget_ms = max_cost_usd, latency_budget_ms

    def route(self, predictions: dict[str, float], input_tokens: int = 512,
              expected_output_tokens: int = 512) -> RoutingDecision:
        return select_model(self.models, predictions, self.threshold, input_tokens,
                            expected_output_tokens, self.max_cost_usd, self.latency_budget_ms)
