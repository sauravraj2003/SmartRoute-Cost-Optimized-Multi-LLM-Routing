"""Shared routing and quality metrics."""
from __future__ import annotations

import numpy as np


def summarize_results(rows):
    if not rows:
        return {"n": 0, "success_rate": 0.0, "mean_cost_usd": 0.0, "cost_per_success_usd": None, "p50_latency_ms": 0.0, "p95_latency_ms": 0.0}
    success = np.asarray([bool(r.get("success", False)) for r in rows], dtype=float)
    costs = np.asarray([float(r.get("cost_usd", 0)) for r in rows], dtype=float)
    latency = np.asarray([float(r.get("latency_ms", 0)) for r in rows], dtype=float)
    summary = {"n": len(rows), "success_rate": float(success.mean()), "mean_cost_usd": float(costs.mean()),
            "cost_per_success_usd": float(costs.sum() / success.sum()) if success.sum() else None,
            "p50_latency_ms": float(np.percentile(latency, 50)), "p95_latency_ms": float(np.percentile(latency, 95)),
            "escalation_rate": float(np.mean([r.get("escalated", False) for r in rows]))}
    if any("first_route_success" in row for row in rows):
        summary["first_route_success_rate"] = float(np.mean([row["first_route_success"] for row in rows]))
        summary["mean_escalation_extra_cost_usd"] = float(np.mean([row.get("extra_cost_usd", 0) for row in rows]))
    calibrated = [row for row in rows if row.get("predicted_success") is not None and row.get("first_route_success") is not None]
    if calibrated:
        summary["first_route_brier_score"] = brier_score(
            [row["predicted_success"] for row in calibrated], [row["first_route_success"] for row in calibrated])
    return summary


def brier_score(probabilities, outcomes) -> float:
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    if p.size != y.size or p.size == 0:
        raise ValueError("probabilities and outcomes must have equal nonzero length")
    return float(np.mean((p - y) ** 2))
