"""Tier-ordered escalation primitives."""
from __future__ import annotations

from typing import Any


def escalation_candidates(models: list[dict[str, Any]], current_model_id: str,
                          attempted: set[str] | None = None, max_escalations: int | None = None,
                          escalation_order: list[int] | None = None) -> list[dict[str, Any]]:
    attempted = attempted or set()
    current = next((m for m in models if m["id"] == current_model_id), None)
    if current is None:
        raise ValueError(f"Unknown model id: {current_model_id}")
    order = escalation_order or sorted({int(m.get("tier", 0)) for m in models})
    rank = {tier: index for index, tier in enumerate(order)}
    current_rank = rank.get(int(current.get("tier", 0)), -1)
    remaining = sorted((m for m in models if m.get("enabled", True) and m["id"] not in attempted
                        and rank.get(int(m.get("tier", 0)), -1) > current_rank),
                       key=lambda m: (rank.get(int(m.get("tier", 0)), len(order)),
                                      m.get("input_cost_per_million_tokens", 0)))
    return remaining if max_escalations is None else remaining[:max(0, max_escalations)]


def next_model(models: list[dict[str, Any]], current_model_id: str, attempted: set[str] | None = None):
    candidates = escalation_candidates(models, current_model_id, attempted)
    return candidates[0] if candidates else None
