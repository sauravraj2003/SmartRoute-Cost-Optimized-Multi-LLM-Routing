"""Simple comparable routing baselines."""
from __future__ import annotations


def cheapest_model(models, input_tokens: int = 512, output_tokens: int = 512):
    if not models:
        raise ValueError("models cannot be empty")
    from router.policy import estimate_cost
    return min(models, key=lambda m: (estimate_cost(m, input_tokens, output_tokens), m.get("tier", 0)))


def strongest_model(models):
    if not models:
        raise ValueError("models cannot be empty")
    return max(models, key=lambda m: (m.get("tier", 0), m.get("input_cost_per_million_tokens", 0)))


def heuristic_model(models, prompt: str):
    text = prompt.lower()
    tier = 3 if len(prompt) > 5000 or any(w in text for w in ("traceback", "race condition", "security")) else 2 if any(w in text for w in ("debug", "algorithm", "refactor")) else 1
    available = [m for m in models if m.get("enabled", True)]
    return min(available, key=lambda m: (abs(m.get("tier", 1) - tier), m.get("input_cost_per_million_tokens", 0)))
