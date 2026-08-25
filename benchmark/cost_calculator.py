"""Token-based USD cost calculations."""
from __future__ import annotations


def calculate_cost(input_tokens: int, output_tokens: int, input_rate: float, output_rate: float) -> float:
    if min(input_tokens, output_tokens, input_rate, output_rate) < 0:
        raise ValueError("Token counts and rates must be non-negative")
    return input_tokens * input_rate / 1_000_000 + output_tokens * output_rate / 1_000_000


def model_cost(model: dict, input_tokens: int, output_tokens: int) -> float:
    from router.policy import estimate_cost
    return estimate_cost(model, input_tokens, output_tokens)
