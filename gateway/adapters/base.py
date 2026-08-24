"""Provider-neutral adapter result and interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any] | None = None


class ModelAdapter(ABC):
    def __init__(self, model: dict[str, Any], api_key: str | None = None,
                 timeout: float = 90.0) -> None:
        self.model, self.api_key, self.timeout = model, api_key, timeout

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerationResult:
        """Generate one completion and normalize usage and latency."""

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return ((input_tokens * float(self.model.get("input_cost_per_million_tokens", 0)) +
                 output_tokens * float(self.model.get("output_cost_per_million_tokens", 0))) / 1_000_000)
