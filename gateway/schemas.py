"""Validated gateway request and response contracts."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    mode: str | None = None
    model: str | None = None
    temperature: float = Field(default=.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, gt=0, le=32768,
                            validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    stream: bool = False
    tests: str | None = Field(default=None, max_length=100_000,
                              description="Optional test code, executed only in the isolated Docker sandbox")
    entry_point: str = Field(default="", max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def normalized_messages(self) -> list[dict[str, str]]:
        if self.messages:
            return [message.model_dump() for message in self.messages]
        if self.prompt:
            return [{"role": "user", "content": self.prompt}]
        raise ValueError("Provide either prompt or messages")

    def user_prompt(self) -> str:
        return "\n".join(m["content"] for m in self.normalized_messages() if m["role"] == "user")


class RouteResponse(BaseModel):
    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    predicted_success: float
    routing_mode: str
    escalated: bool = False
    request_id: str
    routing_reason: str
    verified: bool = False
    actual_success: bool | None = None
    quality_score: float | None = None
    verification_error: str | None = None
    usage_estimated: bool = False
    cache_hit: bool = False


class ModelInfo(BaseModel):
    id: str
    display_name: str
    provider: str
    tier: int
    enabled: bool
