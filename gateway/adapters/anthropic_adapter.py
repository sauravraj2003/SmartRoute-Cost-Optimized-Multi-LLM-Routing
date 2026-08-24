"""Anthropic Messages API adapter."""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import GenerationResult, ModelAdapter


class AnthropicAdapter(ModelAdapter):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerationResult:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Anthropic models")
        system = "\n\n".join(m["content"] for m in messages if m["role"] in {"system", "developer"})
        conversation = []
        for message in messages:
            if message["role"] in {"system", "developer"}:
                continue
            role = "user" if message["role"] == "tool" else message["role"]
            content = f"Tool result:\n{message['content']}" if message["role"] == "tool" else message["content"]
            if conversation and conversation[-1]["role"] == role:
                conversation[-1]["content"] += "\n\n" + content
            else:
                conversation.append({"role": role, "content": content})
        body: dict[str, Any] = {"model": self.model.get("model_name", self.model["id"]),
                                "messages": conversation, "max_tokens": kwargs.get("max_tokens", 1024),
                                "temperature": kwargs.get("temperature", .2)}
        if system:
            body["system"] = system
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.model.get("endpoint", "https://api.anthropic.com").rstrip("/") + "/v1/messages",
                                         headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json=body)
            response.raise_for_status()
            data = response.json()
        elapsed = (time.perf_counter() - started) * 1000
        usage = data.get("usage", {})
        text = "\n".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return GenerationResult(text, data.get("model", body["model"]), int(usage.get("input_tokens", 0)),
                                int(usage.get("output_tokens", 0)), elapsed, data)
