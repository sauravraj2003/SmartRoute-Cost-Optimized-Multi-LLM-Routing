"""OpenAI Chat Completions compatible adapter (also supports Groq/local APIs)."""
from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .base import GenerationResult, ModelAdapter


class OpenAIAdapter(ModelAdapter):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> GenerationResult:
        endpoint = self.model.get("endpoint", "https://api.openai.com/v1").rstrip("/")
        provider = str(self.model.get("provider", "openai")).upper()
        env_key = "GROQ_API_KEY" if provider == "GROQ" else "OPENAI_API_KEY"
        key = self.api_key or os.getenv(env_key)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        body = {"model": self.model.get("model_name", self.model["id"]), "messages": messages,
                "temperature": kwargs.get("temperature", .2),
                "max_completion_tokens": kwargs.get("max_tokens", 1024)}
        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{endpoint}/chat/completions", headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
        elapsed = (time.perf_counter() - started) * 1000
        usage = data.get("usage", {})
        return GenerationResult(text=data["choices"][0]["message"].get("content") or "",
                                model=data.get("model", body["model"]),
                                input_tokens=int(usage.get("prompt_tokens", 0)),
                                output_tokens=int(usage.get("completion_tokens", 0)), latency_ms=elapsed, raw=data)
