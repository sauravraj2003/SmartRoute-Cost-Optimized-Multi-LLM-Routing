"""FastAPI application exposing SmartRoute's routing and OpenAI-compatible API."""
from __future__ import annotations

import os
import json
import time
import uuid
import asyncio
from functools import lru_cache
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway.adapters import AnthropicAdapter, LocalAdapter, OpenAIAdapter
from gateway.middleware import RequestContextMiddleware
from gateway.schemas import RouteRequest, RouteResponse
from gateway.telemetry import log_event, set_span_attributes, setup_tracing
from benchmark.evaluators import evaluate_code
from router.policy import RoutingDecision, estimate_cost, select_model
from router.escalation import escalation_candidates
from router.train import SuccessPredictor
from router.config import load_models_config
from gateway.semantic_cache import SemanticCache, cache_signature


def _read_yaml(path: str, key: str | None = None):
    with open(path, encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    return value.get(key, value) if key else value


@lru_cache(maxsize=1)
def get_models() -> list[dict[str, Any]]:
    path = os.getenv("SMARTROUTE_MODELS_CONFIG") or os.getenv("SMARTRoute_MODELS_CONFIG", "configs/models.yaml")
    pricing_path = os.getenv("SMARTROUTE_PRICING_CONFIG", "configs/pricing.yaml")
    config = load_models_config(path, pricing_path)
    if not config:
        raise RuntimeError(f"No models configured in {path}")
    return [model for model in config if model.get("enabled", True)]


def _mode_config(mode: str) -> dict[str, Any]:
    config = _read_yaml(os.getenv("SMARTROUTE_ROUTING_CONFIG") or os.getenv("SMARTRoute_ROUTING_CONFIG", "configs/routing.yaml"))
    selected = config.get("modes", {}).get(mode)
    if selected is None:
        raise HTTPException(status_code=400, detail=f"Unknown routing mode: {mode}")
    return selected


def _adapter(model: dict[str, Any]):
    provider = model.get("provider", "openai").lower()
    adapters = {"anthropic": AnthropicAdapter, "local": LocalAdapter,
                "openai": OpenAIAdapter, "groq": OpenAIAdapter}
    if provider not in adapters:
        raise HTTPException(status_code=422, detail=f"Unsupported provider adapter: {provider}")
    cls = adapters[provider]
    return cls(model)


@lru_cache(maxsize=1)
def _semantic_cache() -> SemanticCache:
    return SemanticCache(
        os.getenv("SMARTROUTE_CACHE_PATH", "cache/semantic.sqlite3"),
        ttl_seconds=int(os.getenv("SMARTROUTE_CACHE_TTL_SECONDS", "86400")),
        max_entries=int(os.getenv("SMARTROUTE_CACHE_MAX_ENTRIES", "10000")),
    )


app = FastAPI(title="SmartRoute", version="0.1.0", description="Cost-aware multi-model coding assistant gateway")
app.add_middleware(RequestContextMiddleware)
setup_tracing()


@app.get("/health")
async def health():
    return {"status": "ok", "service": "smartroute"}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": m["id"], "object": "model", "owned_by": m.get("provider", "unknown")} for m in get_models()]}


async def _route(body: RouteRequest, request: Request) -> RouteResponse:
    request_started = time.perf_counter()
    if body.stream:
        raise HTTPException(status_code=400, detail="Streaming is not supported")
    if body.tests is not None and not body.tests.strip():
        raise HTTPException(status_code=422, detail="tests must contain at least one test")
    try:
        messages = body.normalized_messages()
        prompt = body.user_prompt()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    models = get_models()
    routing_path = os.getenv("SMARTROUTE_ROUTING_CONFIG") or os.getenv("SMARTRoute_ROUTING_CONFIG", "configs/routing.yaml")
    mode_name = body.mode or os.getenv("SMARTROUTE_MODE") or os.getenv("SMARTRoute_MODE") or _read_yaml(routing_path).get("default_mode", "balanced")
    mode = _mode_config(mode_name)
    model_ids = [m["id"] for m in models]
    routing_text = "\n".join(message["content"] for message in messages)
    predictor = SuccessPredictor(os.getenv("SMARTROUTE_MODEL_DIR") or os.getenv("SMARTRoute_MODEL_DIR", "router/models"))
    predictions = predictor.predict(routing_text, model_ids)
    input_tokens = max(1, (len(routing_text) + 3) // 4)
    routing_config = _read_yaml(routing_path)
    requested = next((m for m in models if m["id"] == body.model), None) if body.model else None
    if body.model and requested is None:
        raise HTTPException(status_code=404, detail=f"Model is not configured or enabled: {body.model}")
    if requested is not None:
        if input_tokens + body.max_tokens > int(requested.get("context_window", input_tokens + body.max_tokens)):
            raise HTTPException(status_code=422, detail="Pinned model context window is too small for this request")
        latency_limit = routing_config.get("latency_budget_ms")
        model_p95 = requested.get("typical_latency_ms", {}).get("p95")
        if latency_limit is not None and (model_p95 is None or model_p95 > latency_limit):
            raise HTTPException(status_code=422, detail="Pinned model does not meet the configured latency limit")
        pinned_cost = estimate_cost(requested, input_tokens, body.max_tokens)
        if mode.get("max_cost_usd") is not None and pinned_cost > float(mode["max_cost_usd"]):
            raise HTTPException(status_code=422, detail="Pinned model exceeds the configured request cost ceiling")
        decision = RoutingDecision(requested, predictions.get(requested["id"], .5), pinned_cost,
                                   "caller pinned a configured model")
    else:
        try:
            decision = select_model(models, predictions, float(mode.get("success_threshold", .72)), input_tokens,
                                    body.max_tokens, mode.get("max_cost_usd"), routing_config.get("latency_budget_ms"))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    selected = decision.model
    features = routing_config.get("features", {})
    cache_config = features.get("semantic_cache", {})
    cache_enabled = cache_config is True or (isinstance(cache_config, dict) and cache_config.get("enabled", False))
    cache_hit = None
    cache_scope = None
    if cache_enabled and body.tests and body.tests.strip():
        cache_scope = cache_signature(messages, body.tests, body.entry_point,
                                      f"{mode_name}|pinned={body.model or ''}",
                                      body.max_tokens, body.temperature, models,
                                      float(mode.get("success_threshold", .72)), mode.get("max_cost_usd"))
        try:
            cache_hit = _semantic_cache().get(
                cache_scope, prompt, float(cache_config.get("similarity_threshold", .96))
                if isinstance(cache_config, dict) else .96)
        except Exception as exc:
            log_event("routing_cache_error", operation="read", error=str(exc))
    if cache_hit:
        cached_model = next((model for model in models if model["id"] == cache_hit["model"]), None)
        if cached_model is not None:
            response = RouteResponse(
                text=cache_hit["text"], model=cached_model["id"],
                provider=cached_model.get("provider", "unknown"), input_tokens=0,
                output_tokens=0, latency_ms=(time.perf_counter() - request_started) * 1000,
                cost_usd=0.0, predicted_success=float(predictions.get(cached_model["id"], .5)),
                routing_mode=mode_name, escalated=False,
                request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                routing_reason="verified semantic cache hit", verified=True,
                actual_success=True, quality_score=1.0, cache_hit=True)
            log_event("routing_cache_hit", **response.model_dump(exclude={"text"}))
            return response
    tried: set[str] = set()
    errors = []
    max_attempts = 1 if requested is not None else max(1, int(mode.get("max_escalations", 0)) + 1)
    result = None
    result_model = None
    all_cost = 0.0
    all_input_tokens = 0
    all_output_tokens = 0
    all_latency = 0.0
    usage_estimated = False
    verified = False
    actual_success = None
    quality_score = None
    verification_error = None
    escalated = False
    selected_reason = decision.reason
    for attempt in range(max_attempts):
        tried.add(selected["id"])
        try:
            candidate = await _adapter(selected).generate(messages, temperature=body.temperature, max_tokens=body.max_tokens)
            result = candidate
            result_model = selected
            call_input_tokens = candidate.input_tokens or input_tokens
            call_output_tokens = candidate.output_tokens or max(1, (len(candidate.text) + 3) // 4)
            usage_estimated = usage_estimated or not bool(candidate.input_tokens and candidate.output_tokens)
            all_input_tokens += call_input_tokens
            all_output_tokens += call_output_tokens
            all_latency += candidate.latency_ms
            all_cost += _adapter(selected).estimate_cost(call_input_tokens, call_output_tokens)
        except Exception as exc:
            errors.append(f"{selected['id']}: {exc}")
            candidate = None
        if candidate is not None and body.tests is not None:
            try:
                verification_started = time.perf_counter()
                check = await asyncio.to_thread(evaluate_code, candidate.text, body.tests,
                                                body.entry_point, 5.0, prompt)
                all_latency += (time.perf_counter() - verification_started) * 1000
                verified = True
                actual_success = bool(check["success"])
                quality_score = float(check["quality_score"])
                if actual_success:
                    break
                if not mode.get("test_based_escalation", True):
                    break
            except Exception as exc:
                verification_error = str(exc)
                # A failed/unavailable verifier is not evidence that the model failed.
                break
        elif candidate is not None:
            break
        if attempt + 1 >= max_attempts:
            break
        stronger = escalation_candidates(models, selected["id"], tried,
                                         escalation_order=routing_config.get("escalation_order"))
        if not stronger:
            break
        if mode.get("max_cost_usd") is not None:
            next_estimate = estimate_cost(stronger[0], input_tokens, body.max_tokens)
            if all_cost + next_estimate > float(mode["max_cost_usd"]):
                selected_reason = "escalation stopped by request cost ceiling"
                break
        selected = stronger[0]
        escalated = True
        selected_reason = "escalated after provider error" if candidate is None else "escalated after verification failure"
    if result is None:
        log_event("generation_failed", request_id=getattr(request.state, "request_id", None), errors=errors)
        raise HTTPException(status_code=502, detail={"message": "All model attempts failed", "errors": errors})
    served_model = result_model or selected
    response = RouteResponse(text=result.text, model=served_model["id"], provider=served_model.get("provider", "unknown"),
                             input_tokens=all_input_tokens, output_tokens=all_output_tokens,
                             latency_ms=all_latency, cost_usd=all_cost,
                             predicted_success=float(predictions.get(served_model["id"], 0)), routing_mode=mode_name,
                             escalated=escalated or served_model["id"] != decision.model["id"],
                             request_id=getattr(request.state, "request_id", str(uuid.uuid4())),
                             routing_reason=selected_reason if selected_reason != decision.reason else decision.reason,
                             verified=verified, actual_success=actual_success, quality_score=quality_score,
                             verification_error=verification_error, usage_estimated=usage_estimated)
    if cache_enabled and cache_scope and verified and actual_success:
        try:
            _semantic_cache().put(cache_scope, prompt, {
                "text": result.text, "model": served_model["id"],
                "verified": True, "actual_success": True,
            })
        except Exception as exc:
            log_event("routing_cache_error", operation="write", error=str(exc))
    log_event("routing_complete", **response.model_dump(exclude={"text"}))
    set_span_attributes(selected_model=response.model, predicted_success=response.predicted_success,
                        routing_mode=mode_name, escalated=response.escalated,
                        actual_success=response.actual_success, input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens, cost_usd=response.cost_usd,
                        latency_ms=response.latency_ms)
    return response


@app.post("/route", response_model=RouteResponse)
async def route(body: RouteRequest, request: Request):
    return await _route(body, request)


@app.post("/v1/chat/completions")
async def chat_completions(body: RouteRequest, request: Request):
    routed = await _route(body, request)
    return JSONResponse({"id": routed.request_id, "object": "chat.completion", "created": int(time.time()),
                         "model": routed.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": routed.text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": routed.input_tokens, "completion_tokens": routed.output_tokens,
                                   "total_tokens": routed.input_tokens + routed.output_tokens},
                         "smartroute": routed.model_dump(exclude={"text"})})


def run_server() -> None:
    import uvicorn
    uvicorn.run("gateway.api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)
