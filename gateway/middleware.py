"""Gateway request ID and optional bearer-token authentication middleware."""
from __future__ import annotations

import os
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from gateway.telemetry import start_span


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or secrets.token_hex(12)
        request.state.request_id = request_id
        expected = os.getenv("SMARTROUTE_API_KEY") or os.getenv("SMARTRoute_API_KEY")
        if expected and request.url.path not in {"/health", "/v1/models"}:
            supplied = request.headers.get("authorization", "")
            if not secrets.compare_digest(supplied, f"Bearer {expected}"):
                return JSONResponse({"detail": "Unauthorized", "request_id": request_id}, status_code=401)
        with start_span("smartroute.http_request") as span:
            if span is not None:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.route", request.url.path)
                span.set_attribute("smartroute.request_id", request_id)
            response = await call_next(request)
            if span is not None:
                span.set_attribute("http.status_code", response.status_code)
            response.headers["x-request-id"] = request_id
            return response
