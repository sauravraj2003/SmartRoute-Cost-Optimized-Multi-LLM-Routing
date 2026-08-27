"""Structured logging and optional OpenTelemetry setup."""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager, nullcontext
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("smartroute")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.setLevel((os.getenv("SMARTROUTE_LOG_LEVEL") or os.getenv("SMARTRoute_LOG_LEVEL", "INFO")).upper())
    return logger


logger = configure_logging()


def log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, default=str, separators=(",", ":")))


def setup_tracing(service_name: str = "smartroute-gateway") -> None:
    """Set up local console spans or OTLP export, without recording prompt contents."""
    if os.getenv("SMARTROUTE_TRACING_ENABLED", "true").lower() not in {"1", "true", "yes"}:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)
    except ImportError:
        logger.warning("Tracing enabled but the OpenTelemetry SDK/exporter is unavailable")


@contextmanager
def start_span(name: str):
    try:
        from opentelemetry import trace
    except ImportError:
        with nullcontext() as span:
            yield span
        return
    with trace.get_tracer("smartroute").start_as_current_span(name) as span:
        yield span


def set_span_attributes(**attributes: Any) -> None:
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span.is_recording():
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(key, value)
    except ImportError:
        return
