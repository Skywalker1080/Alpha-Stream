"""OpenTelemetry setup + small span helpers for the whole app.

Exports traces to Grafana Tempo over OTLP/HTTP (default http://tempo:4318,
override with OTEL_EXPORTER_OTLP_ENDPOINT). Falls back to no-ops when the OTel
packages are not installed so the app never crashes on missing deps.
"""
import os
from contextlib import contextmanager
from functools import wraps
from typing import Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback path
    _OTEL_AVAILABLE = False

_configured = False
tracer = None


def init_telemetry(service_name: Optional[str] = None):
    """Configure the global TracerProvider once. Idempotent. Returns the tracer."""
    global _configured, tracer
    if not _OTEL_AVAILABLE or _configured:
        return tracer

    service = service_name or os.getenv("OTEL_SERVICE_NAME", "crypto-prism-fastapi")
    # No explicit endpoint: let the SDK read OTEL_EXPORTER_OTLP_ENDPOINT, which
    # appends the correct signal path (/v1/traces). Passing endpoint= manually
    # skips that appending and POSTs to "/", which Tempo rejects with 404.
    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(service)
    _configured = True
    return tracer


def current_trace_id() -> str:
    """Hex trace id of the active span (empty string if none)."""
    if not _OTEL_AVAILABLE:
        return ""
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return ""


@contextmanager
def trace_step(name: str, **attributes):
    """Create a child span under the current trace. No-op if OTel is unavailable."""
    if _OTEL_AVAILABLE and tracer is not None:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                span.set_attributes({k: str(v) for k, v in attributes.items()})
            yield span
    else:
        yield None


def traced_node(name: str):
    """Decorator wrapping a LangGraph node function's whole body in a span.

    The node's `ticker` is pulled from the state dict (first positional arg).
    """

    def deco(fn):
        @wraps(fn)
        def wrapper(state, *args, **kwargs):
            attrs = {"node": name}
            if isinstance(state, dict):
                attrs["ticker"] = state.get("ticker", "")
            with trace_step(f"node.{name}", **attrs):
                return fn(state, *args, **kwargs)

        return wrapper

    return deco