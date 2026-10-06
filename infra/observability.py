"""Production observability: Prometheus metrics + optional OpenTelemetry tracing.

The module is deliberately defensive so the deterministic test suite and
zero-infrastructure local mode still work when observability dependencies are
not installed. Production images install them from requirements.txt.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = None
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    OTEL_API_AVAILABLE = True
except ImportError:  # pragma: no cover
    OTEL_API_AVAILABLE = False


SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "razorrisk-api")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

if PROMETHEUS_AVAILABLE:
    HTTP_REQUESTS = Counter(
        "razorrisk_http_requests_total",
        "HTTP requests handled by RazorRisk.",
        ["method", "route", "status"],
    )
    HTTP_LATENCY = Histogram(
        "razorrisk_http_request_duration_seconds",
        "HTTP request latency in seconds.",
        ["method", "route"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
    )
    HTTP_IN_FLIGHT = Gauge(
        "razorrisk_http_requests_in_flight",
        "Current HTTP requests in flight.",
    )
    SCORE_TOTAL = Counter(
        "razorrisk_scores_total",
        "Transactions scored by the risk engine.",
        ["risk_tier", "decision"],
    )
    SCORE_LATENCY = Histogram(
        "razorrisk_score_duration_seconds",
        "Risk scoring latency in seconds.",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    )
    INVESTIGATION_TOTAL = Counter(
        "razorrisk_investigations_total",
        "Investigation jobs by lifecycle outcome.",
        ["status"],
    )
    INVESTIGATION_LATENCY = Histogram(
        "razorrisk_investigation_duration_seconds",
        "Investigation execution latency in seconds.",
        buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900, 3600, 7200),
    )
    INVESTIGATION_QUEUE_DEPTH = Gauge(
        "razorrisk_investigation_queue_depth",
        "Approximate investigation stream length.",
    )
    INVESTIGATION_RETRIES = Counter(
        "razorrisk_investigation_retries_total",
        "Investigation retries requested after failures.",
    )
    RATE_LIMIT_HITS = Counter(
        "razorrisk_rate_limit_hits_total",
        "Requests rejected by distributed rate limiting.",
        ["scope"],
    )
    DEPENDENCY_FAILURES = Counter(
        "razorrisk_dependency_failures_total",
        "External dependency failures.",
        ["dependency"],
    )
    JEV_REQUESTS = Counter("razorrisk_jev_requests_total", "Jev verification calls by outcome.", ["outcome"])
    JEV_FAILURES = Counter("razorrisk_jev_failures_total", "Jev verification failures by reason.", ["reason"])
    JEV_CONSISTENT = Counter("razorrisk_jev_consistent_total", "Successful Jev results classified as CONSISTENT.")
    JEV_REVIEW_RECOMMENDED = Counter("razorrisk_jev_review_recommended_total", "Successful Jev results classified as REVIEW_RECOMMENDED.")
    JEV_UNAVAILABLE = Counter("razorrisk_jev_unavailable_total", "Jev verification requests skipped because unavailable or unconfigured.")
    JEV_AUTO_RESOLVED = Counter("razorrisk_jev_auto_resolved_total", "HITL reviews actually auto-resolved after Jev eligibility checks.")
    JEV_LATENCY = Histogram("razorrisk_jev_latency_seconds", "End-to-end Jev verification latency in seconds.", buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30))
    JEV_ACTION_DISAGREEMENTS = Counter("razorrisk_jev_action_disagreement_total", "Jev action choices that disagree with the investigator.")
    JEV_GROUNDING_FAILURES = Counter("razorrisk_jev_grounding_failure_total", "Jev hypothesis-grounding checks below threshold.")
    JEV_RETRIES = Counter("razorrisk_jev_retries_total", "Retries attempted for transient Jev provider failures.")
    JEV_CIRCUIT_OPEN = Gauge("razorrisk_jev_circuit_open", "Whether the local Jev circuit breaker is open (1) or closed (0).")
else:
    HTTP_REQUESTS = HTTP_LATENCY = HTTP_IN_FLIGHT = None
    SCORE_TOTAL = SCORE_LATENCY = None
    INVESTIGATION_TOTAL = INVESTIGATION_LATENCY = INVESTIGATION_QUEUE_DEPTH = None
    INVESTIGATION_RETRIES = RATE_LIMIT_HITS = DEPENDENCY_FAILURES = None
    JEV_REQUESTS = JEV_FAILURES = JEV_CONSISTENT = JEV_REVIEW_RECOMMENDED = None
    JEV_UNAVAILABLE = JEV_AUTO_RESOLVED = JEV_LATENCY = None
    JEV_ACTION_DISAGREEMENTS = JEV_GROUNDING_FAILURES = JEV_RETRIES = None
    JEV_CIRCUIT_OPEN = None


def setup_tracing() -> None:
    """Configure OpenTelemetry when explicitly requested.

    If an OTLP endpoint is supplied, spans are exported there. Otherwise a
    console exporter is used only when OTEL_CONSOLE_EXPORTER=true, keeping
    production stdout quiet by default.
    """
    if not OTEL_API_AVAILABLE:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    if OTEL_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=OTEL_ENDPOINT)))
        except ImportError:
            pass
    elif os.getenv("OTEL_CONSOLE_EXPORTER", "false").lower() == "true":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)


def instrument_fastapi(app) -> None:
    if not OTEL_API_AVAILABLE:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        return


def tracer():
    if not OTEL_API_AVAILABLE:
        return None
    return trace.get_tracer(SERVICE_NAME)


@contextmanager
def span(name: str, attributes: dict | None = None) -> Iterator[object]:
    """Create an OpenTelemetry span when tracing is installed; otherwise no-op."""
    t = tracer()
    if t is None:
        yield None
        return
    with t.start_as_current_span(name, attributes=attributes or {}) as current:
        yield current


def metrics_response() -> tuple[bytes, str]:
    if not PROMETHEUS_AVAILABLE:
        return b"# prometheus_client is not installed\n", CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST


@contextmanager
def observe_timer(histogram) -> Iterator[None]:
    if histogram is None:
        yield
        return
    with histogram.time():
        yield
