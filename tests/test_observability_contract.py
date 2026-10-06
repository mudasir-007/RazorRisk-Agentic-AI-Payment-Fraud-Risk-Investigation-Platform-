"""Static/runtime checks for the production observability layer."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_observability_module_exists_and_has_core_metrics():
    source = (ROOT / "infra" / "observability.py").read_text()
    for token in (
        "razorrisk_http_requests_total",
        "razorrisk_http_request_duration_seconds",
        "razorrisk_scores_total",
        "razorrisk_investigations_total",
        "razorrisk_investigation_queue_depth",
        "razorrisk_rate_limit_hits_total",
        "setup_tracing",
        "instrument_fastapi",
    ):
        assert token in source


def test_api_exposes_metrics_endpoint():
    source = (ROOT / "api" / "main.py").read_text()
    assert '@app.get("/metrics", include_in_schema=False)' in source
    assert "metrics_response" in source


def test_worker_exposes_metrics_port():
    source = (ROOT / "infra" / "worker.py").read_text()
    assert "start_http_server" in source
    assert "OBSERVABILITY_METRICS_PORT" in source


def test_compose_contains_prometheus_and_grafana():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "prometheus:" in compose
    assert "grafana:" in compose
    assert "prometheus_data:" in compose
    assert "grafana_data:" in compose


def test_observability_docs_and_config_exist():
    assert (ROOT / "prometheus.yml").exists()
    assert (ROOT / "docs" / "OBSERVABILITY.md").exists()
    assert (ROOT / "grafana" / "provisioning" / "datasources" / "prometheus.yml").exists()
    env = (ROOT / ".env.example").read_text()
    for token in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OBSERVABILITY_METRICS_PORT", "GRAFANA_ADMIN_PASSWORD"):
        assert token in env
