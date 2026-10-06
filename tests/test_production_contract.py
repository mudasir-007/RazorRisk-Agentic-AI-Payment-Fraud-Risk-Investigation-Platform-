"""Static production-contract checks that do not require external services."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_production_files_exist():
    required = [
        ROOT / "infra" / "redis_client.py",
        ROOT / "infra" / "rate_limit.py",
        ROOT / "infra" / "jobs.py",
        ROOT / "infra" / "worker.py",
        ROOT / "BUG.md",
        ROOT / "docker-compose.yml",
        ROOT / "static" / "index.html",
        ROOT / "static" / "js" / "app.js",
        ROOT / "static" / "js" / "graph_vis.js",
    ]
    assert all(p.exists() for p in required)


def test_model_artifacts_have_evaluation_contract():
    model_dir = ROOT / "ml" / "models"
    for name in ("aggregator_eval.json", "tabular_eval.json", "gnn_eval.json", "hyperparameters.json"):
        assert (model_dir / name).exists(), name
    data = json.loads((model_dir / "aggregator_eval.json").read_text())
    assert set(data) == {"tabular_only", "gnn_only", "stacked"}


def test_dashboard_uses_async_investigation_queue():
    js = (ROOT / "static" / "js" / "app.js").read_text()
    assert "/api/v1/investigations/enqueue/" in js
    assert "/api/v1/investigations/jobs/" in js


def test_score_endpoint_has_distributed_limiter():
    source = (ROOT / "api" / "routes_transactions.py").read_text()
    assert "enforce_rate_limit" in source
    assert 'scope="transaction-score"' in source


def test_worker_enforces_absolute_sla():
    source = (ROOT / "infra" / "worker.py").read_text()
    assert "asyncio.wait_for" in source
    assert "sla_deadline" in source


def test_compose_declares_redis_and_worker():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "redis:" in compose
    assert "investigation-worker:" in compose
    assert "REDIS_REQUIRED=true" in compose
