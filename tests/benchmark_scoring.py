"""Reproducible latency benchmark for RazorRisk's live scoring pipeline.

Measures wall-clock time for each stage of calculate_composite_risk_score()
against the shipped model artifacts and the current database (SQLite by
default; set DATABASE_URL to point at Postgres instead). This is a local,
single-process measurement of *relative* cost between stages, not a
production load-test or an SLA guarantee — no concurrency, no network hop
to a real Postgres/Redis, whatever CPU happens to be running this process.

Usage
-----
    python tests/benchmark_scoring.py
    python tests/benchmark_scoring.py --user USER_RING1_1 --runs 50

Requires a database already populated (see README's Quick Start) and the
shipped model artifacts under ml/models/.
"""
from __future__ import annotations

import argparse
import datetime
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_raw_sqlite_connection  # noqa: E402
from ml.risk_aggregator import (  # noqa: E402
    _LiveModels,
    calculate_composite_risk_score,
    live_gnn_score_and_evidence,
    live_tabular_score,
)
from ml import risk_graph  # noqa: E402


def _percentiles(times_ms: list[float]) -> tuple[float, float]:
    times_ms = sorted(times_ms)
    n = len(times_ms)
    p50 = times_ms[n // 2]
    p95 = times_ms[min(n - 1, int(n * 0.95))]
    return p50, p95


def _make_txn(user_id: str, device_id: str, ip_address: str, merchant_id: str, amount: float) -> dict:
    return {
        "user_id": user_id, "device_id": device_id, "ip_address": ip_address,
        "merchant_id": merchant_id, "amount": amount, "is_vpn_proxy": True,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="USER_RING1_1")
    parser.add_argument("--device", default="DEV_FRAUD_RING1")
    parser.add_argument("--ip", default="185.220.101.44")
    parser.add_argument("--merchant", default="MCH_042")
    parser.add_argument("--amount", type=float, default=88000)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    def txn() -> dict:
        return _make_txn(args.user, args.device, args.ip, args.merchant, args.amount)

    conn = get_raw_sqlite_connection()

    # Warm up: loads model artifacts into the in-process cache and builds
    # the first graph snapshot. Timed separately since it's a one-time
    # (or once-per-TTL) cost, not paid on every request.
    import time
    t0 = time.perf_counter()
    calculate_composite_risk_score(txn())
    cold_ms = (time.perf_counter() - t0) * 1000
    _LiveModels.ensure_loaded()

    def bench(fn, runs: int) -> tuple[float, float]:
        samples = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)
        return _percentiles(samples)

    print("# RazorRisk Scoring Latency Benchmark\n")
    print(f"Cold call (model load + first graph snapshot): {cold_ms:.1f}ms\n")

    full_p50, full_p95 = bench(lambda: calculate_composite_risk_score(txn()), args.runs)
    print(f"| Stage | p50 | p95 |\n|---|---:|---:|")
    print(f"| Full `calculate_composite_risk_score` (warm) | {full_p50:.2f}ms | {full_p95:.2f}ms |")

    p50, p95 = bench(lambda: live_tabular_score(txn(), velocity_1h=1), args.runs)
    print(f"| `live_tabular_score` (XGBoost, incl. feature build) | {p50:.2f}ms | {p95:.2f}ms |")

    p50, p95 = bench(lambda: live_gnn_score_and_evidence(args.user), args.runs)
    print(f"| `live_gnn_score_and_evidence` (GraphSAGE, cached snapshot) | {p50:.2f}ms | {p95:.2f}ms |")

    def velocity_query():
        c = get_raw_sqlite_connection()
        c.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND timestamp > datetime('now', '-1 hours')",
            (args.user,),
        ).fetchone()
        c.close()

    p50, p95 = bench(velocity_query, args.runs)
    print(f"| Velocity DB query | {p50:.2f}ms | {p95:.2f}ms |")

    coef, intercept = _LiveModels.coef, _LiveModels.intercept

    def stacker_only():
        z = coef[0] * 0.5 + coef[1] * 0.5 + coef[2] * 0.5 + coef[3] * 0.5 + intercept
        return 1 / (1 + 2.71828 ** (-z))

    p50, p95 = bench(stacker_only, max(args.runs, 500))
    print(f"| Stacker blend (logistic regression, pure math) | {p50 * 1000:.1f}\u00b5s | {p95 * 1000:.1f}\u00b5s |")

    graph_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        graph = risk_graph.build_user_graph(conn)
        risk_graph.detect_communities(graph)
        graph_times.append((time.perf_counter() - t0) * 1000)
    print(
        f"\nPeriodic graph rebuild + community detection ({graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges), n=5: p50={statistics.median(graph_times):.1f}ms"
    )


if __name__ == "__main__":
    main()
