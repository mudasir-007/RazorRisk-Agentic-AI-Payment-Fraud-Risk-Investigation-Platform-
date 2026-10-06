"""Comparative Jev benchmark for RazorRisk.

Run from the repository root:

    python tests/benchmarks/benchmark_jev.py --runs 3

The baseline always runs with Jev disabled. The Jev arm is only executed when
TYPESAFE_API_KEY is configured; otherwise it is reported as SKIPPED so the
benchmark remains runnable before API access is available.

This benchmark measures the investigation-layer tradeoff introduced by Jev:
same transaction, same evidence/model path, with Jev disabled versus enabled.
It reports median/p95 latency and the incremental Jev overhead. It does not
claim that Jev improves fraud-model accuracy; that requires labeled outcomes
and is a separate evaluation.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.database import get_raw_sqlite_connection
from ml.risk_aggregator import calculate_composite_risk_score
from agent.graph_agent import investigation_agent
from agent import jev_verifier, mode_state


def _pick_transaction() -> str:
    conn = get_raw_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT transaction_id FROM transactions ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise RuntimeError("No transactions found. Generate the synthetic dataset first.")
    return row[0]


def _load_payload(transaction_id: str) -> dict:
    conn = get_raw_sqlite_connection()
    try:
        row = conn.execute(
            """SELECT t.transaction_id, t.user_id, t.device_id, t.ip_address,
                      t.merchant_id, t.amount, d.is_vpn_proxy, ip.is_suspicious_proxy
               FROM transactions t
               LEFT JOIN devices d ON t.device_id = d.device_id
               LEFT JOIN ip_addresses ip ON t.ip_address = ip.ip_address
               WHERE t.transaction_id = ?""",
            (transaction_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        raise RuntimeError(f"Transaction {transaction_id!r} not found.")
    return {
        "transaction_id": row[0], "user_id": row[1], "device_id": row[2],
        "ip_address": row[3], "merchant_id": row[4], "amount": row[5],
        "is_vpn_proxy": bool(row[6]), "is_suspicious_proxy": bool(row[7]),
    }


def _run_arm(payload: dict, risk: dict, jev_enabled: bool, runs: int) -> list[float]:
    mode_state.set_mode("deterministic")
    mode_state.set_jev_verification_enabled(jev_enabled)
    latencies: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = investigation_agent.investigate(payload, risk)
        elapsed = (time.perf_counter() - started) * 1000
        if jev_enabled and result.get("jev_verification") is None:
            raise RuntimeError("Jev arm returned no verification result.")
        if not jev_enabled and result.get("jev_verification") is not None:
            raise RuntimeError("Baseline unexpectedly ran Jev.")
        latencies.append(elapsed)
    return latencies


def _stats(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return statistics.median(ordered), ordered[p95_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-id", help="Transaction to benchmark; defaults to the newest transaction.")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    transaction_id = args.transaction_id or _pick_transaction()
    payload = _load_payload(transaction_id)
    risk = calculate_composite_risk_score(payload)

    baseline = _run_arm(payload, risk, jev_enabled=False, runs=args.runs)
    median_base, p95_base = _stats(baseline)

    print("# Jev Investigation Benchmark")
    print(f"Transaction: `{transaction_id}`")
    print(f"Runs per arm: {args.runs}")
    print(f"Risk score: {risk.get('risk_score', 'n/a')}")
    print()
    print("| Configuration | Median (ms) | P95 (ms) | Runs |")
    print("|---|---:|---:|---:|")
    print(f"| Without Jev | {median_base:.2f} | {p95_base:.2f} | {len(baseline)} |")

    if not jev_verifier.is_available():
        print("| With Jev | **SKIPPED** | **SKIPPED** | 0 |")
        print("\nWith-Jev benchmark not run: set `TYPESAFE_API_KEY` and rerun this command.")
        return

    try:
        with_jev = _run_arm(payload, risk, jev_enabled=True, runs=args.runs)
    finally:
        mode_state.set_jev_verification_enabled(False)

    median_jev, p95_jev = _stats(with_jev)
    delta = median_jev - median_base
    overhead = (delta / median_base * 100.0) if median_base else 0.0
    print(f"| With Jev | {median_jev:.2f} | {p95_jev:.2f} | {len(with_jev)} |")
    print(f"\nMedian incremental latency: **{delta:+.2f} ms ({overhead:+.1f}%)**")
    print("Note: this is an infrastructure/latency comparison, not a fraud-detection accuracy claim.")


if __name__ == "__main__":
    main()
