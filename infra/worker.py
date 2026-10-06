"""Dedicated investigation worker process.

Run one or more replicas with:
    python -m infra.worker
All workers consume the same Redis queue, so horizontal scaling does not
create per-process queues.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import traceback
import time

from infra.jobs import read_job, reclaim_job, ack_job, requeue_job, get_job, update_job
from db.database import get_raw_sqlite_connection
from ml.risk_aggregator import calculate_composite_risk_score
from ml.decision_policy import apply_decision_policy, MANDATORY_HUMAN_REASONS
from ml.risk_aggregator import HIGH_RISK_THRESHOLD
from agent.graph_agent import investigation_agent
from api.routes_hitl import auto_resolve_review
from utils.logger import get_logger
from infra.observability import INVESTIGATION_TOTAL, INVESTIGATION_LATENCY, INVESTIGATION_RETRIES, span

logger = get_logger("investigation_worker")
_shutdown = False


def load_transaction(transaction_id: str) -> dict | None:
    conn = get_raw_sqlite_connection()
    try:
        row = conn.execute("""
            SELECT t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount,
                   d.is_vpn_proxy, ip.is_suspicious_proxy
            FROM transactions t
            LEFT JOIN devices d ON t.device_id = d.device_id
            LEFT JOIN ip_addresses ip ON t.ip_address = ip.ip_address
            WHERE t.transaction_id = ?
        """, (transaction_id,)).fetchone()
        if not row:
            return None
        return {
            "transaction_id": transaction_id, "user_id": row[0], "device_id": row[1],
            "ip_address": row[2], "merchant_id": row[3], "amount": row[4],
            "is_vpn_proxy": bool(row[5]), "is_suspicious_proxy": bool(row[6]),
        }
    finally:
        conn.close()


def execute_job_sync(job: dict) -> dict:
    txn_id = job["transaction_id"]
    payload = load_transaction(txn_id)
    if payload is None:
        raise ValueError(f"Transaction {txn_id} not found")
    risk = calculate_composite_risk_score(payload)
    policy = apply_decision_policy(payload, risk)
    needs = risk["risk_score"] >= HIGH_RISK_THRESHOLD or policy.get("hitl_required", False)
    if not needs and not job.get("force", False):
        return {"transaction_id": txn_id, "investigation_skipped": True, "risk_evaluation": risk, "policy_evaluation": policy}
    result = investigation_agent.investigate(payload, risk)

    # Mirrors the API endpoint's Jev-driven HITL auto-resolution.
    review_reasons = set(policy.get("review_reasons", []))
    jev_result = result.get("jev_verification")
    if (
        policy.get("hitl_required")
        and jev_result
        and jev_result.get("eligible_for_auto_resolve")
        and not (review_reasons & MANDATORY_HUMAN_REASONS)
    ):
        resolved_id = auto_resolve_review(
            txn_id,
            decision_action=result["recommended_action"],
            rationale=(
                f"Auto-resolved: Jev and the investigator independently agreed on "
                f"'{result['recommended_action']}' "
                f"(confidence={jev_result['independent_action_confidence']:.0%}, "
                f"hypothesis grounding={jev_result['hypothesis_grounded_probability']:.0%}). "
                f"No mandatory-human-review reason was present."
            ),
        )
        result["hitl_auto_resolved_review_id"] = resolved_id
        if resolved_id:
            logger.info(f"HITL review {resolved_id} for {txn_id} auto-resolved via Jev verification.")

    conn = get_raw_sqlite_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO investigation_reports
            (investigation_id, transaction_id, risk_score, evidence_json, fraud_hypothesis, recommended_action, summary_report)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (result["investigation_id"], txn_id, risk["risk_score"], json.dumps(result["evidence"]),
              result["fraud_hypothesis"], result["recommended_action"], result["summary_report"]))
        conn.commit()
    finally:
        conn.close()
    return result

async def process(job_id: str) -> None:
    started = time.perf_counter()
    job = await get_job(job_id)
    if not job:
        return
    if job["status"] not in {"queued", "retrying"}:
        return
    if job["sla_deadline"] < __import__("time").time():
        await update_job(job_id, status="failed", error="SLA deadline exceeded before execution")
        return
    await update_job(job_id, status="running", attempts=int(job.get("attempts", 0)) + 1)
    try:
        # The ML/SQLite path is synchronous; isolate it from the async worker loop.
        # Enforce the absolute job deadline at the orchestration layer as well as
        # inside the provider clients. This prevents a slow dependency from
        # silently consuming the full SLA window.
        remaining = max(0.1, float(job["sla_deadline"]) - time.time())
        with span("razorrisk.investigation", {"job.id": job_id, "transaction.id": job.get("transaction_id", "")}):
            result = await asyncio.wait_for(asyncio.to_thread(execute_job_sync, job), timeout=remaining)
        await update_job(job_id, status="completed", result=result)
        if INVESTIGATION_TOTAL is not None:
            INVESTIGATION_TOTAL.labels("completed").inc()
    except Exception as exc:
        logger.error("Job %s failed: %s\n%s", job_id, exc, traceback.format_exc())
        current = await get_job(job_id) or job
        attempts = int(current.get("attempts", 1))
        if attempts < int(os.getenv("INVESTIGATION_MAX_ATTEMPTS", "3")) and current.get("sla_deadline", 0) > __import__("time").time():
            await update_job(job_id, status="retrying", error=str(exc))
            if INVESTIGATION_TOTAL is not None:
                INVESTIGATION_TOTAL.labels("retrying").inc()
            if INVESTIGATION_RETRIES is not None:
                INVESTIGATION_RETRIES.inc()
            await requeue_job(job_id)
        else:
            await update_job(job_id, status="failed", error=str(exc))
            if INVESTIGATION_TOTAL is not None:
                INVESTIGATION_TOTAL.labels("failed").inc()

async def main() -> None:
    # Prometheus scrapes workers separately because workers do not expose the
    # FastAPI /metrics endpoint. Each worker therefore reports its own latency,
    # retry and completion counters without pretending they are globally shared.
    try:
        from prometheus_client import start_http_server
        metrics_port = int(os.getenv("OBSERVABILITY_METRICS_PORT", "9101"))
        start_http_server(metrics_port)
        logger.info("Worker Prometheus metrics listening on :%s", metrics_port)
    except ImportError:
        logger.warning("prometheus_client not installed; worker metrics disabled")
    logger.info("Redis Streams investigation worker started")
    consumer = f"worker-{os.getpid()}"
    while not _shutdown:
        try:
            stream_id, job_id = await reclaim_job(consumer)
            if not job_id:
                stream_id, job_id = await read_job(consumer, timeout_ms=5000)
            if job_id:
                try:
                    await process(job_id)
                finally:
                    if stream_id:
                        await ack_job(stream_id)
        except Exception:
            logger.error("Worker queue loop failure\n%s", traceback.format_exc())
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
