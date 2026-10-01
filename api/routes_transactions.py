import json
import uuid
import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from typing import Optional, List
from ml.risk_aggregator import calculate_composite_risk_score, HIGH_RISK_THRESHOLD, invalidate_live_graph_snapshot
from ml.decision_policy import apply_decision_policy
from ml.watchlist import refresh_watchlist
from api.routes_hitl import enqueue_review
from ml.graph_builder import graph_builder
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger, bind_correlation_id, clear_correlation_id
from infra.rate_limit import enforce_rate_limit
from infra.observability import SCORE_TOTAL, SCORE_LATENCY, span

logger = get_logger("api_transactions")

router = APIRouter(prefix="/api/v1/transactions", tags=["Transactions & Risk Scoring"])

class TransactionPayload(BaseModel):
    transaction_id: Optional[str] = None
    user_id: str = Field(..., example="USER_0012")
    device_id: str = Field(..., example="DEV_0045")
    ip_address: str = Field(..., example="192.168.1.100")
    merchant_id: str = Field(default="MCH_001", example="MCH_001")
    amount: float = Field(..., example=45000.0)
    currency: str = Field(default="INR")
    # Velocity has an explicit source-selection control for testing and
    # simulation: when velocity_source_client is True, the supplied value is
    # trusted; when False, the backend computes the trailing 1-hour count from
    # persisted transaction history. This makes the frontend toggle mean
    # exactly what it says instead of silently ignoring a client value.
    is_vpn_proxy: bool = Field(default=False)
    is_suspicious_proxy: bool = Field(default=False)
    velocity_enabled: bool = Field(default=False, description="Frontend source toggle: True trusts client velocity; False calculates velocity in backend")
    velocity_1h: Optional[int] = Field(default=None, ge=0, description="Client velocity used only when velocity_enabled is True")

@router.post("/score")
async def score_transaction(request: Request, payload: TransactionPayload):
    """
    Evaluates an incoming payment transaction in real-time.
    Computes Tabular ML fraud probability, GNN node embedding score, and graph topology risk.
    Returns immediately; high-risk/HITL transactions are investigated by a separate follow-up request.
    """
    await enforce_rate_limit(request, scope="transaction-score", limit=120)

    txn_dict = payload.model_dump()
    txn_id = txn_dict.get("transaction_id") or f"TXN_{uuid.uuid4().hex[:8].upper()}"
    txn_dict["transaction_id"] = txn_id

    # Every log line emitted while scoring this transaction — across the
    # tabular model, GNN, aggregator, and (for high-risk ones) the agent —
    # carries this same ID, so `grep <corr_id> logs/*.log` reconstructs the
    # full cross-subsystem trace of one transaction's decision.
    corr_id = bind_correlation_id()
    score_started = __import__("time").perf_counter()
    logger.info(f"Received live transaction scoring request: TxnID:{txn_id}, User:{payload.user_id}, Amt:₹{payload.amount}")

    try:
        # 1. Compute Composite Risk Score
        with span("razorrisk.risk_scoring", {"transaction.id": txn_id, "transaction.amount": payload.amount}):
            risk_res = calculate_composite_risk_score(txn_dict)
        policy_res = apply_decision_policy(txn_dict, risk_res)
        risk_res.update(policy_res)

        # 2. High-risk transactions (>= 70) need an investigation, but that
        # can involve a real LLM call and take several seconds — it's run
        # as a separate follow-up request (POST /api/v1/investigations/run/{id})
        # from the frontend so the risk score itself renders instantly instead
        # of the whole scoring UI blocking on the slower agent step.
        needs_investigation = risk_res["risk_score"] >= HIGH_RISK_THRESHOLD or risk_res.get("hitl_required", False)

        # 3. Persist to Database
        conn = get_raw_sqlite_connection()
        cursor = conn.cursor()

        # Ensure Entities exist
        cursor.execute("INSERT OR IGNORE INTO users (user_id, name, email) VALUES (?, ?, ?)",
                       (payload.user_id, f"User {payload.user_id}", f"{payload.user_id}@example.com"))
        cursor.execute("INSERT OR IGNORE INTO devices (device_id, device_type, os, is_vpn_proxy) VALUES (?, ?, ?, ?)",
                       (payload.device_id, "Mobile-Android", "Android 14", payload.is_vpn_proxy))
        cursor.execute("INSERT OR IGNORE INTO ip_addresses (ip_address, country, city, isp, is_suspicious_proxy) VALUES (?, ?, ?, ?, ?)",
                       (payload.ip_address, "IN", "Mumbai", "Airtel", payload.is_suspicious_proxy))

        # Persist the effective velocity actually used by the scoring path.
        # In client-trust mode this is the supplied simulation value; in
        # backend mode it is the database-derived trailing-1h count.
        cursor.execute("""
            INSERT OR REPLACE INTO transactions
            (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, velocity_enabled, velocity_source, amount_zscore_prior)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            txn_id, payload.user_id, payload.device_id, payload.ip_address, payload.merchant_id,
            payload.amount, payload.currency, datetime.datetime.now().isoformat(), "COMPLETED",
            risk_res["velocity_1h"], risk_res.get("velocity_source") == "CLIENT",
            risk_res.get("velocity_source", "BACKEND"), risk_res["amount_zscore_prior"]
        ))

        # Insert Risk Score. evidence_multiplier is a retired concept (see
        # ml/risk_aggregator.py's module docstring — graph evidence is now a
        # learned stacker input, not a separate rule-based multiplier);
        # hardcoded to 1.0 here purely for schema/history compatibility with
        # the risk_scores table's existing evidence_multiplier column.
        scoring_id = f"SCORE_{uuid.uuid4().hex[:8].upper()}"
        cursor.execute("""
            INSERT INTO risk_scores
            (scoring_id, transaction_id, risk_score, tabular_score, gnn_score, stacker_calibrated_score, velocity_multiplier, evidence_multiplier, risk_tier, decision)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scoring_id, txn_id, risk_res["risk_score"], risk_res["tabular_score"],
            risk_res["gnn_score"], risk_res["stacker_calibrated_score"],
            risk_res["velocity_multiplier"], 1.0,
            risk_res["risk_tier"], risk_res["decision"]
        ))

        conn.commit()
        conn.close()

        # Queue HITL only after the transaction and its risk record exist.
        # This keeps the internal review queue consistent and makes the
        # transaction_id FK resolvable on SQLite configurations that enforce FK checks.
        review_id = enqueue_review(txn_id, risk_res)
        risk_res["review_id"] = review_id

        # MONITOR soft-flags the user for the next transaction (see
        # ml/watchlist.py) — same FK-safety reasoning as enqueue_review
        # above, run after the transaction row exists.
        if risk_res.get("decision") == "MONITOR":
            refresh_watchlist(payload.user_id, txn_id)

        # 4. Fold this transaction into the live in-memory graph immediately
        # (O(1) incremental update) so the Graph Topology tab reflects it
        # without waiting for a full pipeline rebuild.
        graph_builder.add_transaction(
            user_id=payload.user_id, device_id=payload.device_id, ip_address=payload.ip_address,
            merchant_id=payload.merchant_id, amount=payload.amount,
            is_fraud=risk_res["risk_score"] >= HIGH_RISK_THRESHOLD
        )
        # The just-committed transaction must be visible to the NEXT live GNN
        # score. Without invalidation, the 20s graph snapshot could make rapid
        # repeated transactions use stale device/IP/community evidence. The
        # current request remains scored against the pre-insert graph, which
        # prevents the transaction from influencing its own GNN score.
        invalidate_live_graph_snapshot()

        return {
            "transaction_id": txn_id,
            "user_id": payload.user_id,
            "amount": payload.amount,
            "risk_evaluation": risk_res,
            "needs_investigation": needs_investigation,
            "correlation_id": corr_id
        }
    except Exception as e:
        # Full detail goes to the audit log for debugging; the API response
        # stays generic on purpose — internal exception text (module paths,
        # dict keys, stack detail) should never reach the dashboard directly.
        logger.error(f"Scoring failed for TxnID:{txn_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transaction scoring failed (correlation_id={corr_id}). Check logs/risk_engine.log for details.")
    finally:
        elapsed = __import__("time").perf_counter() - score_started
        if SCORE_LATENCY is not None:
            SCORE_LATENCY.observe(elapsed)
        if SCORE_TOTAL is not None and "risk_res" in locals():
            SCORE_TOTAL.labels(risk_res.get("risk_tier", "UNKNOWN"), risk_res.get("decision", "UNKNOWN")).inc()
        clear_correlation_id()

@router.get("/recent")
def get_recent_transactions(limit: int = 20, tier: Optional[str] = None):
    """Retrieves recent transactions with optional filtering by risk tier."""
    conn = get_raw_sqlite_connection()
    cursor = conn.cursor()

    query = """
        SELECT t.transaction_id, t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount, t.timestamp,
               t.velocity_1h, t.velocity_enabled, t.velocity_source,
               r.risk_score, r.tabular_score, r.gnn_score, r.stacker_calibrated_score,
               r.risk_tier, r.decision
        FROM transactions t
        LEFT JOIN risk_scores r ON t.transaction_id = r.transaction_id
    """
    params = []
    if tier:
        query += " WHERE r.risk_tier = ?"
        params.append(tier.upper())
    
    query += " ORDER BY t.timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    result = []
    for r in rows:
        result.append({
            "transaction_id": r[0],
            "user_id": r[1],
            "device_id": r[2],
            "ip_address": r[3],
            "merchant_id": r[4],
            "amount": r[5],
            "timestamp": r[6],
            "velocity_1h": r[7] if r[7] is not None else 0,
            "velocity_enabled": bool(r[8]) if r[8] is not None else False,
            "velocity_source": r[9] or ("CLIENT" if bool(r[8]) else "BACKEND"),
            "risk_score": r[10] if r[10] is not None else 0.0,
            "tabular_score": r[11] if r[11] is not None else 0.0,
            "gnn_score": r[12] if r[12] is not None else 0.0,
            "stacker_calibrated_score": r[13] if r[13] is not None else 0.0,
            "risk_tier": r[14] or "UNSCORED",
            "decision": r[15] or "NONE"
        })

    return {"count": len(result), "transactions": result}
