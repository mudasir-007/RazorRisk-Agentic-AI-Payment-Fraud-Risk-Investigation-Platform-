
"""Guardrailed evidence-provider adapters.

These are local deterministic stand-ins for external intelligence providers.
A production deployment can replace the provider implementation while keeping
the same SecurityGuard boundary and response contract.
"""
from __future__ import annotations

from typing import Any
from db.database import get_raw_sqlite_connection
from security.guardrails import security_guard, GuardrailViolation


def _merchant_intelligence(transaction_id: str, merchant_id: str) -> dict[str, Any]:
    conn = get_raw_sqlite_connection()
    try:
        try:
            row = conn.execute(
                "SELECT merchant_id, category, fraud_rate, name FROM merchants WHERE merchant_id = ?",
                (merchant_id,),
            ).fetchone()
        except Exception:
            return {"provider": "merchant_intelligence", "available": False, "reason": "evidence store unavailable"}
        if not row:
            return {"provider": "merchant_intelligence", "found": False}
        return {
            "provider": "merchant_intelligence", "found": True,
            "merchant_id": row[0], "category": row[1],
            "historical_fraud_rate": float(row[2] or 0), "merchant_name": row[3],
        }
    finally:
        conn.close()


def _device_intelligence(transaction_id: str, user_id: str, device_id: str) -> dict[str, Any]:
    conn = get_raw_sqlite_connection()
    try:
        try:
            account_count = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM transactions WHERE device_id = ?", (device_id,)
            ).fetchone()[0]
            first_seen = conn.execute(
                "SELECT first_seen FROM devices WHERE device_id = ?", (device_id,)
            ).fetchone()
        except Exception:
            return {"provider": "device_intelligence", "available": False, "reason": "evidence store unavailable"}

        first_seen_val = first_seen[0] if first_seen else None
        if hasattr(first_seen_val, "isoformat"):
            first_seen_val = first_seen_val.isoformat()

        return {
            "provider": "device_intelligence", "found": first_seen is not None,
            "device_account_count": int(account_count or 0),
            "device_first_seen": first_seen_val,
            "note": "Account count is evidence only; it is not itself a fraud verdict.",
        }
    finally:
        conn.close()


def _network_intelligence(transaction_id: str, user_id: str, ip_address: str) -> dict[str, Any]:
    conn = get_raw_sqlite_connection()
    try:
        try:
            row = conn.execute(
                "SELECT country, city, isp, is_suspicious_proxy FROM ip_addresses WHERE ip_address = ?",
                (ip_address,),
            ).fetchone()
            users = conn.execute(
                "SELECT COUNT(DISTINCT user_id) FROM transactions WHERE ip_address = ?", (ip_address,)
            ).fetchone()[0]
        except Exception:
            return {"provider": "network_intelligence", "available": False, "reason": "evidence store unavailable"}
        return {
            "provider": "network_intelligence", "found": row is not None,
            "country": row[0] if row else None, "city": row[1] if row else None,
            "isp": row[2] if row else None,
            "suspicious_proxy": bool(row[3]) if row else False,
            "ip_account_count": int(users or 0),
            "note": "Shared IP is weak evidence because NAT/shared networks are common.",
        }
    finally:
        conn.close()


def call_guardrailed_evidence(
    tool_name: str,
    *,
    transaction_id: str,
    user_id: str,
    device_id: str | None = None,
    merchant_id: str | None = None,
    ip_address: str | None = None,
) -> dict[str, Any]:
    try:
        security_guard.authorize(
            tool_name,
            transaction_id=transaction_id,
            user_id=user_id,
            device_id=device_id,
            merchant_id=merchant_id,
            ip_address=ip_address,
        )
        if tool_name == "device_intelligence":
            return _device_intelligence(transaction_id, user_id, device_id or "")
        if tool_name == "merchant_intelligence":
            return _merchant_intelligence(transaction_id, merchant_id or "")
        if tool_name == "network_intelligence":
            return _network_intelligence(transaction_id, user_id, ip_address or "")
        raise GuardrailViolation("Unsupported tool")
    except GuardrailViolation as exc:
        security_guard.audit_denial(tool_name, str(exc))
        return {"provider": tool_name, "blocked_by_guardrail": True, "reason": str(exc)}
