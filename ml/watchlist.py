"""Repeat-MEDIUM-risk watchlist.

Before this existed, `MONITOR` was purely a label: a MEDIUM-tier
transaction was persisted and shown in the dashboard, but nothing about it
carried forward, so a user who tripped MEDIUM risk five times in a row got
scored exactly like a clean user every single time.

This module is the read/write boundary for a lightweight fix: a MONITOR
decision soft-flags its user for `WATCHLIST_TTL_HOURS`. `is_watchlisted()`
is called from the live scoring path (`ml/risk_aggregator.py`) to apply an
explicit, separately-labeled score multiplier — the same overlay pattern
already used for velocity/proxy — on that user's NEXT transaction.
`refresh_watchlist()` is called from the API layer
(`api/routes_transactions.py`) after a MONITOR decision to (re)set the
flag.

Deliberately NOT a HITL trigger by itself (see
`ml/decision_policy.py`'s `MANDATORY_HUMAN_REASONS`) — repeat medium-risk
behavior tightens the automated tiering/auto-block path instead of
creating more human review work. If the multiplier pushes a transaction
into HIGH/CRITICAL, it still goes through the exact same decision policy
as any other transaction: auto-blocked if the stacker is confident and
unambiguous, or routed to a human only if a mandatory reason applies.

One row per user_id, refreshed (not appended) on every MONITOR outcome.
Entries are read but never deleted on a normal request — they age out via
`expires_at`, which keeps this module a pure read/write boundary with no
background job required. `clear_watchlist()` exists for admin/testing use.
"""
from __future__ import annotations

from typing import Optional

from config import WATCHLIST_TTL_HOURS
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("watchlist")


def is_watchlisted(user_id: str) -> bool:
    """True if user_id has an unexpired watchlist entry."""
    if not user_id:
        return False
    conn = get_raw_sqlite_connection()
    try:
        row = conn.execute(
            "SELECT 1 FROM user_watchlist WHERE user_id = ? AND expires_at > datetime('now') LIMIT 1",
            (user_id,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def refresh_watchlist(
    user_id: str,
    transaction_id: str,
    reason: str = "MEDIUM_RISK_MONITOR",
    ttl_hours: Optional[int] = None,
) -> None:
    """(Re)flag user_id, extending the TTL from now. Idempotent per user —
    a user already on the watchlist just gets their expiry pushed out
    again rather than accumulating duplicate rows."""
    if not user_id:
        return
    ttl = ttl_hours if ttl_hours is not None else WATCHLIST_TTL_HOURS
    conn = get_raw_sqlite_connection()
    try:
        conn.execute(
            """INSERT INTO user_watchlist (user_id, reason, source_transaction_id, flagged_at, expires_at)
               VALUES (?, ?, ?, datetime('now'), datetime('now', ?))
               ON CONFLICT(user_id) DO UPDATE SET
                   reason = excluded.reason,
                   source_transaction_id = excluded.source_transaction_id,
                   flagged_at = excluded.flagged_at,
                   expires_at = excluded.expires_at""",
            (user_id, reason, transaction_id, f"+{ttl} hours"),
        )
        conn.commit()
        logger.info(
            "Watchlist refreshed: user=%s txn=%s ttl_hours=%s reason=%s",
            user_id, transaction_id, ttl, reason,
        )
    finally:
        conn.close()


def clear_watchlist(user_id: str) -> None:
    """Remove user_id's watchlist entry outright. Not called from the live
    scoring/API path — entries are left to expire via TTL by design, so a
    human reviewer clearing a case doesn't need to know this table exists.
    Exposed for admin tooling and tests."""
    if not user_id:
        return
    conn = get_raw_sqlite_connection()
    try:
        conn.execute("DELETE FROM user_watchlist WHERE user_id = ?", (user_id,))
        conn.commit()
    finally:
        conn.close()
