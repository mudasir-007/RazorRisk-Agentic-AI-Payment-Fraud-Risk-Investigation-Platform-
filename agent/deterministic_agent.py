"""
RazorRisk — deterministic fallback investigator.

Used when no LLM API key is configured (or the LLM call fails for any
reason). This is plain rule-based pattern matching over the same evidence
the LLM path would have seen — genuinely useful engineering, and labeled
as exactly what it is in every report it produces (agent_mode is always
"deterministic_fallback", never dressed up as agentic reasoning). A demo
running with zero API keys configured still produces a complete,
correctly-reasoned investigation report — it just says plainly that a
human wrote these rules, not a model.

IMPORTANT — evidence confluence:
This module used to branch on shared_device_account_count / shared_ip_
account_count ALONE (>=3 / >=4) to recommend BLOCK/HOLD. That is exactly
the connectivity-only false-positive pattern ml/risk_aggregator.py's
evidence_confluence gate was built to fix (a 7-person hostel or a 40-person
carrier-NAT IP has high shared_ip_account_count with zero fraud behavior).
The fix there did NOT originally propagate here — meaning a hostel/carrier-
NAT/shared-device transaction that reached investigation (e.g. a borderline
overall score, or a manually triggered investigation) could still get a
human-facing report saying "High-confidence device sharing fraud ring
detected" and a BLOCK_ACCOUNT_AND_HOLD_FUNDS recommendation, based on
connectivity alone. Found by actually running the hostel scenario through
this path, not by inspection.

Fixed to require the same confluence risk_aggregator.py requires: strong
fingerprint sharing (shared_device>=3 or shared_ip>=5) AND a behavioral
anomaly (velocity>=5, or amount far above this user's own historical
average — a zscore proxy, since the raw amount_zscore_prior feature isn't
threaded through to this module) before recommending BLOCK/HOLD.
Connectivity alone now produces an explicit "looks like a benign
shared-fingerprint community" hypothesis with a light-touch action, instead
of silence or an escalation it doesn't deserve.
"""

# Mirrors ml/risk_aggregator.py's thresholds — kept in sync deliberately so
# the risk score and the human-facing report never disagree about what
# counts as "strong" fingerprint sharing.
SHARED_DEVICE_THRESHOLD = 3
SHARED_IP_THRESHOLD = 5
VELOCITY_ANOMALY_THRESHOLD = 5
CARDING_VELOCITY_THRESHOLD = 8
# risk_aggregator.py has the real amount_zscore_prior feature available;
# this module only gets historical_avg_amount, so "amount is at least this
# many multiples of the user's own historical average" stands in as the
# zscore proxy. cold-start users (no history) fall back to an absolute
# floor since a multiple-of-zero is undefined.
AMOUNT_MULTIPLE_ANOMALY = 3.0
COLD_START_AMOUNT_FLOOR = 50000.0


def _has_behavioral_anomaly(velocity_1h: int, txn_payload: dict, history_evidence: dict) -> bool:
    if velocity_1h >= VELOCITY_ANOMALY_THRESHOLD:
        return True
    amount = float(txn_payload.get("amount", 0.0))
    hist_avg = history_evidence.get("historical_avg_amount", 0.0)
    if hist_avg > 0:
        return amount >= AMOUNT_MULTIPLE_ANOMALY * hist_avg
    return amount > COLD_START_AMOUNT_FLOOR


def determine_fraud_hypothesis(txn_payload: dict, graph_evidence: dict, history_evidence: dict, device_evidence: dict, velocity_1h: int = 1):
    """Returns (fraud_hypothesis: str, recommended_action: str, action_rationale: str).
    velocity_1h is the server-computed real value (ml/risk_aggregator.py),
    passed explicitly rather than read from txn_payload — txn_payload no
    longer carries a velocity field at all (it used to be client-supplied
    and got removed once that was shown to be a spoofable signal)."""
    device_id = txn_payload.get("device_id", "UNKNOWN_DEV")
    ip_address = txn_payload.get("ip_address", "0.0.0.0")
    amount = float(txn_payload.get("amount", 0.0))

    has_strong_fingerprint_sharing = (
        graph_evidence["shared_device_account_count"] >= SHARED_DEVICE_THRESHOLD
        or graph_evidence["shared_ip_account_count"] >= SHARED_IP_THRESHOLD
    )
    behavioral_anomaly = _has_behavioral_anomaly(velocity_1h, txn_payload, history_evidence)

    if graph_evidence["shared_device_account_count"] >= SHARED_DEVICE_THRESHOLD and behavioral_anomaly:
        hypothesis = (
            f"High-confidence device sharing fraud ring detected. Device '{device_id}' is shared "
            f"across {graph_evidence['shared_device_account_count']} distinct user accounts "
            f"(community fraud ratio: {graph_evidence['community_fraud_ratio']:.0%}), AND this "
            f"transaction itself is behaviorally anomalous (velocity {velocity_1h}/hr vs. historical "
            f"avg amount ₹{history_evidence['historical_avg_amount']:,.2f}). Device fingerprint sharing "
            f"combined with anomalous behavior is characteristic of automated account farming or "
            f"stolen-credential testing — connectivity alone would not be enough to conclude this."
        )
        return hypothesis, "BLOCK_ACCOUNT_AND_HOLD_FUNDS", (
            f"Device fingerprint '{device_id}' exhibits severe account multi-tenancy "
            f"({graph_evidence['shared_device_account_count']} linked accounts) together with an "
            f"anomalous transaction, not connectivity in isolation."
        )

    if (graph_evidence["shared_ip_account_count"] >= SHARED_IP_THRESHOLD or device_evidence["is_suspicious_proxy"]) and behavioral_anomaly:
        hypothesis = (
            f"Proxy/VPN botnet pattern detected. Transaction originates from IP '{ip_address}' "
            f"({device_evidence['isp']}, {device_evidence['country']}), shared by "
            f"{graph_evidence['shared_ip_account_count']} separate user accounts, AND this "
            f"transaction is behaviorally anomalous for this user (velocity {velocity_1h}/hr, "
            f"amount ₹{amount:,.2f} vs. historical avg ₹{history_evidence['historical_avg_amount']:,.2f})."
        )
        return hypothesis, "HOLD_FOR_MANUAL_REVIEW", (
            f"Traffic routed via a proxy/VPN IP '{ip_address}' linked to "
            f"{graph_evidence['shared_ip_account_count']} accounts, combined with anomalous behavior."
        )

    if velocity_1h >= CARDING_VELOCITY_THRESHOLD:
        hypothesis = (
            f"High-velocity carding/micro-transaction pattern. User executed "
            f"{velocity_1h} transactions within one hour, well above their "
            f"historical baseline of {history_evidence['historical_max_velocity_1h']} txns/hr."
        )
        return hypothesis, "TEMPORARY_VELOCITY_FREEZE", (
            f"Abnormal transaction frequency ({velocity_1h} txns/hr) is consistent "
            f"with scripted/automated activity."
        )

    if has_strong_fingerprint_sharing and not behavioral_anomaly:
        hypothesis = (
            f"Shared fingerprint observed ({graph_evidence['shared_device_account_count']} accounts on "
            f"this device, {graph_evidence['shared_ip_account_count']} accounts on this IP), but no "
            f"accompanying behavioral anomaly — velocity ({velocity_1h}/hr) and amount (₹{amount:,.2f} vs. "
            f"historical avg ₹{history_evidence['historical_avg_amount']:,.2f}) are both within this "
            f"user's normal range. This shape matches a benign shared-fingerprint community (family, "
            f"hostel, office device, or carrier-grade NAT) more than a fraud ring — identity overlap "
            f"alone is not evidence of fraud."
        )
        return hypothesis, "APPROVE_WITH_VERIFICATION", (
            f"Connectivity signal present but no behavioral anomaly accompanies it; light-touch "
            f"verification rather than a block, consistent with the evidence-confluence policy."
        )

    hypothesis = (
        f"Elevated transaction risk. Amount (₹{amount:,.2f}) is notably higher than this user's "
        f"historical average (₹{history_evidence['historical_avg_amount']:,.2f}), without a clear "
        f"device/IP-sharing fraud-ring pattern to corroborate it."
    )
    return hypothesis, "REQUIRE_TWO_FACTOR_AUTHENTICATION", (
        f"Amount anomaly (₹{amount:,.2f}) on its own, but no single strong fraud-ring indicator."
    )
