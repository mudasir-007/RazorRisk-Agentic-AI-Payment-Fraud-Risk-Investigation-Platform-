
"""Post-ML decision policy.

ML produces risk evidence. This module decides whether the transaction can be
handled automatically, needs additional evidence, or must enter human review.
It deliberately does not turn graph connectivity into a fraud verdict.
"""
from __future__ import annotations

from typing import Any
from security.evidence_api import call_guardrailed_evidence
from utils.logger import get_logger

logger = get_logger("decision_policy")

HITL_REASONS = {
    "MODEL_UNCERTAINTY",
    "MODEL_DISAGREEMENT",
    "HIGH_IMPACT",
    "EVIDENCE_CONFLICT",
    "NOVEL_BEHAVIOR",
}

# Confidence floor above which the stacker's calibrated probability is
# treated as decisive enough to act on without a human in the loop. Deliberately
# read off the RAW stacker_calibrated_score (calibrated_prob), not the
# velocity-inflated final_risk_score/risk_tier: velocity_mult can push a 0.70
# calibrated probability up to a CRITICAL tier (100 capped), and auto-blocking
# on that inflated number would mean "moved fast" alone could trigger an
# irreversible action. Auto-block only fires on genuine model confidence.
AUTO_BLOCK_THRESHOLD = 0.95

# Reasons that indicate real ambiguity or a business requirement for a human
# in the loop, and therefore should NEVER be bypassed by high model
# confidence alone:
#   - MODEL_UNCERTAINTY / MODEL_DISAGREEMENT / EVIDENCE_CONFLICT: the models
#     (or model vs. evidence) don't agree, so "confidence" isn't trustworthy.
#   - HIGH_IMPACT: dual control on large-dollar transactions is a standard
#     payments/AML control (segregation of duties, audit trail, asymmetric
#     cost of an autonomous false positive on a large transfer) independent
#     of how confident the model is.
# NOVEL_BEHAVIOR (high velocity alone) is deliberately excluded: velocity is
# a pattern signal already folded into the score, not an ambiguity signal,
# so it should not by itself override a high-confidence auto-block.
MANDATORY_HUMAN_REASONS = {
    "MODEL_UNCERTAINTY",
    "MODEL_DISAGREEMENT",
    "EVIDENCE_CONFLICT",
    "HIGH_IMPACT",
}


def apply_decision_policy(txn: dict[str, Any], risk: dict[str, Any]) -> dict[str, Any]:
    tab = float(risk["tabular_score"]) / 100.0
    gnn = float(risk["gnn_score"]) / 100.0
    stack = float(risk["stacker_calibrated_score"]) / 100.0
    amount = float(txn.get("amount", 0))
    # Velocity is always part of the scoring/policy path. The frontend toggle
    # only selects its source: CLIENT (simulation/trust mode) or BACKEND
    # (calculated from transaction history).
    velocity = int(risk.get("effective_velocity_1h", risk.get("velocity_1h", 1)))
    graph = risk.get("graph_evidence", {})

    reasons: list[str] = []
    evidence: dict[str, Any] = {}

    # Bug #30: graph.evidence_available is False only on a user's genuine
    # first-ever transaction (no persisted history to build a graph node
    # from yet — see ml/risk_aggregator.py::live_gnn_score_and_evidence).
    # In that case gnn == 0.0 is an absence of data, not a second model's
    # considered opinion, so comparing it against the tabular score isn't
    # "disagreement" — it's the same information gap MODEL_UNCERTAINTY
    # already exists to name when the *combined* score is ambiguous.
    # Treating a missing signal as a contradicting one used to force every
    # first-time transaction with real tabular signal into human review and
    # cap its calibrated score below AUTO_BLOCK_THRESHOLD, permanently.
    graph_evidence_available = graph.get("evidence_available", True)

    # These are policy triggers, not fraud labels.
    if 0.35 <= stack <= 0.65:
        reasons.append("MODEL_UNCERTAINTY")
    if graph_evidence_available and abs(tab - gnn) >= 0.45:
        reasons.append("MODEL_DISAGREEMENT")
    if amount >= 50000:
        reasons.append("HIGH_IMPACT")
    if (
        float(graph.get("shared_ip_accounts", 1)) >= 5
        and float(graph.get("shared_device_accounts", 1)) <= 2
        and stack < 0.70
    ):
        # Strong shared-IP evidence with otherwise modest model risk is
        # deliberately treated as a conflict, not as fraud.
        reasons.append("EVIDENCE_CONFLICT")
    if velocity >= 10 and stack < 0.55:
        reasons.append("NOVEL_BEHAVIOR")

    # Call external-style evidence only for transactions where it can resolve
    # ambiguity or materially improve an investigation. Every call passes
    # through the allowlist + input validation + rate limiter.
    should_enrich = (
        risk["risk_tier"] in {"MEDIUM", "HIGH", "CRITICAL"}
        or bool(reasons)
        or bool(txn.get("is_vpn_proxy") or txn.get("is_suspicious_proxy"))
    )

    if should_enrich:
        evidence["device"] = call_guardrailed_evidence(
            "device_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            device_id=txn.get("device_id"),
        )
        evidence["network"] = call_guardrailed_evidence(
            "network_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            ip_address=txn.get("ip_address"),
        )
        evidence["merchant"] = call_guardrailed_evidence(
            "merchant_intelligence",
            transaction_id=txn["transaction_id"],
            user_id=txn["user_id"],
            merchant_id=txn.get("merchant_id"),
        )

        # A provider saying "shared by many accounts" is not itself a HITL
        # trigger. It becomes useful when it conflicts with ML evidence.
        if (
            evidence["device"].get("device_account_count", 0) >= 3
            and stack < 0.45
        ):
            if "EVIDENCE_CONFLICT" not in reasons:
                reasons.append("EVIDENCE_CONFLICT")

    hitl_required = bool(reasons) and (
        risk["risk_tier"] in {"MEDIUM", "HIGH", "CRITICAL"} or
        "HIGH_IMPACT" in reasons or
        "MODEL_DISAGREEMENT" in reasons
    )

    # Auto-block override: even when hitl_required tripped above (e.g. a
    # CRITICAL-tier txn with a NOVEL_BEHAVIOR reason), a sufficiently
    # confident, unambiguous stacker score should not still be routed to a
    # human queue. Every fraud case landing on a human is what this override
    # exists to fix: it only fires when confidence is decisive AND none of
    # the mandatory-human reasons (disagreement/conflict/uncertainty/
    # high-impact) are present.
    auto_block = (
        stack >= AUTO_BLOCK_THRESHOLD
        and not (set(reasons) & MANDATORY_HUMAN_REASONS)
    )
    if auto_block:
        hitl_required = False

    if auto_block:
        final_decision = "BLOCK"
    elif hitl_required:
        final_decision = "HUMAN_REVIEW"
    elif risk["risk_tier"] == "CRITICAL":
        final_decision = "BLOCK_PENDING_REVIEW"
    elif risk["risk_tier"] == "HIGH":
        final_decision = "HOLD_FOR_INVESTIGATION"
    elif risk["risk_tier"] == "MEDIUM":
        final_decision = "MONITOR"
    else:
        final_decision = "APPROVE"

    return {
        "decision": final_decision,
        "hitl_required": hitl_required,
        "review_reasons": reasons,
        "external_evidence": evidence,
        "auto_blocked": auto_block,
        "policy_version": "v2.1-confidence-auto-block",
    }
