"""
RazorRisk — Jev (TypeSafe System One) verification pass.

Runs only when TYPESAFE_API_KEY is set AND the dashboard's Jev toggle is on
(agent/mode_state.py — off by default, same "opt-in, never block the report"
posture as the LLM providers in agent/llm_investigator.py). This is a
verification layer, not a decision-maker: it never writes the fraud
hypothesis or chooses the recommended_action for the report. It runs after
agent/llm_investigator.py (or agent/deterministic_agent.py) has already
produced one, and cross-checks it against the same deterministic evidence.

Two independent checks against the evidence dict from agent/graph_agent.py,
never fed the investigator's own answer as an input:

1. independent_action (Choice): Jev picks a recommended_action from the same
   five-option set the investigator uses, from evidence alone. If this
   disagrees with what the investigator (LLM or deterministic) actually
   picked, that's a second opinion the investigator never saw, at a few
   hundredths of a cent and well under a second — cheap enough to run on
   every investigation, unlike a second LLM call, which would double
   cost/latency for a check that's supposed to be the safety net under the
   LLM's own answer.

2. hypothesis_grounded (Noul): given the investigator's fraud_hypothesis
   TEXT plus the evidence, is every specific claim in it traceable to the
   evidence? agent/llm_investigator.py's system prompt already instructs the
   LLM not to invent numbers/IDs, but nothing upstream verifies that
   instruction was followed — this is that check. (On the deterministic path this should normally score high since agent/deterministic_agent.py
   only ever renders evidence fields verbatim; a low score there points at
   a template bug, not a hallucination.)

Disagreement or a low grounding score sets verification_flag to
"REVIEW_RECOMMENDED" instead of silently trusting either path's answer —
mirrors the MODEL_DISAGREEMENT HITL trigger already used for GNN vs. XGBoost
(ml/risk_aggregator.py), applied one layer up.

api/routes_agent.py uses a CONSISTENT verdict, above
JEV_AUTO_RESOLVE_MIN_CONFIDENCE, to auto-resolve an already-queued HITL
review instead of leaving it pending — but only for reviews that weren't
already flagged for one of ml.decision_policy.MANDATORY_HUMAN_REASONS
(HIGH_IMPACT, MODEL_DISAGREEMENT, EVIDENCE_CONFLICT, MODEL_UNCERTAINTY).
Those are compliance/ambiguity triggers, not confidence triggers, so no
amount of Jev+LLM agreement is allowed to bypass them — same principle
ml/decision_policy.py already applies to its own AUTO_BLOCK_THRESHOLD
override. This module only ever computes the verdict; it never touches the
HITL queue itself.

Raises on any failure (bad key, network, non-200, malformed response) —
caller (agent/graph_agent.py) catches and logs, and the investigation report
is still returned in full without a verification section. This is a bonus
signal, never a blocker.

NOTE: Jev is in early access as of Sep 2026 and TypeSafe's own docs mark the
raw API's request/response shape as not yet stable — expect to revisit this
file if a field name here changes upstream.
"""
import requests
import time
import uuid
import threading

from config import (
    TYPESAFE_API_KEY, TYPESAFE_MODEL, TYPESAFE_API_BASE, TYPESAFE_TIMEOUT_SECONDS,
    JEV_AUTO_RESOLVE_MIN_CONFIDENCE,
    JEV_MAX_RETRIES, JEV_RETRY_BACKOFF_SECONDS,
    JEV_CIRCUIT_FAILURE_THRESHOLD, JEV_CIRCUIT_RESET_SECONDS,
)
from utils.logger import get_logger
from infra import observability

logger = get_logger("jev_verifier")

ACTION_CRITERIA = {
    "BLOCK_ACCOUNT_AND_HOLD_FUNDS": "Evidence shows an active, high-confidence fraud ring or carding pattern; freeze the account and hold funds immediately.",
    "HOLD_FOR_MANUAL_REVIEW": "Evidence is suspicious but not conclusive enough to act on unilaterally; a human analyst should look at it.",
    "TEMPORARY_VELOCITY_FREEZE": "The main signal is an abnormal transaction velocity spike rather than a confirmed fraud ring; freeze new transactions temporarily.",
    "REQUIRE_TWO_FACTOR_AUTHENTICATION": "Some risk signals are present, but a stronger identity check could resolve them without blocking the user outright.",
    "APPROVE_WITH_VERIFICATION": "Evidence overall looks like a false positive or low residual risk; approve after a light verification step.",
}

GROUNDING_THRESHOLD = 0.5
SUPPORTED_ACTIONS = frozenset(ACTION_CRITERIA)

_CIRCUIT_LOCK = threading.Lock()
_CIRCUIT_FAILURES = 0
_CIRCUIT_OPENED_AT = 0.0

def is_available() -> bool:
    """Whether Jev is configured and available for an attempted verification."""
    return bool(TYPESAFE_API_KEY)


def reset_circuit_breaker():
    global _CIRCUIT_FAILURES, _CIRCUIT_OPENED_AT
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES = 0
        _CIRCUIT_OPENED_AT = 0.0
        if observability.JEV_CIRCUIT_OPEN is not None:
            observability.JEV_CIRCUIT_OPEN.set(0)

def _circuit_open():
    global _CIRCUIT_FAILURES, _CIRCUIT_OPENED_AT
    with _CIRCUIT_LOCK:
        if _CIRCUIT_OPENED_AT and time.monotonic() - _CIRCUIT_OPENED_AT >= JEV_CIRCUIT_RESET_SECONDS:
            _CIRCUIT_FAILURES = 0
            _CIRCUIT_OPENED_AT = 0.0
            if observability.JEV_CIRCUIT_OPEN is not None:
                observability.JEV_CIRCUIT_OPEN.set(0)
        return bool(_CIRCUIT_OPENED_AT)

def _record_failure():
    global _CIRCUIT_FAILURES, _CIRCUIT_OPENED_AT
    with _CIRCUIT_LOCK:
        _CIRCUIT_FAILURES += 1
        if _CIRCUIT_FAILURES >= max(1, JEV_CIRCUIT_FAILURE_THRESHOLD):
            _CIRCUIT_OPENED_AT = time.monotonic()
            if observability.JEV_CIRCUIT_OPEN is not None:
                observability.JEV_CIRCUIT_OPEN.set(1)

def _record_success():
    reset_circuit_breaker()
def _call_systemone(state: str, questions: dict) -> dict:
    if _circuit_open():
        if observability.JEV_FAILURES is not None: observability.JEV_FAILURES.labels(reason="circuit_open").inc()
        raise RuntimeError("Jev circuit breaker is open; verification temporarily unavailable.")
    attempts = max(0, JEV_MAX_RETRIES) + 1
    for attempt in range(attempts):
        try:
            resp = requests.post(f"{TYPESAFE_API_BASE}/v1/systemone", headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}", "Content-Type": "application/json"}, json={"model": TYPESAFE_MODEL, "state": state, "questions": questions}, timeout=TYPESAFE_TIMEOUT_SECONDS)
            status_code = getattr(resp, "status_code", None)
            if isinstance(status_code, int) and not 200 <= status_code < 300:
                error = requests.HTTPError(f"HTTP {status_code}", response=resp)
                if status_code == 429 or status_code >= 500:
                    raise error
                raise RuntimeError(f"Jev HTTP request failed: HTTP {status_code}") from error
            data = resp.json(); _record_success(); return data
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
        except requests.HTTPError as exc:
            if not (exc.response is not None and (exc.response.status_code == 429 or exc.response.status_code >= 500)):
                raise RuntimeError(f"Jev HTTP request failed: {exc}") from exc
            last_exc = exc
        if attempt < attempts - 1:
            if observability.JEV_RETRIES is not None: observability.JEV_RETRIES.inc()
            time.sleep(JEV_RETRY_BACKOFF_SECONDS * (2 ** attempt))
    _record_failure()
    if observability.JEV_FAILURES is not None: observability.JEV_FAILURES.labels(reason="transient_exhausted").inc()
    raise RuntimeError(f"Jev request failed after {attempts} attempt(s): {last_exc}") from last_exc

def verify_investigation(txn_payload: dict, risk_summary: dict, evidence: dict,
                          fraud_hypothesis: str, recommended_action: str) -> dict:
    """Return a verification result; provider failures remain fail-open to the caller."""
    if not TYPESAFE_API_KEY:
        if observability.JEV_UNAVAILABLE is not None: observability.JEV_UNAVAILABLE.inc()
        raise RuntimeError("TYPESAFE_API_KEY is not configured.")

    evidence_state = {
        "risk_scores": risk_summary,
        "graph_evidence": evidence["graph_evidence"],
        "history_evidence": evidence["history_evidence"],
        "device_evidence": evidence["device_evidence"],
        "model_evidence": evidence["model_evidence"],
    }

    action_resp = _call_systemone(
        state=str(evidence_state),
        questions={
            "recommended_action": {
                "type": "choice",
                "instructions": "Given this fraud-risk evidence for one transaction, which action best fits?",
                "criteria": ACTION_CRITERIA,
            }
        },
    )
    try:
        action_answer = action_resp["answers"]["recommended_action"]
        independent_action = action_answer["choice"]
        if independent_action not in SUPPORTED_ACTIONS: raise ValueError(f"unsupported action: {independent_action!r}")
        confidence = float(action_answer.get("confidence", 0.0) or 0.0)
        if not 0.0 <= confidence <= 1.0: raise ValueError(f"invalid action confidence: {confidence!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Jev action response: {exc}") from exc

    grounding_resp = _call_systemone(
        state=f"Evidence:\n{evidence_state}\n\nInvestigator's fraud hypothesis:\n{fraud_hypothesis}",
        questions={
            "hypothesis_grounded": {
                "type": "noul",
                "instructions": (
                    "Is every specific claim in the investigator's fraud hypothesis "
                    "(numbers, counts, device/IP/account details) traceable to the "
                    "evidence provided, with nothing invented?"
                ),
            }
        },
    )
    try:
        grounded_probability = float(grounding_resp["answers"]["hypothesis_grounded"]["noul"])
        if not 0.0 <= grounded_probability <= 1.0: raise ValueError(f"invalid grounding probability: {grounded_probability!r}")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Malformed Jev grounding response: {exc}") from exc

    actions_agree = independent_action == recommended_action
    is_grounded = grounded_probability >= GROUNDING_THRESHOLD
    verification_flag = "CONSISTENT" if (actions_agree and is_grounded) else "REVIEW_RECOMMENDED"
    eligible_for_auto_resolve = verification_flag == "CONSISTENT" and confidence >= JEV_AUTO_RESOLVE_MIN_CONFIDENCE

    result = {
        "jev_request_id": f"JEV_{uuid.uuid4().hex[:12]}",
        "jev_model": action_resp.get("model", TYPESAFE_MODEL),
        "independent_action": independent_action,
        "independent_action_confidence": confidence,
        "independent_action_probabilities": action_answer.get("probabilities"),
        "actions_agree": actions_agree,
        "hypothesis_grounded_probability": grounded_probability,
        "hypothesis_grounded": is_grounded,
        "verification_flag": verification_flag,
        "eligible_for_auto_resolve": eligible_for_auto_resolve,
    }

    if observability.JEV_REQUESTS is not None: observability.JEV_REQUESTS.labels(outcome="success").inc()
    if verification_flag == "CONSISTENT" and observability.JEV_CONSISTENT is not None: observability.JEV_CONSISTENT.inc()
    if verification_flag == "REVIEW_RECOMMENDED" and observability.JEV_REVIEW_RECOMMENDED is not None: observability.JEV_REVIEW_RECOMMENDED.inc()
    if not actions_agree and observability.JEV_ACTION_DISAGREEMENTS is not None: observability.JEV_ACTION_DISAGREEMENTS.inc()
    if not is_grounded and observability.JEV_GROUNDING_FAILURES is not None: observability.JEV_GROUNDING_FAILURES.inc()
    if verification_flag == "REVIEW_RECOMMENDED":
        logger.warning(
            f"[Jev] Flagged for review — investigator picked '{recommended_action}', "
            f"Jev independently picked '{action_answer['choice']}' (agree={actions_agree}); "
            f"hypothesis grounding={grounded_probability:.2f}."
        )
    else:
        logger.info(
            f"[Jev] Verification consistent (confidence={confidence:.2f}, "
            f"auto-resolve eligible={eligible_for_auto_resolve})."
        )

    return result
