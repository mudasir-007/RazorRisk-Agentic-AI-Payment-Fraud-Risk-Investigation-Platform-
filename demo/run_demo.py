"""
RazorRisk live demo script.

Fires 5 real transactions at the running API (POST /api/v1/transactions/score)
covering LOW -> MEDIUM -> HIGH -> CRITICAL, plus a bonus case showing the
dual-control safety rail on large-value transactions. Each one uses a fresh,
never-before-seen identity, so this exercises the "first transaction from a
brand-new customer" cold-start path -- the same path Bug #30 (see
PROJECT_WORKFLOW.md) fixed: without that fix, the BLOCK scenario below would
have dead-ended in HUMAN_REVIEW instead of an automatic BLOCK, because a
brand-new user's GNN score used to be forced to 0.0 and read by the stacker
as a confident "not fraud" vote, permanently capping the calibrated
probability below the auto-block threshold.

Why this script checks the clock
---------------------------------
The live /score endpoint has no `timestamp` field -- by design: a
real-time scoring endpoint should always score a transaction using the
ACTUAL current time, not a client-supplied one (accepting a client
timestamp would let a fraud attempt disguise a 2 AM transaction as an
afternoon one and dodge the model's time-of-day signal). That's correct
behavior for a live product, but it does mean this demo can't force a
specific simulated hour the way ml/risk_aggregator.py's own functions and
test suite can (see Bug #29 in PROJECT_WORKFLOW.md).

The tabular model weighs hour-of-day heavily -- heavily enough that the
SAME transaction can swing from single digits to 90%+ purely based on
whether it's run at 2 PM or 2 AM. That's a real, disclosed finding (see
PROJECT_WORKFLOW.md's note on is_night over-reliance), not something this
script tries to hide. So instead of picking one fixed set of demo amounts
and hoping you happen to run this at the right hour, it checks the real
current hour and picks from two pre-validated parameter sets so every
scenario reliably lands on its intended tier either way. The one exception
is BLOCK: a brand-new identity's calibrated probability can only clear the
95% auto-block bar during night hours in the currently shipped model --
during the day the best HONEST outcome is CRITICAL / BLOCK_PENDING_REVIEW
(held for a mandatory human sign-off rather than instantly executed). The
script tells you which case you're in rather than silently substituting one
for the other.

Usage
-----
1. Start the API in one terminal:
     python run.py
   (or: uvicorn api.main:app --reload --port 8000)

2. In another terminal:
     python demo/run_demo.py

3. Open http://localhost:8000/dashboard/ in a browser BEFORE step 2 so you
   can point at the Recent Transactions / Graph Topology / HITL Review tabs
   live as each request lands, right after this script prints it.

Each scenario prints a one-line narration, the payload sent, the model
breakdown, and the final decision. A closing table recaps all five.
"""
from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime

import requests

API_BASE = "http://localhost:8000"
SCORE_URL = f"{API_BASE}/api/v1/transactions/score"

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
COLORS = {
    "LOW": "\033[92m",         # green
    "MEDIUM": "\033[93m",      # yellow
    "HIGH": "\033[38;5;208m",  # orange
    "CRITICAL": "\033[91m",    # red
}

# hour_of_day in {0..5} is what ml/train_tabular_model.py treats as
# is_night=1 -- matched here so this script's day/night split lines up with
# the model's own, not an arbitrary guess.
IS_NIGHT = datetime.now().hour in {0, 1, 2, 3, 4, 5}


def _fresh_id(label: str) -> str:
    """A guaranteed-never-seen-before identity for this demo run, so every
    run genuinely exercises the cold-start path rather than accumulating
    history across repeated demo runs."""
    return f"{label}_{uuid.uuid4().hex[:6].upper()}"


# Both parameter sets were swept directly against this project's shipped
# ml/models/ artifacts and razor_risk.db (see PROJECT_WORKFLOW.md Bug #30)
# -- not guessed. "night" always lands the intended tier/decision; "day"
# does too, EXCEPT scenario 4, where it honestly lands on
# BLOCK_PENDING_REVIEW instead of an instant BLOCK (see module docstring).
NIGHT = dict(normal=2000, medium=5000, high=(8000, 1), block=(12000, 1), dualctrl=65000)
DAY = dict(normal=2000, medium=8000, high=(5000, 2), block=(15000, 2), dualctrl=65000)
PARAMS = NIGHT if IS_NIGHT else DAY

SCENARIOS = [
    dict(
        title="1. NORMAL — everyday purchase",
        narration=(
            "A brand-new customer's first transaction: an ordinary amount, "
            "no repeat activity. Should sail through automatically."
        ),
        user_label="CUSTOMER_NORMAL",
        amount=PARAMS["normal"],
        velocity_1h=1,
        is_vpn_proxy=False,
        is_suspicious_proxy=False,
        merchant_id="MCH_010",
        expect="LOW / APPROVE",
    ),
    dict(
        title="2. MEDIUM — model isn't confident either way",
        narration=(
            "A larger amount from a first-ever identity. The model can't "
            "confidently clear or condemn it, so it's routed to a human "
            "reviewer instead of auto-decided."
        ),
        user_label="CUSTOMER_MEDIUM",
        amount=PARAMS["medium"],
        velocity_1h=1,
        is_vpn_proxy=False,
        is_suspicious_proxy=False,
        merchant_id="MCH_010",
        expect="MEDIUM / HUMAN_REVIEW",
    ),
    dict(
        title="3. HIGH — escalated amount / repeat activity",
        narration=(
            "Escalated further. The model is now confident enough to flag "
            "it HIGH and hold it for investigation, without yet needing a "
            "human just to decide."
        ),
        user_label="CUSTOMER_HIGH",
        amount=PARAMS["high"][0],
        velocity_1h=PARAMS["high"][1],
        is_vpn_proxy=False,
        is_suspicious_proxy=False,
        merchant_id="MCH_010",
        expect="HIGH / HOLD_FOR_INVESTIGATION",
    ),
    dict(
        title="4. BLOCK — largest amount + VPN",
        narration=(
            "Escalated further again, plus a VPN/proxy. This is deep in "
            "CRITICAL territory either way -- watch below whether it clears "
            "the auto-block bar or gets held for a mandatory human sign-off; "
            "the model's confidence for a brand-new identity is sensitive "
            "enough to real-world time-of-day (a disclosed finding, see "
            "PROJECT_WORKFLOW.md) that this genuinely depends on the exact "
            "hour you're running this, not on anything scripted."
        ),
        user_label="CUSTOMER_BLOCK",
        amount=PARAMS["block"][0],
        velocity_1h=PARAMS["block"][1],
        is_vpn_proxy=True,
        is_suspicious_proxy=True,
        merchant_id="MCH_010",
        expect="CRITICAL, and either BLOCK or BLOCK_PENDING_REVIEW",
    ),
    dict(
        title="5. BONUS — large amount = mandatory dual control",
        narration=(
            "Even when the model is extremely confident, a transaction "
            "above the high-value threshold is never auto-blocked -- it's "
            "always sent to a human. This is a deliberate payments/AML "
            "control (segregation of duties), independent of how sure the "
            "model is."
        ),
        user_label="CUSTOMER_DUALCTRL",
        amount=PARAMS["dualctrl"],
        velocity_1h=1,
        is_vpn_proxy=False,
        is_suspicious_proxy=False,
        merchant_id="MCH_010",
        expect="CRITICAL / HUMAN_REVIEW (HIGH_IMPACT dual control)",
    ),
]


def check_server() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.ok
    except requests.exceptions.ConnectionError:
        return False


def run_scenario(scenario: dict) -> dict | None:
    user_id = _fresh_id(scenario["user_label"])
    payload = {
        "user_id": user_id,
        "device_id": f"DEV_{user_id}",
        "ip_address": f"103.21.58.{(hash(user_id) % 200) + 10}",
        "merchant_id": scenario["merchant_id"],
        "amount": scenario["amount"],
        "currency": "INR",
        "is_vpn_proxy": scenario["is_vpn_proxy"],
        "is_suspicious_proxy": scenario["is_suspicious_proxy"],
        "velocity_enabled": True,
        "velocity_1h": scenario["velocity_1h"],
    }

    print(f"\n{BOLD}{scenario['title']}{RESET}")
    print(f"{DIM}{scenario['narration']}{RESET}")
    print(f"  -> POST /api/v1/transactions/score")
    print(f"     user_id={user_id}  amount=₹{scenario['amount']:,}  "
          f"velocity_1h={scenario['velocity_1h']}  "
          f"vpn={scenario['is_vpn_proxy']}  suspicious_proxy={scenario['is_suspicious_proxy']}")
    print(f"     {DIM}expected: {scenario['expect']}{RESET}")

    try:
        resp = requests.post(SCORE_URL, json=payload, timeout=15)
    except requests.exceptions.RequestException as e:
        print(f"  {COLORS['CRITICAL']}Request failed: {e}{RESET}")
        return None

    if not resp.ok:
        print(f"  {COLORS['CRITICAL']}HTTP {resp.status_code}: {resp.text}{RESET}")
        return None

    data = resp.json()
    risk = data["risk_evaluation"]
    tier = risk["risk_tier"]
    color = COLORS.get(tier, "")

    print(f"  {color}{BOLD}Tier: {tier}   Score: {risk['risk_score']}/100   "
          f"Decision: {risk['decision']}{RESET}")
    print(f"     tabular={risk['tabular_score']}%  gnn={risk['gnn_score']}%  "
          f"stacker={risk['stacker_calibrated_score']}%  "
          f"velocity_mult={risk['velocity_multiplier']}x")
    if risk.get("review_reasons"):
        print(f"     review reasons: {', '.join(risk['review_reasons'])}")
    if risk.get("auto_blocked"):
        print(f"     {COLORS['CRITICAL']}auto-blocked by model confidence (no human needed){RESET}")
    elif scenario["user_label"] == "CUSTOMER_BLOCK" and tier == "CRITICAL":
        print(f"     {DIM}CRITICAL, but confidence landed just under the 95% "
              f"auto-block bar this run -- held for BLOCK_PENDING_REVIEW "
              f"(a mandatory human sign-off) instead of executed instantly. "
              f"See PROJECT_WORKFLOW.md Bug #29/#30.{RESET}")
    if data.get("needs_investigation"):
        print(f"     {DIM}-> would trigger agent investigation "
              f"(POST /api/v1/investigations/run/{data['transaction_id']}){RESET}")
    print(f"     transaction_id={data['transaction_id']}")

    return {
        "title": scenario["title"],
        "user_id": user_id,
        "amount": scenario["amount"],
        "tier": tier,
        "score": risk["risk_score"],
        "decision": risk["decision"],
    }


def main():
    print(f"{BOLD}RazorRisk — live risk-tier demo{RESET}")
    print(f"Target: {API_BASE}")
    print(f"{DIM}Current server hour: {datetime.now().strftime('%H:%M')} "
          f"({'night-tuned' if IS_NIGHT else 'day-tuned'} parameter set)"
          f"{RESET}\n")

    if not check_server():
        print(f"{COLORS['CRITICAL']}Server not reachable at {API_BASE}.{RESET}")
        print("Start it first:  python run.py   (or)  uvicorn api.main:app --port 8000")
        sys.exit(1)

    print(f"{COLORS['LOW']}Server is up.{RESET} "
          f"Open {API_BASE}/dashboard/ now if you want to watch these land live.\n")
    time.sleep(1)

    results = []
    for scenario in SCENARIOS:
        results.append(run_scenario(scenario))
        time.sleep(0.6)

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Summary{RESET}")
    print(f"{'Scenario':<45}{'Tier':<11}{'Score':<8}{'Decision'}")
    print("-" * 72)
    for r in results:
        if r is None:
            continue
        color = COLORS.get(r["tier"], "")
        print(f"{r['title']:<45}{color}{r['tier']:<11}{RESET}{r['score']:<8}{r['decision']}")
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"\n{DIM}Every identity above was fresh, so this exercised the cold-start "
          f"path end to end (Bug #30). See PROJECT_WORKFLOW.md for Bug #29 and "
          f"Bug #30 -- why this script checks the clock instead of assuming one.{RESET}")


if __name__ == "__main__":
    main()
