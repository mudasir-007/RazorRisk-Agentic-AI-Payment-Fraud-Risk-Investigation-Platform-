"""
RazorRisk — risk aggregator: stacker training + live composite scoring.

Two responsibilities, kept in one file because they share the same
contract (tabular_score, gnn_score, shared_device_norm, shared_ip_norm ->
combined probability):

1. train_stacker(): offline. Combines the synthetic-domain tabular model's,
the synthetic-domain GNN's, AND
   live graph-evidence signals (shared device/IP account counts, normalized)
   via a small logistic regression stacker — LEARNED combination weights,
   not an arbitrary hand-picked average or a rule bolted onto the output.
   History of this file's aggregation logic, oldest to newest:
     - fixed 0.35/0.45/0.20 weighted sum of tabular+GNN only — an
       unvalidated hand-picked number.
     - a learned 2-input (tabular, gnn) stacker, with graph evidence
       (shared_device/shared_ip counts) computed every request but only
       ever attached to the API response for investigator display — never
       touching the score. A separate rule-based "evidence_confluence"
       overlay was layered on top of THAT to catch the gap (raise the tier
       when strong connectivity co-occurred with a behavioral anomaly).
     - THIS version: graph evidence is now a real 3rd/4th input to the SAME
       learned model, and the rule-based overlay is retired. Reasoning: a
       hand-picked overlay threshold is exactly the untested-manual-weight
       failure mode the 0.35/0.45/0.20 -> learned-stacker change was
       already fixed for once — bolting a second hand-picked rule onto the
       learned score's output just reintroduces that same problem one
       layer up. The synthetic dataset was specifically built with paired
       high-connectivity BENIGN communities (hostel, family, carrier-NAT,
       event-spike, shared-device, bill-split) alongside high-connectivity
       FRAUD rings so the stacker has real examples of both to learn the
       distinction from, rather than a human guessing a threshold. Verify
       this actually held before trusting it — see
       tests/test_edge_case_matrix.py, which runs the 17-case adversarial
       matrix against the trained model, not just rule logic.

2. calculate_composite_risk_score(): online. The live per-transaction
   scoring path the API calls. Loads the trained stacker's coefficients
   and combines a fresh tabular prediction, a fresh inductive GNN
   prediction, and fresh graph-evidence counts. The velocity/VPN-proxy
   multiplier is applied AFTER the calibrated probability as an explicit,
   separately-labeled rule-based overlay, not folded into the "learned"
   score — VPN/proxy flags aren't in either model's feature set, so
   multiplying them in is a deliberate rule, and pretending it was learned
   would misrepresent what actually happened.

   Previously this also had a second "fast path" that skipped the graph/GNN
   call entirely for small, unremarkable transactions from established
   users — a second lever on top of the graph-snapshot cache below, added
   back when the cache didn't exist yet and a full graph rebuild really did
   happen on every single request. It's been removed: the graph cache
   already gets a warm-cache full-path call down to ~15ms (measured), which
   is fast enough that a second "skip evidence entirely" mechanism stopped
   earning its complexity. It was also quietly relying on txn_payload's
   client-supplied velocity_1h to decide eligibility — the exact spoofable
   read this file no longer trusts anywhere. One scoring path, always fed
   real evidence, backed by one cache — simpler and strictly more accurate.
"""
import os
import json
import time
import datetime as _dt

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.common import classification_report_dict, load_tuned_hyperparameters
from ml.train_tabular_model import train_tabular_model, predict_tabular_fraud_prob, MERCHANT_RATES_PATH
from ml.train_gnn import train_gnn, GraphSAGEInference, GNN_WEIGHTS_PATH
from ml.risk_graph import build_user_graph, detect_communities, fetch_node_features, build_adjacency
from config import PRIOR_AMOUNT_WINDOW_DAYS, WATCHLIST_SCORE_MULTIPLIER
from db.database import get_raw_sqlite_connection
from ml.watchlist import is_watchlisted
from utils.logger import get_logger

logger = get_logger("risk_aggregator")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
AGGREGATOR_PATH = os.path.join(MODEL_DIR, "aggregator.npz")
AGGREGATOR_EVAL_PATH = os.path.join(MODEL_DIR, "aggregator_eval.json")

HIGH_RISK_THRESHOLD = 70.0
CRITICAL_THRESHOLD = 90.0
MEDIUM_THRESHOLD = 40.0

# Cap for normalizing raw shared-account counts into the stacker's input
# range. Chosen from this dataset's own fraud rings (largest is ~7-8
# accounts); the largest BENIGN look-alike community (the carrier-NAT/event
# scenarios) intentionally goes well past this cap (40-60 accounts) so the
# stacker has to learn "high count alone isn't the signal" rather than just
# raising the cap to cover them — a real deployment retraining on real ring
# sizes should revisit this number rather than assume it transfers.
SHARED_ACCOUNT_CAP = 9

# Graph snapshot (topology + communities + node features + adjacency + GNN
# forward pass) is rebuilt at most once per this many seconds instead of
# once per request — see _GraphSnapshotCache. This is now the ONLY
# performance lever in this file (see module docstring for why the earlier
# separate fast-path was retired).
GRAPH_CACHE_TTL_SECONDS = 20.0


def _normalize_shared_signal(count: int) -> float:
    """Raw shared-account counts (e.g. 1, 2, 7) live on a totally different
    scale than the two 0-1 model probabilities they're stacked with — feeding
    them in raw would let one feature's magnitude dominate the logistic
    regression's gradient regardless of its actual signal strength. -1
    because a solo account's device/IP query returns 1 (itself) as the
    baseline, so a normal, unconnected user normalizes to 0.0, not 0.11."""
    return min(max(count - 1, 0), SHARED_ACCOUNT_CAP) / SHARED_ACCOUNT_CAP


class _GraphSnapshotCache:
    """Caches the expensive graph-build -> community-detection -> node-
    feature -> adjacency -> GNN-forward-pass pipeline for
    GRAPH_CACHE_TTL_SECONDS. Rebuilding this from scratch is the dominant
    cost in live scoring (full scan of `transactions`, Louvain on the
    resulting graph); nothing about it changes meaningfully between one
    transaction and the next a few seconds later, so per-request rebuilds
    were pure waste. Every transaction still gets a real, current-enough
    GNN score computed against this snapshot — nothing about detection is
    skipped, only how often the snapshot underneath it is refreshed."""
    user_ids = None
    G = None
    community_size = None
    scores = None
    built_at = 0.0

    @classmethod
    def invalidate(cls):
        cls.user_ids = None
        cls.built_at = 0.0

    @classmethod
    def get(cls, conn):
        now = time.monotonic()
        if cls.user_ids is None or (now - cls.built_at) > GRAPH_CACHE_TTL_SECONDS:
            G = build_user_graph(conn)
            communities, community_size = detect_communities(G)
            user_ids, X_raw = fetch_node_features(conn, G, community_size)
            A_mean = build_adjacency(G, user_ids)
            _LiveModels.ensure_loaded()
            scores = _LiveModels.gnn.score_all(X_raw, A_mean)

            cls.G = G
            cls.community_size = community_size
            cls.user_ids = user_ids
            cls.scores = scores
            cls.built_at = now
            logger.info(f"Graph snapshot rebuilt: {len(user_ids)} users, TTL={GRAPH_CACHE_TTL_SECONDS}s")
        return cls.G, cls.community_size, cls.user_ids, cls.scores



def invalidate_live_graph_snapshot():
    """Invalidate the live GNN snapshot after a transaction is persisted.

    A scored transaction must be visible to the next scoring request; otherwise
    the 20-second cache can make repeated transactions use stale graph
    topology/evidence. The current request is scored against the pre-insert
    state (avoiding self-influence), and the next request rebuilds from the
    newly committed state.
    """
    _GraphSnapshotCache.invalidate()


def train_stacker():
    """Runs the full training sequence (tabular -> GNN -> stacker) and
    persists all three artifacts. This is the one function the admin
    pipeline endpoints call after a reseed/ingest."""
    tabular_df = train_tabular_model()
    user_ids, gnn_scores, y_user, train_user_mask, test_user_mask = train_gnn()

    gnn_by_user = pd.DataFrame({"user_id": user_ids, "gnn_score": gnn_scores})
    df = tabular_df.merge(gnn_by_user, on="user_id", how="left")
    if df["gnn_score"].isna().any():
        fallback = float(np.mean(gnn_scores))
        logger.warning(f"{df['gnn_score'].isna().sum()} transactions had no matching GNN score — filling with population mean.")
        df["gnn_score"] = df["gnn_score"].fillna(fallback)

    # Graph evidence (shared device/IP counts) as REAL stacker inputs, not a
    # rule bolted onto the output — see module docstring for the full
    # history of why. Computed here from each user's MOST RECENT
    # transaction's device/IP (matching what live scoring sees for a new
    # incoming transaction from that user).
    conn = get_raw_sqlite_connection()
    shared_rows = conn.execute("""
        SELECT t1.user_id,
               (SELECT COUNT(DISTINCT t2.user_id) FROM transactions t2
                WHERE t2.device_id = t1.last_device) AS shared_device_accounts,
               (SELECT COUNT(DISTINCT t3.user_id) FROM transactions t3
                WHERE t3.ip_address = t1.last_ip) AS shared_ip_accounts
        FROM (
            SELECT user_id,
                   (SELECT device_id FROM transactions t WHERE t.user_id = outer_t.user_id ORDER BY timestamp DESC LIMIT 1) AS last_device,
                   (SELECT ip_address FROM transactions t WHERE t.user_id = outer_t.user_id ORDER BY timestamp DESC LIMIT 1) AS last_ip
            FROM transactions outer_t
            GROUP BY user_id
        ) t1
    """).fetchall()
    conn.close()
    shared_df = pd.DataFrame(shared_rows, columns=["user_id", "shared_device_accounts", "shared_ip_accounts"])
    shared_df["shared_device_norm"] = shared_df["shared_device_accounts"].apply(_normalize_shared_signal)
    shared_df["shared_ip_norm"] = shared_df["shared_ip_accounts"].apply(_normalize_shared_signal)
    df = df.merge(shared_df[["user_id", "shared_device_norm", "shared_ip_norm"]], on="user_id", how="left")
    df[["shared_device_norm", "shared_ip_norm"]] = df[["shared_device_norm", "shared_ip_norm"]].fillna(0.0)

    STACKER_FEATURES = ["tabular_score", "gnn_score", "shared_device_norm", "shared_ip_norm"]

    train_mask = (df["split"] == "train").to_numpy()
    test_mask = (df["split"] == "test").to_numpy()

    X_train = df.loc[train_mask, STACKER_FEATURES]
    y_train = df.loc[train_mask, "is_fraud"]
    X_test = df.loc[test_mask, STACKER_FEATURES]
    y_test = df.loc[test_mask, "is_fraud"]

    if y_train.nunique() < 2:
        logger.warning("Training split has only one class — stacker cannot be fit; using equal-weight fallback coefficients.")
        coef = np.array([2.0, 2.0, 2.0, 2.0])
        intercept = np.array([-2.0])
    else:
        # The base models use their own imbalance controls (XGBoost
        # scale_pos_weight and GNN positive-class loss weighting). The stacker
        # has a separate objective, so it explicitly reweights the minority
        # class with sklearn's inverse-frequency balanced weighting. This is
        # critical here because the stacker sees already-compressed model
        # probabilities and would otherwise optimize mostly for the negative
        # class.
        class_counts = np.bincount(y_train.astype(int), minlength=2)
        logger.info(
            f"Balanced stacker training: negatives={int(class_counts[0])}, "
            f"positives={int(class_counts[1])}, "
            f"positive_weight={len(y_train) / max(2 * class_counts[1], 1):.2f}"
        )
        # Bug #28: read the CV-search result instead of the hardcoded
        # literal — see ml/common.py::load_tuned_hyperparameters(). Falls
        # back to the same C=0.05/balanced default if the search hasn't
        # been run yet.
        tuned = load_tuned_hyperparameters()
        stacker_params = tuned.get("stacker", {}).get("best_params", {"C": 0.05, "class_weight": "balanced"})
        stacker = LogisticRegression(max_iter=1000, random_state=42, **stacker_params)
        stacker.fit(X_train, y_train)
        coef = stacker.coef_[0]
        intercept = stacker.intercept_
        logger.info(
            f"Stacker learned weights: tabular_coef={coef[0]:.4f} gnn_coef={coef[1]:.4f} "
            f"shared_device_coef={coef[2]:.4f} shared_ip_coef={coef[3]:.4f} intercept={intercept[0]:.4f}"
        )

    eval_metrics = {"note": "insufficient held-out fraud examples to evaluate"}
    if len(y_test) and y_test.sum() > 0:
        z = (coef[0] * X_test["tabular_score"] + coef[1] * X_test["gnn_score"]
             + coef[2] * X_test["shared_device_norm"] + coef[3] * X_test["shared_ip_norm"] + intercept[0])
        combined_scores = 1 / (1 + np.exp(-z))
        eval_metrics = {
            "tabular_only": classification_report_dict(y_test, X_test["tabular_score"]),
            "gnn_only": classification_report_dict(y_test, X_test["gnn_score"]),
            "stacked": classification_report_dict(y_test, combined_scores),
        }
        logger.info(f"Aggregator comparison (tabular-only vs GNN-only vs stacked): {eval_metrics}")
    else:
        logger.warning("Skipping stacker held-out evaluation — not enough fraud examples in the test split.")

    np.savez(AGGREGATOR_PATH, coef=coef, intercept=intercept)
    with open(AGGREGATOR_EVAL_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"Aggregator weights saved to {AGGREGATOR_PATH}")
    return eval_metrics


class _LiveModels:
    """Loaded lazily on first score request and cached for the process
    lifetime. Call reset() after a pipeline retrain so the next request
    picks up newly trained weights instead of stale in-memory ones."""
    gnn = None
    coef = None
    intercept = None
    merchant_rates = None
    merchant_global_rate = None

    @classmethod
    def reset(cls):
        cls.gnn = None
        cls.coef = None
        cls.intercept = None
        cls.merchant_rates = None
        cls.merchant_global_rate = None
        # A retrained GNN/stacker invalidates any cached graph snapshot too
        # — its scores were computed with the OLD weights.
        _GraphSnapshotCache.invalidate()

    @classmethod
    def ensure_loaded(cls):
        if cls.gnn is None:
            if not os.path.exists(GNN_WEIGHTS_PATH):
                train_gnn()
            cls.gnn = GraphSAGEInference()
        if cls.coef is None:
            if not os.path.exists(AGGREGATOR_PATH):
                train_stacker()
            data = np.load(AGGREGATOR_PATH)
            cls.coef, cls.intercept = data["coef"], float(data["intercept"][0])
        if cls.merchant_rates is None:
            if not os.path.exists(MERCHANT_RATES_PATH):
                train_tabular_model()
            with open(MERCHANT_RATES_PATH) as f:
                data = json.load(f)
            cls.merchant_rates = data["rates"]
            cls.merchant_global_rate = data["global_rate"]


def _merchant_fraud_rate(merchant_id: str) -> float:
    _LiveModels.ensure_loaded()
    return _LiveModels.merchant_rates.get(str(merchant_id), _LiveModels.merchant_global_rate)


def live_gnn_score_and_evidence(user_id: str):
    """Returns the cached graph snapshot's GNN score + connectivity evidence
    for user_id, rebuilding the snapshot first only if it's missing or
    older than GRAPH_CACHE_TTL_SECONDS (see _GraphSnapshotCache above)."""
    conn = get_raw_sqlite_connection()
    try:
        G, community_size, user_ids, scores = _GraphSnapshotCache.get(conn)

        if user_id not in user_ids:
            # Bug #30: a user with NO transaction history yet (this is their
            # first-ever transaction, so they can't be in a graph snapshot
            # built from persisted transactions) used to return a hard 0.0
            # here. That 0.0 was then fed into the learned stacker as if the
            # GNN had confidently voted "not fraud" — but the GNN never
            # looked at this user at all; there was nothing to look at. Two
            # concrete failures followed: (1) calculate_composite_risk_score
            # below combines this with the stacker's fitted GNN coefficient
            # (~2.26), so the calibrated probability for a first-time user
            # was structurally capped around ~0.70 (sigmoid(coef[0]*1.0 +
            # coef[2]*0 - intercept-ish), no matter how confident the
            # tabular signal was — well under the 0.95 AUTO_BLOCK_THRESHOLD,
            # so a brand-new identity could never be auto-blocked, only ever
            # routed to human review; (2) ml/decision_policy.py's
            # MODEL_DISAGREEMENT check (abs(tabular - gnn) >= 0.45) fired on
            # nearly every first-time transaction with any real tabular
            # signal, since gnn was artificially pinned at 0 — flagging
            # "disagreement" that was actually just absent graph data, not
            # two models reaching different conclusions. Reproduced directly
            # against a never-before-seen synthetic identity (not one of the
            # golden-matrix users, which all have persisted history and so
            # never hit this branch — this is why none of the existing 69
            # tests caught it). Golden-matrix G10's own stated intent is
            # "tabular dominates" for structurally isolated users; this
            # fallback now honors that literally instead of only in tier,
            # by NOT asserting a confident graph opinion that doesn't exist.
            # graph_evidence_available=False downstream (a) makes
            # calculate_composite_risk_score skip the 4-input stacker for
            # this request and use the tabular probability directly as the
            # calibrated score, and (b) makes decision_policy.py skip
            # MODEL_DISAGREEMENT, since there's no second opinion to
            # disagree with.
            return 0.0, {
                "graph_degree": 0, "community_size": 1,
                "shared_device_accounts": 1, "shared_ip_accounts": 1,
                "graph_evidence_available": False,
            }

        idx = user_ids.index(user_id)
        gnn_score = float(scores[idx])
        degree = G.degree(user_id) if user_id in G else 0
        comm_size = community_size.get(user_id, 1)

        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) FROM transactions
            WHERE device_id = (SELECT device_id FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1)
        """, (user_id,))
        shared_device = cur.fetchone()[0] or 1
        cur.execute("""
            SELECT COUNT(DISTINCT user_id) FROM transactions
            WHERE ip_address = (SELECT ip_address FROM transactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1)
        """, (user_id,))
        shared_ip = cur.fetchone()[0] or 1

        return gnn_score, {
            "graph_degree": degree, "community_size": comm_size,
            "shared_device_accounts": shared_device, "shared_ip_accounts": shared_ip,
            "graph_evidence_available": True,
        }
    finally:
        conn.close()


def live_tabular_score(txn_payload: dict, velocity_1h: int) -> tuple[float, dict]:
    """Returns (fraud_probability, feature_row). velocity_1h is passed in —
    computed ONCE, server-side, by calculate_composite_risk_score — rather
    than read from txn_payload here. An earlier version defaulted to
    `txn_payload.get("velocity_1h", ...)`, which is client-controlled: a
    caller (or a fraudster's own client) could just always send a low
    number and suppress a signal that specifically exists to catch rapid
    repeated activity. The one DB-computed value is what actually drives
    scoring now, and it's the same value used everywhere else in this
    request (persisted to the DB, shown in the API response, passed to the
    investigation agent) — one source of truth instead of several places
    each reading (or not reading) the client's claim independently."""
    conn = get_raw_sqlite_connection()
    try:
        user_id = txn_payload["user_id"]
        amount = float(txn_payload.get("amount", 0.0))
        cur = conn.cursor()

        # Bug #29: hour_of_day/day_of_week/is_night used to call
        # datetime.now() directly — correct for a transaction genuinely
        # happening right now (the real live-API case), but wrong for
        # re-scoring an already-occurred transaction (tests, investigation
        # replay, evaluation), since it silently swapped in whatever the
        # real wall-clock happened to be at call time instead of when the
        # transaction actually occurred. Training computes these from the
        # transaction's own stored `timestamp` column
        # (ml/train_tabular_model.py, `strftime('%H', t.timestamp)`) — this
        # resolves the same way at inference: honor an explicit
        # `timestamp` in the payload if present, otherwise default to
        # real "now" (unchanged behavior for the live API, which has no
        # other notion of "when" a brand-new transaction occurred).
        ref_time = txn_payload.get("timestamp")
        if isinstance(ref_time, str):
            ref_time = _dt.datetime.fromisoformat(ref_time)
        elif not isinstance(ref_time, _dt.datetime):
            ref_time = _dt.datetime.now()

        # Bounded to PRIOR_AMOUNT_WINDOW_DAYS (default 90) — matches
        # ml/train_tabular_model.py's FEATURE_SQL exactly, so amount_zscore_
        # prior is computed the identical way at train time and live-scoring
        # time. An earlier version used an unbounded lifetime window here;
        # switching to a rolling window is a deliberate choice (recent
        # behavior should matter more than all-time history) kept in sync
        # via the shared PRIOR_AMOUNT_WINDOW_DAYS constant in config.py.
        cur.execute(
            "SELECT AVG(amount), COUNT(amount) FROM transactions WHERE user_id = ? AND timestamp > datetime(?, ?)",
            (user_id, ref_time.isoformat(sep=" "), f"-{PRIOR_AMOUNT_WINDOW_DAYS} days")
        )
        prior_avg, prior_count = cur.fetchone()
        cur.execute(
            "SELECT amount FROM transactions WHERE user_id = ? AND timestamp > datetime(?, ?)",
            (user_id, ref_time.isoformat(sep=" "), f"-{PRIOR_AMOUNT_WINDOW_DAYS} days")
        )
        prior_amounts = [r[0] for r in cur.fetchall()]
        prior_std = float(np.std(prior_amounts, ddof=1)) if prior_count and prior_count > 1 else 0.0
        amount_zscore_prior = 0.0 if not prior_std else (amount - float(prior_avg)) / prior_std

        # Same rationale as train_tabular_model.FEATURE_SQL: velocity_1h is a
        # single fixed 60-minute window — an actor who paces transactions
        # more than an hour apart resets it to near-zero even mid-pattern.
        # Counting distinct device fingerprints used by the SAME user over a
        # much longer 7-day trailing window survives that pacing evasion.
        cur.execute("""
            SELECT COUNT(DISTINCT device_id) FROM transactions
            WHERE user_id = ? AND timestamp > datetime(?, '-7 days')
        """, (user_id, ref_time.isoformat(sep=" ")))
        prior_distinct_devices = cur.fetchone()[0] or 0
        incoming_device = txn_payload.get("device_id")
        cur.execute("SELECT 1 FROM transactions WHERE user_id = ? AND device_id = ? LIMIT 1", (user_id, incoming_device))
        distinct_devices_7d = prior_distinct_devices if cur.fetchone() else prior_distinct_devices + 1

        # distinct_merchants_1h: lets the model tell "5 transactions to 5
        # different merchants" (ordinary busy shopping) apart from "5
        # transactions to the SAME merchant" (card-testing/structuring) —
        # velocity_1h alone can't distinguish these, since it only counts
        # transactions, not who they went to. Same trailing-1h window.
        incoming_merchant = txn_payload.get("merchant_id", "")
        cur.execute("""
            SELECT COUNT(DISTINCT merchant_id) FROM transactions
            WHERE user_id = ? AND timestamp > datetime(?, '-1 hours')
        """, (user_id, ref_time.isoformat(sep=" ")))
        prior_distinct_merchants = cur.fetchone()[0] or 0
        cur.execute(
            "SELECT 1 FROM transactions WHERE user_id = ? AND merchant_id = ? AND timestamp > datetime(?, '-1 hours') LIMIT 1",
            (user_id, incoming_merchant, ref_time.isoformat(sep=" "))
        )
        merchant_seen = bool(cur.fetchone())
        distinct_merchants_1h = prior_distinct_merchants + (0 if merchant_seen else 1)

        # Count previous transactions for the incoming merchant in the same
        # one-hour window. This is intentionally separate from total velocity
        # so card-testing / repeated authorization attempts become a visible
        # tabular pattern even when total velocity is only moderate.
        cur.execute("""
            SELECT COUNT(*) FROM transactions
            WHERE user_id = ? AND merchant_id = ?
              AND timestamp > datetime(?, '-1 hours')
        """, (user_id, incoming_merchant, ref_time.isoformat(sep=" ")))
        same_merchant_1h = int(cur.fetchone()[0] or 0) + 1

        feature_row = {
            "amount_log": float(np.log1p(amount)),
            "hour_of_day": ref_time.hour,
            "day_of_week": ref_time.weekday(),
            "velocity_1h": velocity_1h,
            "amount_zscore_prior": amount_zscore_prior,
            "merchant_fraud_rate": _merchant_fraud_rate(txn_payload.get("merchant_id", "")),
            "distinct_devices_7d": distinct_devices_7d,
            "distinct_merchants_1h": distinct_merchants_1h,
            "same_merchant_1h": same_merchant_1h,
            "amount_to_prior_avg": float(np.clip(amount / float(prior_avg), 0, 100)) if prior_avg else 1.0,
            "is_night": int(ref_time.hour in {0,1,2,3,4,5}),
            "is_round_amount": int(amount >= 5000 and abs(amount % 500) < 1e-6),
        }
        return predict_tabular_fraud_prob(feature_row), amount_zscore_prior
    finally:
        conn.close()


def calculate_composite_risk_score(txn_payload: dict) -> dict:
    """Live per-transaction scoring — the API's hot path."""
    user_id = txn_payload.get("user_id", "")
    amount = float(txn_payload.get("amount", 0.0))

    # Velocity source is explicit. Frontend toggle ON (velocity_enabled=True)
    # means this is a simulation/testing path that trusts the supplied client
    # velocity. Toggle OFF means the backend calculates the trailing 1-hour
    # count from persisted transaction history. In both modes the selected
    # velocity is used consistently by the tabular model, velocity rules,
    # policy, persistence, audit log, and API response.
    trust_client_velocity = bool(txn_payload.get("velocity_enabled", False))
    client_velocity = txn_payload.get("velocity_1h")
    if client_velocity is not None:
        client_velocity = int(client_velocity)
        if client_velocity < 0:
            raise ValueError("velocity_1h must be >= 0")

    if trust_client_velocity:
        if client_velocity is None:
            raise ValueError("velocity_1h is required when velocity_enabled=true")
        velocity_1h = client_velocity
        velocity_source = "CLIENT"
    else:
        conn = get_raw_sqlite_connection()
        # Use the transaction timestamp when replaying/scoring historical
        # transactions; live API calls without a timestamp naturally fall
        # back to the current wall clock inside live_tabular_score().
        ref_time = txn_payload.get("timestamp")
        if isinstance(ref_time, str):
            ref_time = _dt.datetime.fromisoformat(ref_time)
        elif not isinstance(ref_time, _dt.datetime):
            ref_time = _dt.datetime.now()
        velocity_1h = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND timestamp > datetime(?, '-1 hours')",
            (user_id, ref_time.isoformat(sep=" "))
        ).fetchone()[0] + 1
        conn.close()
        velocity_source = "BACKEND"

    tabular_prob, amount_zscore_prior = live_tabular_score(txn_payload, velocity_1h)
    gnn_prob, graph_evidence = live_gnn_score_and_evidence(user_id)

    # Graph evidence as a real stacker input (see train_stacker()'s and this
    # module's docstrings for why this replaced an earlier rule-based
    # overlay) — same normalization used at training time.
    shared_device_norm = _normalize_shared_signal(graph_evidence["shared_device_accounts"])
    shared_ip_norm = _normalize_shared_signal(graph_evidence["shared_ip_accounts"])
    graph_evidence_available = graph_evidence.get("graph_evidence_available", True)

    _LiveModels.ensure_loaded()
    coef = _LiveModels.coef

    if graph_evidence_available:
        z = float(
            coef[0] * tabular_prob + coef[1] * gnn_prob
            + coef[2] * shared_device_norm + coef[3] * shared_ip_norm
            + _LiveModels.intercept
        )
        calibrated_prob = 1 / (1 + np.exp(-z))
    else:
        # Bug #30 fix: the stacker's coefficients were fit assuming a real
        # GNN opinion is always present. Running gnn_prob=0.0 through it
        # anyway silently treats "no data yet" as "the GNN actively cleared
        # this transaction," which both caps the achievable calibrated
        # score far below AUTO_BLOCK_THRESHOLD and — via decision_policy.py
        # — manufactures a false MODEL_DISAGREEMENT on almost every
        # first-time transaction with real tabular signal. For a
        # first-ever transaction from this identity there is no graph
        # evidence to combine, so the calibrated probability is the
        # tabular model's own probability, unmodified — the same "tabular
        # dominates" principle the golden matrix's G10 case already
        # documents as this project's intended behavior for
        # structurally-isolated users, now honored for genuinely unseen
        # ones too instead of only in the final tier.
        calibrated_prob = tabular_prob

    velocity_mult = 1.0
    effective_velocity_1h = velocity_1h
    if velocity_1h >= 10:
        velocity_mult = 1.5
    elif velocity_1h >= 5:
        velocity_mult = 1.25
    if txn_payload.get("is_vpn_proxy", False) or txn_payload.get("is_suspicious_proxy", False):
        velocity_mult *= 1.15

    # Repeat-MEDIUM-risk overlay (see ml/watchlist.py). Explicit and
    # separately labeled, same as velocity_mult above — not folded into the
    # learned stacker, since "this user tripped MONITOR recently" isn't a
    # feature either model was trained on.
    watchlist_flagged = is_watchlisted(user_id)
    watchlist_mult = WATCHLIST_SCORE_MULTIPLIER if watchlist_flagged else 1.0

    raw_score = calibrated_prob * velocity_mult * watchlist_mult * 100.0
    final_risk_score = round(float(min(max(raw_score, 0.0), 100.0)), 1)

    if final_risk_score >= CRITICAL_THRESHOLD:
        tier, decision = "CRITICAL", "BLOCK_AND_INVESTIGATE"
    elif final_risk_score >= HIGH_RISK_THRESHOLD:
        tier, decision = "HIGH", "HOLD_FOR_INVESTIGATION"
    elif final_risk_score >= MEDIUM_THRESHOLD:
        tier, decision = "MEDIUM", "MONITOR"
    else:
        tier, decision = "LOW", "APPROVE"

    result = {
        "risk_score": final_risk_score,
        "tabular_score": round(tabular_prob * 100.0, 1),
        "gnn_score": round(gnn_prob * 100.0, 1),
        "stacker_calibrated_score": round(float(calibrated_prob) * 100.0, 1),
        "velocity_multiplier": round(velocity_mult, 2),
        "velocity_enabled": trust_client_velocity,
        "velocity_source": velocity_source,
        "watchlist_flagged": watchlist_flagged,
        "watchlist_multiplier": round(watchlist_mult, 2),
        "effective_velocity_1h": effective_velocity_1h,
        # The selected velocity is exposed with its source so the dashboard
        # and audit trail can distinguish CLIENT simulation mode from BACKEND
        # calculated mode.
        "velocity_1h": velocity_1h,
        "amount_zscore_prior": round(amount_zscore_prior, 3),
        "risk_tier": tier,
        "decision": decision,
        "graph_evidence": {
            "shared_device_accounts": graph_evidence["shared_device_accounts"],
            "shared_ip_accounts": graph_evidence["shared_ip_accounts"],
            "community_size": graph_evidence["community_size"],
            "graph_degree": graph_evidence["graph_degree"],
            "evidence_available": graph_evidence_available,
        },
    }
    logger.info(
        "Transaction User:%s Amt:₹%s -> GNNNodeEmbedding:%s%% TabularML:%s%% "
        "StackerCalibrated:%s%% Velocity1h:%s VelocitySource:%s VelocityMultiplier:%s "
        "Watchlisted:%s WatchlistMultiplier:%s "
        "RiskScore:%s/100 [%s] Action:%s",
        user_id, amount, result["gnn_score"], result["tabular_score"],
        result["stacker_calibrated_score"], velocity_1h, velocity_source,
        result["velocity_multiplier"], watchlist_flagged, result["watchlist_multiplier"],
        final_risk_score, tier, decision,
    )
    return result


if __name__ == "__main__":
    train_stacker()
    sample_txn = {
        "user_id": "USER_RING1_1", "device_id": "DEV_FRAUD_RING1",
        "ip_address": "185.220.101.44", "merchant_id": "MCH_042",
        "amount": 88000, "is_vpn_proxy": True,
    }
    print("Risk Score Output:", calculate_composite_risk_score(sample_txn))
