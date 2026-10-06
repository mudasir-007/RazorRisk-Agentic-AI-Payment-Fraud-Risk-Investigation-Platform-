"""
RazorRisk — GNN risk model.

Implements a 2-layer GraphSAGE-style mean-aggregation network FROM SCRATCH
in NumPy (forward pass + analytic backprop), rather than pulling in
PyTorch/PyTorch-Geometric.

Why from scratch, not torch: an earlier version of this project depended on
torch for a graph of ~1500 nodes — a 500MB+ wheel (plus CUDA-adjacent
dependency weight) for a model this describes as "2-layer mean aggregation,
concat with self, linear + ReLU". The math below is exactly what SAGEConv
does; swapping to torch_geometric at production scale is a drop-in
replacement of the Layer class, not a redesign. Worth saying exactly this
if asked in an interview — it's a deliberate scale-appropriate call, not a
limitation being glossed over. It also means this module (and the whole
live-scoring path) has zero heavy ML framework dependencies.

Task: binary node classification — is this user part of a fraud ring?
Ground truth (is_fraud_ground_truth on transactions, rolled up to the user
level) is used ONLY here, for training labels and evaluation. Live scoring
does not have access to it — a real deployment doesn't get handed the
answer.

Inductive at inference: GraphSAGEInference.score_all() takes whatever
X/A_mean the CURRENT graph produces and runs a pure forward pass with the
already-trained weights — any user present in the graph gets a real score,
including ones that didn't exist at training time, with no retraining and
no per-user cache to go stale. This replaced an earlier transductive
design (a user_id -> cached probability lookup table with a heuristic
fallback for cache misses) — inductive inference is strictly better here
since it needs no fallback path at all.
"""
import os
import json

import numpy as np
import sqlite3

from ml.common import user_level_split, classification_report_dict, load_tuned_hyperparameters, RNG_SEED
from ml.risk_graph import build_user_graph, detect_communities, fetch_node_features, build_adjacency
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("gnn_training")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
GNN_WEIGHTS_PATH = os.path.join(MODEL_DIR, "gnn_model.npz")
GNN_EVAL_PATH = os.path.join(MODEL_DIR, "gnn_eval.json")

HIDDEN_DIM_1 = 16
HIDDEN_DIM_2 = 8
LEARNING_RATE = 0.03
EPOCHS = 350


# ---------------------------------------------------------------------------
# Model: 2-layer GraphSAGE-style network, manual forward + backward
# ---------------------------------------------------------------------------

class SAGELayer:
    def __init__(self, in_dim, out_dim, rng):
        limit = np.sqrt(6 / (2 * in_dim + out_dim))  # Xavier-ish init
        self.W = rng.uniform(-limit, limit, size=(2 * in_dim, out_dim))
        self.b = np.zeros(out_dim)

    def forward(self, H, A_mean):
        self.H_in = H
        self.Agg = A_mean @ H
        self.Concat = np.hstack([H, self.Agg])
        self.Z = self.Concat @ self.W + self.b
        self.out = np.maximum(0, self.Z)  # ReLU
        return self.out

    def backward(self, dOut, A_mean, lr):
        dZ = dOut * (self.Z > 0)
        dW = self.Concat.T @ dZ
        db = dZ.sum(axis=0)
        dConcat = dZ @ self.W.T
        d_in, d_agg = np.split(dConcat, [self.H_in.shape[1]], axis=1)
        dH_from_agg = A_mean.T @ d_agg
        dH_total = d_in + dH_from_agg

        self.W -= lr * dW
        self.b -= lr * db
        return dH_total


class RiskGNN:
    """Training-time model: forward + backward. Kept separate from
    GraphSAGEInference below on purpose — inference and training are
    different concerns and shouldn't share a class."""

    def __init__(self, in_dim, seed=RNG_SEED, hidden_dim_1=None, hidden_dim_2=None):
        rng = np.random.default_rng(seed)
        h1 = hidden_dim_1 or HIDDEN_DIM_1
        h2 = hidden_dim_2 or HIDDEN_DIM_2
        self.layer1 = SAGELayer(in_dim, h1, rng)
        self.layer2 = SAGELayer(h1, h2, rng)
        limit = np.sqrt(6 / (h2 + 1))
        self.Wc = rng.uniform(-limit, limit, size=(h2, 1))
        self.bc = np.zeros(1)

    def forward(self, X, A_mean):
        self.H1 = self.layer1.forward(X, A_mean)
        self.H2 = self.layer2.forward(self.H1, A_mean)
        self.logits = (self.H2 @ self.Wc + self.bc).flatten()
        self.probs = 1 / (1 + np.exp(-self.logits))
        return self.probs

    def backward(self, y, train_mask, pos_weight, A_mean, lr):
        n_train = train_mask.sum()
        sample_weight = np.where(y == 1, pos_weight, 1.0)
        dLogits = np.zeros_like(y)
        dLogits[train_mask] = (
            sample_weight[train_mask] * (self.probs[train_mask] - y[train_mask])
        ) / max(n_train, 1)
        dLogits = dLogits.reshape(-1, 1)

        dWc = self.H2.T @ dLogits
        dbc = dLogits.sum(axis=0)
        dH2 = dLogits @ self.Wc.T

        self.Wc -= lr * dWc
        self.bc -= lr * dbc

        dH1 = self.layer2.backward(dH2, A_mean, lr)
        self.layer1.backward(dH1, A_mean, lr)

    def loss(self, y, train_mask, pos_weight):
        if train_mask.sum() == 0:
            return 0.0
        p = np.clip(self.probs[train_mask], 1e-9, 1 - 1e-9)
        yt = y[train_mask]
        w = np.where(yt == 1, pos_weight, 1.0)
        return float(np.mean(-w * (yt * np.log(p) + (1 - yt) * np.log(1 - p))))


class GraphSAGEInference:
    """Forward-pass-only reimplementation of RiskGNN using trained weights.
    Loaded once by ml/risk_aggregator.py at API startup and reused across
    every live scoring request — the actual per-request cost is just a
    handful of numpy matmuls, not a database round-trip to a cache."""

    def __init__(self, weights_path=GNN_WEIGHTS_PATH):
        data = np.load(weights_path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.Wc, self.bc = data["Wc"], data["bc"]
        self.mu, self.sigma = data["mu"], data["sigma"]

    def score_all(self, X_raw: np.ndarray, A_mean: np.ndarray) -> np.ndarray:
        """Inductive scoring: any node present in X_raw/A_mean gets a score
        using the already-trained weights, no retraining needed."""
        X = (X_raw - self.mu) / self.sigma
        Agg0 = A_mean @ X
        H1 = np.maximum(0, np.hstack([X, Agg0]) @ self.W1 + self.b1)
        Agg1 = A_mean @ H1
        H2 = np.maximum(0, np.hstack([H1, Agg1]) @ self.W2 + self.b2)
        logits = (H2 @ self.Wc + self.bc).flatten()
        return 1 / (1 + np.exp(-logits))


def _user_fraud_labels(conn, user_ids):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_id FROM transactions WHERE is_fraud_ground_truth = 1")
    fraud_users = {r[0] for r in cur.fetchall()}
    return np.array([1.0 if uid in fraud_users else 0.0 for uid in user_ids]), fraud_users


def train_gnn():
    conn = get_raw_sqlite_connection()
    G = build_user_graph(conn)
    communities, community_size = detect_communities(G)
    user_ids, X = fetch_node_features(conn, G, community_size)
    A_mean = build_adjacency(G, user_ids)
    y, fraud_users = _user_fraud_labels(conn, user_ids)

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    logger.info(f"GNN training set: {len(user_ids)} users ({n_pos} fraud-ring members, {n_neg} others).")

    train_mask, test_mask = user_level_split(user_ids, y)
    logger.info(f"Train: {int(train_mask.sum())} users ({int(y[train_mask].sum())} fraud) | "
                f"Test: {int(test_mask.sum())} users ({int(y[test_mask].sum())} fraud)")

    mu = X[train_mask].mean(axis=0) if train_mask.sum() else X.mean(axis=0)
    sigma = X[train_mask].std(axis=0) if train_mask.sum() else X.std(axis=0)
    sigma[sigma == 0] = 1
    X_norm = (X - mu) / sigma

    # Bug #28: read the CV-search result instead of the hardcoded
    # module-level defaults — see ml/common.py::load_tuned_hyperparameters().
    # Falls back to HIDDEN_DIM_1/HIDDEN_DIM_2/LEARNING_RATE/EPOCHS above if
    # the search hasn't been run yet.
    tuned = load_tuned_hyperparameters()
    gnn_params = tuned.get("gnn", {}).get("best_params", {})
    hidden_dim_1 = gnn_params.get("hidden_dim_1", HIDDEN_DIM_1)
    hidden_dim_2 = gnn_params.get("hidden_dim_2", HIDDEN_DIM_2)
    learning_rate = gnn_params.get("learning_rate", LEARNING_RATE)
    epochs = gnn_params.get("epochs", EPOCHS)

    pos_weight = n_neg / max(n_pos, 1)
    model = RiskGNN(in_dim=X.shape[1], hidden_dim_1=hidden_dim_1, hidden_dim_2=hidden_dim_2)

    logger.info(f"Training GraphSAGE ({epochs} epochs, lr={learning_rate}, pos_weight={pos_weight:.1f})...")
    for epoch in range(1, epochs + 1):
        model.forward(X_norm, A_mean)
        model.backward(y, train_mask, pos_weight, A_mean, learning_rate)
        if epoch % 100 == 0 or epoch == 1:
            l = model.loss(y, train_mask, pos_weight)
            logger.info(f"  epoch {epoch:4d}  train_loss={l:.4f}")

    final_scores = model.forward(X_norm, A_mean)

    eval_metrics = {"note": "insufficient held-out fraud examples to evaluate"}
    if test_mask.sum() > 0 and y[test_mask].sum() > 0:
        eval_metrics = classification_report_dict(y[test_mask], final_scores[test_mask])
        logger.info(f"GNN held-out test metrics: {eval_metrics}")
    else:
        logger.warning("Skipping GNN held-out evaluation — not enough fraud examples in the test split.")

    np.savez(
        GNN_WEIGHTS_PATH,
        W1=model.layer1.W, b1=model.layer1.b,
        W2=model.layer2.W, b2=model.layer2.b,
        Wc=model.Wc, bc=model.bc,
        mu=mu, sigma=sigma,
    )
    with open(GNN_EVAL_PATH, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    logger.info(f"GNN weights saved to {GNN_WEIGHTS_PATH}")

    # Per-user scores for the risk_aggregator's stacker (broadcast to each
    # user's transactions when training the stacker).
    return user_ids, final_scores, y, train_mask, test_mask


if __name__ == "__main__":
    train_gnn()
