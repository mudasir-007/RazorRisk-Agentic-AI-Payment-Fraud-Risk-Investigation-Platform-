"""
Shared utilities across ml/. Split logic lives here (rather than duplicated
in train_gnn.py and train_tabular_model.py) so both models are evaluated on
the exact same held-out users — required for the risk_aggregator's stacker
to combine their scores on a fair, leak-free basis, and ported directly from
the more rigorous train/test discipline in the merged reference project.
"""
import json
import os

import numpy as np

RNG_SEED = 42
TRAIN_FRAC = 0.7

HYPERPARAMETERS_PATH = os.path.join(os.path.dirname(__file__), "models", "hyperparameters.json")


def load_tuned_hyperparameters() -> dict:
    """Reads ml/models/hyperparameters.json (written by
    ml/hyperparameter_search.py) if it exists, else returns {}.

    Bug #28: this file used to be written by the search script and then
    never read by anything — train_tabular_model(), train_gnn(), and
    train_stacker() each hardcoded their own literal default matching
    whatever the search happened to find *at the time someone last ran it
    and copied the numbers in by hand*. Re-running the search with a wider
    grid or new data would silently have no effect on the next `train_*()`
    call unless someone noticed and manually edited three files. This is
    the one place all three now read from, with the same hardcoded values
    kept as fallback defaults so training still works with no search
    artifact present (e.g. a fresh clone before anyone has run the search).
    """
    if not os.path.exists(HYPERPARAMETERS_PATH):
        return {}
    try:
        with open(HYPERPARAMETERS_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def user_level_split(user_ids, y, seed=RNG_SEED, train_frac=TRAIN_FRAC):
    """
    Stratified split by USER (not by transaction).

    All transactions belonging to the same user are guaranteed to remain
    entirely in either train or test.

    Users are stratified according to whether they have at least one
    fraud transaction.

    Returns:
        (train_mask, test_mask)

    Both are boolean arrays aligned to user_ids/y.

    NOTE:
        This is a user-level random split. It does NOT perform a temporal
        split. A separate temporal holdout should be added if production-
        style temporal validation is required.
    """
    user_ids = np.asarray(user_ids)
    y = np.asarray(y)

    if len(user_ids) != len(y):
        raise ValueError(
            f"user_ids and y must have the same length: "
            f"{len(user_ids)} != {len(y)}"
        )

    if len(user_ids) == 0:
        raise ValueError("Cannot split an empty dataset.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1.")

    # ---------------------------------------------------------
    # Build one label per USER.
    #
    # A user is considered fraud-associated if ANY of their
    # transactions is fraudulent.
    # ---------------------------------------------------------
    unique_users, inverse = np.unique(
        user_ids,
        return_inverse=True,
    )

    user_labels = np.zeros(len(unique_users), dtype=int)

    # If any transaction belonging to a user is fraud,
    # that user receives label 1.
    np.maximum.at(
        user_labels,
        inverse,
        y.astype(int),
    )

    # ---------------------------------------------------------
    # Stratified USER split
    # ---------------------------------------------------------
    rng = np.random.default_rng(seed)

    fraud_users = np.where(user_labels == 1)[0]
    normal_users = np.where(user_labels == 0)[0]

    rng.shuffle(fraud_users)
    rng.shuffle(normal_users)

    # Keep approximately the same fraction of fraud-associated
    # and normal users in training.
    n_fraud_train = (
        max(1, int(len(fraud_users) * train_frac))
        if len(fraud_users)
        else 0
    )

    n_normal_train = int(len(normal_users) * train_frac)

    train_user_idx = np.concatenate([
        fraud_users[:n_fraud_train],
        normal_users[:n_normal_train],
    ])

    train_user_mask = np.zeros(
        len(unique_users),
        dtype=bool,
    )

    train_user_mask[train_user_idx] = True

    test_user_mask = ~train_user_mask

    # ---------------------------------------------------------
    # Convert USER masks → TRANSACTION masks
    # ---------------------------------------------------------
    train_mask = train_user_mask[inverse]
    test_mask = test_user_mask[inverse]

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------
    assert np.all(train_mask | test_mask)
    assert not np.any(train_mask & test_mask)

    train_users = set(user_ids[train_mask])
    test_users = set(user_ids[test_mask])

    assert train_users.isdisjoint(test_users), (
        "User leakage detected: train and test contain "
        "overlapping users."
    )

    return train_mask, test_mask


def classification_report_dict(y_true, y_score, threshold=0.5):
    """Held-out evaluation metrics, returned as a dict so callers can both
    print and log/persist them (rather than only printing, which the
    deployed API can't capture)."""
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix,
    )
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score) if len(set(y_true.tolist())) > 1 else float("nan")
    ap = average_precision_score(y_true, y_score)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(auc) if auc == auc else None,  # NaN-safe
        "pr_auc": float(ap),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }
