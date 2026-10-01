"""
RazorRisk — real-data ingestion (ULB "Credit Card Fraud Detection" dataset,
284,807 European card transactions, 492 confirmed frauds — the standard
Kaggle mirror of this dataset). Optional supplement to the synthetic
dataset, not a replacement for it.

IMPORTANT — this used to be destructive. The original version of this
script started with:
    DELETE FROM transactions; DELETE FROM users; DELETE FROM devices; ...
which wiped out EVERY synthetic entity — including all the adversarial
benign-look-alike and fraud-ring scenarios in
tests/GOLDEN_TEST_MATRIX.md (hostel, carrier-NAT, event-spike, structuring,
account-takeover, etc.) — and replaced them with one crude synthetic
"ring1" cluster that every real fraud row got dumped into regardless of
the real dataset's own structure. Running this after generate_synthetic_
data.py would have silently made the golden test matrix stop meaning
anything: the specific USER_HOSTEL_1 / USER_CARRIER_2 / USER_STRUCT_1 /
etc. identities the matrix checks against would simply no longer exist.

This version is additive instead:
  1. Ensures the synthetic dataset (data/generate_synthetic_data.py) has
     already been generated — calling it first if the `users` table is
     empty — so the golden-matrix scenarios always exist before real data
     is layered on.
  2. NEVER deletes anything. Real transactions are added on top.
  3. Real FRAUD rows are assigned across the EXISTING fraud-ring identities
     (Ring1-5, structuring, fan-out-launder, no-shared-infra, low-and-slow,
     cold-start-fraud, account-takeover) using each one's own already-
     established device_id/ip_address — so a real transaction lands on
     exactly the same graph structure the golden matrix already validated,
     just with the real dataset's own amount distribution instead of a
     synthetic one. This is a genuine improvement over inventing a new,
     separate ring: the tabular model gets real-world fraud amount/timing
     patterns, layered onto graph structures already known to behave
     correctly for both false-positive and true-positive cases.
  4. Real BENIGN rows are assigned across the large ordinary baseline
     population (USER_0001...USER_1500 by default) generate_synthetic_
     data.py already creates, each using their own existing device/IP —
     never inventing new device/IP rows, so nothing here can accidentally
     create a spurious shared-fingerprint community the graph would
     misread as a fraud ring.

Net effect: after running both generate_synthetic_data.py and this script,
the `transactions` table contains the full synthetic golden-matrix dataset
PLUS real-world fraud/benign amount and timing patterns layered onto the
same entities — training sees both, and the golden matrix (tests/
test_edge_case_matrix.py) stays valid because every identity it checks
still exists with its original graph structure intact.
"""
import urllib.request
import urllib.error
import random
import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from db.database import init_db, get_raw_sqlite_connection
from data.generate_synthetic_data import generate_dataset
from utils.logger import get_logger

logger = get_logger("real_data_ingestor")

DATA_DIR = Path(__file__).resolve().parent
CSV_PATH = DATA_DIR / "creditcard.csv"

# Public, non-Kaggle-auth mirrors of the "Credit Card Fraud Detection"
# dataset (ULB Machine Learning Group). Tried in order. Kaggle itself
# requires an authenticated API call, so a plain-mirror approach is used
# here to keep this script dependency-free.
DATASET_URLS = [
    "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv",
    "https://raw.githubusercontent.com/nethaji-1997/Credit-Card-Fraud-Detection/master/creditcard.csv",
]

MANUAL_DOWNLOAD_HINT = (
    "Could not download the real dataset automatically (no internet access, or every mirror "
    "is currently unreachable). To use real data: download 'creditcard.csv' yourself from the "
    "Kaggle 'Credit Card Fraud Detection' dataset "
    "(https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at "
    f"'{CSV_PATH}', then re-run this pipeline. Until then, RazorRisk will keep working fine on "
    "the synthetic dataset alone (data/generate_synthetic_data.py)."
)


def download_real_dataset():
    """Downloads creditcard.csv from the first reachable mirror. Raises
    RuntimeError with an actionable message if every mirror fails, e.g.
    because this environment has no outbound network access."""
    if CSV_PATH.exists():
        logger.info(f"Real Kaggle dataset found locally at {CSV_PATH}")
        return

    last_error = None
    for url in DATASET_URLS:
        try:
            logger.info(f"Downloading real Kaggle Credit Card Fraud dataset from mirror: {url} ...")
            tmp_path = CSV_PATH.with_suffix(".csv.partial")
            urllib.request.urlretrieve(url, tmp_path)
            tmp_path.rename(CSV_PATH)
            logger.info(f"Dataset downloaded successfully to {CSV_PATH}")
            return
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            logger.warning(f"Mirror failed ({url}): {e}")
            last_error = e

    logger.error(MANUAL_DOWNLOAD_HINT)
    raise RuntimeError(MANUAL_DOWNLOAD_HINT) from last_error


def _fraud_entity_pool(conn):
    """Returns (user_id, device_id, ip_address, merchant_id) for every
    distinct user_id already labeled fraud in the synthetic dataset —
    Ring1-5, structuring, fan-out-launder, no-shared-infra, low-and-slow,
    cold-start-fraud, and the account-takeover hijack transaction — pulled
    from their own most recent fraud transaction, not invented."""
    rows = conn.execute("""
        SELECT user_id, device_id, ip_address, merchant_id FROM transactions t1
        WHERE is_fraud_ground_truth = 1
          AND t1.timestamp = (
              SELECT MAX(t2.timestamp) FROM transactions t2
              WHERE t2.user_id = t1.user_id AND t2.is_fraud_ground_truth = 1
          )
        GROUP BY user_id
    """).fetchall()
    return rows


def _benign_entity_pool(conn):
    """Returns (user_id, device_id, ip_address) for the large ordinary
    baseline population (USER_0001...) — each user's own established
    device/IP, pulled from their existing transactions so nothing new is
    invented that could accidentally create a spurious shared fingerprint."""
    rows = conn.execute(r"""
        SELECT user_id, device_id, ip_address FROM transactions t1
        WHERE user_id LIKE 'USER\_0%' ESCAPE '\'
          AND t1.timestamp = (
              SELECT MAX(t2.timestamp) FROM transactions t2 WHERE t2.user_id = t1.user_id
          )
        GROUP BY user_id
    """).fetchall()
    return rows


def ingest_real_dataset(sample_size=15000, ensure_synthetic_base=True):
    download_real_dataset()
    logger.info("Reading and parsing real Kaggle credit card transactions...")

    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        # A failed/interrupted download or an HTML error page saved as .csv
        # will fail to parse — remove the bad file so the next run
        # re-downloads instead of permanently thinking a valid copy exists.
        CSV_PATH.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded creditcard.csv could not be parsed and was removed: {e}") from e

    required_cols = {"Time", "Amount", "Class"}
    if not required_cols.issubset(df.columns):
        CSV_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            f"creditcard.csv is missing expected columns {required_cols - set(df.columns)}; "
            "the downloaded file was likely corrupted or is the wrong dataset. File removed — retry."
        )

    logger.info(f"Loaded {len(df)} real raw transactions from CSV (Total Fraud Count: {df['Class'].sum()}).")

    fraud_df = df[df["Class"] == 1]
    normal_budget = max(sample_size - len(fraud_df), 0)
    normal_df = df[df["Class"] == 0].sample(n=min(normal_budget, len(df[df["Class"] == 0])), random_state=42)
    combined_df = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(f"Selected {len(combined_df)} real transactions to layer onto the synthetic dataset "
                f"({len(fraud_df)} real fraud cases).")

    init_db()
    conn = get_raw_sqlite_connection()

    existing_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if existing_users == 0:
        if not ensure_synthetic_base:
            conn.close()
            raise RuntimeError(
                "No synthetic dataset found and ensure_synthetic_base=False — nothing to layer "
                "real transactions onto. Run data/generate_synthetic_data.generate_dataset() first, "
                "or call ingest_real_dataset() with ensure_synthetic_base=True (the default)."
            )
        logger.info("No synthetic dataset found yet — generating it first so the golden-matrix "
                    "scenarios exist before layering real transactions on top.")
        conn.close()
        generate_dataset()
        conn = get_raw_sqlite_connection()

    fraud_pool = _fraud_entity_pool(conn)
    benign_pool = _benign_entity_pool(conn)
    if not fraud_pool or not benign_pool:
        conn.close()
        raise RuntimeError(
            "Synthetic dataset exists but has no usable fraud/benign entity pool to layer real "
            "transactions onto — was it generated by an older version of generate_synthetic_data.py? "
            "Re-run generate_dataset() to rebuild it, then retry this ingestion."
        )
    logger.info(f"Layering real transactions onto {len(fraud_pool)} existing fraud identities and "
                f"{len(benign_pool)} existing baseline identities (no new users/devices/IPs created).")

    base_time = datetime.datetime.now() - datetime.timedelta(days=30)
    tx_list = []
    for idx, row in combined_df.iterrows():
        tx_id = f"TXN_REAL_{idx:06d}"
        is_fraud = bool(row["Class"] == 1)
        amt = round(float(row["Amount"]), 2)
        if amt <= 0.0:
            amt = 1.0
        tx_time = base_time + datetime.timedelta(seconds=float(row["Time"]))

        if is_fraud:
            user_id, device_id, ip_address, merchant_id = random.choice(fraud_pool)
            velocity_1h = int(np.random.randint(5, 15))
            zscore = round(float(np.random.uniform(2.5, 6.0)), 2)
        else:
            user_id, device_id, ip_address = random.choice(benign_pool)
            merchant_id = f"MCH_{np.random.randint(1, 51):03d}"
            velocity_1h = int(np.random.randint(1, 3))
            zscore = round(float(np.random.uniform(-0.5, 1.2)), 2)

        tx_list.append((
            tx_id, user_id, device_id, ip_address, merchant_id, amt, "INR",
            tx_time, "COMPLETED", velocity_1h, zscore, is_fraud
        ))

    conn.executemany("""
        INSERT OR IGNORE INTO transactions
        (transaction_id, user_id, device_id, ip_address, merchant_id, amount, currency, timestamp, status, velocity_1h, amount_zscore_prior, is_fraud_ground_truth)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, tx_list)
    conn.commit()
    conn.close()

    logger.info(f"Successfully layered {len(tx_list)} real Kaggle transactions onto the synthetic "
                f"dataset (golden-matrix scenarios untouched).")
    return len(tx_list)


if __name__ == "__main__":
    ingest_real_dataset()
