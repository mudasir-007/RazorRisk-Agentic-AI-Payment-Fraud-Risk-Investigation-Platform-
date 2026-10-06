"""Regression tests for bugs found during manual RazorRisk validation."""
import json
from pathlib import Path
import unittest

from db.database import get_raw_sqlite_connection
from ml.risk_aggregator import calculate_composite_risk_score
from api.routes_hitl import enqueue_review, review_transaction, ReviewDecision
from ml.decision_policy import apply_decision_policy


class TestVelocityRegressions(unittest.TestCase):
    def _base(self):
        return {
            "transaction_id": "TXN_REGRESSION_VEL",
            "user_id": "USER_REGRESSION_VEL",
            "device_id": "DEV_REGRESSION_VEL",
            "ip_address": "10.240.240.10",
            "merchant_id": "MCH_041",
            "amount": 1000,
            "is_vpn_proxy": False,
            "is_suspicious_proxy": False,
        }

    def tearDown(self):
        conn = get_raw_sqlite_connection()
        conn.execute("DELETE FROM transactions WHERE user_id = 'USER_REGRESSION_VEL'")
        conn.commit()
        conn.close()

    def test_backend_velocity_counts_prior_rows_plus_current(self):
        observed = []
        for i in range(1, 6):
            risk = calculate_composite_risk_score({**self._base(), "transaction_id": f"TXN_REG_VEL_{i}", "velocity_enabled": False})
            observed.append(risk["velocity_1h"])
            conn = get_raw_sqlite_connection()
            conn.execute(
                """INSERT INTO transactions
                (transaction_id,user_id,device_id,ip_address,merchant_id,amount,timestamp,velocity_1h,velocity_enabled,velocity_source)
                VALUES (?,?,?,?,?,?,datetime('now'),?,0,'BACKEND')""",
                (f"TXN_REG_VEL_{i}", self._base()["user_id"], self._base()["device_id"], self._base()["ip_address"], self._base()["merchant_id"], 1000, risk["velocity_1h"]),
            )
            conn.commit(); conn.close()
        self.assertEqual(observed, [1, 2, 3, 4, 5])

    def test_client_mode_trusts_value_and_backend_mode_ignores_it(self):
        client = calculate_composite_risk_score({**self._base(), "velocity_enabled": True, "velocity_1h": 999})
        backend = calculate_composite_risk_score({**self._base(), "velocity_enabled": False, "velocity_1h": 999})
        self.assertEqual((client["velocity_source"], client["velocity_1h"]), ("CLIENT", 999))
        self.assertEqual(backend["velocity_source"], "BACKEND")
        self.assertNotEqual(backend["velocity_1h"], 999)

    def test_client_mode_requires_velocity_and_rejects_negative(self):
        with self.assertRaises(ValueError):
            calculate_composite_risk_score({**self._base(), "velocity_enabled": True})
        with self.assertRaises(ValueError):
            calculate_composite_risk_score({**self._base(), "velocity_enabled": True, "velocity_1h": -1})

    def test_velocity_thresholds_are_explicit(self):
        for value, expected in [(0, 1.0), (4, 1.0), (5, 1.25), (9, 1.25), (10, 1.5)]:
            risk = calculate_composite_risk_score({**self._base(), "velocity_enabled": True, "velocity_1h": value})
            self.assertEqual(risk["velocity_multiplier"], expected)

    def test_proxy_overlay_is_additive_to_velocity_multiplier(self):
        risk = calculate_composite_risk_score({**self._base(), "velocity_enabled": True, "velocity_1h": 10, "is_vpn_proxy": True})
        self.assertEqual(risk["velocity_multiplier"], 1.72)


class TestHITLRegressions(unittest.TestCase):
    def setUp(self):
        self.txn_id = "TXN_REGRESSION_HITL"
        conn = get_raw_sqlite_connection()
        conn.execute("DELETE FROM human_reviews WHERE transaction_id = ?", (self.txn_id,))
        conn.execute("DELETE FROM risk_scores WHERE transaction_id = ?", (self.txn_id,))
        conn.execute("DELETE FROM transactions WHERE transaction_id = ?", (self.txn_id,))
        conn.execute(
            """INSERT INTO transactions(transaction_id,user_id,device_id,ip_address,merchant_id,amount,timestamp)
            VALUES(?,?,?,?,?,100000,datetime('now'))""",
            (self.txn_id, "USER_HITL_REG", "DEV_HITL_REG", "10.250.250.10", "MCH_001"),
        )
        conn.execute(
            """INSERT INTO risk_scores(scoring_id,transaction_id,risk_score,tabular_score,gnn_score,stacker_calibrated_score,velocity_multiplier,evidence_multiplier,risk_tier,decision)
            VALUES('SCORE_HITL_REG',?,?,?,?,?,1.0,1.0,'LOW','APPROVE')""",
            (self.txn_id, 0.0, 0.0, 0.0, 0.0),
        )
        conn.commit(); conn.close()

    def tearDown(self):
        conn = get_raw_sqlite_connection()
        conn.execute("DELETE FROM human_reviews WHERE transaction_id = ?", (self.txn_id,))
        conn.execute("DELETE FROM risk_scores WHERE transaction_id = ?", (self.txn_id,))
        conn.execute("DELETE FROM transactions WHERE transaction_id = ?", (self.txn_id,))
        conn.commit(); conn.close()

    def test_hitl_queue_is_idempotent(self):
        risk = {"hitl_required": True, "risk_score": 95, "review_reasons": ["HIGH_IMPACT"], "external_evidence": {}}
        first = enqueue_review(self.txn_id, risk)
        second = enqueue_review(self.txn_id, risk)
        self.assertEqual(first, second)
        conn = get_raw_sqlite_connection()
        count = conn.execute("SELECT COUNT(*) FROM human_reviews WHERE transaction_id = ? AND status='PENDING'", (self.txn_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_hitl_resolution_updates_risk_decision(self):
        risk = {"hitl_required": True, "risk_score": 95, "review_reasons": ["HIGH_IMPACT"], "external_evidence": {}}
        review_id = enqueue_review(self.txn_id, risk)
        response = review_transaction(review_id, ReviewDecision(decision="BLOCK", reviewer="tester", rationale="Regression test"))
        self.assertEqual(response["status"], "RESOLVED")
        conn = get_raw_sqlite_connection()
        decision = conn.execute("SELECT decision FROM risk_scores WHERE transaction_id = ?", (self.txn_id,)).fetchone()[0]
        status = conn.execute("SELECT status FROM human_reviews WHERE review_id = ?", (review_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(decision, "BLOCK")
        self.assertEqual(status, "RESOLVED")

    def test_resolved_review_cannot_be_resolved_twice(self):
        risk = {"hitl_required": True, "risk_score": 95, "review_reasons": ["HIGH_IMPACT"], "external_evidence": {}}
        review_id = enqueue_review(self.txn_id, risk)
        review_transaction(review_id, ReviewDecision(decision="HOLD", reviewer="tester", rationale="First decision"))
        with self.assertRaises(Exception):
            review_transaction(review_id, ReviewDecision(decision="BLOCK", reviewer="tester", rationale="Second decision"))


class TestDocumentationAndFrontendContract(unittest.TestCase):
    def test_frontend_contains_velocity_source_toggle_and_model_columns(self):
        html = Path("static/index.html").read_text(encoding="utf-8")
        js = Path("static/js/app.js").read_text(encoding="utf-8")
        for token in ("velocity_enabled", "velocity_1h", "Trust client-provided velocity", "Calculate from backend history"):
            self.assertIn(token, html + js)
        for token in ("GNN node embedding", "Tabular ML", "Stacker calibrated"):
            self.assertIn(token, html)

    def test_documentation_mentions_regressions(self):
        readme = Path("README.md").read_text(encoding="utf-8")
        workflow = Path("PROJECT_WORKFLOW.md").read_text(encoding="utf-8")
        for token in ("velocity", "HITL", "stale GNN", "human review"):
            self.assertIn(token.lower(), (readme + workflow).lower())


class TestVelocityFieldErrorHandlingContract(unittest.TestCase):
    """Bug #26 regression: velocity_1h is the only field on the scoring form
    with backend-side validation (Pydantic's `ge=0`, plus risk_aggregator.py's
    "required when velocity_enabled=true" check). handleTransactionScore()
    used to call res.json() unconditionally with no res.ok check, so a
    validation failure on this field handed FastAPI's raw error body
    (a string, or a list of {loc, msg, ...} objects) straight to
    updateRiskDisplay(data.risk_evaluation) — which is undefined on an error
    response — crashing and leaking that raw structure into the user-facing
    alert. These check the source for the specific fix, not just that some
    error handling exists, since a shallower fix (e.g. only checking res.ok
    without extracting a clean message) would still leak the raw body."""

    def setUp(self):
        self.js = Path("static/js/app.js").read_text(encoding="utf-8")

    def test_score_response_checks_ok_before_using_the_body(self):
        self.assertIn("{ ok: res.ok, status: res.status, data }", self.js)
        self.assertIn("if (!ok) {", self.js)

    def test_error_message_extractor_handles_both_fastapi_error_shapes(self):
        self.assertIn("function extractErrorMessage(data, status)", self.js)
        # Plain-string `detail` (risk_aggregator.py's manually-raised ValueError).
        self.assertIn("typeof detail === 'string'", self.js)
        # List-of-objects `detail` (Pydantic's own validation-error shape).
        self.assertIn("Array.isArray(detail)", self.js)

    def test_invalid_client_velocity_is_rejected_before_the_network_call(self):
        self.assertIn("Number.isNaN(parsedVelocity)", self.js)
        self.assertIn("parsedVelocity < 0", self.js)

    def test_preset_switch_resets_stale_velocity_value(self):
        self.assertIn("document.getElementById('velocity_1h').value = '1';", self.js)

    def test_recent_transactions_table_has_a_velocity_fallback(self):
        self.assertIn("t.velocity_1h ?? 0", self.js)


class TestGraphFreshnessContract(unittest.TestCase):
    def test_transaction_path_invalidates_live_graph_after_commit(self):
        source = Path("api/routes_transactions.py").read_text(encoding="utf-8")
        self.assertIn("invalidate_live_graph_snapshot()", source)
        self.assertIn("conn.commit()", source)
        self.assertLess(source.index("conn.commit()"), source.index("invalidate_live_graph_snapshot()"))


class TestRealDataIngestionIsAdditive(unittest.TestCase):
    """Bug #22 regression: ingest_real_kaggle_dataset.py used to open with
    DELETE FROM transactions/users/devices/... before loading the real CSV,
    silently wiping every synthetic golden-matrix scenario. This proves the
    current version never deletes anything and layers real rows onto the
    existing fraud-ring/baseline identities instead."""

    def test_ingestion_never_deletes_and_preserves_golden_matrix_users(self):
        import numpy as np
        import pandas as pd
        from data.ingest_real_kaggle_dataset import ingest_real_dataset, CSV_PATH

        conn = get_raw_sqlite_connection()
        before_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        before_hostel = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = 'USER_HOSTEL_1'"
        ).fetchone()[0]
        before_ring1 = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = 'USER_RING1_1'"
        ).fetchone()[0]
        conn.close()
        self.assertGreater(before_hostel, 0, "golden-matrix fixture missing — run generate_synthetic_data first")

        np.random.seed(7)
        n_normal, n_fraud = 200, 10
        df = pd.DataFrame({
            "Time": np.sort(np.random.uniform(0, 86400, n_normal + n_fraud)),
            "Amount": np.concatenate([np.random.exponential(50, n_normal), np.random.exponential(300, n_fraud)]),
            "Class": np.concatenate([np.zeros(n_normal, dtype=int), np.ones(n_fraud, dtype=int)]),
        })
        wrote_temp_csv = not CSV_PATH.exists()
        if wrote_temp_csv:
            df.to_csv(CSV_PATH, index=False)
        try:
            ingest_real_dataset(sample_size=len(df))
        finally:
            if wrote_temp_csv:
                CSV_PATH.unlink(missing_ok=True)

        conn = get_raw_sqlite_connection()
        after_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        after_hostel = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = 'USER_HOSTEL_1'"
        ).fetchone()[0]
        after_ring1 = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = 'USER_RING1_1'"
        ).fetchone()[0]
        real_rows_added = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE transaction_id LIKE 'TXN_REAL_%'"
        ).fetchone()[0]
        conn.close()

        self.assertGreaterEqual(after_users, before_users, "ingestion must never reduce the user count")
        self.assertEqual(after_hostel, before_hostel, "golden-matrix hostel scenario must survive ingestion untouched")
        self.assertGreaterEqual(after_ring1, before_ring1, "Ring1 fraud identity must survive, and may gain real-fraud rows")
        self.assertGreater(real_rows_added, 0, "ingestion should have added real transactions on top")


if __name__ == "__main__":
    unittest.main()


