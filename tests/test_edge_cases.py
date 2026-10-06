
"""RazorRisk adversarial false-positive / uncertainty regression suite.

These tests encode the design requirement that connectivity is evidence, not
a verdict, and that uncertain/conflicting cases enter HITL.
"""
import unittest
from ml.decision_policy import apply_decision_policy


class TestAdversarialDecisionPolicy(unittest.TestCase):
    def risk(self, tab, gnn, stack, tier="LOW", graph=None):
        return {
            "risk_score": stack * 100,
            "tabular_score": tab * 100,
            "gnn_score": gnn * 100,
            "stacker_calibrated_score": stack * 100,
            "risk_tier": tier,
            "graph_evidence": graph or {
                "shared_device_accounts": 1,
                "shared_ip_accounts": 1,
            },
        }

    def txn(self, amount=500, velocity=1):
        return {
            "transaction_id": "TXN_EDGE_TEST",
            "user_id": "USER_EDGE",
            "device_id": "DEV_EDGE",
            "ip_address": "203.0.113.10",
            "merchant_id": "MCH_001",
            "amount": amount,
            "velocity_1h": velocity,
        }

    def test_shared_ip_alone_is_not_hitl(self):
        risk = self.risk(
            0.08, 0.10, 0.09, "LOW",
            {"shared_device_accounts": 1, "shared_ip_accounts": 50},
        )
        result = apply_decision_policy(self.txn(), risk)
        self.assertFalse(result["hitl_required"])

    def test_model_disagreement_enters_hitl(self):
        risk = self.risk(0.90, 0.15, 0.70, "HIGH")
        result = apply_decision_policy(self.txn(), risk)
        self.assertTrue(result["hitl_required"])
        self.assertIn("MODEL_DISAGREEMENT", result["review_reasons"])

    def test_high_impact_enters_hitl(self):
        risk = self.risk(0.20, 0.20, 0.20, "LOW")
        result = apply_decision_policy(self.txn(amount=100000), risk)
        self.assertTrue(result["hitl_required"])
        self.assertIn("HIGH_IMPACT", result["review_reasons"])

    def test_shared_device_is_not_a_verdict(self):
        risk = self.risk(
            0.10, 0.12, 0.11, "LOW",
            {"shared_device_accounts": 5, "shared_ip_accounts": 5},
        )
        result = apply_decision_policy(self.txn(), risk)
        self.assertFalse(result["hitl_required"])

    def test_uncertain_medium_case_enters_hitl(self):
        risk = self.risk(0.50, 0.52, 0.51, "MEDIUM")
        result = apply_decision_policy(self.txn(), risk)
        self.assertTrue(result["hitl_required"])
        self.assertIn("MODEL_UNCERTAINTY", result["review_reasons"])


if __name__ == "__main__":
    unittest.main()
