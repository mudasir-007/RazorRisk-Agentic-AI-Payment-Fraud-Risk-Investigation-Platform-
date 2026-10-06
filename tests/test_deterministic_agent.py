"""Regression tests for agent/deterministic_agent.py's evidence-confluence
fix. Added after actually running the hostel benign scenario through the
investigation path and finding it produced a false "fraud ring detected" /
BLOCK_ACCOUNT_AND_HOLD_FUNDS hypothesis from connectivity alone. These pin
the fix down so it can't silently regress back to the connectivity-only
behavior. velocity_1h is passed explicitly (matching the current function
signature) rather than embedded in the txn dict — this module no longer
reads a velocity field from txn_payload at all.
"""
import unittest

from agent.deterministic_agent import determine_fraud_hypothesis


class TestDeterministicAgentConfluence(unittest.TestCase):
    def history(self, avg_amount=500.0, max_velocity=1):
        return {
            "historical_avg_amount": avg_amount,
            "historical_max_velocity_1h": max_velocity,
        }

    def device(self, is_suspicious_proxy=False, isp="Test ISP", country="IN"):
        return {"is_suspicious_proxy": is_suspicious_proxy, "isp": isp, "country": country}

    # --- Connectivity alone, no behavioral anomaly: must NOT escalate ---

    def test_shared_ip_alone_does_not_block_or_hold(self):
        """A 7-person hostel or a 40-person carrier-NAT IP: high shared_ip,
        ordinary velocity and amount. This used to trigger HOLD_FOR_MANUAL_
        REVIEW purely on shared_ip_account_count >= 4."""
        txn = {"device_id": "DEV_1", "ip_address": "1.2.3.4", "amount": 400}
        graph_evidence = {
            "shared_device_account_count": 1, "shared_ip_account_count": 40,
            "community_fraud_ratio": 0.0,
        }
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=380), self.device(), velocity_1h=1)
        self.assertNotIn(action, ("BLOCK_ACCOUNT_AND_HOLD_FUNDS", "HOLD_FOR_MANUAL_REVIEW"))
        self.assertEqual(action, "APPROVE_WITH_VERIFICATION")

    def test_shared_device_alone_does_not_block(self):
        """A 4-5 user shared office/POS device with ordinary spending.
        This used to trigger BLOCK_ACCOUNT_AND_HOLD_FUNDS purely on
        shared_device_account_count >= 3."""
        txn = {"device_id": "DEV_OFFICE", "ip_address": "5.6.7.8", "amount": 300}
        graph_evidence = {
            "shared_device_account_count": 5, "shared_ip_account_count": 1,
            "community_fraud_ratio": 0.0,
        }
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=280), self.device(), velocity_1h=1)
        self.assertNotEqual(action, "BLOCK_ACCOUNT_AND_HOLD_FUNDS")
        self.assertEqual(action, "APPROVE_WITH_VERIFICATION")

    def test_suspicious_proxy_alone_without_anomaly_does_not_hold(self):
        txn = {"device_id": "DEV_2", "ip_address": "9.9.9.9", "amount": 300}
        graph_evidence = {"shared_device_account_count": 1, "shared_ip_account_count": 1, "community_fraud_ratio": 0.0}
        _, action, _ = determine_fraud_hypothesis(
            txn, graph_evidence, self.history(avg_amount=280), self.device(is_suspicious_proxy=True), velocity_1h=1
        )
        self.assertNotEqual(action, "HOLD_FOR_MANUAL_REVIEW")

    # --- Connectivity + behavioral anomaly together: SHOULD escalate ---

    def test_shared_device_with_anomaly_blocks(self):
        txn = {"device_id": "DEV_RING", "ip_address": "1.1.1.1", "amount": 88000}
        graph_evidence = {
            "shared_device_account_count": 7, "shared_ip_account_count": 7,
            "community_fraud_ratio": 0.8,
        }
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=300), self.device(), velocity_1h=12)
        self.assertEqual(action, "BLOCK_ACCOUNT_AND_HOLD_FUNDS")

    def test_shared_ip_with_amount_anomaly_holds(self):
        txn = {"device_id": "DEV_3", "ip_address": "2.2.2.2", "amount": 96000}
        graph_evidence = {"shared_device_account_count": 1, "shared_ip_account_count": 6, "community_fraud_ratio": 0.5}
        _, action, _ = determine_fraud_hypothesis(
            txn, graph_evidence, self.history(avg_amount=1000), self.device(is_suspicious_proxy=True), velocity_1h=1
        )
        self.assertEqual(action, "HOLD_FOR_MANUAL_REVIEW")

    # --- Standalone velocity carding: no connectivity needed ---

    def test_high_velocity_alone_freezes(self):
        txn = {"device_id": "DEV_4", "ip_address": "3.3.3.3", "amount": 50}
        graph_evidence = {"shared_device_account_count": 1, "shared_ip_account_count": 1, "community_fraud_ratio": 0.0}
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=45), self.device(), velocity_1h=10)
        self.assertEqual(action, "TEMPORARY_VELOCITY_FREEZE")

    # --- Cold start: no history, amount is the only signal available ---

    def test_cold_start_large_amount_is_treated_as_anomalous(self):
        txn = {"device_id": "DEV_5", "ip_address": "4.4.4.4", "amount": 90000}
        graph_evidence = {"shared_device_account_count": 4, "shared_ip_account_count": 1, "community_fraud_ratio": 0.0}
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=0.0), self.device(), velocity_1h=1)
        self.assertEqual(action, "BLOCK_ACCOUNT_AND_HOLD_FUNDS")

    def test_cold_start_small_amount_is_not_anomalous(self):
        txn = {"device_id": "DEV_6", "ip_address": "5.5.5.5", "amount": 300}
        graph_evidence = {"shared_device_account_count": 4, "shared_ip_account_count": 1, "community_fraud_ratio": 0.0}
        _, action, _ = determine_fraud_hypothesis(txn, graph_evidence, self.history(avg_amount=0.0), self.device(), velocity_1h=1)
        self.assertEqual(action, "APPROVE_WITH_VERIFICATION")


if __name__ == "__main__":
    unittest.main()
