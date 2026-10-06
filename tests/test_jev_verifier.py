"""Tests for agent/jev_verifier.py and the HITL auto-resolution it feeds."""
import unittest
from unittest.mock import patch, MagicMock
import requests
from agent import jev_verifier
from api.routes_hitl import auto_resolve_review, jev_auto_resolve_eligible, _ACTION_TO_HITL_DECISION
from db.database import get_raw_sqlite_connection

def _evidence():
    return {
        "graph_evidence": {"shared_device_account_count": 6, "shared_ip_account_count": 6, "community_size": 6},
        "history_evidence": {"total_historical_txns": 2, "historical_avg_amount": 500.0},
        "device_evidence": {"is_suspicious_proxy": True, "os": "Android", "device_type": "Mobile", "city": "Mumbai", "country": "IN", "isp": "Test ISP"},
        "model_evidence": {"tabular_score": 91.0, "gnn_score": 95.0},
    }

def _mock_response(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp

class TestJevVerifierAvailability(unittest.TestCase):
    def test_unavailable_without_key(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", ""):
            self.assertFalse(jev_verifier.is_available())
    def test_available_with_key(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123"):
            self.assertTrue(jev_verifier.is_available())
    def test_raises_cleanly_when_called_without_key(self):
        with patch.object(jev_verifier, "TYPESAFE_API_KEY", ""):
            with self.assertRaises(RuntimeError):
                jev_verifier.verify_investigation({}, {}, _evidence(), "some hypothesis", "HOLD_FOR_MANUAL_REVIEW")

class TestJevVerifierAgreement(unittest.TestCase):
    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_agreement_and_high_confidence_is_consistent_and_auto_resolve_eligible(self, mock_post):
        mock_post.side_effect = [
            _mock_response({"model": "jev-1.13.0", "answers": {"recommended_action": {"type": "choice", "choice": "HOLD_FOR_MANUAL_REVIEW", "probabilities": {"HOLD_FOR_MANUAL_REVIEW": 0.91}, "confidence": 0.91}}}),
            _mock_response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.97}}}),
        ]
        result = jev_verifier.verify_investigation({"transaction_id": "TXN_1"}, {"risk_score": 82}, _evidence(), fraud_hypothesis="Shared device/IP cluster with 6 linked accounts and a suspicious proxy.", recommended_action="HOLD_FOR_MANUAL_REVIEW")
        self.assertEqual(result["verification_flag"], "CONSISTENT"); self.assertTrue(result["actions_agree"]); self.assertTrue(result["hypothesis_grounded"]); self.assertTrue(result["eligible_for_auto_resolve"]); self.assertEqual(mock_post.call_count, 2)

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_disagreement_flags_review_and_is_not_auto_resolve_eligible(self, mock_post):
        mock_post.side_effect = [
            _mock_response({"model": "jev-1.13.0", "answers": {"recommended_action": {"type": "choice", "choice": "BLOCK_ACCOUNT_AND_HOLD_FUNDS", "probabilities": {"BLOCK_ACCOUNT_AND_HOLD_FUNDS": 0.88}, "confidence": 0.88}}}),
            _mock_response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.9}}}),
        ]
        result = jev_verifier.verify_investigation({"transaction_id": "TXN_2"}, {"risk_score": 82}, _evidence(), fraud_hypothesis="Looks like a shared-device false positive.", recommended_action="APPROVE_WITH_VERIFICATION")
        self.assertEqual(result["verification_flag"], "REVIEW_RECOMMENDED"); self.assertFalse(result["actions_agree"]); self.assertFalse(result["eligible_for_auto_resolve"])

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_low_grounding_flags_review_even_on_action_agreement(self, mock_post):
        mock_post.side_effect = [
            _mock_response({"model": "jev-1.13.0", "answers": {"recommended_action": {"type": "choice", "choice": "HOLD_FOR_MANUAL_REVIEW", "probabilities": {"HOLD_FOR_MANUAL_REVIEW": 0.95}, "confidence": 0.95}}}),
            _mock_response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.2}}}),
        ]
        result = jev_verifier.verify_investigation({"transaction_id": "TXN_3"}, {"risk_score": 82}, _evidence(), fraud_hypothesis="Claims a 14-account ring across 3 countries not present in evidence.", recommended_action="HOLD_FOR_MANUAL_REVIEW")
        self.assertTrue(result["actions_agree"]); self.assertFalse(result["hypothesis_grounded"]); self.assertEqual(result["verification_flag"], "REVIEW_RECOMMENDED"); self.assertFalse(result["eligible_for_auto_resolve"])

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch.object(jev_verifier, "JEV_AUTO_RESOLVE_MIN_CONFIDENCE", 0.85)
    @patch("agent.jev_verifier.requests.post")
    def test_agreement_below_confidence_floor_is_not_auto_resolve_eligible(self, mock_post):
        mock_post.side_effect = [
            _mock_response({"model": "jev-1.13.0", "answers": {"recommended_action": {"type": "choice", "choice": "HOLD_FOR_MANUAL_REVIEW", "probabilities": {"HOLD_FOR_MANUAL_REVIEW": 0.60}, "confidence": 0.60}}}),
            _mock_response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": 0.9}}}),
        ]
        result = jev_verifier.verify_investigation({"transaction_id": "TXN_4"}, {"risk_score": 82}, _evidence(), fraud_hypothesis="Ambiguous case.", recommended_action="HOLD_FOR_MANUAL_REVIEW")
        self.assertEqual(result["verification_flag"], "CONSISTENT"); self.assertFalse(result["eligible_for_auto_resolve"])

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_unsupported_action_is_rejected(self, mock_post):
        mock_post.side_effect = [_mock_response({"answers": {"recommended_action": {"type": "choice", "choice": "DO_SOMETHING_DANGEROUS", "probabilities": {}, "confidence": 0.99}}})]
        with self.assertRaisesRegex(RuntimeError, "unsupported action"):
            jev_verifier.verify_investigation({"transaction_id": "TXN_BAD_ACTION"}, {"risk_score": 80}, _evidence(), "Evidence-backed hypothesis.", "HOLD_FOR_MANUAL_REVIEW")

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_malformed_grounding_probability_is_rejected(self, mock_post):
        mock_post.side_effect = [
            _mock_response({"answers": {"recommended_action": {"type": "choice", "choice": "HOLD_FOR_MANUAL_REVIEW", "probabilities": {"HOLD_FOR_MANUAL_REVIEW": 0.9}, "confidence": 0.9}}}),
            _mock_response({"answers": {"hypothesis_grounded": {"type": "noul", "noul": "not-a-probability"}}}),
        ]
        with self.assertRaisesRegex(RuntimeError, "Malformed Jev grounding response"):
            jev_verifier.verify_investigation({"transaction_id": "TXN_BAD_GROUNDING"}, {"risk_score": 80}, _evidence(), "Evidence-backed hypothesis.", "HOLD_FOR_MANUAL_REVIEW")

class TestJevAutoResolveEligibility(unittest.TestCase):
    def test_mandatory_human_reason_blocks_auto_resolve(self):
        self.assertFalse(jev_auto_resolve_eligible({"hitl_required": True, "review_reasons": ["HIGH_IMPACT"]}, {"jev_verification": {"eligible_for_auto_resolve": True}}))
    def test_non_mandatory_review_can_be_auto_resolved(self):
        self.assertTrue(jev_auto_resolve_eligible({"hitl_required": True, "review_reasons": ["RISK_TIER"]}, {"jev_verification": {"eligible_for_auto_resolve": True}}))
    def test_missing_jev_result_cannot_auto_resolve(self):
        self.assertFalse(jev_auto_resolve_eligible({"hitl_required": True, "review_reasons": ["RISK_TIER"]}, {}))

class TestActionToHitlDecisionMapping(unittest.TestCase):
    def test_all_five_investigator_actions_map_to_a_valid_hitl_decision(self):
        valid={"APPROVE","HOLD","BLOCK"}; expected={"BLOCK_ACCOUNT_AND_HOLD_FUNDS","HOLD_FOR_MANUAL_REVIEW","TEMPORARY_VELOCITY_FREEZE","REQUIRE_TWO_FACTOR_AUTHENTICATION","APPROVE_WITH_VERIFICATION"}
        self.assertEqual(set(_ACTION_TO_HITL_DECISION.keys()), expected); self.assertTrue(all(v in valid for v in _ACTION_TO_HITL_DECISION.values()))
    def test_unknown_action_defaults_to_hold_not_approve(self):
        self.assertEqual(_ACTION_TO_HITL_DECISION.get("SOME_FUTURE_ACTION","HOLD"),"HOLD")

class TestAutoResolveReviewAgainstRealDb(unittest.TestCase):
    def _seed_pending_review(self, txn_id, review_id):
        conn=get_raw_sqlite_connection()
        conn.execute("INSERT OR IGNORE INTO users (user_id, name, email) VALUES (?, ?, ?)",(f"USER_{txn_id}","Test User","test@example.com"))
        conn.execute("INSERT OR IGNORE INTO devices (device_id, device_type, os) VALUES (?, ?, ?)",(f"DEV_{txn_id}","Mobile","Android"))
        conn.execute("INSERT OR IGNORE INTO ip_addresses (ip_address, country) VALUES (?, ?)",(f"IP_{txn_id}","IN"))
        conn.execute("INSERT OR REPLACE INTO transactions (transaction_id, user_id, device_id, ip_address, amount) VALUES (?, ?, ?, ?, ?)",(txn_id,f"USER_{txn_id}",f"DEV_{txn_id}",f"IP_{txn_id}",1000.0))
        conn.execute("INSERT OR REPLACE INTO human_reviews (review_id, transaction_id, status, risk_score, reasons_json, evidence_json) VALUES (?, ?, 'PENDING', ?, ?, ?)",(review_id,txn_id,82.0,"[]","{}")); conn.commit(); conn.close()
    def test_auto_resolve_marks_review_resolved_with_mapped_decision(self):
        txn_id,review_id="TXN_AUTORESOLVE_1","REV_AUTORESOLVE_1"; self._seed_pending_review(txn_id,review_id)
        self.assertEqual(auto_resolve_review(txn_id,"HOLD_FOR_MANUAL_REVIEW","Auto-resolved by test: Jev and investigator agreed."),review_id)
        conn=get_raw_sqlite_connection(); row=conn.execute("SELECT status, reviewer, reviewer_decision FROM human_reviews WHERE review_id = ?",(review_id,)).fetchone(); conn.close()
        self.assertEqual(row[0],"RESOLVED"); self.assertEqual(row[1],"jev_auto_triage"); self.assertEqual(row[2],"HOLD")
    def test_auto_resolve_is_a_noop_when_nothing_pending(self):
        txn_id,review_id="TXN_AUTORESOLVE_2","REV_AUTORESOLVE_2"; self._seed_pending_review(txn_id,review_id)
        first=auto_resolve_review(txn_id,"APPROVE_WITH_VERIFICATION","first pass"); second=auto_resolve_review(txn_id,"APPROVE_WITH_VERIFICATION","second pass")
        self.assertEqual(first,review_id); self.assertIsNone(second)
    def test_auto_resolve_returns_none_when_no_review_was_ever_queued(self):
        self.assertIsNone(auto_resolve_review("TXN_NEVER_QUEUED","HOLD_FOR_MANUAL_REVIEW","n/a"))


    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch.object(jev_verifier, "JEV_MAX_RETRIES", 2)
    @patch("agent.jev_verifier.requests.post")
    def test_transient_failure_retries_then_succeeds(self, mock_post):
        mock_post.side_effect = [requests.Timeout(), _mock_response({"answers": {"recommended_action": {"choice": "HOLD_FOR_MANUAL_REVIEW", "confidence": 0.9}}}), _mock_response({"answers": {"hypothesis_grounded": {"noul": 0.9}}})]
        result = jev_verifier.verify_investigation({}, {"risk_score": 80}, _evidence(), "Evidence-backed.", "HOLD_FOR_MANUAL_REVIEW")
        self.assertEqual(result["verification_flag"], "CONSISTENT"); self.assertEqual(mock_post.call_count, 3)

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch.object(jev_verifier, "JEV_MAX_RETRIES", 1)
    @patch.object(jev_verifier, "JEV_CIRCUIT_FAILURE_THRESHOLD", 1)
    @patch("agent.jev_verifier.requests.post")
    def test_circuit_breaker_opens_after_transient_exhaustion(self, mock_post):
        mock_post.side_effect = requests.Timeout()
        jev_verifier.reset_circuit_breaker()
        with self.assertRaises(RuntimeError): jev_verifier.verify_investigation({}, {}, _evidence(), "x", "HOLD_FOR_MANUAL_REVIEW")
        with self.assertRaisesRegex(RuntimeError, "circuit breaker"): jev_verifier.verify_investigation({}, {}, _evidence(), "x", "HOLD_FOR_MANUAL_REVIEW")
        self.assertEqual(mock_post.call_count, 2)
        jev_verifier.reset_circuit_breaker()

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_non_transient_4xx_is_not_retried(self, mock_post):
        resp=_mock_response({}); resp.status_code=400
        mock_post.return_value=resp
        with self.assertRaisesRegex(RuntimeError, "HTTP request failed"): jev_verifier.verify_investigation({}, {}, _evidence(), "x", "HOLD_FOR_MANUAL_REVIEW")
        self.assertEqual(mock_post.call_count, 1)

    @patch.object(jev_verifier, "TYPESAFE_API_KEY", "sk_test_123")
    @patch("agent.jev_verifier.requests.post")
    def test_nan_action_confidence_is_rejected(self, mock_post):
        import math
        mock_post.return_value=_mock_response({"answers": {"recommended_action": {"choice": "HOLD_FOR_MANUAL_REVIEW", "confidence": math.nan}}})
        with self.assertRaisesRegex(RuntimeError, "invalid action confidence"): jev_verifier.verify_investigation({}, {}, _evidence(), "x", "HOLD_FOR_MANUAL_REVIEW")

if __name__=="__main__":
    unittest.main()
