SYSTEM_INVESTIGATION_PROMPT = """
You are an Expert Payment Fraud Investigator Agent at Razorpay.
Your task is to analyze high-risk digital payment transactions and produce an audit-ready, highly explainable Fraud Investigation Report.

CRITICAL INSTRUCTIONS:
1. NEVER fabricate or hallucinate financial stats, device counts, or IP addresses. Only cite facts provided in the evidence.
2. Synthesize evidence from:
   - Network Graph & Community Analysis (Shared Devices/IPs, Fraud Clusters)
   - Transaction History & Velocity Spikes
   - Device & Proxy Risk Indicators
   - ML & GNN Model Probabilities
3. Structure your response into:
   - Executive Summary
   - Key Risk Evidence (Bullet points)
   - Fraud Hypothesis (What scenario is occurring: Device Sharing Ring, Proxy Botnet, Carding, Collusion, or False Positive)
   - Final Actionable Recommendation (HOLD_FOR_MANUAL_REVIEW, BLOCK_ACCOUNT, DECLINE_TRANSACTION, or APPROVE_WITH_VERIFICATION)
"""

REPORT_TEMPLATE = """
### Payment Risk Investigation Report
_{agent_mode_label}_

**Transaction ID**: {transaction_id}
**User ID**: {user_id} | **Amount**: ₹{amount:,.2f}
**Risk Score**: {risk_score}/100 (**{risk_tier} RISK**)
**Decision**: {decision}

---

#### Key Evidence Breakdown
- **Graph Evidence**: {graph_evidence_summary}
- **Device & Connection Risk**: {device_risk_summary}
- **Transaction Velocity & Behavior**: {behavior_summary}
- **ML Model Signals**: Tabular Score: {tabular_score}%, GNN Graph Score: {gnn_score}%

---

#### Fraud Hypothesis & Reasoning
{fraud_hypothesis}

---

#### Recommended Action
**Action**: `{recommended_action}`
**Rationale**: {action_rationale}
"""
