import json
from db.database import get_raw_sqlite_connection
from ml.graph_builder import graph_builder
from utils.logger import get_logger

logger = get_logger("agent_tools")

class GraphTool:
    """Deterministic tool to inspect network graph topology and shared entity linkages."""
    name = "GraphTool"
    description = "Queries shared device/IP connections, community size, and connected account counts."

    @staticmethod
    def run(user_id: str) -> dict:
        logger.info(f"[GraphTool] Invoked for User: {user_id}")
        g_feat = graph_builder.extract_user_graph_features(user_id)
        
        # Query specific connected user IDs
        conn = get_raw_sqlite_connection()
        cursor = conn.cursor()
        
        # Get shared device users
        cursor.execute("""
            SELECT DISTINCT t2.user_id 
            FROM transactions t1
            JOIN transactions t2 ON t1.device_id = t2.device_id AND t1.user_id != t2.user_id
            WHERE t1.user_id = ?
            LIMIT 10
        """, (user_id,))
        shared_device_users = [r[0] for r in cursor.fetchall()]

        # Get shared IP users
        cursor.execute("""
            SELECT DISTINCT t2.user_id 
            FROM transactions t1
            JOIN transactions t2 ON t1.ip_address = t2.ip_address AND t1.user_id != t2.user_id
            WHERE t1.user_id = ?
            LIMIT 10
        """, (user_id,))
        shared_ip_users = [r[0] for r in cursor.fetchall()]

        conn.close()

        result = {
            "user_id": user_id,
            "shared_device_account_count": g_feat["shared_device_accounts"],
            "shared_device_users": shared_device_users,
            "shared_ip_account_count": g_feat["shared_ip_accounts"],
            "shared_ip_users": shared_ip_users,
            "community_id": g_feat["community_id"],
            "community_size": g_feat["community_size"],
            "community_fraud_ratio": g_feat["community_fraud_ratio"],
            "suspicious_network_cluster": False,
            "connectivity_strength": "strong" if (g_feat["shared_device_accounts"] >= 3 or g_feat["shared_ip_accounts"] >= 5) else "weak_or_normal"
        }
        logger.info(f"[GraphTool] Result for User:{user_id} -> ClusterSuspicious:{result['suspicious_network_cluster']}")
        return result

class TransactionHistoryTool:
    """Deterministic tool to query historical transaction baseline and velocity stats."""
    name = "TransactionHistoryTool"
    description = "Retrieves user's 30-day historical transaction volume, average spend, and max hourly velocity."

    @staticmethod
    def run(user_id: str) -> dict:
        logger.info(f"[TransactionHistoryTool] Invoked for User: {user_id}")
        conn = get_raw_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                COUNT(*) as total_txns,
                COALESCE(AVG(amount), 0.0) as avg_amount,
                COALESCE(MAX(amount), 0.0) as max_amount,
                COALESCE(MAX(velocity_1h), 1) as max_velocity
            FROM transactions
            WHERE user_id = ?
        """, (user_id,))
        row = cursor.fetchone()
        conn.close()

        result = {
            "user_id": user_id,
            "total_historical_txns": row[0],
            "historical_avg_amount": round(row[1], 2),
            "historical_max_amount": round(row[2], 2),
            "historical_max_velocity_1h": row[3]
        }
        logger.info(f"[TransactionHistoryTool] Result for User:{user_id} -> TotalTxns:{row[0]}, AvgAmt:INR {row[1]:.2f}")
        return result

class DeviceRiskTool:
    """Deterministic tool to check device fingerprint, proxy/VPN indicators, and location risk."""
    name = "DeviceRiskTool"
    description = "Checks device type, VPN/proxy flags, and device reuse count across accounts."

    @staticmethod
    def run(device_id: str, ip_address: str) -> dict:
        logger.info(f"[DeviceRiskTool] Invoked for Device: {device_id}, IP: {ip_address}")
        conn = get_raw_sqlite_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT device_type, os, is_vpn_proxy FROM devices WHERE device_id = ?", (device_id,))
        dev_row = cursor.fetchone()

        cursor.execute("SELECT country, city, isp, is_suspicious_proxy FROM ip_addresses WHERE ip_address = ?", (ip_address,))
        ip_row = cursor.fetchone()

        # Count accounts on this device
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM transactions WHERE device_id = ?", (device_id,))
        dev_account_count = cursor.fetchone()[0]

        conn.close()

        result = {
            "device_id": device_id,
            "device_type": dev_row[0] if dev_row else "Unknown",
            "os": dev_row[1] if dev_row else "Unknown",
            "is_vpn_proxy": bool(dev_row[2]) if dev_row else False,
            "ip_address": ip_address,
            "country": ip_row[0] if ip_row else "Unknown",
            "city": ip_row[1] if ip_row else "Unknown",
            "isp": ip_row[2] if ip_row else "Unknown",
            "is_suspicious_proxy": bool(ip_row[3]) if ip_row else False,
            "device_account_reuse_count": dev_account_count,
            "high_risk_device": dev_account_count >= 3 or (dev_row and dev_row[2])
        }
        logger.info(f"[DeviceRiskTool] Result -> DevAccounts:{dev_account_count}, Proxy:{result['is_suspicious_proxy']}")
        return result

class FraudModelTool:
    """Deterministic tool to retrieve model probabilities and top risk factors."""
    name = "FraudModelTool"
    description = "Returns tabular ML model fraud probability, GNN score, and top risk factors."

    @staticmethod
    def run(txn_payload: dict, velocity_1h: int) -> dict:
        """velocity_1h is the server-computed value from risk_summary
        (ml/risk_aggregator.py), passed in by the caller — txn_payload no
        longer carries a client-suppliable velocity field at all."""
        from ml.risk_aggregator import live_tabular_score, live_gnn_score_and_evidence

        user_id = txn_payload.get("user_id", "")
        tab_score, _amount_zscore = live_tabular_score(txn_payload, velocity_1h)
        gnn_score, _evidence = live_gnn_score_and_evidence(user_id)

        top_reasons = []
        if txn_payload.get("amount", 0) > 50000:
            top_reasons.append("High Transaction Amount (> ₹50,000)")
        if velocity_1h >= 5:
            top_reasons.append(f"Elevated Hourly Velocity ({velocity_1h} txns/hr)")
        if gnn_score > 0.5:
            top_reasons.append("GNN Risk Embedding triggered by suspicious graph neighborhood")
        if txn_payload.get("is_vpn_proxy", False) or txn_payload.get("is_suspicious_proxy", False):
            top_reasons.append("Connection originated from TOR/Proxy/VPN exit node")

        return {
            "tabular_fraud_probability": round(tab_score, 4),
            "gnn_fraud_probability": round(gnn_score, 4),
            "primary_risk_factors": top_reasons
        }
