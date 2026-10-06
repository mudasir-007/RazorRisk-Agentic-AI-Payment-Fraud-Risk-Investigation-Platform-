import threading
import networkx as nx
import pandas as pd
import numpy as np
from db.database import get_raw_sqlite_connection, read_sql_query
from utils.logger import get_logger

logger = get_logger("graph_builder")

class TransactionGraphBuilder:
    def __init__(self):
        self.G = nx.Graph()
        self.user_communities = {}
        self.node_features = {}
        # Rebuilds happen on a background thread pool (FastAPI sync routes).
        # This lock only serializes *writers* (build_graph/detect_communities/
        # incremental updates) against each other — readers always see either
        # the fully-old or fully-new graph, never a half-built one, because
        # build_graph() constructs into a local object and swaps `self.G`
        # with a single atomic reference assignment at the end.
        self._lock = threading.Lock()

    def build_graph(self):
        """Construct heterogeneous network graph from database transactions.
        Builds into a fresh local graph and atomically swaps it in, so any
        concurrent request reading self.G never sees a partially-cleared
        graph (the previous in-place self.G.clear() + repopulate pattern
        could race with readers running on other thread-pool threads)."""
        logger.info("Extracting graph topology from transactions...")
        conn = get_raw_sqlite_connection()

        # Query recent 30-day transactions and entity mappings
        query = """
            SELECT t.transaction_id, t.user_id, t.device_id, t.ip_address, t.merchant_id, t.amount, t.is_fraud_ground_truth
            FROM transactions t
        """
        df = read_sql_query(query)
        conn.close()

        new_G = nx.Graph()

        # Build nodes and edges
        for _, row in df.iterrows():
            u_node = f"User:{row['user_id']}"
            d_node = f"Device:{row['device_id']}"
            ip_node = f"IP:{row['ip_address']}"
            m_node = f"Merchant:{row['merchant_id']}"

            # Add node types and attributes
            new_G.add_node(u_node, type="User", is_fraud=row['is_fraud_ground_truth'])
            new_G.add_node(d_node, type="Device")
            new_G.add_node(ip_node, type="IP")
            new_G.add_node(m_node, type="Merchant")

            # Add edges with weights
            new_G.add_edge(u_node, d_node, relation="USES_DEVICE", weight=2.0)
            new_G.add_edge(u_node, ip_node, relation="USES_IP", weight=1.0)
            new_G.add_edge(u_node, m_node, relation="TRANSACTS_WITH", weight=0.5, amount=row['amount'])

        with self._lock:
            self.G = new_G

        logger.info(f"Graph constructed successfully with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges.")
        return self.G

    def add_transaction(self, user_id: str, device_id: str, ip_address: str, merchant_id: str,
                         amount: float = 0.0, is_fraud: bool = False):
        """Incrementally fold one new live transaction into the current graph
        without a full rebuild. Used after a live /transactions/score call so
        the graph stays current between full rebuilds (which only need to
        happen after a data-pipeline reseed/ingest)."""
        u_node = f"User:{user_id}"
        d_node = f"Device:{device_id}"
        ip_node = f"IP:{ip_address}"
        m_node = f"Merchant:{merchant_id}"

        with self._lock:
            self.G.add_node(u_node, type="User", is_fraud=is_fraud)
            self.G.add_node(d_node, type="Device")
            self.G.add_node(ip_node, type="IP")
            self.G.add_node(m_node, type="Merchant")
            self.G.add_edge(u_node, d_node, relation="USES_DEVICE", weight=2.0)
            self.G.add_edge(u_node, ip_node, relation="USES_IP", weight=1.0)
            self.G.add_edge(u_node, m_node, relation="TRANSACTS_WITH", weight=0.5, amount=amount)

    def detect_communities(self):
        """Run modularity community detection to identify fraud rings.
        Builds into a local dict and swaps it in atomically for the same
        thread-safety reason as build_graph()."""
        if self.G.number_of_nodes() == 0:
            self.build_graph()

        logger.info("Running Louvain modularity community detection...")
        G_snapshot = self.G  # single reference read; stable for the rest of this call

        # Project onto User-User co-existence graph
        user_nodes = [n for n, d in G_snapshot.nodes(data=True) if d.get('type') == 'User']

        # Build user projection graph based on shared devices & IPs
        user_graph = nx.Graph()
        for u in user_nodes:
            user_graph.add_node(u)

        for n, data in G_snapshot.nodes(data=True):
            if data.get('type') in ['Device', 'IP']:
                neighbors = [nbr for nbr in G_snapshot.neighbors(n) if G_snapshot.nodes[nbr].get('type') == 'User']
                if len(neighbors) > 1:
                    # Shared entity! Create weighted user-user edges
                    weight = 3.0 if data.get('type') == 'Device' else 1.5
                    for i in range(len(neighbors)):
                        for j in range(i+1, len(neighbors)):
                            u1, u2 = neighbors[i], neighbors[j]
                            if user_graph.has_edge(u1, u2):
                                user_graph[u1][u2]['weight'] += weight
                            else:
                                user_graph.add_edge(u1, u2, weight=weight)

        # Detect communities using Louvain / Greedy Modularity
        communities = list(nx.community.greedy_modularity_communities(user_graph))

        new_communities = {}
        community_stats = []

        for comm_id, comm_members in enumerate(communities):
            members_list = list(comm_members)
            comm_size = len(members_list)

            # Count shared devices/IPs in community
            fraud_count = sum(1 for m in members_list if G_snapshot.nodes[m].get('is_fraud', False))
            fraud_ratio = fraud_count / comm_size if comm_size > 0 else 0.0

            for m in members_list:
                user_id = m.replace("User:", "")
                new_communities[user_id] = {
                    "community_id": comm_id,
                    "community_size": comm_size,
                    "community_fraud_ratio": round(fraud_ratio, 4)
                }

            if comm_size > 3 or fraud_count > 0:
                community_stats.append({
                    "comm_id": comm_id,
                    "size": comm_size,
                    "fraud_members": fraud_count,
                    "fraud_ratio": round(fraud_ratio, 2)
                })

        with self._lock:
            self.user_communities = new_communities

        logger.info(f"Community detection completed: Found {len(communities)} user clusters. {len(community_stats)} high-density risk clusters identified.")
        return self.user_communities

    def extract_user_graph_features(self, user_id: str) -> dict:
        """Extract graph topology metrics for a specific user ID."""
        G_snapshot = self.G  # single reference read — immune to a concurrent build_graph() swap mid-call
        u_node = f"User:{user_id}"
        if not G_snapshot.has_node(u_node):
            return {
                "graph_degree": 0,
                "shared_device_accounts": 1,
                "shared_ip_accounts": 1,
                "community_id": -1,
                "community_size": 1,
                "community_fraud_ratio": 0.0,
                "risk_topology_score": 0.0
            }

        neighbors = list(G_snapshot.neighbors(u_node))
        graph_degree = len(neighbors)

        shared_device_accounts = 1
        shared_ip_accounts = 1

        for nbr in neighbors:
            nbr_type = G_snapshot.nodes[nbr].get("type")
            connected_users = [u for u in G_snapshot.neighbors(nbr) if G_snapshot.nodes[u].get("type") == "User"]
            if nbr_type == "Device":
                shared_device_accounts = max(shared_device_accounts, len(connected_users))
            elif nbr_type == "IP":
                shared_ip_accounts = max(shared_ip_accounts, len(connected_users))

        comm_info = self.user_communities.get(user_id, {
            "community_id": -1,
            "community_size": 1,
            "community_fraud_ratio": 0.0
        })

        # Calculate heuristic risk score based on network topology
        topology_risk = 0.0
        if shared_device_accounts > 2:
            topology_risk += min(0.4 + (shared_device_accounts * 0.1), 0.75)
        if shared_ip_accounts > 4:
            topology_risk += min(0.3 + (shared_ip_accounts * 0.05), 0.5)
        if comm_info["community_size"] > 4 and comm_info["community_fraud_ratio"] > 0.3:
            topology_risk += 0.3

        topology_risk = min(topology_risk, 0.99)

        return {
            "graph_degree": graph_degree,
            "shared_device_accounts": shared_device_accounts,
            "shared_ip_accounts": shared_ip_accounts,
            "community_id": comm_info["community_id"],
            "community_size": comm_info["community_size"],
            "community_fraud_ratio": comm_info["community_fraud_ratio"],
            "risk_topology_score": round(topology_risk, 4)
        }

graph_builder = TransactionGraphBuilder()

if __name__ == "__main__":
    graph_builder.build_graph()
    graph_builder.detect_communities()
    print("Sample User Graph Features:", graph_builder.extract_user_graph_features("USER_RING1_1"))
