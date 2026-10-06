"""
RazorRisk — canonical User-User risk graph.

Builds an undirected weighted graph over USERS ONLY:
    edge(u1, u2, weight) if u1 and u2 share a device and/or an IP
    weight = (DEVICE_SHARE_WEIGHT if shared device) + (IP_SHARE_WEIGHT if shared ip)

Device-sharing is weighted higher than IP-sharing because a shared IP can be
entirely benign (same wifi network, office, cafe), while a shared device
fingerprint is much stronger fraud-ring evidence. This weighting is a
modeling choice, not a fact — flagged here so it can be defended in review.

This is deliberately separate from ml/graph_builder.py, which builds a
richer User-Device-IP-Merchant multi-type graph for the dashboard's visual
topology explorer. The two used to be the same graph, which caused a real
bug: 2-hop traversal through Merchant nodes (a popular merchant almost every
user transacts with) made a 7-person fraud ring balloon into a 692-node
unreadable subgraph. The fix is structural, not just a traversal cap on the
visualization side: the graph that actually drives GNN training, community
detection, and risk scoring should never have included Merchant nodes (or
raw device/IP fan-out) as hops in the first place — a merchant used by
thousands of unrelated users isn't fraud-ring evidence, so it was never a
real graph *feature*.

Runs Louvain community detection on this graph; community_id/community_size
per user becomes a GNN node feature.
"""
import numpy as np
import networkx as nx

from utils.logger import get_logger

logger = get_logger("risk_graph")

DEVICE_SHARE_WEIGHT = 2
IP_SHARE_WEIGHT = 1


def _fetch_shared_groups(cur, group_col):
    """Returns {shared_value: [user_id, ...]} for groups of size > 1, derived
    straight from the transactions table (no separate join tables needed —
    a user "uses" a device/IP if they have a transaction with it)."""
    cur.execute(f"""
        SELECT {group_col}, user_id
        FROM transactions
        GROUP BY {group_col}, user_id
    """)
    groups = {}
    for key, user_id in cur.fetchall():
        groups.setdefault(key, set()).add(user_id)
    return {k: sorted(v) for k, v in groups.items() if len(v) > 1}


def build_user_graph(conn) -> nx.Graph:
    """Builds the canonical User-only weighted risk graph from the current
    database contents. Cheap enough (pure SQL group-by + in-memory edge
    accumulation) to call on every training run and every live scoring
    request without a background job — no torch/GPU involved anywhere in
    this module."""
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users")
    all_users = [r[0] for r in cur.fetchall()]

    G = nx.Graph()
    G.add_nodes_from(all_users)

    edge_weight = {}

    for device_id, users in _fetch_shared_groups(cur, "device_id").items():
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                key = (users[i], users[j])
                edge_weight[key] = edge_weight.get(key, 0) + DEVICE_SHARE_WEIGHT

    for ip_addr, users in _fetch_shared_groups(cur, "ip_address").items():
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                key = (users[i], users[j])
                edge_weight[key] = edge_weight.get(key, 0) + IP_SHARE_WEIGHT

    for (u1, u2), w in edge_weight.items():
        G.add_edge(u1, u2, weight=w)

    logger.info(f"User risk graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    return G


def detect_communities(G: nx.Graph):
    """Louvain community detection. Returns (communities_list, community_size_by_user)."""
    if G.number_of_edges() == 0:
        return [], {}
    communities = list(nx.community.louvain_communities(G, weight="weight", seed=42))
    community_size = {}
    for c in communities:
        for u in c:
            community_size[u] = len(c)
    return communities, community_size


def fetch_node_features(conn, G: nx.Graph, community_size: dict):
    """Per-user node features for the GNN: log1p(txn_count), log1p(avg_amount),
    log1p(std_amount), log1p(max_hourly_txns), log1p(graph_degree),
    log1p(community_size). Returns (user_ids, X) with rows aligned to
    G's node order for the given user_ids list."""
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users ORDER BY user_id")
    user_ids = [r[0] for r in cur.fetchall()]

    # Fetch transaction amounts per user and calculate standard deviation
    # in Python because SQLite may not provide the SQRT() math function.
    cur.execute("""
        SELECT user_id, amount
        FROM transactions
        ORDER BY user_id
    """)

    amounts_by_user = {}
    for user_id, amount in cur.fetchall():
        amounts_by_user.setdefault(user_id, []).append(amount)

    txn_stats = {}
    for user_id, amounts in amounts_by_user.items():
        amounts = np.asarray(amounts, dtype=np.float64)
        txn_count = len(amounts)
        avg_amount = float(np.mean(amounts))

        # Sample standard deviation, matching:
        # SQRT(SUM((x - mean)^2) / (n - 1))
        if txn_count > 1:
            std_amount = float(np.std(amounts, ddof=1))
        else:
            std_amount = 0.0

        txn_stats[user_id] = (
            txn_count,
            avg_amount,
            std_amount,
        )

    # Max transactions observed in any 1-hour bucket (hour-truncated timestamp)
    cur.execute("""
        SELECT user_id, MAX(cnt) FROM (
            SELECT user_id, strftime('%Y-%m-%d %H', timestamp) AS hb, COUNT(*) AS cnt
            FROM transactions GROUP BY user_id, hb
        ) AS hourly_counts
        GROUP BY user_id
    """)
    max_hourly = {r[0]: r[1] for r in cur.fetchall()}

    features = []
    for uid in user_ids:
        txn_count, avg_amount, std_amount = txn_stats.get(
            uid, (0, 0.0, 0.0)
        )
        degree = G.degree(uid) if uid in G else 0
        comm_size = community_size.get(uid, 1)

        features.append([
            np.log1p(txn_count),
            np.log1p(max(avg_amount, 0)),
            np.log1p(max(std_amount, 0)),
            np.log1p(max_hourly.get(uid, 0)),
            np.log1p(degree),
            np.log1p(comm_size),
        ])

    X = np.array(features, dtype=np.float64)
    return user_ids, X


def build_adjacency(G: nx.Graph, user_ids: list) -> np.ndarray:
    """Row-normalized weighted adjacency matrix (mean aggregator for GraphSAGE).
    Isolated nodes (no shared device/IP with anyone) get an explicit zero row —
    their GNN embedding then depends only on their own features via the
    self-concatenation term, not a division-by-zero neighbor average."""
    W = nx.to_numpy_array(G, nodelist=user_ids, weight="weight")
    deg = W.sum(axis=1, keepdims=True)
    deg_safe = np.where(deg == 0, 1, deg)
    A_mean = W / deg_safe
    A_mean[(deg.flatten() == 0)] = 0
    return A_mean
