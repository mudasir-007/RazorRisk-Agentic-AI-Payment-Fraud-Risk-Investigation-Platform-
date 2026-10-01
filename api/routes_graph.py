from fastapi import APIRouter
from ml.graph_builder import graph_builder
from db.database import get_raw_sqlite_connection
from utils.logger import get_logger

logger = get_logger("api_graph")

router = APIRouter(prefix="/api/v1/graph", tags=["Graph & Topology Engine"])

@router.get("/topology/{user_id}")
def get_user_graph_topology(user_id: str, depth: int = 2):
    """
    Returns a bounded, fraud-signal-focused neighborhood (Users, Devices, IPs,
    Merchants) formatted for frontend Vis.js visualizer rendering.
    """
    # Reuse the already-built shared graph (kept current by API startup,
    # live-transaction incremental updates, and the admin pipeline) instead
    # of rebuilding the entire multi-thousand-node graph from the database
    # on every single tab click. A per-request full rebuild here was both
    # slow and, under concurrent requests, could race with any other reader
    # of graph_builder.G.
    if graph_builder.G.number_of_nodes() == 0:
        graph_builder.build_graph()

    G_snapshot = graph_builder.G
    u_node = f"User:{user_id}"

    if not G_snapshot.has_node(u_node):
        return {"user_id": user_id, "nodes": [], "edges": []}

    # A plain BFS here used to walk through every edge type equally,
    # including TRANSACTS_WITH — so any user who happened to shop at the same
    # popular merchant, or share a coincidentally-common IP, got pulled into
    # a "2-hop neighborhood" that ballooned to hundreds of unrelated nodes
    # and buried the actual fraud ring in noise. Shared DEVICE/IP is the real
    # fraud signal (it's what risk_aggregator/extract_user_graph_features
    # actually score); merchant fan-out is not, so the traversal below only
    # expands a second hop through Device/IP nodes, and caps how many
    # co-users any single shared entity can contribute.
    MAX_FANOUT_PER_ENTITY = 12
    MAX_TOTAL_NODES = 80

    visited_nodes = {u_node}
    hop1_entities = list(G_snapshot.neighbors(u_node))
    visited_nodes.update(hop1_entities)

    truncated = False
    for entity in hop1_entities:
        if G_snapshot.nodes[entity].get("type") not in ("Device", "IP"):
            continue  # skip Merchant fan-out — not a fraud signal, just noise
        co_users = [n for n in G_snapshot.neighbors(entity) if G_snapshot.nodes[n].get("type") == "User"]
        if len(co_users) > MAX_FANOUT_PER_ENTITY:
            truncated = True
            co_users = co_users[:MAX_FANOUT_PER_ENTITY]
        visited_nodes.update(co_users)
        if len(visited_nodes) >= MAX_TOTAL_NODES:
            truncated = True
            break

    if len(visited_nodes) > MAX_TOTAL_NODES:
        truncated = True
        capped = set(list(visited_nodes)[:MAX_TOTAL_NODES])
        capped.add(u_node)
        visited_nodes = capped

    subgraph = G_snapshot.subgraph(visited_nodes)

    vis_nodes = []
    vis_edges = []

    # Calm, type-coded palette — color encodes *entity type* by default.
    # Red is reserved exclusively for nodes with a confirmed fraud label or
    # membership in a high-fraud-ratio community; it is not the default
    # color for "User" nodes just because they showed up in this subgraph.
    color_map = {
        "User": "#5B8DEF",      # signal blue
        "Device": "#9B7EDE",    # muted violet
        "IP": "#E0A63C",        # amber
        "Merchant": "#3FBF8F"   # teal-green
    }
    FRAUD_COLOR = "#EF4A63"
    QUERIED_BORDER = "#F8FAFC"

    shape_map = {
        "User": "dot",
        "Device": "square",
        "IP": "diamond",
        "Merchant": "triangle"
    }

    for node, data in subgraph.nodes(data=True):
        node_type = data.get("type", "User")
        label = node.split(":")[-1]

        is_queried_user = node == u_node
        # A user is "fraud-flagged" only on real signal: a confirmed ground-truth
        # label on this node, or membership in a community the graph has scored
        # as high-fraud-ratio — never a naive string match on the label text.
        comm_info = graph_builder.user_communities.get(label, {})
        is_fraud_flagged = bool(data.get("is_fraud")) or comm_info.get("community_fraud_ratio", 0) > 0.3

        node_color = FRAUD_COLOR if (node_type == "User" and is_fraud_flagged) else color_map.get(node_type, "#94a3b8")

        vis_nodes.append({
            "id": node,
            "label": label,
            "group": node_type,
            "color": {
                "background": node_color,
                "border": QUERIED_BORDER if is_queried_user else node_color,
                "highlight": {"background": node_color, "border": QUERIED_BORDER}
            },
            "borderWidth": 3 if is_queried_user else 1,
            "shape": shape_map.get(node_type, "dot"),
            "size": 26 if is_queried_user else (18 if is_fraud_flagged else 14),
            "font": {"color": "#E2E8F0", "size": 13, "vadjust": -18 if not is_queried_user else -22},
            "title": f"{node_type}: {label}" + (" — queried user" if is_queried_user else "") + (" — fraud-flagged" if is_fraud_flagged else "")
        })

    # Edge labels are relation types (USES_DEVICE / USES_IP / TRANSACTS_WITH)
    # repeated on every one of what can be dozens of edges in a fraud-ring
    # subgraph — rendering all of them permanently is what produced the
    # illegible label soup. They live in `title` (hover tooltip) instead;
    # the node shape (square/diamond/triangle) already encodes the relation.
    for u, v, data in subgraph.edges(data=True):
        vis_edges.append({
            "from": u,
            "to": v,
            "title": data.get("relation", ""),
            "color": {"color": "#37415199", "highlight": "#64748B"}
        })

    return {
        "user_id": user_id,
        "node_count": len(vis_nodes),
        "edge_count": len(vis_edges),
        "truncated": truncated,
        "nodes": vis_nodes,
        "edges": vis_edges
    }

@router.get("/communities")
def get_graph_communities():
    """Returns detected fraud communities and high-density clusters."""
    if graph_builder.G.number_of_nodes() == 0:
        graph_builder.build_graph()
    if not graph_builder.user_communities:
        graph_builder.detect_communities()
    comm_dict = graph_builder.user_communities

    # Aggregate by community ID
    clusters = {}
    for uid, data in comm_dict.items():
        cid = data["community_id"]
        if cid not in clusters:
            clusters[cid] = {
                "community_id": cid,
                "size": data["community_size"],
                "fraud_ratio": data["community_fraud_ratio"],
                "members": []
            }
        if len(clusters[cid]["members"]) < 10:
            clusters[cid]["members"].append(uid)

    sorted_clusters = sorted(clusters.values(), key=lambda x: (x["fraud_ratio"], x["size"]), reverse=True)
    return {"total_communities": len(clusters), "clusters": sorted_clusters[:15]}
