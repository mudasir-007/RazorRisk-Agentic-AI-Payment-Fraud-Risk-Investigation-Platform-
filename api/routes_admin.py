import threading
from fastapi import APIRouter, HTTPException
from data.generate_synthetic_data import generate_dataset
from ml.graph_builder import graph_builder
from ml.risk_aggregator import train_stacker, _LiveModels
from utils.logger import get_logger

logger = get_logger("api_admin")

router = APIRouter(prefix="/api/v1/admin", tags=["Admin & Data Pipeline"])

# Serializes full data-pipeline runs (reseed/ingest + retrain). These touch
# the whole SQLite DB (bulk delete + insert) and every ML model file, so two
# runs firing at once — e.g. a double-clicked button — would corrupt each
# other rather than just race on the in-memory graph.
_pipeline_lock = threading.Lock()


def _rebuild_graph_and_retrain():
    """Shared post-ingestion step: rebuild the dashboard's visualization
    graph, then run the full tabular -> GNN -> stacker training sequence
    (ml.risk_aggregator.train_stacker) on whatever data now sits in the
    database, and drop the live-scoring process's cached model weights so
    the very next scored transaction picks up the freshly trained ones
    instead of stale in-memory copies."""
    graph_builder.build_graph()
    graph_builder.detect_communities()
    eval_metrics = train_stacker()
    _LiveModels.reset()
    return eval_metrics


@router.post("/pipeline/synthetic")
def run_synthetic_pipeline(num_users: int = 1500, num_transactions: int = 12000):
    """
    Regenerates the synthetic fraud-ring dataset (4 injected fraud scenarios)
    and retrains the tabular ML model + GraphSAGE GNN on it. Safe to call
    repeatedly — it replaces existing transaction data.
    """
    try:
        logger.info("Admin: running SYNTHETIC data pipeline...")
        if not _pipeline_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A data pipeline run is already in progress — please wait for it to finish.")
        try:
            count = generate_dataset(num_users=num_users, num_transactions=num_transactions)
            eval_metrics = _rebuild_graph_and_retrain()
        finally:
            _pipeline_lock.release()
        return {"status": "OK", "mode": "synthetic", "transactions_generated": count, "eval_metrics": eval_metrics}
    except HTTPException:
        raise
    except Exception as e:
        # Full traceback goes to the log; the client gets a short, stable
        # message rather than a raw exception string (which can contain
        # internal identifiers and reads as an alarming stack trace in the UI).
        logger.error(f"Synthetic pipeline failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Synthetic data pipeline failed. Check logs/app.log for details.")


@router.post("/rebuild-graph")
def rebuild_graph():
    """Rebuilds the in-memory entity graph + community detection from
    whatever is currently in the database, without touching the ML models.
    Useful after live transactions have added new nodes/edges."""
    graph_builder.build_graph()
    communities = graph_builder.detect_communities()
    return {
        "status": "OK",
        "nodes": graph_builder.G.number_of_nodes(),
        "edges": graph_builder.G.number_of_edges(),
        "users_with_community": len(communities),
    }
