import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response

from config import (
    BASE_DIR,
    IS_SERVERLESS,
    SQLITE_DB_PATH,
    DATABASE_BACKEND,
    ALLOWED_ORIGINS,
    API_KEY,
)

from db.database import init_db, get_raw_sqlite_connection
from ml.graph_builder import graph_builder
from utils.logger import get_logger
from infra.redis_client import redis_health, close_redis
from infra.observability import (
    setup_tracing,
    instrument_fastapi,
    metrics_response,
    HTTP_REQUESTS,
    HTTP_LATENCY,
    HTTP_IN_FLIGHT,
)

from api.routes_transactions import router as transactions_router
from api.routes_graph import router as graph_router
from api.routes_agent import router as agent_router
from api.routes_logs import router as logs_router
from api.routes_admin import router as admin_router
from api.routes_hitl import router as hitl_router


logger = get_logger("api_main")


# ---------------------------------------------------------------------------
# Observability / tracing
# ---------------------------------------------------------------------------

setup_tracing()


# ---------------------------------------------------------------------------
# Startup readiness state
# ---------------------------------------------------------------------------
#
# The heavy startup work below (DB connect/init, dataset seeding, in-memory
# graph build) used to run inline inside lifespan(), which FastAPI/uvicorn
# run to completion BEFORE the server binds its port and starts accepting
# connections. If any step blocked — e.g. an unreachable DATABASE_URL — the
# whole process never finished starting, so nothing (not even /health)
# responded, and PaaS platforms killed it as "still starting" with no
# useful signal.
#
# lifespan() now fires this work off as a background task and returns
# immediately, so the port opens right away. Routes that need the DB/graph
# check _startup_ready (via require_ready below) and return a clean 503
# instead of a confusing raw exception while warm-up is still running.

_startup_ready = asyncio.Event()
_startup_error: Optional[str] = None


async def _run_startup_work() -> None:
    """The real startup sequence — now runs in the background, off the
    critical path of the port coming up."""

    global _startup_error

    try:
        # ------------------------------------------------------------
        # 1. Initialize database
        # ------------------------------------------------------------

        init_db()

        # ------------------------------------------------------------
        # 2. Seed database if empty
        # ------------------------------------------------------------

        conn = get_raw_sqlite_connection()

        try:
            txn_count = conn.execute(
                "SELECT COUNT(*) FROM transactions"
            ).fetchone()[0]
        finally:
            conn.close()

        if txn_count == 0:
            logger.info(
                f"Empty {DATABASE_BACKEND} database detected "
                "— seeding RazorRisk dataset..."
            )

            from data.generate_synthetic_data import generate_dataset

            generate_dataset(
                num_users=1500,
                num_transactions=12000,
                seed=42,
            )

        # ------------------------------------------------------------
        # 3. Build in-memory transaction graph
        # ------------------------------------------------------------

        logger.info(
            "Building in-memory transaction graph "
            "from current database state..."
        )

        graph_builder.build_graph()

        # ------------------------------------------------------------
        # 4. Detect graph communities
        # ------------------------------------------------------------

        graph_builder.detect_communities()

        logger.info(
            f"Graph ready: "
            f"{graph_builder.G.number_of_nodes()} nodes, "
            f"{graph_builder.G.number_of_edges()} edges."
        )

        _startup_ready.set()

        logger.info(
            "RazorRisk background startup complete — "
            "DB-backed endpoints are now live."
        )

    except Exception as exc:
        # Deliberately not re-raised: the port stays open and /health
        # reports the failure with a real message instead of the process
        # hanging silently (the old failure mode) or dying with logs that
        # never got flushed.
        _startup_error = str(exc)
        logger.error(
            "RazorRisk background startup failed: %s",
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Startup:
        Kicks off DB init / seeding / graph build as a background task
        and returns immediately — the port opens without waiting on any
        of it. See _run_startup_work() and _startup_ready above.

    Shutdown:
        1. Cancel any still-running background startup work
        2. Close Redis connection/resources
    """

    logger.info("Starting RazorRisk FastAPI Backend Engine...")

    startup_task = asyncio.create_task(_run_startup_work())

    yield

    # ============================================================
    # SHUTDOWN
    # ============================================================

    logger.info("Shutting down RazorRisk FastAPI Backend Engine...")

    startup_task.cancel()

    await close_redis()

    logger.info("RazorRisk shutdown complete.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RazorRisk — Agentic AI Payment Fraud & Risk Investigation Platform",
    description=(
        "Graph Neural Network + Tabular ML + "
        "LangGraph Agentic Investigation Engine"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Instrument FastAPI
# ---------------------------------------------------------------------------

instrument_fastapi(app)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

# Defaults to "*" for zero-config local development and Vercel
# static-site deployment.
#
# Production deployments should set ALLOWED_ORIGINS to the actual
# frontend origin(s).

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Observability middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """
    Record RED metrics for every request without changing
    request semantics.
    """

    started = __import__("time").perf_counter()
    route = request.url.path

    if HTTP_IN_FLIGHT is not None:
        HTTP_IN_FLIGHT.inc()

    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response

    except Exception:
        status = "500"
        raise

    finally:
        elapsed = __import__("time").perf_counter() - started

        # Use FastAPI's route template when available to avoid
        # high-cardinality labels such as:
        #
        # /api/v1/investigations/jobs/JOB_<random-id>

        route_obj = request.scope.get("route")

        metric_route = getattr(
            route_obj,
            "path",
            route,
        ) or route

        if HTTP_REQUESTS is not None:
            HTTP_REQUESTS.labels(
                request.method,
                metric_route,
                status,
            ).inc()

        if HTTP_LATENCY is not None:
            HTTP_LATENCY.labels(
                request.method,
                metric_route,
            ).observe(elapsed)

        if HTTP_IN_FLIGHT is not None:
            HTTP_IN_FLIGHT.dec()


# ---------------------------------------------------------------------------
# API key middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def api_key_gate(request: Request, call_next):
    """
    Optional API-key gate for /api/v1/*.

    If API_KEY is configured:
        Every /api/v1/* request must provide:

            X-API-Key: <API_KEY>

    /health and /dashboard are intentionally not gated.
    """

    if API_KEY and request.url.path.startswith("/api/v1/"):

        if request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Invalid or missing X-API-Key header."
                    )
                },
            )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Readiness gate
# ---------------------------------------------------------------------------

@app.middleware("http")
async def readiness_gate(request: Request, call_next):
    """
    DB/graph-backed routes (/api/v1/*) return a clean 503 while the
    background startup work (see _run_startup_work) is still running,
    instead of a raw exception hitting an uninitialized table/graph.

    /health, /metrics, /dashboard, and / are intentionally exempt so they
    stay usable — and truthful about current status — during warm-up.
    """

    if request.url.path.startswith("/api/v1/") and not _startup_ready.is_set():
        if _startup_error:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "RazorRisk failed to start up.",
                    "error": _startup_error,
                },
            )

        return JSONResponse(
            status_code=503,
            headers={"Retry-After": "5"},
            content={
                "detail": (
                    "RazorRisk is still warming up "
                    "(database init / dataset seeding / graph build "
                    "in progress). Check /health, retry shortly."
                )
            },
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(transactions_router)
app.include_router(graph_router)
app.include_router(agent_router)
app.include_router(logs_router)
app.include_router(admin_router)
app.include_router(hitl_router)


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)

app.mount(
    "/dashboard",
    StaticFiles(
        directory=str(static_dir),
        html=True,
    ),
    name="static",
)


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Redirect the bare domain to the RazorRisk dashboard.
    """

    return RedirectResponse(
        url="/dashboard/"
    )


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

@app.get("/metrics", include_in_schema=False)
def metrics():
    """
    Prometheus scrape endpoint.

    Intentionally outside the API-key gate.
    """

    payload, content_type = metrics_response()

    return Response(
        content=payload,
        media_type=content_type,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    redis_ok = await redis_health()

    if _startup_error:
        startup_state = "ERROR"
    elif _startup_ready.is_set():
        startup_state = "READY"
    else:
        startup_state = "STARTING"

    overall = "HEALTHY" if (redis_ok and startup_state == "READY") else (
        "ERROR" if startup_state == "ERROR" else "DEGRADED"
    )

    payload = {
        "status": overall,
        "system": "RazorRisk AI Engine",
        "version": "1.0.0",
        "dependencies": {
            "redis": "UP" if redis_ok else "DOWN",
            "database": DATABASE_BACKEND.upper(),
        },
        # DB init / dataset seeding / graph build run in the background
        # (see _run_startup_work) so the port and this endpoint respond
        # immediately even before they finish. "STARTING" here is expected
        # and normal for the first several seconds/minutes after boot.
        "startup": startup_state,
        "distributed_mode": (
            redis_ok
            and DATABASE_BACKEND == "postgresql"
        ),
    }

    if _startup_error:
        payload["startup_error"] = _startup_error

    return payload


# ---------------------------------------------------------------------------
# System statistics
# ---------------------------------------------------------------------------

@app.get("/api/v1/stats")
def get_system_stats():
    conn = get_raw_sqlite_connection()

    try:
        cursor = conn.cursor()

        cursor.execute(
            "SELECT COUNT(*) FROM transactions"
        )
        total_txns = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM users"
        )
        total_users = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM devices"
        )
        total_devices = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM risk_scores
            WHERE risk_tier IN ('HIGH', 'CRITICAL')
            """
        )
        high_risk_txns = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM investigation_reports"
        )
        investigations_count = cursor.fetchone()[0]

    finally:
        conn.close()

    return {
        "total_transactions": total_txns,
        "total_users": total_users,
        "total_devices": total_devices,
        "high_risk_transactions": high_risk_txns,
        "investigations_conducted": investigations_count,
    }