import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Vercel's Python runtime, Hugging Face Spaces' Docker runtime, and
# Antideploy (built on Google Cloud Run) all deploy onto a filesystem that's
# either read-only outside /tmp or gets wiped on every cold start/redeploy.
# Vercel sets VERCEL=1, HF Spaces sets SPACE_ID, and Cloud Run itself sets
# K_SERVICE on every service unconditionally (it's how Cloud Run identifies
# itself, documented and unlikely to change) — any one of these flips this
# flag, redirecting the SQLite DB and log files to /tmp instead of losing
# data unpredictably or crashing on the first write. Render (and local dev)
# use a real writable, persistent process, so BASE_DIR is fine there.
IS_RESTRICTED_FS = bool(os.getenv("VERCEL") or os.getenv("SPACE_ID") or os.getenv("K_SERVICE"))
IS_SERVERLESS = IS_RESTRICTED_FS  # kept as an alias — existing call sites (api/main.py) use this name
RUNTIME_DIR = Path("/tmp") if IS_RESTRICTED_FS else BASE_DIR

# Settings
SQLITE_DB_PATH = RUNTIME_DIR / "razor_risk.db"
# PostgreSQL is the production/default data plane. Set DATABASE_URL to a
# managed Postgres/Supabase connection string in production. SQLite is an
# explicit fallback for tests and zero-infrastructure local work only.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgrespassword@localhost:5432/razor_risk")
DATABASE_BACKEND = "postgresql" if DATABASE_URL.startswith(("postgresql://", "postgres://", "postgresql+")) else "sqlite"
DB_POOL_MIN = int(os.getenv("DB_POOL_MIN", "2"))
DB_POOL_MAX = int(os.getenv("DB_POOL_MAX", "10"))
# libpq's own connect_timeout has no default, so an unreachable/firewalled
# Postgres host (wrong DATABASE_URL, DB paused, etc.) hangs psycopg.connect()
# on the TCP handshake indefinitely instead of raising. Since init_db() runs
# inside the FastAPI lifespan — before the port is ever bound — that
# hang is what stalls the whole boot on PaaS platforms. 5s is generous for
# any real network path; a bad host now fails loudly in seconds instead of
# silently for minutes.
DB_CONNECT_TIMEOUT_SECONDS = int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "5"))
LOG_DIR = RUNTIME_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

APP_LOG_PATH = LOG_DIR / "app.log"
RISK_ENGINE_LOG_PATH = LOG_DIR / "risk_engine.log"
AGENT_LOG_PATH = LOG_DIR / "agent_investigations.log"
ML_TRAINING_LOG_PATH = LOG_DIR / "ml_training.log"
GRAPH_LOG_PATH = LOG_DIR / "graph.log"
DATABASE_LOG_PATH = LOG_DIR / "database.log"
PIPELINE_LOG_PATH = LOG_DIR / "pipeline.log"
FRONTEND_LOG_PATH = LOG_DIR / "frontend_client.log"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # reserved, not currently used
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Per-provider model overrides (agent/llm_investigator.py). Defaults are
# current, non-deprecated production models as of Aug 2026 — Groq retired
# llama-3.1/3.3-versatile from its free/developer tier in June 2026, so
# openai/gpt-oss-120b (Groq's recommended migration target) is the default
# rather than the older Llama name.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))

# Jev (TypeSafe AI System One model) — optional post-investigation
# verification pass, agent/jev_verifier.py. Off (agent/mode_state.py) and
# unconfigured by default; set TYPESAFE_API_KEY and flip the dashboard
# toggle to enable it. Jev is a structured-decision model, not a chat LLM —
# it returns typed answers with calibrated probabilities in well under a
# second, so it's cheap enough to run as a per-investigation guardrail
# rather than a sampled spot-check.
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
TYPESAFE_API_BASE = os.getenv("TYPESAFE_API_BASE", "https://api.typesafe.ai")
TYPESAFE_MODEL = os.getenv("TYPESAFE_MODEL", "jev-latest")
# Jev's own docs quote 70-500ms end to end; 5s is a generous ceiling before
# treating it as unavailable, matching LLM_TIMEOUT_SECONDS's role above.
TYPESAFE_TIMEOUT_SECONDS = float(os.getenv("TYPESAFE_TIMEOUT_SECONDS", "5"))
# Minimum independent_action_confidence (0-1) required, on top of action
# agreement and a grounded hypothesis, before agent/graph_agent.py's
# verification result is allowed to auto-resolve a pending HITL review
# (api/routes_agent.py). Kept high on purpose — this only ever *removes*
# a human from a queue, so it should fire on genuinely unambiguous cases.
JEV_AUTO_RESOLVE_MIN_CONFIDENCE = float(os.getenv("JEV_AUTO_RESOLVE_MIN_CONFIDENCE", "0.85"))
# Resilience controls for the external Jev verification dependency. These are
# deliberately conservative: retry only transient provider failures, then
# fail open so Jev can never block the underlying investigation.
JEV_MAX_RETRIES = int(os.getenv("JEV_MAX_RETRIES", "2"))
JEV_RETRY_BACKOFF_SECONDS = float(os.getenv("JEV_RETRY_BACKOFF_SECONDS", "0.25"))
JEV_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("JEV_CIRCUIT_FAILURE_THRESHOLD", "3"))
JEV_CIRCUIT_RESET_SECONDS = float(os.getenv("JEV_CIRCUIT_RESET_SECONDS", "30"))

# CORS: defaults to "*" so the dashboard and Vercel's static-only deployment
# (which calls this API cross-origin, see README's Deployment section) work
# with zero config out of the box. Set ALLOWED_ORIGINS to a comma-separated
# list (e.g. "https://your-app.vercel.app") to lock this down in production.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

# Optional API key gate for /api/v1/* (api/main.py). Empty (the default)
# means no auth is enforced, matching this project's existing permissive
# local/demo posture. Set API_KEY to require an `X-API-Key` header matching
# it on every /api/v1/* request; /health and /dashboard stay open either way.
API_KEY = os.getenv("API_KEY", "")

# Risk threshold triggering agent investigation
HIGH_RISK_THRESHOLD = 70.0

# --- Repeat-MEDIUM-risk watchlist ---
# A transaction that resolves to MONITOR (MEDIUM tier, no HITL, no
# confidence auto-block) soft-flags its user for WATCHLIST_TTL_HOURS. Their
# next transaction gets an explicit, separately-labeled score multiplier —
# the same pattern as the velocity/proxy overlay — so consecutive
# medium-risk behavior compounds instead of resetting to a clean slate on
# every request. See ml/watchlist.py. Deliberately NOT a HITL trigger by
# itself: this tightens the automated tiering/auto-block path rather than
# creating more human review work (ml/decision_policy.py's
# MANDATORY_HUMAN_REASONS is unaffected by it).
WATCHLIST_TTL_HOURS = int(os.getenv("WATCHLIST_TTL_HOURS", 24))
WATCHLIST_SCORE_MULTIPLIER = float(os.getenv("WATCHLIST_SCORE_MULTIPLIER", 1.2))

# Rolling window (days) for the amount_zscore_prior feature (both offline
# training in ml/train_tabular_model.py and live scoring in
# ml/risk_aggregator.py use this SAME constant) — a user's "normal" amount
# is judged against their trailing 90 days, not their entire lifetime.
# Keeping this in one place is what keeps train-time and live-scoring-time
# feature computation from silently drifting apart.
PRIOR_AMOUNT_WINDOW_DAYS = int(os.getenv("PRIOR_AMOUNT_WINDOW_DAYS", 90))