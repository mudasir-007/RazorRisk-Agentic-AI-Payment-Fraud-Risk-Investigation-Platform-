from fastapi import APIRouter
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from config import (
    APP_LOG_PATH, RISK_ENGINE_LOG_PATH, AGENT_LOG_PATH,
    ML_TRAINING_LOG_PATH, GRAPH_LOG_PATH, DATABASE_LOG_PATH, PIPELINE_LOG_PATH,
)
from utils.logger import get_logger, frontend_logger

logger = get_logger("api_logs")

router = APIRouter(prefix="/api/v1/logs", tags=["Audit System Logs"])

def read_tail(file_path: Path, max_lines: int = 50) -> list:
    if not file_path.exists():
        return [f"Log file {file_path.name} does not exist yet."]
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return [line.strip() for line in lines[-max_lines:]]
    except Exception as e:
        return [f"Error reading log file {file_path.name}: {str(e)}"]

@router.get("/stream")
def get_log_streams(lines: int = 30):
    """Returns tail of every subsystem log channel."""
    return {
        "app_logs": read_tail(APP_LOG_PATH, lines),
        "risk_engine_logs": read_tail(RISK_ENGINE_LOG_PATH, lines),
        "agent_logs": read_tail(AGENT_LOG_PATH, lines),
        "ml_training_logs": read_tail(ML_TRAINING_LOG_PATH, lines),
        "graph_logs": read_tail(GRAPH_LOG_PATH, lines),
        "database_logs": read_tail(DATABASE_LOG_PATH, lines),
        "pipeline_logs": read_tail(PIPELINE_LOG_PATH, lines),
    }


class ClientLogPayload(BaseModel):
    level: str = "error"
    message: str
    context: Optional[str] = None


@router.post("/client")
def log_client_error(payload: ClientLogPayload):
    """Lets the dashboard's own JS report errors (a failed fetch, a render
    exception) into the server-side audit trail, instead of those only
    ever landing in a browser console no one but the person hitting the
    bug ever sees — a real gap for a tool whose whole pitch is
    auditability. See static/js/app.js's window.onerror hook."""
    msg = f"[client:{payload.context or 'unknown'}] {payload.message}"
    if payload.level == "warning":
        frontend_logger.warning(msg)
    else:
        frontend_logger.error(msg)
    return {"status": "logged"}
