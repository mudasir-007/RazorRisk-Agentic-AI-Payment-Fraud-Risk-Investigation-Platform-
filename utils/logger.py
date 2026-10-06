"""
RazorRisk — structured logging.

Two things this module is responsible for, both promised in the original
project spec but not actually implemented until now:

1. Per-subsystem log files. One shared app.log made it impossible to, say,
   tail just model-training runs without agent chatter interleaved. Every
   subsystem gets its own rotating file; get_logger(name) routes by prefix
   match against __name__-style module names, so callers don't need to
   know which file they're writing to.

2. Correlation IDs. A single scored transaction touches the tabular model,
   the GNN, the aggregator, and (for high-risk ones) the agent — four
   modules, four log lines minimum, previously with no shared identifier
   tying them together. bind_correlation_id() sets a contextvar for the
   current request; every log line emitted while it's set gets a
   [corr_id=...] prefix automatically via a logging.Filter, so grepping
   logs/*.log for one correlation ID reconstructs the full trace of a
   single transaction across every subsystem it touched.
"""
import logging
import sys
import uuid
import contextvars
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import (
    APP_LOG_PATH, RISK_ENGINE_LOG_PATH, AGENT_LOG_PATH,
    ML_TRAINING_LOG_PATH, GRAPH_LOG_PATH, DATABASE_LOG_PATH,
    PIPELINE_LOG_PATH, FRONTEND_LOG_PATH, LOG_DIR,
)

LOG_DIR.mkdir(parents=True, exist_ok=True)

_correlation_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("correlation_id", default="")


def bind_correlation_id(corr_id: str = None) -> str:
    """Call at the start of a request/pipeline run. Every log line emitted
    on this thread/task until clear_correlation_id() carries this ID."""
    corr_id = corr_id or uuid.uuid4().hex[:10].upper()
    _correlation_id_var.set(corr_id)
    return corr_id


def clear_correlation_id():
    _correlation_id_var.set("")


class _CorrelationFilter(logging.Filter):
    def filter(self, record):
        corr_id = _correlation_id_var.get()
        record.corr_id = f"[{corr_id}] " if corr_id else ""
        return True


class FormatterWithColor(logging.Formatter):
    """Custom formatter with ANSI colors for console display."""
    grey, yellow, red, bold_red, cyan, reset = (
        "\x1b[38;20m", "\x1b[33;20m", "\x1b[31;20m", "\x1b[31;1m", "\x1b[36;20m", "\x1b[0m"
    )
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(corr_id)s%(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format_str)
        return logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


def _setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger

    corr_filter = _CorrelationFilter()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(FormatterWithColor())
    ch.addFilter(corr_filter)
    logger.addHandler(ch)

    file_formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(corr_id)s%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(file_formatter)
    fh.addFilter(corr_filter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger


# Per-subsystem channels
app_logger = _setup_logger("RazorRisk.App", APP_LOG_PATH)
risk_logger = _setup_logger("RazorRisk.RiskEngine", RISK_ENGINE_LOG_PATH)
agent_logger = _setup_logger("RazorRisk.Agent", AGENT_LOG_PATH)
ml_logger = _setup_logger("RazorRisk.MLTraining", ML_TRAINING_LOG_PATH)
graph_logger = _setup_logger("RazorRisk.Graph", GRAPH_LOG_PATH)
db_logger = _setup_logger("RazorRisk.Database", DATABASE_LOG_PATH)
pipeline_logger = _setup_logger("RazorRisk.Pipeline", PIPELINE_LOG_PATH)
frontend_logger = _setup_logger("RazorRisk.Frontend", FRONTEND_LOG_PATH)

# Ordered prefix -> channel routing. Checked in order, first match wins —
# order matters for names that could match more than one substring
# (e.g. "risk_aggregator" contains "risk" AND is training-adjacent; it's
# scoring-hot-path code, so it belongs in risk_engine.log, checked first).
_ROUTES = [
    ("risk_aggregator", risk_logger),
    ("risk_engine", risk_logger),
    ("agent", agent_logger),
    ("tool", agent_logger),
    ("investigat", agent_logger),
    ("train_gnn", ml_logger),
    ("train_tabular", ml_logger),
    ("gnn_training", ml_logger),
    ("tabular_training", ml_logger),
    ("ml_training", ml_logger),
    ("risk_graph", graph_logger),
    ("graph_builder", graph_logger),
    ("graph_vis", graph_logger),
    ("database", db_logger),
    ("db.", db_logger),
    ("pipeline", pipeline_logger),
    ("admin", pipeline_logger),
    ("frontend", frontend_logger),
    ("client", frontend_logger),
]


def get_logger(module_name: str) -> logging.Logger:
    """Routes a module's __name__ (or any descriptive string) to the right
    subsystem log file by prefix match. Falls back to app.log for anything
    that doesn't match a known subsystem — general API/server lifecycle
    logging."""
    lname = module_name.lower()
    for needle, target in _ROUTES:
        if needle in lname:
            return target
    return app_logger
