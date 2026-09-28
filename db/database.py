"""Database abstraction for RazorRisk.

Production uses PostgreSQL (including managed Supabase PostgreSQL). SQLite is
kept only as an explicit local/test fallback so the ML/evaluation suite can
run without infrastructure. Application call sites intentionally keep their
historical ``get_raw_sqlite_connection`` name for backwards compatibility.
"""
import os
import re
import sqlite3
from pathlib import Path
from config import DATABASE_URL, BASE_DIR, SQLITE_DB_PATH, DB_CONNECT_TIMEOUT_SECONDS
from utils.logger import get_logger

logger = get_logger("database")


def is_postgres() -> bool:
    return DATABASE_URL.startswith(("postgresql://", "postgres://", "postgresql+psycopg://", "postgresql+psycopg2://"))


def _translate_sql(sql: str) -> str:
    """Translate the small SQLite SQL dialect still used by legacy call sites."""
    # Parameter style.
    sql = sql.replace("?", "%s")

    # SQLite datetime() helpers -> PostgreSQL timestamp/interval expressions.
    #
    # The legacy scoring queries use forms such as:
    #   datetime('now')
    #   datetime('now', '-1 hours')
    #   datetime(timestamp, '-1 hours')
    #   datetime(?, '-1 hours')
    #
    # The parameter-style conversion above has already changed '?' to '%s',
    # so parameterized SQLite datetime expressions must be translated too.
    sql = re.sub(
        r"datetime\(\s*'now'\s*\)",
        "CURRENT_TIMESTAMP",
        sql,
        flags=re.I,
    )
    sql = re.sub(
        r"datetime\(\s*'now'\s*,\s*'([^']+)'\s*\)",
        r"CURRENT_TIMESTAMP + INTERVAL '\1'",
        sql,
        flags=re.I,
    )
    sql = re.sub(
        r"datetime\(\s*%s\s*,\s*'([^']+)'\s*\)",
        r"%s::timestamp + INTERVAL '\1'",
        sql,
        flags=re.I,
    )
    sql = re.sub(
        r"datetime\(\s*%s\s*,\s*%s\s*\)",
        r"%s::timestamp + (%s)::interval",
        sql,
        flags=re.I,
    )
    sql = re.sub(
        r"datetime\(\s*([A-Za-z_][\w.]*)\s*,\s*'([^']+)'\s*\)",
        r"\1 + INTERVAL '\2'",
        sql,
        flags=re.I,
    )

    # SQLite strftime expressions used by the feature query.
    sql = re.sub(r"CAST\(strftime\(\s*'%H'\s*,\s*([^\)]+)\)\s+AS\s+INTEGER\)", r"EXTRACT(HOUR FROM \1)::INTEGER", sql, flags=re.I)
    sql = re.sub(r"CAST\(strftime\(\s*'%w'\s*,\s*([^\)]+)\)\s+AS\s+INTEGER\)", r"EXTRACT(DOW FROM \1)::INTEGER", sql, flags=re.I)
    # Hour-bucketing strftime used by ml/risk_graph.py::fetch_node_features
    # (e.g. GROUP BY user_id, strftime('%Y-%m-%d %H', timestamp)). Left
    # untranslated, the literal '%Y'/'%H' in the string survive into the
    # final SQL text and psycopg's %-style parameter substitution rejects
    # them outright ("only '%s', '%b', '%t' are allowed as placeholders").
    sql = re.sub(r"strftime\(\s*'%Y-%m-%d %H'\s*,\s*([^\)]+)\)", r"to_char(\1, 'YYYY-MM-DD HH24')", sql, flags=re.I)

    # SQLite has no real boolean type — it stores/compares BOOLEAN columns
    # as integers, so legacy call sites across ml/, data/, and tests/ wrote
    # literal `col = 1` / `col = 0` comparisons against this schema's four
    # actual BOOLEAN columns. Postgres enforces the real type and rejects
    # `boolean = integer` outright, so translate the literal form here
    # rather than patching every call site (new ones would silently regress).
    _BOOLEAN_COLUMNS = ("is_fraud_ground_truth", "is_vpn_proxy", "is_suspicious_proxy", "velocity_enabled")
    sql = re.sub(r"\b(" + "|".join(_BOOLEAN_COLUMNS) + r")\s*=\s*1\b", r"\1 = TRUE", sql, flags=re.I)
    sql = re.sub(r"\b(" + "|".join(_BOOLEAN_COLUMNS) + r")\s*=\s*0\b", r"\1 = FALSE", sql, flags=re.I)

    # SQLite INSERT OR IGNORE / REPLACE equivalents. These are intentionally
    # generic because the schema has a primary/unique constraint on every
    # table touched by the application.
    if re.match(r"\s*INSERT\s+OR\s+IGNORE\b", sql, flags=re.I):
        sql = re.sub(r"INSERT\s+OR\s+IGNORE\b", "INSERT", sql, count=1, flags=re.I)
        if "ON CONFLICT" not in sql.upper():
            sql = sql.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
    elif re.match(r"\s*INSERT\s+OR\s+REPLACE\b", sql, flags=re.I):
        sql = re.sub(r"INSERT\s+OR\s+REPLACE\b", "INSERT", sql, count=1, flags=re.I)
        if "ON CONFLICT" not in sql.upper():
            m = re.search(r"INSERT\s+INTO\s+([\w\".]+)\s*\(([^)]+)\)\s*VALUES\s*\(([^)]+)\)", sql, flags=re.I | re.S)
            if m:
                cols = [c.strip().strip('"') for c in m.group(2).split(",")]
                # Every call site in this codebase lists the table's single-
                # column PRIMARY KEY first (transaction_id, review_id,
                # investigation_id, ...) — SQLite's INSERT OR REPLACE doesn't
                # need a conflict target, but Postgres's ON CONFLICT DO
                # UPDATE requires one, and errors out without it.
                conflict_col = cols[0]
                assignments = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols[1:])
                sql = sql.rstrip().rstrip(";") + f" ON CONFLICT ({conflict_col}) DO UPDATE SET {assignments}"

    # Catch-all: psycopg applies %-style parameter substitution client-side
    # to every query text, whether or not it has params, so any literal '%'
    # left over from SQL itself — e.g. a LIKE pattern's wildcard, as in
    # data/ingest_real_kaggle_dataset.py's "LIKE 'USER\_0%' ESCAPE '\'" —
    # is misread as a malformed placeholder unless escaped to '%%'. This
    # must run last, after every translation above that legitimately
    # produces '%s' placeholders (from the '?' pass) or column expressions,
    # so it only touches genuinely stray '%' characters.
    sql = re.sub(r"%(?![sbt])", "%%", sql)
    return sql


class _PGCursor:
    def __init__(self, cursor):
        self._cursor = cursor

    def execute(self, sql, params=None):
        self._cursor.execute(_translate_sql(sql), params or ())
        return self

    def executemany(self, sql, seq):
        self._cursor.executemany(_translate_sql(sql), seq)
        return self

    def executescript(self, script):
        # sqlite3 cursors natively support executescript(); psycopg cursors
        # don't. Mirror _PGConnection.executescript() here so call sites that
        # historically got a cursor (not a connection) via get_raw_sqlite_connection()
        # and called .executescript() on it work identically under Postgres.
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)

    def fetchone(self):
        return self._cursor.fetchone()

    def fetchall(self):
        return self._cursor.fetchall()

    def __iter__(self):
        return iter(self._cursor)

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def close(self):
        self._cursor.close()


class _PGConnection:
    def __init__(self, raw):
        self.raw = raw

    def cursor(self):
        return _PGCursor(self.raw.cursor())

    def execute(self, sql, params=None):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executemany(self, sql, seq):
        cur = self.cursor()
        cur.executemany(sql, seq)
        return cur

    def executescript(self, script):
        # psycopg supports multi-statements, but splitting gives cleaner error
        # localization and avoids transaction surprises around DDL.
        for statement in script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)

    def commit(self):
        self.raw.commit()

    def rollback(self):
        self.raw.rollback()

    def close(self):
        self.raw.close()


def _connect_postgres():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL is configured but psycopg is not installed. Install dependencies or set "
            "DATABASE_URL=sqlite:///... only for local/test execution."
        ) from exc
    url = DATABASE_URL
    for prefix in ("postgresql+psycopg://", "postgresql+psycopg2://"):
        if url.startswith(prefix):
            url = "postgresql://" + url[len(prefix):]
    try:
        return _PGConnection(psycopg.connect(url, connect_timeout=DB_CONNECT_TIMEOUT_SECONDS))
    except Exception as exc:
        logger.error(
            "Could not reach Postgres at target host within %ss "
            "(DATABASE_URL host is unreachable/wrong, or the DB is down): %s",
            DB_CONNECT_TIMEOUT_SECONDS,
            exc,
        )
        raise


def get_raw_sqlite_connection():
    """Return the configured database connection.

    Name retained for compatibility with the existing application modules.
    Production is PostgreSQL; SQLite is only selected when DATABASE_URL is an
    explicit sqlite URL (used by tests and zero-infrastructure local work).
    """
    if is_postgres():
        return _connect_postgres()
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL[len("sqlite:///"):]
        return sqlite3.connect(db_path)
    return sqlite3.connect(str(SQLITE_DB_PATH))


def init_db():
    """Apply the database schema and lightweight compatibility migrations."""
    logger.info("Initializing database using connection target: %s", DATABASE_URL.split("@")[0])
    schema_path = BASE_DIR / "db" / "schema.sql"
    conn = get_raw_sqlite_connection()
    try:
        if is_postgres():
            schema = schema_path.read_text(encoding="utf-8")
            # schema.sql is PostgreSQL-first; apply a tiny compatibility transform for the test/local SQLite fallback.
            schema = schema.replace("BIGSERIAL", "INTEGER").replace("DOUBLE PRECISION", "FLOAT")
            # SQLite-specific migration logic
            # below is intentionally bypassed for production.
            conn.executescript(schema)
        else:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = f.read().replace("BIGSERIAL", "INTEGER").replace("DOUBLE PRECISION", "FLOAT")
                conn.executescript(schema)
            def add_column_if_missing(table, column, definition):
                cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
                if column not in cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            add_column_if_missing("transactions", "velocity_enabled", "BOOLEAN DEFAULT 0")
            add_column_if_missing("transactions", "velocity_source", "TEXT DEFAULT 'BACKEND'")
            add_column_if_missing("risk_scores", "stacker_calibrated_score", "FLOAT NOT NULL DEFAULT 0.0")
        conn.commit()
    finally:
        conn.close()
    logger.info("Database schema applied successfully (%s)", "PostgreSQL" if is_postgres() else "SQLite test/local mode")


def get_sqlalchemy_engine():
    """SQLAlchemy engine for pandas/analytics workloads."""
    from sqlalchemy import create_engine
    if is_postgres():
        url = DATABASE_URL
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        elif url.startswith("postgres://"):
            url = "postgresql+psycopg://" + url[len("postgres://"):]
        elif url.startswith("postgresql+asyncpg://"):
            url = "postgresql+psycopg://" + url[len("postgresql+asyncpg://"):]
        return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    return create_engine(DATABASE_URL)


def read_sql_query(sql, params=None):
    """Pandas-compatible SQL reader with the same PostgreSQL SQL translation."""
    import pandas as pd
    engine = get_sqlalchemy_engine()
    try:
        query = _translate_sql(sql) if is_postgres() else sql
        return pd.read_sql_query(query, engine, params=params)
    finally:
        engine.dispose()
        