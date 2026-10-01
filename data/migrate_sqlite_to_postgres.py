"""One-time migration helper for users upgrading from a pre-v26 SQLite DB."""
import argparse
import sqlite3
from pathlib import Path
import psycopg

TABLES = [
    "users", "devices", "ip_addresses", "merchants", "transactions",
    "risk_scores", "investigation_reports", "system_logs", "human_reviews", "user_watchlist",
]


def main():
    parser = argparse.ArgumentParser(description="Migrate an existing RazorRisk SQLite database into PostgreSQL")
    parser.add_argument("sqlite_path", type=Path)
    parser.add_argument("--database-url", required=True)
    args = parser.parse_args()

    src = sqlite3.connect(args.sqlite_path)
    dst = psycopg.connect(args.database_url)
    try:
        with dst.cursor() as cur:
            schema = Path(__file__).resolve().parents[1] / "db" / "schema.sql"
            for statement in schema.read_text().split(";"):
                if statement.strip():
                    cur.execute(statement)

            # Foreign-key order: parents first, children last.
            for table in TABLES:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
                if not rows:
                    continue
                col_info = src.execute(f"PRAGMA table_info({table})").fetchall()
                columns = [r[1] for r in col_info]
                # SQLite stores BOOLEAN columns as plain integers (0/1) and
                # PRAGMA table_info still reports the declared type name, so
                # use it to cast those columns to real Python bools before
                # binding — psycopg won't implicitly coerce an int parameter
                # into a Postgres BOOLEAN column (DatatypeMismatch).
                bool_col_idx = {i for i, r in enumerate(col_info) if (r[2] or "").upper() == "BOOLEAN"}
                if bool_col_idx:
                    rows = [
                        tuple(bool(v) if i in bool_col_idx and v is not None else v for i, v in enumerate(row))
                        for row in rows
                    ]
                placeholders = ",".join(["%s"] * len(columns))
                col_sql = ",".join(columns)
                cur.executemany(
                    f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders}) ON CONFLICT DO NOTHING",
                    rows,
                )
                print(f"{table}: migrated {len(rows)} rows")
        dst.commit()
    finally:
        src.close()
        dst.close()


if __name__ == "__main__":
    main()
