"""Pytest bootstrap: tests use isolated SQLite; production uses PostgreSQL."""
import os
from pathlib import Path

# Keep the deterministic unit/regression suite infrastructure-free. Production
# deployments must provide DATABASE_URL pointing at PostgreSQL/Supabase.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path('/tmp') / 'razorrisk_test.db'}")

from db.database import init_db


def pytest_sessionstart(session):
    init_db()
    # Seed the deterministic golden fixture once so tests that inspect the
    # synthetic communities directly do not depend on the shipped SQLite DB.
    from data.generate_synthetic_data import generate_dataset
    # Train the checked-in scoring pipeline against the same deterministic
    # fixture used by the golden matrix. This prevents the tests from
    # evaluating a model artifact trained on a different dataset snapshot.
    generate_dataset(num_users=1500, num_transactions=12000, seed=42)
    from ml.risk_aggregator import train_stacker
    train_stacker()
