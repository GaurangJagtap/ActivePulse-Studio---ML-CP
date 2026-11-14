"""Initialize or migrate the ActivePulse Studio database schema."""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.db import DatabaseManager, normalize_db_url


def init_database(db_url: str | None = None, experiment_id: str = "default") -> str:
    target_url = normalize_db_url(db_url or os.environ.get("DATABASE_URL") or "sqlite:///al_platform.db")
    db = DatabaseManager(db_url=target_url, experiment_id=experiment_id)
    db.close()
    return target_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize ActivePulse Studio database schema.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="Target SQLAlchemy or PostgreSQL database URL (defaults to DATABASE_URL env or sqlite:///al_platform.db).",
    )
    args = parser.parse_args()
    resolved = init_database(args.db_url)
    print(f"Initialized ActivePulse database schema at: {resolved}")


if __name__ == "__main__":
    main()

