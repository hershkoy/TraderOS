#!/usr/bin/env python3
"""Apply init-scripts/19-fundamentals-quarterly.sql to TimescaleDB."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import psycopg2


def get_connection_string() -> str:
    host = os.getenv("TIMESCALEDB_HOST", "localhost")
    port = os.getenv("TIMESCALEDB_PORT", "5432")
    database = os.getenv("TIMESCALEDB_DATABASE", "backtrader")
    user = os.getenv("TIMESCALEDB_USER", "backtrader_user")
    password = os.getenv("TIMESCALEDB_PASSWORD", "backtrader_password")
    return f"host={host} port={port} dbname={database} user={user} password={password}"


def main() -> int:
    migration_file = ROOT / "init-scripts" / "19-fundamentals-quarterly.sql"
    if not migration_file.exists():
        print(f"Migration not found: {migration_file}")
        return 1
    sql = migration_file.read_text(encoding="utf-8")
    print(f"Applying {migration_file}...")
    with psycopg2.connect(get_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'fundamentals_quarterly'
                ORDER BY ordinal_position
                """
            )
            cols = cur.fetchall()
    if not cols:
        print("Table fundamentals_quarterly not found after migration")
        return 2
    print(f"OK fundamentals_quarterly ({len(cols)} columns)")
    for name, dtype in cols:
        print(f"  {name}: {dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
