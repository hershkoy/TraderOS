"""
Ingest OptionsDX SPY/VIX EOD 7z archives into option_contracts + option_quotes.

Usage (from repo root, venv on):
    python scripts/db/ingest_optionsdx.py
    python scripts/db/ingest_optionsdx.py --src temp/optionsdx --keep-extracted
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import os
import shutil
import sys
from pathlib import Path

import psycopg2
import py7zr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.options.optionsdx import (  # noqa: E402
    CONTRACT_COLUMNS,
    QUOTE_COLUMNS,
    iter_file_quotes,
    underlying_from_archive_name,
)

os.makedirs("logs/db", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/db/ingest_optionsdx.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("ingest_optionsdx")

QUOTE_BATCH = 80000


def db_connect():
    conn = psycopg2.connect(
        host=os.getenv("TIMESCALEDB_HOST", "localhost"),
        port=int(os.getenv("TIMESCALEDB_PORT", "5432")),
        dbname=os.getenv("TIMESCALEDB_DATABASE", "backtrader"),
        user=os.getenv("TIMESCALEDB_USER", "backtrader_user"),
        password=os.getenv("TIMESCALEDB_PASSWORD", "backtrader_password"),
        connect_timeout=10,
        options="-c statement_timeout=0",
    )
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("SET synchronous_commit = off")
        cur.execute("SET work_mem = '256MB'")
    conn.commit()
    return conn


def prepare_db(conn) -> None:
    sql_path = ROOT / "init-scripts" / "13-drop-option-quotes-retention.sql"
    with conn.cursor() as cur:
        cur.execute(sql_path.read_text(encoding="utf-8"))
        cur.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS optionsdx_quotes_stage (
                ts TIMESTAMPTZ,
                option_id TEXT,
                bid NUMERIC(18,6),
                ask NUMERIC(18,6),
                last NUMERIC(18,6),
                volume BIGINT,
                open_interest BIGINT,
                iv NUMERIC(12,6),
                delta NUMERIC(12,6),
                gamma NUMERIC(12,6),
                theta NUMERIC(12,6),
                vega NUMERIC(12,6),
                snapshot_type TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS optionsdx_contracts_stage (
                option_id TEXT,
                underlying TEXT,
                expiration DATE,
                strike_cents INT,
                option_right CHAR(1),
                multiplier INT,
                first_seen TIMESTAMPTZ,
                last_seen TIMESTAMPTZ
            )
            """
        )
    conn.commit()


def _copy_rows(cur, table: str, columns, rows) -> None:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    for row in rows:
        writer.writerow(["" if row.get(c) is None else row.get(c) for c in columns])
    buf.seek(0)
    cur.copy_expert(
        "COPY %s (%s) FROM STDIN WITH (FORMAT csv, NULL '')"
        % (table, ", ".join(columns)),
        buf,
    )


def flush_quotes(cur, rows) -> int:
    if not rows:
        return 0
    cur.execute("TRUNCATE optionsdx_quotes_stage")
    _copy_rows(cur, "optionsdx_quotes_stage", QUOTE_COLUMNS, rows)
    cur.execute(
        """
        INSERT INTO option_quotes (
            ts, option_id, bid, ask, last, volume, open_interest,
            iv, delta, gamma, theta, vega, snapshot_type
        )
        SELECT DISTINCT ON (option_id, ts)
            ts, option_id, bid, ask, last, volume, open_interest,
            iv, delta, gamma, theta, vega, snapshot_type
        FROM optionsdx_quotes_stage
        ORDER BY option_id, ts
        ON CONFLICT (option_id, ts) DO UPDATE SET
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            last = EXCLUDED.last,
            volume = EXCLUDED.volume,
            open_interest = EXCLUDED.open_interest,
            iv = EXCLUDED.iv,
            delta = EXCLUDED.delta,
            gamma = EXCLUDED.gamma,
            theta = EXCLUDED.theta,
            vega = EXCLUDED.vega,
            snapshot_type = EXCLUDED.snapshot_type
        """
    )
    return len(rows)


def flush_contracts(cur, contracts_by_id: dict) -> int:
    if not contracts_by_id:
        return 0
    rows = list(contracts_by_id.values())
    cur.execute("TRUNCATE optionsdx_contracts_stage")
    _copy_rows(cur, "optionsdx_contracts_stage", CONTRACT_COLUMNS, rows)
    cur.execute(
        """
        INSERT INTO option_contracts (
            option_id, underlying, expiration, strike_cents, option_right,
            multiplier, first_seen, last_seen
        )
        SELECT
            option_id, underlying, expiration, strike_cents, option_right,
            multiplier, first_seen, last_seen
        FROM optionsdx_contracts_stage
        ON CONFLICT (option_id) DO UPDATE SET
            last_seen = GREATEST(option_contracts.last_seen, EXCLUDED.last_seen),
            first_seen = LEAST(
                COALESCE(option_contracts.first_seen, EXCLUDED.first_seen),
                EXCLUDED.first_seen
            ),
            underlying = EXCLUDED.underlying,
            expiration = EXCLUDED.expiration,
            strike_cents = EXCLUDED.strike_cents,
            option_right = EXCLUDED.option_right,
            multiplier = EXCLUDED.multiplier
        """
    )
    return len(rows)


def ingest_txt(conn, path: Path, underlying: str) -> tuple:
    n_quotes = 0
    n_contracts = 0
    batch = []
    contracts = {}
    with conn.cursor() as cur:
        for quote, contract in iter_file_quotes(path, underlying):
            oid = contract["option_id"]
            prev = contracts.get(oid)
            if prev is None:
                contracts[oid] = contract
            else:
                if contract["last_seen"] > prev["last_seen"]:
                    prev["last_seen"] = contract["last_seen"]
                if contract["first_seen"] < prev["first_seen"]:
                    prev["first_seen"] = contract["first_seen"]
            batch.append(quote)
            if len(batch) >= QUOTE_BATCH:
                n_quotes += flush_quotes(cur, batch)
                batch = []
        n_quotes += flush_quotes(cur, batch)
        n_contracts += flush_contracts(cur, contracts)
    conn.commit()
    return n_quotes, n_contracts


def extract_archive(archive: Path, dest: Path) -> list:
    dest.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, mode="r") as z:
        z.extractall(path=dest)
    return sorted(p for p in dest.rglob("*") if p.is_file() and p.suffix.lower() in (".txt", ".csv"))


def ingest_archive(conn, archive: Path, extract_root: Path, keep: bool) -> tuple:
    underlying = underlying_from_archive_name(archive.name)
    dest = extract_root / archive.stem
    logger.info("Extracting %s -> %s (%s)", archive.name, dest, underlying)
    files = extract_archive(archive, dest)
    q_total = 0
    c_total = 0
    try:
        for path in files:
            logger.info("Ingesting %s", path.name)
            qn, cn = ingest_txt(conn, path, underlying)
            q_total += qn
            c_total += cn
            logger.info("  quotes=%s unique_contracts_in_file=%s", qn, cn)
    finally:
        if not keep and dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
    return q_total, c_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest OptionsDX EOD 7z into TimescaleDB")
    parser.add_argument("--src", default="temp/optionsdx", help="Folder of .7z archives")
    parser.add_argument(
        "--extract-dir",
        default="temp/optionsdx/extracted",
        help="Temporary extract folder",
    )
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument(
        "--only",
        default="",
        help="Optional substring filter on archive filename (e.g. spy_eod_2022)",
    )
    args = parser.parse_args()

    src = Path(args.src)
    archives = sorted(src.glob("*.7z"))
    if args.only:
        archives = [a for a in archives if args.only.lower() in a.name.lower()]
    if not archives:
        logger.error("No .7z archives in %s", src)
        return 1

    logger.info("Archives: %s", len(archives))
    conn = db_connect()
    try:
        prepare_db(conn)
        q_all = 0
        c_all = 0
        extract_root = Path(args.extract_dir)
        extract_root.mkdir(parents=True, exist_ok=True)
        for i, archive in enumerate(archives, 1):
            logger.info("[%s/%s] %s", i, len(archives), archive.name)
            qn, cn = ingest_archive(conn, archive, extract_root, args.keep_extracted)
            q_all += qn
            c_all += cn
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM option_quotes")
            nq = cur.fetchone()[0]
            cur.execute(
                "SELECT underlying, COUNT(*) FROM option_contracts GROUP BY 1 ORDER BY 1"
            )
            contracts = cur.fetchall()
            cur.execute("SELECT MIN(ts), MAX(ts) FROM option_quotes")
            rng = cur.fetchone()
        logger.info("Done. quotes_upserted=%s contract_rows_touched=%s", q_all, c_all)
        logger.info("DB option_quotes=%s range=%s", nq, rng)
        logger.info("DB contracts by underlying: %s", contracts)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
