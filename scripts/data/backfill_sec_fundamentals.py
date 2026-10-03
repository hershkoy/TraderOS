#!/usr/bin/env python3
"""Backfill SEC companyfacts into fundamentals_quarterly.

Download (resume-friendly) -> parse TTM/ratios -> join mkt_cap from market_data -> upsert.

Examples (Windows CMD)::

    venv\\Scripts\\activate && python scripts/db/apply_fundamentals_schema.py
    venv\\Scripts\\activate && python scripts/data/backfill_sec_fundamentals.py --symbols AAPL,MSFT,META
    venv\\Scripts\\activate && python scripts/data/backfill_sec_fundamentals.py --universe daily --max-symbols 20
    venv\\Scripts\\activate && python scripts/data/backfill_sec_fundamentals.py --universe daily
    venv\\Scripts\\activate && python scripts/data/backfill_sec_fundamentals.py --parse-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.api.sec.companyfacts_client import CompanyfactsClient, DEFAULT_CACHE_ROOT
from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.sec_fundamentals import (
    FundamentalRow,
    attach_mkt_caps,
    build_fundamental_rows,
    passes_four_filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("backfill_sec_fundamentals")

UPSERT_SQL = """
INSERT INTO fundamentals_quarterly (
    symbol, cik, period_end, filed, fy, fp,
    ocf, capex, fcf, total_debt, assets, shares_out,
    buybacks_ttm, issuance_ttm, net_buybacks_ttm, mkt_cap,
    ocf_to_debt, capex_to_ocf, net_buyback_pct, fcf_roa,
    sic, sic_desc, exclude_sector, debt_zero,
    tag_ocf, tag_capex, tag_debt, source, updated_at
) VALUES (
    %(symbol)s, %(cik)s, %(period_end)s, %(filed)s, %(fy)s, %(fp)s,
    %(ocf)s, %(capex)s, %(fcf)s, %(total_debt)s, %(assets)s, %(shares_out)s,
    %(buybacks_ttm)s, %(issuance_ttm)s, %(net_buybacks_ttm)s, %(mkt_cap)s,
    %(ocf_to_debt)s, %(capex_to_ocf)s, %(net_buyback_pct)s, %(fcf_roa)s,
    %(sic)s, %(sic_desc)s, %(exclude_sector)s, %(debt_zero)s,
    %(tag_ocf)s, %(tag_capex)s, %(tag_debt)s, %(source)s, NOW()
)
ON CONFLICT (symbol, period_end, fp) DO UPDATE SET
    cik = EXCLUDED.cik,
    filed = EXCLUDED.filed,
    fy = EXCLUDED.fy,
    ocf = EXCLUDED.ocf,
    capex = EXCLUDED.capex,
    fcf = EXCLUDED.fcf,
    total_debt = EXCLUDED.total_debt,
    assets = EXCLUDED.assets,
    shares_out = EXCLUDED.shares_out,
    buybacks_ttm = EXCLUDED.buybacks_ttm,
    issuance_ttm = EXCLUDED.issuance_ttm,
    net_buybacks_ttm = EXCLUDED.net_buybacks_ttm,
    mkt_cap = EXCLUDED.mkt_cap,
    ocf_to_debt = EXCLUDED.ocf_to_debt,
    capex_to_ocf = EXCLUDED.capex_to_ocf,
    net_buyback_pct = EXCLUDED.net_buyback_pct,
    fcf_roa = EXCLUDED.fcf_roa,
    sic = EXCLUDED.sic,
    sic_desc = EXCLUDED.sic_desc,
    exclude_sector = EXCLUDED.exclude_sector,
    debt_zero = EXCLUDED.debt_zero,
    tag_ocf = EXCLUDED.tag_ocf,
    tag_capex = EXCLUDED.tag_capex,
    tag_debt = EXCLUDED.tag_debt,
    source = EXCLUDED.source,
    updated_at = NOW()
"""


def _finite(v: Any) -> Any:
    """Replace NaN/inf with None for Postgres float columns."""
    if v is None:
        return None
    if isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
        return None
    return v


def row_to_params(row: FundamentalRow) -> Dict[str, Any]:
    d = row.to_db_dict()
    for k in (
        "ocf",
        "capex",
        "fcf",
        "total_debt",
        "assets",
        "shares_out",
        "buybacks_ttm",
        "issuance_ttm",
        "net_buybacks_ttm",
        "mkt_cap",
        "ocf_to_debt",
        "capex_to_ocf",
        "net_buyback_pct",
        "fcf_roa",
    ):
        d[k] = _finite(d.get(k))
    return d


def daily_universe_symbols(client) -> List[str]:
    """Distinct symbols with 1d bars in market_data."""
    sql = """
    SELECT DISTINCT symbol
    FROM market_data
    WHERE timeframe = '1d'
    ORDER BY symbol
    """
    rows = client.execute_query(sql)
    if not rows:
        return []
    return [str(r[0]).upper() for r in rows if r and r[0]]


def load_closes(client, symbol: str) -> Dict[date, float]:
    """Load 1d closes for one symbol (simple filter; last write wins per day)."""
    if not client.ensure_connection():
        raise RuntimeError("DB connection failed")
    # Prefer ALPACA then any; avoid DISTINCT ON full-sort cost
    sql = """
    SELECT (DATE_TRUNC('day', ts AT TIME ZONE 'UTC'))::date AS d, close
    FROM market_data
    WHERE symbol = %s
      AND timeframe = '1d'
      AND close IS NOT NULL
      AND close > 0
      AND provider = 'ALPACA'
    ORDER BY ts
    """
    rows = client.execute_query(sql, (symbol,))
    if not rows:
        sql2 = """
        SELECT (DATE_TRUNC('day', ts AT TIME ZONE 'UTC'))::date AS d, close
        FROM market_data
        WHERE symbol = %s
          AND timeframe = '1d'
          AND close IS NOT NULL
          AND close > 0
        ORDER BY ts
        """
        rows = client.execute_query(sql2, (symbol,))
    out: Dict[date, float] = {}
    if not rows:
        return out
    for r in rows:
        d, px = r[0], r[1]
        if d is None or px is None:
            continue
        if isinstance(d, datetime):
            out[d.date()] = float(px)
        elif isinstance(d, date):
            out[d] = float(px)
        else:
            try:
                out[date.fromisoformat(str(d)[:10])] = float(px)
            except ValueError:
                continue
    return out


def upsert_rows(db, rows: Sequence[FundamentalRow], *, retries: int = 5) -> int:
    if not rows:
        return 0
    params = [row_to_params(r) for r in rows]
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        if not db.ensure_connection():
            last_err = RuntimeError("DB connection failed")
            time.sleep(min(30, 2 ** attempt))
            continue
        cur = db.connection.cursor()
        try:
            from psycopg2.extras import execute_batch

            execute_batch(cur, UPSERT_SQL, params, page_size=100)
            db.connection.commit()
            return len(params)
        except Exception as e:
            last_err = e
            try:
                db.connection.rollback()
            except Exception:
                pass
            logger.warning("upsert attempt %d/%d failed: %s", attempt, retries, e)
            time.sleep(min(30, 2 ** attempt))
        finally:
            try:
                cur.close()
            except Exception:
                pass
    raise RuntimeError(f"upsert failed after retries: {last_err}")


def resolve_symbols(args, db, ticker_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return list of (symbol, cik)."""
    if args.symbols:
        want = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe == "daily":
        want = daily_universe_symbols(db)
        logger.info("Daily universe: %d symbols with 1d bars", len(want))
    else:
        raise SystemExit("Provide --symbols or --universe daily")

    pairs: List[Tuple[str, str]] = []
    missing = 0
    for sym in want:
        cik = ticker_map.get(sym)
        if not cik:
            missing += 1
            continue
        pairs.append((sym, cik))
    if args.max_symbols and args.max_symbols > 0:
        pairs = pairs[: args.max_symbols]
    logger.info(
        "Resolved %d symbols with CIK (%d missing from SEC map)",
        len(pairs),
        missing,
    )
    return pairs


def download_one(
    sec: CompanyfactsClient,
    cik: str,
    *,
    fresh_days: Optional[float],
    force: bool,
) -> Tuple[bool, bool]:
    facts_ok = sec.fetch_companyfacts(cik, force=force, fresh_days=fresh_days) is not None
    sub_ok = sec.fetch_submissions(cik, force=force, fresh_days=fresh_days) is not None
    return facts_ok, sub_ok


def parse_one(
    sec: CompanyfactsClient,
    db,
    symbol: str,
    cik: str,
) -> Tuple[List[FundamentalRow], Dict[str, Any]]:
    facts = sec.load_cached_companyfacts(cik)
    if facts is None:
        return [], {"symbol": symbol, "cik": cik, "error": "no_cached_facts"}
    subs = sec.load_cached_submissions(cik)
    sic, sic_desc = sec.sic_from_submissions(subs)
    rows = build_fundamental_rows(symbol, cik, facts, sic=sic, sic_desc=sic_desc)
    closes = load_closes(db, symbol)
    attach_mkt_caps(rows, closes)

    n_ocf = sum(1 for r in rows if r.ocf is not None)
    n_mkt = sum(1 for r in rows if r.mkt_cap is not None)
    n_all4 = 0
    for r in rows:
        if r.exclude_sector:
            continue
        flags = passes_four_filters(r)
        if all(
            flags.get(k) is True
            for k in ("f1_ocf_debt", "f2_capex_ocf", "f3_buybacks", "f4_fcf_roa")
        ):
            n_all4 += 1

    meta = {
        "symbol": symbol,
        "cik": cik,
        "sic": sic,
        "exclude_sector": bool(rows and rows[0].exclude_sector),
        "n_rows": len(rows),
        "n_ocf": n_ocf,
        "n_mkt_cap": n_mkt,
        "n_pass_all4": n_all4,
        "tag_ocf": rows[0].tag_ocf if rows else None,
        "tag_capex": rows[0].tag_capex if rows else None,
        "tag_debt": rows[0].tag_debt if rows else None,
        "n_closes": len(closes),
    }
    return rows, meta


def write_summary(summary: Dict[str, Any], outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = outdir / f"sec_fundamentals_backfill_{stamp}.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", choices=["daily"], default=None)
    ap.add_argument("--symbols", default=None, help="Comma-separated tickers")
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--parse-only", action="store_true", help="Skip download; parse cache only")
    ap.add_argument("--download-only", action="store_true", help="Download only; skip parse/upsert")
    ap.add_argument("--reset-resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-download even if cached")
    ap.add_argument(
        "--fresh-days",
        type=float,
        default=None,
        help="Re-fetch if cache older than N days (default: keep forever unless --force)",
    )
    ap.add_argument("--cache-root", default=str(DEFAULT_CACHE_ROOT))
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--workers", type=int, default=4, help="Parallel download workers")
    ap.add_argument("--no-upsert", action="store_true")
    ap.add_argument("--log-dir", default="logs/data")
    args = ap.parse_args()

    if not args.symbols and not args.universe:
        ap.error("Provide --symbols or --universe daily")

    t0 = time.perf_counter()
    sec = CompanyfactsClient(cache_root=args.cache_root, sleep_s=args.sleep)
    if args.reset_resume:
        sec.clear_resume()
        logger.info("Cleared resume files")

    db = get_timescaledb_client()
    if not db.connect():
        logger.error("Cannot connect to TimescaleDB")
        return 1

    ticker_map = sec.load_ticker_map(force_refresh=bool(args.force))
    logger.info("SEC ticker map: %d tickers", len(ticker_map))
    pairs = resolve_symbols(args, db, ticker_map)
    if not pairs:
        logger.error("No symbols to process")
        return 2

    dl_ok = dl_fail = 0
    if not args.parse_only:
        workers = max(1, int(args.workers))
        logger.info("Downloading with %d workers (sleep=%.3fs)", workers, args.sleep)

        def _job(item):
            sym, cik = item
            try:
                f_ok, _s_ok = download_one(
                    sec, cik, fresh_days=args.fresh_days, force=args.force
                )
                return sym, True, f_ok, None
            except Exception as e:
                return sym, False, False, e

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_job, item) for item in pairs]
            for fut in as_completed(futs):
                sym, ok, f_ok, err = fut.result()
                done += 1
                if ok and f_ok:
                    dl_ok += 1
                else:
                    dl_fail += 1
                    if err is not None:
                        logger.warning("Download %s failed: %s", sym, err)
                if done % 25 == 0 or done == len(pairs):
                    logger.info(
                        "Download progress %d/%d ok=%d fail=%d last=%s",
                        done,
                        len(pairs),
                        dl_ok,
                        dl_fail,
                        sym,
                    )
    else:
        logger.info("Parse-only: skipping downloads")

    if args.download_only:
        summary = {
            "mode": "download_only",
            "n_symbols": len(pairs),
            "download_ok": dl_ok,
            "download_fail": dl_fail,
            "elapsed_s": time.perf_counter() - t0,
        }
        path = write_summary(summary, Path(args.log_dir))
        logger.info("Wrote %s", path)
        return 0 if dl_fail == 0 else 3

    metas: List[Dict[str, Any]] = []
    n_upserted = 0
    parse_fail = 0
    for i, (sym, cik) in enumerate(pairs, 1):
        try:
            rows, meta = parse_one(sec, db, sym, cik)
            metas.append(meta)
            if meta.get("error"):
                parse_fail += 1
            elif rows and not args.no_upsert:
                n_upserted += upsert_rows(db, rows)
            if i % 50 == 0 or i == len(pairs):
                logger.info(
                    "Parse progress %d/%d upserted_rows=%d fail=%d last=%s rows=%s",
                    i,
                    len(pairs),
                    n_upserted,
                    parse_fail,
                    sym,
                    meta.get("n_rows"),
                )
        except Exception as e:
            err_s = str(e)
            if "connection" in err_s.lower() or "server closed" in err_s.lower():
                logger.warning("Parse %s DB issue, sleeping 10s then retry once: %s", sym, e)
                time.sleep(10)
                try:
                    if not db.ensure_connection():
                        raise RuntimeError("DB connection failed")
                    rows, meta = parse_one(sec, db, sym, cik)
                    metas.append(meta)
                    if meta.get("error"):
                        parse_fail += 1
                    elif rows and not args.no_upsert:
                        n_upserted += upsert_rows(db, rows)
                    continue
                except Exception as e2:
                    parse_fail += 1
                    logger.warning("Parse %s retry failed: %s", sym, e2)
                    metas.append({"symbol": sym, "cik": cik, "error": str(e2)})
                    continue
            parse_fail += 1
            logger.warning("Parse %s failed: %s", sym, e)
            metas.append({"symbol": sym, "cik": cik, "error": str(e)})

    with_ocf = sum(1 for m in metas if (m.get("n_ocf") or 0) > 0)
    with_mkt = sum(1 for m in metas if (m.get("n_mkt_cap") or 0) > 0)
    excluded = sum(1 for m in metas if m.get("exclude_sector"))
    tag_ocf_counts: Dict[str, int] = {}
    for m in metas:
        t = m.get("tag_ocf") or "none"
        tag_ocf_counts[t] = tag_ocf_counts.get(t, 0) + 1

    elapsed = time.perf_counter() - t0
    summary = {
        "mode": "parse_only" if args.parse_only else "download_and_parse",
        "n_symbols": len(pairs),
        "download_ok": dl_ok,
        "download_fail": dl_fail,
        "parse_fail": parse_fail,
        "symbols_with_ocf": with_ocf,
        "symbols_with_mkt_cap": with_mkt,
        "symbols_exclude_sector": excluded,
        "rows_upserted": n_upserted,
        "tag_ocf_counts": tag_ocf_counts,
        "elapsed_s": round(elapsed, 2),
        "per_symbol": metas,
    }
    path = write_summary(summary, Path(args.log_dir))
    logger.info(
        "Done symbols=%d upserted_rows=%d with_ocf=%d with_mkt=%d excluded=%d elapsed=%.1fs",
        len(pairs),
        n_upserted,
        with_ocf,
        with_mkt,
        excluded,
        elapsed,
    )
    logger.info("Wrote %s", path)
    return 0 if parse_fail == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
