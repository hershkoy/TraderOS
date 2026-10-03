"""Fill mkt_cap / net_buyback_pct on fundamentals_quarterly from market_data 1d closes.

Batched close loads (ANY(%s)) — much faster than one query per symbol.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from psycopg2.extras import execute_batch

from utils.db.timescaledb_client import get_timescaledb_client
from utils.research.sec_fundamentals import FundamentalRow, attach_mkt_caps

UPDATE_SQL = """
UPDATE fundamentals_quarterly
SET mkt_cap = %(mkt_cap)s,
    net_buyback_pct = %(net_buyback_pct)s,
    updated_at = NOW()
WHERE symbol = %(symbol)s
  AND period_end = %(period_end)s
  AND fp = %(fp)s
"""


def _as_date(d) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d)[:10])


def load_closes_batch(db, symbols: List[str]) -> Dict[str, Dict[date, float]]:
    if not symbols:
        return {}
    if not db.ensure_connection():
        raise RuntimeError("DB connection failed")
    sql = """
    SELECT symbol, (ts AT TIME ZONE 'UTC')::date AS d, close
    FROM market_data
    WHERE symbol = ANY(%s)
      AND timeframe = '1d'
      AND provider = 'ALPACA'
      AND close IS NOT NULL
      AND close > 0
    ORDER BY symbol, ts
    """
    rows = db.execute_query(sql, (symbols,))
    out: Dict[str, Dict[date, float]] = defaultdict(dict)
    for r in rows or []:
        sym, d, px = r[0], r[1], r[2]
        if sym is None or d is None or px is None:
            continue
        out[str(sym).upper()][_as_date(d)] = float(px)
    # Fallback for symbols with no ALPACA bars
    missing = [s for s in symbols if s not in out]
    if missing:
        sql2 = """
        SELECT symbol, (ts AT TIME ZONE 'UTC')::date AS d, close
        FROM market_data
        WHERE symbol = ANY(%s)
          AND timeframe = '1d'
          AND close IS NOT NULL
          AND close > 0
        ORDER BY symbol, ts
        """
        rows2 = db.execute_query(sql2, (missing,))
        for r in rows2 or []:
            sym, d, px = r[0], r[1], r[2]
            if sym is None or d is None or px is None:
                continue
            out[str(sym).upper()][_as_date(d)] = float(px)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-symbols", type=int, default=100)
    ap.add_argument("--max-symbols", type=int, default=0)
    ap.add_argument("--all-with-shares", action="store_true")
    args = ap.parse_args()

    db = get_timescaledb_client()
    if not db.connect():
        print("Cannot connect to TimescaleDB")
        return 1

    where = (
        "shares_out IS NOT NULL AND shares_out > 0"
        if args.all_with_shares
        else "shares_out IS NOT NULL AND shares_out > 0 AND mkt_cap IS NULL"
    )
    syms = db.execute_query(
        f"SELECT DISTINCT symbol FROM fundamentals_quarterly WHERE {where} ORDER BY 1"
    )
    symbols = [str(r[0]).upper() for r in (syms or [])]
    if args.max_symbols:
        symbols = symbols[: args.max_symbols]
    print(f"Symbols to fill: {len(symbols)} batch={args.batch_symbols}", flush=True)

    t0 = time.perf_counter()
    n_updated = 0
    n_no_px = 0
    bs = max(1, args.batch_symbols)

    for i in range(0, len(symbols), bs):
        batch = symbols[i : i + bs]
        closes_by_sym = load_closes_batch(db, batch)

        # Pull fund rows for this batch
        rows = db.execute_query(
            f"""
            SELECT symbol, period_end, fp, filed, shares_out, net_buybacks_ttm, cik
            FROM fundamentals_quarterly
            WHERE symbol = ANY(%s) AND {where}
            """,
            (batch,),
        )
        by_sym: Dict[str, List] = defaultdict(list)
        for r in rows or []:
            by_sym[str(r[0]).upper()].append(r)

        params = []
        for sym in batch:
            closes = closes_by_sym.get(sym) or {}
            if not closes:
                n_no_px += 1
                continue
            fund_rows = []
            for symbol, period_end, fp, filed, shares_out, net_bb, cik in by_sym.get(sym, []):
                fund_rows.append(
                    FundamentalRow(
                        symbol=sym,
                        cik=str(cik or ""),
                        period_end=_as_date(period_end),
                        filed=_as_date(filed),
                        fy=None,
                        fp=str(fp),
                        shares_out=float(shares_out) if shares_out is not None else None,
                        net_buybacks_ttm=float(net_bb) if net_bb is not None else None,
                    )
                )
            attach_mkt_caps(fund_rows, closes)
            for r in fund_rows:
                if r.mkt_cap is None:
                    continue
                params.append(
                    {
                        "symbol": r.symbol,
                        "period_end": r.period_end,
                        "fp": r.fp,
                        "mkt_cap": r.mkt_cap,
                        "net_buyback_pct": r.net_buyback_pct,
                    }
                )

        if params:
            if not db.ensure_connection():
                raise RuntimeError("DB connection failed")
            cur = db.connection.cursor()
            try:
                execute_batch(cur, UPDATE_SQL, params, page_size=500)
                db.connection.commit()
                n_updated += len(params)
            except Exception as e:
                db.connection.rollback()
                print(f"  WARN batch upsert: {e}", flush=True)
            finally:
                cur.close()

        done = min(i + bs, len(symbols))
        print(
            f"  {done}/{len(symbols)} updated_rows={n_updated} no_price={n_no_px} "
            f"elapsed={time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    summary = db.execute_query(
        "SELECT COUNT(1), COUNT(mkt_cap), COUNT(shares_out) FROM fundamentals_quarterly"
    )
    print(
        f"Done updated_rows={n_updated} no_price_symbols={n_no_px} "
        f"elapsed={time.perf_counter() - t0:.1f}s "
        f"table_n_mkt_shares={summary[0] if summary else None}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
