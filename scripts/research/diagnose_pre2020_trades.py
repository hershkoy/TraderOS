"""Diagnose why channel-touch has no pre-2020 kept trades."""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research.backtest_channel_touch_trades import (  # noqa: E402
    enrich_rs,
    filter_trades,
    select_same_day_rs,
)
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.db.timescaledb_client import get_timescaledb_client

START = datetime(2018, 11, 1)
END = datetime(2026, 8, 27)


def year_counts(df: pd.DataFrame, col: str = "buy_date") -> Counter:
    if df.empty or col not in df.columns:
        return Counter()
    y = pd.to_datetime(df[col]).dt.year
    return Counter(y.tolist())


def main() -> None:
    csv = ROOT / "reports/ascending_channels/channel_touch_trades_20260827_223039.csv"
    kept = pd.read_csv(csv)
    print("=== KEPT TRADES (final CSV) ===")
    print(dict(sorted(year_counts(kept).items())))

    # Raw trades before post-filters: reload from log says 2280 trades before filter
    # Quick DB: count IB prefix coverage
    client = get_timescaledb_client()
    client.ensure_connection()
    cur = client.connection.cursor()
    cur.execute(
        """
        SELECT COUNT(DISTINCT symbol),
               MIN(ts)::date,
               MAX(ts)::date
        FROM market_data
        WHERE provider='IB' AND timeframe='1d'
        """
    )
    ib_syms, ib_min, ib_max = cur.fetchone()
    print(f"\n=== IB 1d DB === symbols={ib_syms} range={ib_min}..{ib_max}")

    cur.execute(
        """
        WITH alp AS (
          SELECT symbol, MIN(ts) AS amin FROM market_data
          WHERE provider='ALPACA' AND timeframe='1d' GROUP BY symbol
        ),
        ib AS (
          SELECT symbol, MIN(ts) AS imin FROM market_data
          WHERE provider='IB' AND timeframe='1d' GROUP BY symbol
        )
        SELECT COUNT(*) FROM alp a
        JOIN ib i ON i.symbol=a.symbol
        WHERE i.imin < a.amin
        """
    )
    print(f"symbols with IB prefix before Alpaca: {cur.fetchone()[0]}")
    cur.close()

    # Load SPY with fallback like backtest RS path
    spy_panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="ALPACA",
        start=START,
        end=END,
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
        workers=1,
    )
    spy = spy_panels.get("SPY")
    if spy is not None and not spy.empty:
        print(f"\n=== SPY merged panel === rows={len(spy)} min={spy.index.min()} max={spy.index.max()}")
    else:
        print("\n=== SPY merged panel === EMPTY")

    # Sample: raw trades file doesn't exist; parse log or re-run slice
    # Instead analyze pre-filter from a small re-scan isn't feasible full universe
    # Check RS on kept vs what buy dates have rs
    if "rs_spy_126d" in kept.columns:
        kept2 = kept.copy()
        kept2["buy_date"] = pd.to_datetime(kept2["buy_date"])
        print("\n=== RS on kept trades ===")
        print(f"null rs_spy_126d: {kept2['rs_spy_126d'].isna().sum()}/{len(kept2)}")
        early = kept2[kept2["buy_date"] < "2020-01-01"]
        print(f"kept before 2020: {len(early)}")

    # Load raw trades from backtest cache if any - skip
    # Diagnose RS gate: count buy_dates in 2019 from unfiltered - need raw trades
    print("\n=== Checking raw trade pool (re-load subset) ===")
    # Load panels for symbols that had 2019 channels - use AAPL MSFT
    symbols = ["AAPL", "MSFT", "NVDA", "AMD", "SPY"]
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=START,
        end=END,
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
        workers=1,
    )
    for s, p in panels.items():
        if p is not None and not p.empty:
            print(f"  {s}: {p.index.min().date()} .. {p.index.max().date()} n={len(p)}")


if __name__ == "__main__":
    main()
