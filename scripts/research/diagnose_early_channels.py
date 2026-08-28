"""Find why raw channel-touch buys start at 2020."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research.backtest_channel_touch_trades import trades_for_symbol
from scripts.research.find_ascending_channels import find_channels
from utils.data.ohlcv_loader import load_ohlcv_many

START = datetime(2018, 11, 1)
END = datetime(2026, 8, 27)
BASE = dict(
    entry_touch=3,
    trail_pct=0.10,
    trail_pct_wide=0.18,
    squeeze_adaptive=True,
    squeeze_pctile=75.0,
    squeeze_lookback=100,
    stop_pct=0.03,
    pivot_len=15,
    atr_stop_mult=2.0,
    error_pct=1.2,
    min_rally_pct=4.0,
    min_total_rise_pct=3.0,
    entry_mode="pivot",
)


def main() -> None:
    syms = ["AAPL", "MSFT", "NVDA", "AMD", "META", "GOOGL", "AMZN", "JPM", "XOM", "CALM"]
    panels = load_ohlcv_many(
        syms,
        timeframe="1d",
        provider="ALPACA",
        start=START,
        end=END,
        fallback_provider="IB",
        merge_mode="prefix",
        use_cache=True,
        workers=1,
    )

    for sym in syms:
        df = panels.get(sym)
        if df is None or df.empty:
            print(f"{sym}: NO PANEL")
            continue
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_convert(None)
        print(f"\n{sym}: panel {idx.min().date()} .. {idx.max().date()} n={len(df)}")
        chs = find_channels(
            df,
            pivot_len=15,
            error_pct=1.2,
            min_intervening_rally_pct=4.0,
            min_total_rise_pct=3.0,
        )
        print(f"  channels={len(chs)}")
        if chs:
            starts = [c["start_date"] for c in chs]
            ends = [c["end_date"] for c in chs]
            print(f"  channel start range: {min(starts)} .. {max(starts)}")
            print(f"  channel end range:   {min(ends)} .. {max(ends)}")
            pre20_ch = [c for c in chs if c["end_date"] < "2020-01-01"]
            print(f"  channels ending before 2020: {len(pre20_ch)}")

        trades = trades_for_symbol(sym, df, **BASE)
        if not trades:
            print("  trades=0")
            continue
        tdf = pd.DataFrame(trades)
        tdf["buy_date"] = pd.to_datetime(tdf["buy_date"])
        pre = tdf[tdf["buy_date"] < "2020-01-01"]
        print(f"  trades={len(tdf)} earliest_buy={tdf['buy_date'].min().date()} pre2020={len(pre)}")
        if not pre.empty:
            print(pre[["buy_date", "touch_date", "channel_start", "channel_end"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
