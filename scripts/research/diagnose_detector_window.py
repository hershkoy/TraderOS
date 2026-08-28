"""Compare raw trade years: default vs windowed detector scan."""
from __future__ import annotations

import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research.backtest_channel_touch_trades import _scan_trades, list_symbols_fast
from utils.data.ohlcv_loader import load_ohlcv_many

START = datetime(2018, 11, 1)
END = datetime(2026, 8, 27)


def run(label: str, extra: dict) -> None:
    syms = list_symbols_fast("ALPACA", "1d")
    panels = load_ohlcv_many(
        syms,
        timeframe="1d",
        provider="ALPACA",
        start=START,
        end=END,
        use_cache=True,
        workers=8,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    base = {
        "entry_touch": 3,
        "trail_pct": 0.10,
        "trail_pct_wide": 0.18,
        "squeeze_adaptive": True,
        "squeeze_pctile": 75.0,
        "squeeze_lookback": 100,
        "stop_pct": 0.03,
        "pivot_len": 15,
        "atr_stop_mult": 2.0,
        "entry_mode": "pivot",
        "channel_kwargs": {
            "error_pct": 1.2,
            "flat_pct": 0.04,
            "min_bars_apart": 15,
            "min_intervening_rally_pct": 4.0,
            "min_intervening_pullback_pct": 3.0,
            "min_total_rise_pct": 3.0,
            "max_low_pivots": 16,
        },
        **extra,
    }
    t0 = time.perf_counter()
    raw = _scan_trades(panels, symbols=syms, workers=4, base=base)
    y = Counter(pd.to_datetime(raw["buy_date"]).dt.year.tolist())
    pre = int((pd.to_datetime(raw["buy_date"]) < "2020-01-01").sum())
    print(f"\n{label}: n={len(raw)} pre2020={pre} sec={time.perf_counter()-t0:.1f}s")
    print(dict(sorted(y.items())[:8]), "...")


if __name__ == "__main__":
    run("max_low_pivots=128", {
        "channel_kwargs": {
            "error_pct": 1.2, "flat_pct": 0.04, "min_bars_apart": 15,
            "min_intervening_rally_pct": 4.0, "min_intervening_pullback_pct": 3.0,
            "min_total_rise_pct": 3.0, "max_low_pivots": 128,
        }
    })
