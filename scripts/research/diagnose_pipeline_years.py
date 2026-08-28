"""Count channel-touch trades by year at each pipeline stage."""
from __future__ import annotations

import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.research.backtest_channel_touch_trades import (  # noqa: E402
    _scan_trades,
    enrich_rs,
    filter_trades,
    list_symbols_fast,
    select_same_day_rs,
)
from utils.data.ohlcv_loader import load_ohlcv_many

START = datetime(2018, 11, 1)
END = datetime(2026, 8, 27)


def yr(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    y = pd.to_datetime(df["buy_date"]).dt.year
    return dict(sorted(Counter(y.tolist()).items()))


def pre2020(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int((pd.to_datetime(df["buy_date"]) < "2020-01-01").sum())


def main() -> None:
    symbols = list_symbols_fast("ALPACA", "1d")
    rs_symbol = "SPY"
    if rs_symbol not in symbols:
        symbols = list(symbols) + [rs_symbol]

    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=START,
        end=END,
        use_cache=True,
        chunk_size=50,
        workers=8,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    print(f"Loaded {len(panels)} panels in {time.perf_counter()-t0:.1f}s")

    spy_df = panels[rs_symbol]
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
        "error_pct": 1.2,
        "min_rally_pct": 4.0,
        "min_total_rise_pct": 3.0,
        "entry_mode": "pivot",
    }

    t1 = time.perf_counter()
    raw = _scan_trades(panels, symbols=symbols, workers=4, base=base)
    print(f"\n1 RAW scan: n={len(raw)} pre2020={pre2020(raw)} by_year={yr(raw)} ({time.perf_counter()-t1:.1f}s)")

    t2 = time.perf_counter()
    with_rs = enrich_rs(raw, panels, spy_df, lookbacks=(63, 126), bars_per_session=1)
    null_rs = with_rs["rs_spy_126d"].isna().sum() if not with_rs.empty else 0
    pre = with_rs[pd.to_datetime(with_rs["buy_date"]) < "2020-01-01"] if not with_rs.empty else with_rs
    pre_null = pre["rs_spy_126d"].isna().sum() if not pre.empty else 0
    print(
        f"2 After RS enrich: n={len(with_rs)} pre2020={pre2020(with_rs)} "
        f"pre2020_null_rs={pre_null}/{len(pre)} total_null_rs={null_rs} ({time.perf_counter()-t2:.1f}s)"
    )

    t3 = time.perf_counter()
    geom = filter_trades(
        with_rs,
        require_in_channel=True,
        max_channel_span_days=365.0,
    )
    print(
        f"3 After in-channel+span365: n={len(geom)} pre2020={pre2020(geom)} by_year={yr(geom)} ({time.perf_counter()-t3:.1f}s)"
    )

    t4 = time.perf_counter()
    top1 = select_same_day_rs(geom, rs_col="rs_spy_126d", max_per_day=1)
    print(
        f"4 After RS top1/day: n={len(top1)} pre2020={pre2020(top1)} by_year={yr(top1)} ({time.perf_counter()-t4:.1f}s)"
    )

    # Counterfactual: no RS top1
    all_day = select_same_day_rs(geom, rs_col="rs_spy_126d", max_per_day=0)
    print(f"5 No RS cap (max_per_day=0): n={len(all_day)} pre2020={pre2020(all_day)} by_year={yr(all_day)}")

    # Counterfactual: no geometry filters on raw
    no_geo = select_same_day_rs(with_rs, rs_col="rs_spy_126d", max_per_day=1)
    print(f"6 No geometry filters + RS top1: n={len(no_geo)} pre2020={pre2020(no_geo)} by_year={yr(no_geo)}")


if __name__ == "__main__":
    main()
