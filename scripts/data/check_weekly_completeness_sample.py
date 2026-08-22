#!/usr/bin/env python3
"""Quick weekly-bar completeness sample for Phase 1 example symbols."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.find_weekly_bigvol_examples import load_symbols
from utils.scanning.squeeze import to_weekly

SYMBOLS = ["UBER", "MSFT", "NVDA", "WMT", "CRM", "QCOM", "AMZN", "INTC", "META", "V"]


def week_session_counts(daily: pd.DataFrame) -> pd.Series:
    d = daily.copy()
    if d.index.tz is not None:
        d.index = d.index.tz_convert(None)
    # Count trading days per W-FRI week
    return d["close"].groupby(pd.Grouper(freq="W-FRI")).count()


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def main() -> int:
    t0 = time.perf_counter()
    start = datetime(2018, 1, 1)
    end = datetime(2025, 11, 26)
    data = load_symbols(SYMBOLS, "1d", "ALPACA", start, end)
    print(f"Loaded {len(data)}/{len(SYMBOLS)} symbols\n")
    print(f"{'symbol':<8} {'weeks':>6} {'lt5':>6} {'lt5%':>7} {'lt4':>6} {'range'}")
    rows = []
    for sym, df in sorted(data.items()):
        counts = week_session_counts(df)
        # Drop empty weeks
        counts = counts[counts > 0]
        lt5 = int((counts < 5).sum())
        lt4 = int((counts < 4).sum())
        n = len(counts)
        pct = 100.0 * lt5 / n if n else 0.0
        rng = f"{df.index.min().date()} -> {df.index.max().date()}"
        print(f"{sym:<8} {n:6d} {lt5:6d} {pct:6.1f}% {lt4:6d} {rng}")
        rows.append({"symbol": sym, "weeks": n, "weeks_lt5": lt5, "pct_lt5": pct, "weeks_lt4": lt4})
    out = Path("reports/examples")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "weekly_completeness_sample_20260822.csv", index=False)
    print("\nNote: weeks with <5 sessions include holidays (normal) and true gaps.")
    print("weeks_lt4 is a stricter gap proxy.")
    print(f"Timing: {_format_elapsed(time.perf_counter() - t0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
