#!/usr/bin/env python3
"""Score weekly_bigvol full-setup CSV against assessing_strategies-style gates."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd


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


def score(path: Path, split_filter: float = 0.45) -> None:
    t0 = time.perf_counter()
    df = pd.read_csv(path)
    if df.empty:
        print("No full setups in", path)
        return

    # Deduplicate by symbol+confirm_week (multiple ignitions -> same confirm)
    dedup = df.drop_duplicates(subset=["symbol", "confirm_week"], keep="first").copy()

    def _stats(name: str, x: pd.DataFrame) -> None:
        n = len(x)
        print(f"\n=== {name} (n={n}) ===")
        if n == 0:
            return
        for col in ("fwd_4w", "fwd_13w"):
            s = x[col].dropna()
            if s.empty:
                print(f"{col}: n/a")
                continue
            wins = (s > 0).mean() * 100
            print(
                f"{col}: median={s.median():.2%} mean={s.mean():.2%} "
                f"win%={wins:.1f} p25={s.quantile(0.25):.2%} p75={s.quantile(0.75):.2%}"
            )
        # Crude expectancy proxy in R-less % terms
        for col in ("fwd_4w", "fwd_13w"):
            s = x[col].dropna()
            if len(s) == 0:
                continue
            print(f"expectancy({col}) ~= mean return {s.mean():.2%} per setup")

    print(f"File: {path}")
    print(f"Raw full-setup rows: {len(df)}")
    print(f"Unique symbol+confirm: {len(dedup)}")
    print(f"Unique symbols: {dedup['symbol'].nunique()}")
    print(f"Delay weeks: median={dedup['delay_weeks'].median():.0f} "
          f"mean={dedup['delay_weeks'].mean():.1f}")

    _stats("All unique confirms", dedup)

    # Filter likely splits / bad data (|return| extreme)
    clean = dedup[
        (dedup["fwd_4w"].abs() < split_filter) & (dedup["fwd_13w"].abs() < split_filter * 1.5)
    ].copy()
    # also keep rows where fwd is NaN separately counted
    _stats(f"After |fwd| filter (<{split_filter:.0%} 4w)", clean)

    # Year split if possible
    dedup["year"] = pd.to_datetime(dedup["confirm_week"]).dt.year
    print("\nSetups by confirm year:")
    print(dedup.groupby("year").size().to_string())

    print("\nGate checklist (setup-level, not portfolio BT):")
    n = len(clean)
    med13 = clean["fwd_13w"].median() if n else float("nan")
    win13 = (clean["fwd_13w"] > 0).mean() * 100 if n else float("nan")
    print(f"  Trade/setup count (clean unique): {n}  (want ~300+ for confidence)")
    print(f"  Median 13w forward: {med13:.2%}  (want >0)")
    print(f"  13w win rate: {win13:.1f}%")
    print("  Note: this is hold-13w curiosity, not strategy exits (MA10/30/stop).")
    print(f"Timing: {_format_elapsed(time.perf_counter() - t0)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=str, help="Path to weekly_bigvol_full_setups_*.csv")
    ap.add_argument("--split-filter", type=float, default=0.45)
    args = ap.parse_args()
    score(Path(args.csv), split_filter=args.split_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
