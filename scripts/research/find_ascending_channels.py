#!/usr/bin/env python3
"""
Find classical ascending channels (Edwards/Magee-style heuristics).

Not Trendoscope ACP. Stricter than v1:
  - Distinct swing lows separated by meaningful intervening rallies
  - Parallel resistance confirmed by >=2 swing highs (not one floating peak)
  - Support mostly respected inside the pattern window
  - Reject near-flat accumulation bases

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\find_ascending_channels.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\find_ascending_channels.py --n-symbols 100 --min-touches 3
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.db.timescaledb_client import get_timescaledb_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("find_ascending_channels")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _pick_symbols(n: int, provider: str, timeframe: str, min_bars: int) -> List[str]:
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    sql = """
        WITH recent AS (
            SELECT symbol, COUNT(*) AS n_bars, AVG(close * volume) AS avg_dollar_vol
            FROM market_data
            WHERE provider = %s AND timeframe = %s AND ts >= NOW() - INTERVAL '400 days'
            GROUP BY symbol
            HAVING COUNT(*) >= %s
        )
        SELECT symbol FROM recent
        ORDER BY avg_dollar_vol DESC NULLS LAST
        LIMIT %s
    """
    cur = client.connection.cursor()
    try:
        cur.execute(sql, (provider.upper(), timeframe, int(min_bars), int(n)))
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()


def list_symbols_fast(provider: str = "ALPACA", timeframe: str = "1d") -> List[str]:
    """Full-universe symbol list via DISTINCT only (no heavy HAVING aggregates)."""
    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Failed to connect to TimescaleDB")
    return client.get_available_symbols(provider=provider.upper(), timeframe=timeframe)


def _pivots(high: np.ndarray, low: np.ndarray, length: int) -> Tuple[List[int], List[int]]:
    """Confirmed fractal pivots via centered rolling extrema."""
    n = len(high)
    if n < 2 * length + 1:
        return [], []
    hs = pd.Series(high)
    ls = pd.Series(low)
    win = 2 * length + 1
    roll_max = hs.rolling(win, center=True, min_periods=win).max()
    roll_min = ls.rolling(win, center=True, min_periods=win).min()
    is_high = (hs == roll_max) & (hs > hs.shift(1)) & (hs >= hs.shift(-1))
    is_low = (ls == roll_min) & (ls < ls.shift(1)) & (ls <= ls.shift(-1))
    valid = roll_max.notna() & roll_min.notna()
    highs = [int(i) for i in np.flatnonzero((is_high & valid).to_numpy())]
    lows = [int(i) for i in np.flatnonzero((is_low & valid).to_numpy())]
    return highs, lows


def _line_at(y1: float, x1: int, slope: float, x: int) -> float:
    return y1 + slope * (x - x1)


def _touch_ok(y_hat: float, px: float, error_pct: float) -> bool:
    if px <= 0:
        return False
    return abs(y_hat - px) / px * 100.0 <= error_pct


def _intervening_rally_ok(
    high: np.ndarray,
    low_i: int,
    next_low_i: int,
    support_at_low: float,
    min_rally_pct: float,
) -> bool:
    """Require a meaningful rally between two support touches (classical distinct swings)."""
    if next_low_i <= low_i + 1:
        return False
    peak = float(np.nanmax(high[low_i + 1 : next_low_i]))
    if support_at_low <= 0:
        return False
    rally_pct = (peak - support_at_low) / support_at_low * 100.0
    return rally_pct >= min_rally_pct


def _intervening_pullback_ok(
    low: np.ndarray,
    high_i: int,
    next_high_i: int,
    resist_at_high: float,
    min_pullback_pct: float,
) -> bool:
    if next_high_i <= high_i + 1:
        return False
    trough = float(np.nanmin(low[high_i + 1 : next_high_i]))
    if resist_at_high <= 0:
        return False
    pb_pct = (resist_at_high - trough) / resist_at_high * 100.0
    return pb_pct >= min_pullback_pct


def find_channels(
    df: pd.DataFrame,
    *,
    pivot_len: int = 15,
    min_touches: int = 3,
    min_top_touches: int = 2,
    error_pct: float = 1.2,
    flat_pct: float = 0.04,
    min_bars_apart: int = 15,
    min_intervening_rally_pct: float = 4.0,
    min_intervening_pullback_pct: float = 3.0,
    max_support_violation_frac: float = 0.08,
    max_low_pivots: int = 16,
) -> List[dict]:
    """
    Classical ascending channel heuristics:
      - >= min_touches distinct higher lows on an ascending support line
      - intervening rallies between lows (not one consolidation base)
      - >= min_top_touches swing highs on a parallel return line
      - intervening pullbacks between those highs
      - closes mostly stay above support inside the window
    """
    if df is None or df.empty or len(df) < pivot_len * 4 + 40:
        return []

    if (
        isinstance(df.index, pd.DatetimeIndex)
        and df.index.tz is None
        and bool(df.index.is_monotonic_increasing)
        and all(c in df.columns for c in ("high", "low", "close"))
    ):
        out = df
    else:
        out = df.copy()
        if not isinstance(out.index, pd.DatetimeIndex):
            out.index = pd.DatetimeIndex(out.index)
        if out.index.tz is not None:
            out.index = out.index.tz_convert(None)
        out = out.sort_index()

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    dates = out.index

    high_piv, low_piv = _pivots(high, low, pivot_len)
    if len(low_piv) < min_touches or len(high_piv) < min_top_touches:
        return []

    low_piv = low_piv[-max_low_pivots:]
    results: List[dict] = []
    last_end = -1
    nL = len(low_piv)

    for a in range(0, nL - min_touches + 1):
        for c in range(a + min_touches - 1, nL):
            i1 = low_piv[a]
            in_ = low_piv[c]
            if in_ - i1 < min_bars_apart * (min_touches - 1):
                continue
            if i1 <= last_end:
                continue

            y1 = float(low[i1])
            yn = float(low[in_])
            slope = (yn - y1) / float(in_ - i1)
            # Must rise enough over the whole span (reject flat bases)
            total_rise_pct = (yn - y1) / y1 * 100.0
            if slope <= 0 or (slope / y1 * 100.0) < flat_pct or total_rise_pct < 3.0:
                continue

            # Candidate bottom touches on the line, with distinct intervening rallies
            touch_idxs = [i1]
            ok_swings = True
            for m in range(a + 1, c):
                im = low_piv[m]
                prev = touch_idxs[-1]
                if im - prev < min_bars_apart:
                    continue
                ym = float(low[im])
                y_hat = _line_at(y1, i1, slope, im)
                if not _touch_ok(y_hat, ym, error_pct):
                    continue
                # higher low vs previous touch price (classical ascending)
                if ym < float(low[prev]) * 0.995:
                    continue
                if not _intervening_rally_ok(
                    high, prev, im, float(low[prev]), min_intervening_rally_pct
                ):
                    continue
                touch_idxs.append(im)

            # Always include last if it fits and has a real rally from prior touch
            prev = touch_idxs[-1]
            if in_ != prev:
                if in_ - prev < min_bars_apart:
                    ok_swings = False
                elif not _touch_ok(_line_at(y1, i1, slope, in_), yn, error_pct):
                    ok_swings = False
                elif yn < float(low[prev]) * 0.995:
                    ok_swings = False
                elif not _intervening_rally_ok(
                    high, prev, in_, float(low[prev]), min_intervening_rally_pct
                ):
                    ok_swings = False
                else:
                    touch_idxs.append(in_)
            if not ok_swings or len(touch_idxs) < min_touches:
                continue

            # Refit support through first/last accepted touches
            i1 = touch_idxs[0]
            in_ = touch_idxs[-1]
            y1 = float(low[i1])
            yn = float(low[in_])
            slope = (yn - y1) / float(in_ - i1)

            # Parallel return line: choose width so >=2 swing highs touch it,
            # preferring the smallest width that gets min_top_touches (avoids one spike)
            window_highs = [h for h in high_piv if i1 < h < in_]
            if len(window_highs) < min_top_touches:
                continue

            dists = sorted(
                float(high[h]) - _line_at(y1, i1, slope, h) for h in window_highs
            )
            dists = [d for d in dists if d > 0]
            if len(dists) < min_top_touches:
                continue

            best_width = None
            best_upper_idxs: List[int] = []
            # Try widths from each high's distance; pick tightest with enough touches
            for width in sorted(dists):
                upper_idxs = []
                for h in window_highs:
                    y_top = _line_at(y1, i1, slope, h) + width
                    if _touch_ok(y_top, float(high[h]), error_pct * 1.25):
                        upper_idxs.append(h)
                if len(upper_idxs) < min_top_touches:
                    continue
                # Distinct highs: intervening pullbacks between successive upper touches
                filtered = [upper_idxs[0]]
                for h in upper_idxs[1:]:
                    prev_h = filtered[-1]
                    if h - prev_h < min_bars_apart:
                        continue
                    resist_prev = _line_at(y1, i1, slope, prev_h) + width
                    if _intervening_pullback_ok(
                        low, prev_h, h, resist_prev, min_intervening_pullback_pct
                    ):
                        filtered.append(h)
                if len(filtered) >= min_top_touches:
                    best_width = width
                    best_upper_idxs = filtered
                    break  # smallest viable width

            if best_width is None:
                continue

            # Support integrity inside [first touch, last touch]
            viol = 0
            total = 0
            for i in range(i1, in_ + 1):
                total += 1
                sup = _line_at(y1, i1, slope, i)
                # allow small penetration; count meaningful closes below
                if float(close[i]) < sup * (1.0 - error_pct / 100.0):
                    viol += 1
            if total <= 0 or (viol / total) > max_support_violation_frac:
                continue

            touch_dates = [dates[i].strftime("%Y-%m-%d") for i in touch_idxs]
            touch_prices = [round(float(low[i]), 4) for i in touch_idxs]
            upper_dates = [dates[i].strftime("%Y-%m-%d") for i in best_upper_idxs]
            support_now = _line_at(y1, i1, slope, len(out) - 1)
            resist_now = support_now + best_width
            results.append(
                {
                    "start_date": dates[i1].strftime("%Y-%m-%d"),
                    "end_date": dates[in_].strftime("%Y-%m-%d"),
                    "bottom_touches": len(touch_idxs),
                    "top_touches": len(best_upper_idxs),
                    "touch_dates": "|".join(touch_dates),
                    "touch_prices": "|".join(str(p) for p in touch_prices),
                    "touch_indices": list(touch_idxs),
                    "upper_touch_dates": "|".join(upper_dates),
                    "support_x0": int(i1),
                    "support_y0": float(y1),
                    "support_slope": float(slope),
                    "channel_width": float(best_width),
                    "slope_pct_per_bar": round(slope / y1 * 100.0, 4),
                    "total_rise_pct": round((yn - y1) / y1 * 100.0, 2),
                    "channel_width_pct": round(best_width / y1 * 100.0, 2),
                    "support_violation_frac": round(viol / total, 3),
                    "support_last": round(float(support_now), 4),
                    "resist_last": round(float(resist_now), 4),
                    "close_last": round(float(close[-1]), 4),
                    "bars_span": int(in_ - i1),
                    "pivot_len": int(pivot_len),
                }
            )
            last_end = in_

    return results


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Find classical ascending channels (strict swing validation)"
    )
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument("--min-touches", type=int, default=3)
    ap.add_argument("--min-top-touches", type=int, default=2)
    ap.add_argument("--pivot-len", type=int, default=15)
    ap.add_argument("--error-pct", type=float, default=1.2)
    ap.add_argument("--flat-pct", type=float, default=0.04)
    ap.add_argument("--min-bars-apart", type=int, default=15)
    ap.add_argument("--min-rally-pct", type=float, default=4.0)
    ap.add_argument("--min-pullback-pct", type=float, default=3.0)
    ap.add_argument("--max-violation-frac", type=float, default=0.08)
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2025-11-26")
    ap.add_argument("--min-bars", type=int, default=180)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    ap.add_argument("--symbols", default="", help="Comma-separated override list")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = _pick_symbols(args.n_symbols, args.provider, args.timeframe, args.min_bars)
    if not symbols:
        logger.error("No symbols")
        return 1

    logger.info(
        "Loading %d symbols (classical rules: pivot=%d rally>=%.1f%% tops>=%d)",
        len(symbols),
        args.pivot_len,
        args.min_rally_pct,
        args.min_top_touches,
    )
    panels = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        use_cache=True,
        chunk_size=50,
    )

    rows: List[dict] = []
    for sym in symbols:
        df = panels.get(sym)
        if df is None:
            continue
        for ch in find_channels(
            df,
            pivot_len=args.pivot_len,
            min_touches=args.min_touches,
            min_top_touches=args.min_top_touches,
            error_pct=args.error_pct,
            flat_pct=args.flat_pct,
            min_bars_apart=args.min_bars_apart,
            min_intervening_rally_pct=args.min_rally_pct,
            min_intervening_pullback_pct=args.min_pullback_pct,
            max_support_violation_frac=args.max_violation_frac,
        ):
            rows.append({"stock": sym.upper(), **ch})

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = args.outdir / f"ascending_channel_classical_{stamp}.csv"

    if not rows:
        logger.warning("No classical ascending channels found under strict rules")
        pd.DataFrame(
            columns=[
                "stock",
                "start_date",
                "end_date",
                "bottom_touches",
                "top_touches",
                "touch_dates",
                "upper_touch_dates",
                "bars_span",
            ]
        ).to_csv(out_csv, index=False)
        print(f"\nNo matches. Empty report: {out_csv}")
        return 0

    res = pd.DataFrame(rows).sort_values(
        ["bottom_touches", "top_touches", "bars_span"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    # Drop in-memory geometry fields from the public CSV
    drop_cols = [
        c
        for c in (
            "touch_indices",
            "support_x0",
            "support_y0",
            "support_slope",
            "channel_width",
            "pivot_len",
        )
        if c in res.columns
    ]
    res.drop(columns=drop_cols).to_csv(out_csv, index=False)

    elapsed = time.perf_counter() - t0
    logger.info("Found %d channels across %d symbols in %.1fs", len(res), len(panels), elapsed)
    logger.info("Wrote %s", out_csv)

    cols = [
        "stock",
        "start_date",
        "end_date",
        "bottom_touches",
        "top_touches",
        "touch_dates",
        "upper_touch_dates",
        "bars_span",
        "channel_width_pct",
        "support_violation_frac",
    ]
    print("\nClassical ascending channel matches:")
    print(res[cols].head(30).to_string(index=False))
    print(f"\nFull report: {out_csv}")
    # Explicitly note JNJ rejection if present in universe
    if "JNJ" in {s.upper() for s in symbols}:
        jnj = res[res["stock"] == "JNJ"]
        if jnj.empty:
            print("\nJNJ: rejected under classical rules (as expected).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
