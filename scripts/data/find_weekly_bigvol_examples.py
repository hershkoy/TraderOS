#!/usr/bin/env python3
"""
Find historical Weekly BigVol + TTM Squeeze examples in TimescaleDB.

Stage 0 example hunter (pandas, not Backtrader):
  Condition A: weekly volume ignition
  Condition B: TTM momentum zero-cross after squeeze-on lookback
  Condition C: trend filter on confirmation week
  Full setup: A then B+C within max_delay_weeks

Usage (Windows CMD):
  venv\\Scripts\\activate && python scripts\\data\\find_weekly_bigvol_examples.py
  venv\\Scripts\\activate && python scripts\\data\\find_weekly_bigvol_examples.py --symbols AAPL MSFT NVDA
  venv\\Scripts\\activate && python scripts\\data\\find_weekly_bigvol_examples.py --max-symbols 300
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from indicators.ttm_squeeze import calculate_squeeze_momentum
from utils.db.timescaledb_client import get_timescaledb_client
from utils.scanning.squeeze import to_weekly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("find_weekly_bigvol_examples")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)


def _format_elapsed(seconds: float) -> str:
    """Human-readable duration for timing logs."""
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

# Defaults aligned with WeeklyBigVolTTMSqueeze params
VOL_SHORT_LOOKBACK = 12
VOL_WEIGHT_DECAY = 0.85
VOL_ZSCORE_MIN = 2.5
VOL_ROBUST_Z_MIN = 2.0
VOL_ROC_MIN = 0.5
BODY_POS_MIN = 0.5
LENGTH_KC = 20
SQUEEZE_LOOKBACK = 10
MAX_DELAY_WEEKS = 26
MA10_PERIOD = 10
MA30_PERIOD = 30
MAX_EXTENDED_PCT = 0.25
MOM_SLOPE_MIN = 0.0

DEFAULT_LIQUID = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "AVGO", "CRM", "ORCL", "ADBE", "COST", "WMT", "JPM", "BAC",
    "XOM", "CVX", "UNH", "JNJ", "V", "MA", "HD", "PG", "KO", "PEP", "DIS",
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMCI", "PLTR", "COIN",
    "SHOP", "UBER", "ABNB", "SNOW", "NET", "CRWD", "PANW", "MU", "QCOM", "TXN",
]


def _exp_weights(period: int, decay: float) -> np.ndarray:
    raw = np.array([decay ** (period - 1 - i) for i in range(period)], dtype=float)
    return raw / raw.sum()


def _rolling_weighted_mean_std(vol: pd.Series, period: int, decay: float) -> Tuple[pd.Series, pd.Series]:
    """Stats on prior `period` bars (excludes current), matching WeightedVolStats."""
    weights = _exp_weights(period, decay)
    values = vol.to_numpy(dtype=float)
    n = len(values)
    means = np.full(n, np.nan)
    stds = np.full(n, np.nan)
    for i in range(period, n):
        window = values[i - period : i]
        wmean = float(np.dot(weights, window))
        wvar = float(np.dot(weights, (window - wmean) ** 2))
        means[i] = wmean
        stds[i] = math.sqrt(wvar) if wvar > 0 else 0.0
    return pd.Series(means, index=vol.index), pd.Series(stds, index=vol.index)


def _rolling_median_mad(vol: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    """Median/MAD on prior `period` bars (excludes current), matching RobustVolStats."""
    values = vol.to_numpy(dtype=float)
    n = len(values)
    meds = np.full(n, np.nan)
    mads = np.full(n, np.nan)
    for i in range(period, n):
        window = values[i - period : i]
        med = float(np.median(window))
        mad = float(np.median(np.abs(window - med)))
        if mad < 1e-9:
            mad = 1e-9
        meds[i] = med
        mads[i] = mad
    return pd.Series(meds, index=vol.index), pd.Series(mads, index=vol.index)


def _volume_tests(i: int, vol: pd.Series, median: pd.Series, mad: pd.Series,
                  wmean: pd.Series, wstd: pd.Series) -> Tuple[List[str], dict]:
    passed: List[str] = []
    details: dict = {}
    v = float(vol.iloc[i])
    med = median.iloc[i]
    mad_v = mad.iloc[i]
    if pd.notna(med) and pd.notna(mad_v) and mad_v > 1e-9 and med > 0:
        robust_z = 0.6745 * (v - med) / mad_v
        details["robust_z"] = robust_z
        details["median"] = float(med)
        details["mad"] = float(mad_v)
        if robust_z >= VOL_ROBUST_Z_MIN:
            passed.append("robust_z")
    mean_s = wmean.iloc[i]
    std_s = wstd.iloc[i]
    if pd.notna(mean_s) and pd.notna(std_s) and std_s > 0 and mean_s > 0:
        weighted_z = (v - mean_s) / std_s
        details["weighted_z_short"] = weighted_z
        if weighted_z >= VOL_ZSCORE_MIN:
            passed.append("weighted_z_short")
    if pd.notna(med) and med > 0:
        roc = (v / med) - 1.0
        details["roc"] = roc
        if roc >= VOL_ROC_MIN:
            passed.append("roc")
    return passed, details


def find_ignitions(df_w: pd.DataFrame) -> List[dict]:
    if len(df_w) < MA30_PERIOD + VOL_SHORT_LOOKBACK + 2:
        return []

    vol = df_w["volume"]
    median, mad = _rolling_median_mad(vol, VOL_SHORT_LOOKBACK)
    wmean, wstd = _rolling_weighted_mean_std(vol, VOL_SHORT_LOOKBACK, VOL_WEIGHT_DECAY)
    ma10 = df_w["close"].rolling(MA10_PERIOD).mean()
    ma30 = df_w["close"].rolling(MA30_PERIOD).mean()

    events: List[dict] = []
    start = max(MA30_PERIOD, VOL_SHORT_LOOKBACK)
    for i in range(start, len(df_w)):
        passed, details = _volume_tests(i, vol, median, mad, wmean, wstd)
        if len(passed) < 2:
            continue

        hi = float(df_w["high"].iloc[i])
        lo = float(df_w["low"].iloc[i])
        cl = float(df_w["close"].iloc[i])
        rng = hi - lo
        if rng <= 0:
            continue
        body_pos = (cl - lo) / rng
        if body_pos < BODY_POS_MIN:
            continue

        ma10_v = ma10.iloc[i]
        ma30_v = ma30.iloc[i]
        if pd.isna(ma30_v):
            continue
        if pd.notna(ma10_v):
            if cl <= float(ma10_v) and cl <= float(ma30_v):
                continue
        elif cl <= float(ma30_v):
            continue

        events.append({
            "ignition_week": df_w.index[i],
            "ignition_idx": i,
            "close": cl,
            "vol": float(vol.iloc[i]),
            "body_pos": body_pos,
            "ma10": float(ma10_v) if pd.notna(ma10_v) else None,
            "ma30": float(ma30_v),
            "vol_tests": "|".join(passed),
            "robust_z": details.get("robust_z"),
            "weighted_z_short": details.get("weighted_z_short"),
            "roc": details.get("roc"),
            "median_vol": details.get("median"),
        })
    return events


def _is_squeeze_on(mom: pd.Series, idx: int) -> bool:
    try:
        return abs(float(mom.iloc[idx])) < 0.5
    except (TypeError, ValueError, IndexError):
        return False


def find_ttm_crosses(df_w: pd.DataFrame) -> List[dict]:
    mom = calculate_squeeze_momentum(df_w, lengthKC=LENGTH_KC, use_logging=False)
    if mom is None or mom.empty:
        return []
    crosses: List[dict] = []
    for i in range(1, len(mom)):
        prev_v = mom.iloc[i - 1]
        curr_v = mom.iloc[i]
        if pd.isna(prev_v) or pd.isna(curr_v):
            continue
        if not (float(prev_v) <= 0 and float(curr_v) > 0):
            continue
        if float(curr_v) <= MOM_SLOPE_MIN:
            continue
        squeeze_found = False
        for j in range(1, SQUEEZE_LOOKBACK + 1):
            lookback = i - j
            if lookback < 0:
                break
            if _is_squeeze_on(mom, lookback):
                squeeze_found = True
                break
        if not squeeze_found:
            continue
        crosses.append({
            "confirm_week": mom.index[i],
            "confirm_idx": i,
            "mom_prev": float(prev_v),
            "mom": float(curr_v),
        })
    return crosses


def trend_ok(df_w: pd.DataFrame, idx: int) -> Tuple[bool, dict]:
    if idx < 1 or idx >= len(df_w):
        return False, {}
    ma30 = df_w["close"].rolling(MA30_PERIOD).mean()
    cl = float(df_w["close"].iloc[idx])
    ma_curr = ma30.iloc[idx]
    ma_prev = ma30.iloc[idx - 1]
    if pd.isna(ma_curr) or pd.isna(ma_prev):
        return False, {}
    ma_curr_f = float(ma_curr)
    ma_prev_f = float(ma_prev)
    if cl <= ma_curr_f:
        return False, {}
    if ma_curr_f <= ma_prev_f:
        return False, {}
    max_ext = ma_curr_f * (1.0 + MAX_EXTENDED_PCT)
    if cl > max_ext:
        return False, {}
    return True, {"close": cl, "ma30": ma_curr_f, "ma30_prev": ma_prev_f}


def forward_return(df_w: pd.DataFrame, idx: int, weeks: int) -> Optional[float]:
    j = idx + weeks
    if j >= len(df_w):
        return None
    c0 = float(df_w["close"].iloc[idx])
    c1 = float(df_w["close"].iloc[j])
    if c0 <= 0:
        return None
    return (c1 / c0) - 1.0


def analyze_symbol(symbol: str, raw_df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    df_w = to_weekly(raw_df)
    if df_w.empty or len(df_w) < MA30_PERIOD + LENGTH_KC + 5:
        return [], []

    ignitions = find_ignitions(df_w)
    crosses = find_ttm_crosses(df_w)
    cross_by_idx = {c["confirm_idx"]: c for c in crosses}

    ignition_rows: List[dict] = []
    for ig in ignitions:
        row = {"symbol": symbol, **ig}
        row["fwd_4w"] = forward_return(df_w, ig["ignition_idx"], 4)
        row["fwd_13w"] = forward_return(df_w, ig["ignition_idx"], 13)
        ignition_rows.append(row)

    full_rows: List[dict] = []
    for ig in ignitions:
        i0 = ig["ignition_idx"]
        for di in range(1, MAX_DELAY_WEEKS + 1):
            j = i0 + di
            if j not in cross_by_idx:
                continue
            ok, trend = trend_ok(df_w, j)
            if not ok:
                continue
            cross = cross_by_idx[j]
            full_rows.append({
                "symbol": symbol,
                "ignition_week": ig["ignition_week"],
                "confirm_week": cross["confirm_week"],
                "delay_weeks": di,
                "ignition_vol": ig["vol"],
                "body_pos": ig["body_pos"],
                "vol_tests": ig["vol_tests"],
                "robust_z": ig.get("robust_z"),
                "mom_prev": cross["mom_prev"],
                "mom": cross["mom"],
                "confirm_close": trend["close"],
                "ma30": trend["ma30"],
                "fwd_4w": forward_return(df_w, j, 4),
                "fwd_13w": forward_return(df_w, j, 13),
            })
            break  # first valid confirm after this ignition
    return ignition_rows, full_rows


def load_symbols(
    symbols: Iterable[str],
    timeframe: str,
    provider: str,
    start: datetime,
    end: datetime,
    *,
    use_cache: bool = True,
    chunk_size: int = 50,
    workers: int = 4,
) -> Dict[str, pd.DataFrame]:
    """Load OHLCV via batch SQL + optional parquet cache (see utils.data.ohlcv_loader)."""
    from utils.data.ohlcv_loader import load_ohlcv_many

    return load_ohlcv_many(
        list(symbols),
        timeframe=timeframe,
        provider=provider,
        start=start,
        end=end,
        use_cache=use_cache,
        chunk_size=chunk_size,
        workers=workers,
    )


def resolve_symbol_list(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        return [s.upper() for s in args.symbols]
    if args.max_symbols and args.max_symbols > 0 and not args.liquid_only:
        client = get_timescaledb_client()
        if not client.ensure_connection():
            raise RuntimeError("Cannot connect to TimescaleDB")
        # Prefer symbols that actually have daily bars (not ticker_universe alone)
        q = """
            SELECT symbol
            FROM (
                SELECT DISTINCT symbol
                FROM market_data
                WHERE provider = %s AND timeframe = %s
            ) s
            ORDER BY symbol
            LIMIT %s OFFSET %s
        """
        rows = client.execute_query(
            q, (args.provider, args.timeframe, args.max_symbols, args.start_index)
        )
        client.disconnect()
        if not rows:
            return []
        # execute_query may return dicts or tuples depending on cursor factory
        out = []
        for r in rows:
            if isinstance(r, dict):
                out.append(str(r["symbol"]).upper())
            else:
                out.append(str(r[0]).upper())
        return out
    return list(DEFAULT_LIQUID)


def main() -> int:
    parser = argparse.ArgumentParser(description="Find Weekly BigVol + TTM Squeeze examples")
    parser.add_argument("--symbols", nargs="+", help="Explicit symbol list")
    parser.add_argument("--liquid-only", action="store_true", default=True,
                        help="Use default liquid subset (default)")
    parser.add_argument("--no-liquid-only", action="store_false", dest="liquid_only")
    parser.add_argument("--max-symbols", type=int, default=0,
                        help="If >0 and not liquid-only, take first N symbols from DB")
    parser.add_argument("--start-index", type=int, default=0,
                        help="OFFSET into DB symbol list (with --max-symbols)")
    parser.add_argument("--provider", default="ALPACA")
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--fromdate", default="2018-01-01")
    parser.add_argument("--todate", default="2025-11-26")
    parser.add_argument("--outdir", default="reports/examples")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable parquet OHLCV cache under data/cache/ohlcv")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Symbols per batch SQL query (default 50)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel batch fetch workers (default 4; 1=serial)")
    args = parser.parse_args()

    t_run0 = time.perf_counter()
    symbols = resolve_symbol_list(args)
    start = datetime.strptime(args.fromdate, "%Y-%m-%d")
    end = datetime.strptime(args.todate, "%Y-%m-%d")
    logger.info("Hunting examples for %d symbols (%s -> %s)", len(symbols), args.fromdate, args.todate)

    t_load0 = time.perf_counter()
    data = load_symbols(
        symbols,
        args.timeframe,
        args.provider,
        start,
        end,
        use_cache=not args.no_cache,
        chunk_size=args.chunk_size,
        workers=args.workers,
    )
    load_s = time.perf_counter() - t_load0
    logger.info(
        "Loaded OHLCV for %d / %d symbols in %s",
        len(data),
        len(symbols),
        _format_elapsed(load_s),
    )

    all_ignitions: List[dict] = []
    all_full: List[dict] = []
    t_analyze0 = time.perf_counter()
    for sym, df in sorted(data.items()):
        try:
            ign, full = analyze_symbol(sym, df)
            all_ignitions.extend(ign)
            all_full.extend(full)
            logger.info("%s: ignitions=%d full_setups=%d bars=%d", sym, len(ign), len(full), len(df))
        except Exception as exc:
            logger.exception("%s: failed (%s)", sym, exc)
    analyze_s = time.perf_counter() - t_analyze0
    logger.info("Analyzed %d symbols in %s", len(data), _format_elapsed(analyze_s))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ign_path = outdir / f"weekly_bigvol_ignitions_{ts}.csv"
    full_path = outdir / f"weekly_bigvol_full_setups_{ts}.csv"

    ign_df = pd.DataFrame(all_ignitions)
    full_df = pd.DataFrame(all_full)
    if not ign_df.empty:
        ign_df = ign_df.sort_values(["symbol", "ignition_week"])
        for col in ("ignition_week",):
            ign_df[col] = pd.to_datetime(ign_df[col]).dt.strftime("%Y-%m-%d")
    if not full_df.empty:
        full_df = full_df.sort_values(["symbol", "ignition_week", "confirm_week"])
        for col in ("ignition_week", "confirm_week"):
            full_df[col] = pd.to_datetime(full_df[col]).dt.strftime("%Y-%m-%d")

    ign_df.to_csv(ign_path, index=False)
    full_df.to_csv(full_path, index=False)

    total_s = time.perf_counter() - t_run0
    per_symbol = (total_s / len(symbols)) if symbols else 0.0

    print("")
    print("=" * 60)
    print("WEEKLY BIGVOL + TTM SQUEEZE - EXAMPLE HUNT SUMMARY")
    print("=" * 60)
    print(f"Symbols requested : {len(symbols)}")
    print(f"Symbols with data : {len(data)}")
    print(f"Ignition events   : {len(ign_df)}")
    print(f"Full setups (A+B+C): {len(full_df)}")
    print(f"Ignitions CSV     : {ign_path}")
    print(f"Full setups CSV   : {full_path}")
    print(f"Timing            : total={_format_elapsed(total_s)} "
          f"(load={_format_elapsed(load_s)}, analyze={_format_elapsed(analyze_s)}, "
          f"avg/symbol={_format_elapsed(per_symbol)})")
    logger.info(
        "Run timing: total=%s load=%s analyze=%s avg/symbol=%s",
        _format_elapsed(total_s),
        _format_elapsed(load_s),
        _format_elapsed(analyze_s),
        _format_elapsed(per_symbol),
    )

    if not full_df.empty:
        print("")
        print("Full setups (all):")
        cols = ["symbol", "ignition_week", "confirm_week", "delay_weeks", "mom", "fwd_4w", "fwd_13w"]
        print(full_df[cols].to_string(index=False))
        print("")
        print("Fwd return (confirm week) median 4w / 13w:")
        print(f"  4w : {full_df['fwd_4w'].median(skipna=True):.2%}" if full_df["fwd_4w"].notna().any() else "  4w : n/a")
        print(f"  13w: {full_df['fwd_13w'].median(skipna=True):.2%}" if full_df["fwd_13w"].notna().any() else "  13w: n/a")
    elif not ign_df.empty:
        print("")
        print("No full setups. Sample ignitions:")
        cols = ["symbol", "ignition_week", "vol", "body_pos", "vol_tests", "fwd_4w"]
        print(ign_df[cols].head(20).to_string(index=False))
    else:
        print("No ignition events found on this subset.")

    if not ign_df.empty:
        print("")
        print("Ignitions per symbol (top):")
        print(ign_df.groupby("symbol").size().sort_values(ascending=False).head(15).to_string())

    return 0


if __name__ == "__main__":
    # Avoid emoji in logs on Windows charmap
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    raise SystemExit(main())
