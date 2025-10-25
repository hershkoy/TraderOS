# utils/squeeze_scanner.py
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, List

# -----------------------------
# Data helpers (same style as HL scanner)
# -----------------------------

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime")
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)
    wk = df.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return wk

def latest_price(raw_df: pd.DataFrame) -> float:
    return float(raw_df["close"].iloc[-1])

# -----------------------------
# LazyBear Squeeze "val" (histogram) reimplementation
# We only need val and the 0-line cross up (red->green).
# Pine:
# val = linreg(close - avg(avg(highest(high,L), lowest(low,L)), sma(close,L)), L, 0)
# -----------------------------

def rolling_linreg_yhat(x: pd.Series, length: int) -> pd.Series:
    """
    Rolling least-squares fit (degree=1) evaluated at the most-recent x.
    Equivalent to Pine's linreg(x, length, 0).
    """
    # index positions 0..length-1 for regression
    idx = np.arange(length, dtype=float)

    def _fit(arr):
        y = np.asarray(arr, dtype=float)
        # polyfit on [0..length-1], return predicted y at last x (length-1)
        m, b = np.polyfit(idx, y, 1)
        return m * (length - 1) + b

    return x.rolling(length, min_periods=length).apply(_fit, raw=False)

def squeeze_val(df_w: pd.DataFrame, lengthKC: int = 20) -> pd.Series:
    src = df_w["close"]
    # component = close - avg( avg(highest(high,L), lowest(low,L)), sma(close,L) )
    highest_h = df_w["high"].rolling(lengthKC, min_periods=lengthKC).max()
    lowest_l  = df_w["low"].rolling(lengthKC,  min_periods=lengthKC).min()
    mid_hl    = (highest_h + lowest_l) / 2.0
    sma_c     = src.rolling(lengthKC, min_periods=lengthKC).mean()
    baseline  = (mid_hl + sma_c) / 2.0
    x = src - baseline
    return rolling_linreg_yhat(x, lengthKC)

# -----------------------------
# Match & scan
# -----------------------------

@dataclass
class SqueezeCross:
    symbol: str
    cross_date: pd.Timestamp   # week (W-FRI) that CLOSED with val > 0 while prev <= 0
    cross_value: float         # val on that weekly close
    prev_value: float          # val on the prior week
    last_price: float          # latest price (for display)
    timeframe: str             # underlying raw timeframe used to build weeklies

def _find_latest_zero_cross_up(val: pd.Series) -> Optional[tuple]:
    """
    Return (date, prev_val, curr_val) of the most recent WEEKLY close that crossed above 0.
    We require val.shift(1) <= 0 and val > 0 at the same index.
    """
    if val is None or val.empty:
        return None
    cond = (val > 0) & (val.shift(1) <= 0)
    idx = np.where(cond.values)[0]
    if len(idx) == 0:
        return None
    i = idx[-1]  # most recent cross-up
    ts = val.index[i]
    return ts, float(val.iloc[i-1]), float(val.iloc[i])

def scan_symbol_for_squeeze(
    symbol: str,
    raw_df: pd.DataFrame,
    lengthKC: int = 20,
    confirm_on_close: bool = True,
    timeframe_label: str = "1d",
) -> Optional[SqueezeCross]:
    """
    - Resamples to weekly (W-FRI)
    - Computes LazyBear 'val'
    - Triggers when the *last closed* weekly bar crossed the 0 line upward
      (prev <= 0, curr > 0). If confirm_on_close=True (default), we only use
      completed weekly bars (like TradingView on 1W chart).
    """
    if raw_df is None or raw_df.empty:
        return None

    df_w = to_weekly(raw_df)
    val = squeeze_val(df_w, lengthKC=lengthKC)

    # Use only confirmed weekly bar (last row is the most recent Friday close).
    # If you ever want to allow an in-progress week check, you can add an option
    # to look at the latest partial calculation; default keeps it "confirmed".
    cross = _find_latest_zero_cross_up(val)
    if not cross:
        return None

    cross_ts, prev_v, curr_v = cross
    lp = latest_price(raw_df)

    # Ensure the crossing is the last closed weekly bar (confirmation)
    # i.e., cross_ts must be the last index in df_w (latest Friday).
    if confirm_on_close:
        if cross_ts != df_w.index[-1]:
            # Most recent CROSS is not the last closed week → ignore (older signal)
            return None

    return SqueezeCross(
        symbol=symbol,
        cross_date=cross_ts,
        cross_value=curr_v,
        prev_value=prev_v,
        last_price=lp,
        timeframe=timeframe_label,
    )

def scan_universe(
    ohlcv_by_symbol: Dict[str, pd.DataFrame],
    lengthKC: int = 20,
    confirm_on_close: bool = True,
    timeframe_label: str = "1d",
) -> List[SqueezeCross]:
    results: List[SqueezeCross] = []
    for symbol, df in ohlcv_by_symbol.items():
        try:
            m = scan_symbol_for_squeeze(
                symbol, df, lengthKC=lengthKC,
                confirm_on_close=confirm_on_close,
                timeframe_label=timeframe_label
            )
            if m:
                results.append(m)
        except Exception:
            pass
    return results

# (Optional) TimescaleDB loader in the same style as your HL scanner
def load_from_timescaledb(symbols: Iterable[str], timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
    try:
        from utils.timescaledb_client import get_timescaledb_client
    except Exception:
        from timescaledb_client import get_timescaledb_client  # type: ignore

    client = get_timescaledb_client()
    if not client.ensure_connection():
        raise RuntimeError("Cannot connect to TimescaleDB")

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = client.get_market_data(sym, timeframe)
        if df is None or df.empty:
            continue

        # Normalize to OHLCV indexed by datetime (db returns 'ts')
        ts_series = pd.to_datetime(df["ts"], errors="coerce")
        if ts_series.dt.tz is not None:
            ts_series = ts_series.dt.tz_convert(None)

        valid = ~ts_series.isna()
        df = df[valid]
        ts_series = ts_series[valid]

        d = pd.DataFrame(
            {
                "open": [float(x) for x in df["open"]],
                "high": [float(x) for x in df["high"]],
                "low":  [float(x) for x in df["low"]],
                "close":[float(x) for x in df["close"]],
                "volume":[float(x) for x in df["volume"]],
            },
            index=ts_series,
        ).dropna()

        out[sym] = d

    client.disconnect()
    return out
