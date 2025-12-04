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
    import logging
    logger = logging.getLogger(__name__)
    
    src = df_w["close"]
    logger.debug(f"Calculating squeeze val with lengthKC={lengthKC}")
    
    # component = close - avg( avg(highest(high,L), lowest(low,L)), sma(close,L) )
    highest_h = df_w["high"].rolling(lengthKC, min_periods=lengthKC).max()
    lowest_l  = df_w["low"].rolling(lengthKC,  min_periods=lengthKC).min()
    mid_hl    = (highest_h + lowest_l) / 2.0
    sma_c     = src.rolling(lengthKC, min_periods=lengthKC).mean()
    baseline  = (mid_hl + sma_c) / 2.0
    x = src - baseline
    
    # Log key intermediate values for the last few periods
    if len(df_w) >= 3:
        logger.debug(f"Last 3 periods - Close: {src.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Highest H: {highest_h.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Lowest L: {lowest_l.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Mid HL: {mid_hl.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - SMA C: {sma_c.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Baseline: {baseline.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - X (close-baseline): {x.tail(3).tolist()}")
    
    val = rolling_linreg_yhat(x, lengthKC)
    
    # Log the final squeeze values
    valid_vals = val.dropna()
    if len(valid_vals) > 0:
        logger.debug(f"Final squeeze val range: {valid_vals.min():.4f} to {valid_vals.max():.4f}")
        logger.debug(f"Last 3 squeeze vals: {valid_vals.tail(3).tolist()}")
    
    return val

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
    import logging
    logger = logging.getLogger(__name__)
    
    if raw_df is None or raw_df.empty:
        logger.warning(f"{symbol}: No data available")
        return None

    logger.info(f"{symbol}: Starting squeeze analysis with lengthKC={lengthKC}")
    
    # Convert to weekly data
    df_w = to_weekly(raw_df)
    logger.info(f"{symbol}: Weekly data shape: {df_w.shape}, date range: {df_w.index[0]} to {df_w.index[-1]}")
    
    # Calculate squeeze values
    val = squeeze_val(df_w, lengthKC=lengthKC)
    
    # Log key squeeze values
    valid_vals = val.dropna()
    if len(valid_vals) > 0:
        logger.info(f"{symbol}: Squeeze val range: {valid_vals.min():.4f} to {valid_vals.max():.4f}")
        logger.info(f"{symbol}: Last 5 squeeze vals: {valid_vals.tail(5).tolist()}")
        
        # Check for any zero crosses in the data
        zero_crosses = ((val > 0) & (val.shift(1) <= 0)).sum()
        logger.info(f"{symbol}: Total zero-cross up events in data: {zero_crosses}")
    else:
        logger.warning(f"{symbol}: No valid squeeze values calculated")
        return None

    # Use only confirmed weekly bar (last row is the most recent Friday close).
    # If you ever want to allow an in-progress week check, you can add an option
    # to look at the latest partial calculation; default keeps it "confirmed".
    cross = _find_latest_zero_cross_up(val)
    if not cross:
        logger.info(f"{symbol}: No zero-cross up detected")
        return None

    cross_ts, prev_v, curr_v = cross
    lp = latest_price(raw_df)
    
    logger.info(f"{symbol}: Zero-cross detected at {cross_ts}")
    logger.info(f"{symbol}: Cross values - prev: {prev_v:.4f}, curr: {curr_v:.4f}")
    logger.info(f"{symbol}: Latest price: {lp:.2f}")

    # Ensure the crossing is the last closed weekly bar (confirmation)
    # i.e., cross_ts must be the last index in df_w (latest Friday).
    if confirm_on_close:
        if cross_ts != df_w.index[-1]:
            # Most recent CROSS is not the last closed week → ignore (older signal)
            logger.info(f"{symbol}: Cross not on latest closed week ({cross_ts} != {df_w.index[-1]}) - ignoring")
            return None
        else:
            logger.info(f"{symbol}: Cross confirmed on latest closed week")

    logger.info(f"{symbol}: SQUEEZE PATTERN DETECTED!")
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
        from ..db.timescaledb_client import get_timescaledb_client
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
