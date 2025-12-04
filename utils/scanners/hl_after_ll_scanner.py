from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict

# ------------------------------------------------------------
# Small data helpers
# ------------------------------------------------------------

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample any OHLCV DataFrame (indexed by timezone-aware or naive datetimes)
    to weekly bars that end on Friday (W-FRI). Keeps columns: open, high, low, close, volume.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by datetime")

    # If your DB returns tz-aware, drop tz to keep things simple
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert(None)

    wk = df.resample("W-FRI").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()

    return wk


def latest_price_for_monday_check(raw_df: pd.DataFrame) -> float:
    """
    A 'current price' proxy for Monday runs:
      - If you pass daily or intraday raw data for the *current* week,
        we use the very latest 'close' from raw_df.
      - If you only pass weeklies, this returns the last weekly close.
    """
    return float(raw_df["close"].iloc[-1])


# ------------------------------------------------------------
# Pivot detection with lb=1, rb=2 (same idea as Pine pivothigh/pivotlow)
# ------------------------------------------------------------

def pivots_lb1_rb2(high: pd.Series, low: pd.Series) -> pd.DataFrame:
    """
    Return a DataFrame with columns:
      ph: price at pivot high (NaN otherwise)
      pl: price at pivot low  (NaN otherwise)
      ph_idx / pl_idx: boolean masks (True where pivot confirmed)

    Definition (confirmed):
      pivot high at i  if high[i]  > high[i-1]  and high[i] >= high[i+1] and high[i] >= high[i+2]
      pivot low  at i  if low[i]   < low[i-1]   and low[i]  <= low[i+1]  and low[i]  <= low[i+2]

    Because rb=2, the detection for index i is only *knowable* at i+2.
    """
    h = high.values
    l = low.values

    n = len(high)
    ph_mask = np.zeros(n, dtype=bool)
    pl_mask = np.zeros(n, dtype=bool)

    for i in range(2, n - 2):  # we need i-1 and i+1, i+2 to exist
        # pivot high (strict on left, non-strict on right like many Pine scripts)
        if h[i] > h[i - 1] and h[i] >= h[i + 1] and h[i] >= h[i + 2]:
            ph_mask[i] = True

        # pivot low (strict on left, non-strict on right)
        if l[i] < l[i - 1] and l[i] <= l[i + 1] and l[i] <= l[i + 2]:
            pl_mask[i] = True

    out = pd.DataFrame(index=high.index)
    out["ph"] = np.where(ph_mask, high, np.nan)
    out["pl"] = np.where(pl_mask, low, np.nan)
    out["ph_idx"] = ph_mask
    out["pl_idx"] = pl_mask
    return out


# ------------------------------------------------------------
# Classify pivots as HH/HL/LH/LL
# ------------------------------------------------------------

@dataclass
class PivotPoint:
    ts: pd.Timestamp
    kind: str  # 'H' or 'L' for swing kind
    price: float
    label: str  # 'HH','HL','LH','LL'


def classify_pivots_weekly(df_w: pd.DataFrame) -> List[PivotPoint]:
    """
    Build an alternating sequence of confirmed swing highs/lows (with lb=1, rb=2),
    then label each swing vs the *previous same-type swing*:
      - For a swing High:  HH if price > previous High's price else LH
      - For a swing Low:   HL if price > previous Low's price  else LL
    """
    piv = pivots_lb1_rb2(df_w["high"], df_w["low"])

    # Build alternating swings in chronological order
    swings: List[PivotPoint] = []
    lastH: Optional[float] = None
    lastL: Optional[float] = None
    last_kind: Optional[str] = None  # to help alternate (H/L)

    for i, (ts, row) in enumerate(piv.iterrows()):
        if row["ph_idx"]:
            # optional alternation filter: if last kind was 'H', skip consecutive highs
            if last_kind == "H":
                # keep only the more extreme of consecutive highs
                if swings and row["ph"] > swings[-1].price:
                    swings[-1] = PivotPoint(ts, "H", float(row["ph"]), "TMP")
                continue
            # label vs last high
            label = "HH" if (lastH is not None and row["ph"] > lastH) else "LH"
            swings.append(PivotPoint(ts, "H", float(row["ph"]), label))
            lastH = float(row["ph"])
            last_kind = "H"

        if row["pl_idx"]:
            if last_kind == "L":
                if swings and row["pl"] < swings[-1].price:
                    swings[-1] = PivotPoint(ts, "L", float(row["pl"]), "TMP")
                continue
            label = "HL" if (lastL is not None and row["pl"] > lastL) else "LL"
            swings.append(PivotPoint(ts, "L", float(row["pl"]), label))
            lastL = float(row["pl"])
            last_kind = "L"

    # Clean any 'TMP' placeholders (shouldn’t occur with the simple alternation guards)
    for s in swings:
        if s.label == "TMP":
            s.label = "HH" if s.kind == "H" else "LL"

    return swings


# ------------------------------------------------------------
# Pattern scan: LL → HH → HL (optionally LL → HH → HH → HL)
# ------------------------------------------------------------

@dataclass
class Match:
    symbol: str
    ll_date: pd.Timestamp
    ll_price: float
    hh_dates: List[pd.Timestamp]
    hh_prices: List[float]
    hl_date: pd.Timestamp
    hl_price: float
    last_price: float  # for the Monday check


def find_ll_hh_hl(swings: List[PivotPoint], last_price: float, symbol: str) -> Optional[Match]:
    """
    Look for the most recent patterns by checking the last few pivots in sequence:
      (1)   LL→HH→HL (last 3 pivots)
      (2)   LL→HH→HH→HL (last 4 pivots)
      (3)   LL→(LH|HH)3→HL (flexible pattern: LL, up to 3 LH/HH, then HL)

    The pattern must be found in the most recent pivot sequence, not scattered throughout history.

    Monday rule: only return a match if last_price >= HL price (HL not broken on the new week).
    """
    if not swings:
        return None
    
    # Sort swings by date descending (most recent first)
    sorted_swings = sorted(swings, key=lambda x: x.ts, reverse=True)
    
    # Check for pattern 1: LL→HH→HL (last 3 pivots)
    if len(sorted_swings) >= 3:
        if (sorted_swings[0].kind == "L" and sorted_swings[0].label == "HL" and
            sorted_swings[1].kind == "H" and sorted_swings[1].label == "HH" and
            sorted_swings[2].kind == "L" and sorted_swings[2].label == "LL"):
            
            hl = sorted_swings[0]
            hh = sorted_swings[1]
            ll = sorted_swings[2]
            
            # Monday check: don't accept if current price already broke the HL
            if last_price >= hl.price:
                return Match(
                    symbol=symbol,
                    ll_date=ll.ts, ll_price=ll.price,
                    hh_dates=[hh.ts], hh_prices=[hh.price],
                    hl_date=hl.ts, hl_price=hl.price,
                    last_price=last_price
                )
    
    # Check for pattern 2: LL→HH→HH→HL (last 4 pivots)
    if len(sorted_swings) >= 4:
        if (sorted_swings[0].kind == "L" and sorted_swings[0].label == "HL" and
            sorted_swings[1].kind == "H" and sorted_swings[1].label == "HH" and
            sorted_swings[2].kind == "H" and sorted_swings[2].label == "HH" and
            sorted_swings[3].kind == "L" and sorted_swings[3].label == "LL"):
            
            hl = sorted_swings[0]
            hh1 = sorted_swings[1]
            hh2 = sorted_swings[2]
            ll = sorted_swings[3]
            
            # Monday check: don't accept if current price already broke the HL
            if last_price >= hl.price:
                return Match(
                    symbol=symbol,
                    ll_date=ll.ts, ll_price=ll.price,
                    hh_dates=[hh1.ts, hh2.ts], hh_prices=[hh1.price, hh2.price],
                    hl_date=hl.ts, hl_price=hl.price,
                    last_price=last_price
                )
    
    # Check for pattern 3: LL→(LH|HH)3→HL (flexible pattern)
    # Look for this pattern in the most recent pivot sequence only
    # Check patterns of different lengths starting from the most recent pivots
    for pattern_length in [3, 4, 5]:  # Check patterns of different lengths
        if len(sorted_swings) < pattern_length:
            continue
        
        # Extract the most recent pattern sequence
        pattern_sequence = sorted_swings[:pattern_length]
        
        # The first pivot in the sequence must be HL (most recent in time)
        if pattern_sequence[0].kind != "L" or pattern_sequence[0].label != "HL":
            continue
        
        # The last pivot in the sequence must be LL (oldest in time)
        if pattern_sequence[-1].kind != "L" or pattern_sequence[-1].label != "LL":
            continue
        
        # All middle pivots must be LH or HH
        middle_pivots = pattern_sequence[1:-1]
        valid_middle = all(
            (pivot.kind == "H" and pivot.label in ["HH", "LH"]) or
            (pivot.kind == "L" and pivot.label in ["LH", "HL"])
            for pivot in middle_pivots
        )
        
        if not valid_middle:
            continue
        
        # Check that we have at least one LH or HH in the middle
        has_lh_or_hh = any(
            pivot.label in ["LH", "HH"] for pivot in middle_pivots
        )
        
        if not has_lh_or_hh:
            continue
        
        # Pattern found! Extract the components
        hl = pattern_sequence[0]  # Most recent (first in sorted list)
        ll = pattern_sequence[-1]  # Oldest (last in sorted list)
        
        # Monday check: don't accept if current price already broke the HL
        if last_price >= hl.price:
            return Match(
                symbol=symbol,
                ll_date=ll.ts, ll_price=ll.price,
                hh_dates=[pivot.ts for pivot in middle_pivots], 
                hh_prices=[pivot.price for pivot in middle_pivots],
                hl_date=hl.ts, hl_price=hl.price,
                last_price=last_price
            )
    
    return None


# ------------------------------------------------------------
# Public scanner API
# ------------------------------------------------------------

def scan_symbol_for_setup(
    symbol: str,
    raw_df: pd.DataFrame,
) -> Optional[Match]:
    """
    raw_df: datetime-indexed OHLCV (any timeframe). We resample to weekly internally.
    Returns a Match or None.
    """
    if raw_df is None or raw_df.empty:
        return None

    df_w = to_weekly(raw_df)
    swings = classify_pivots_weekly(df_w)

    # “Monday” current-price check uses the most recent close from raw_df
    last_px = latest_price_for_monday_check(raw_df)

    return find_ll_hh_hl(swings, last_px, symbol)


def scan_universe(
    ohlcv_by_symbol: Dict[str, pd.DataFrame]
) -> List[Match]:
    """
    Pass a dict {symbol: raw_df}. The raw_df may be daily or hourly; we’ll resample to weeklies.
    """
    results: List[Match] = []
    for symbol, df in ohlcv_by_symbol.items():
        try:
            m = scan_symbol_for_setup(symbol, df)
            if m:
                results.append(m)
        except Exception:
            # keep the scanner robust; skip bad symbols silently or log if you prefer
            pass
    return results


# ------------------------------------------------------------
# (Optional) TimescaleDB wiring — if you want to pull directly from your DB
# ------------------------------------------------------------

def load_from_timescaledb(symbols: Iterable[str], timeframe: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Example loader using your TimescaleDB client. Returns {symbol: df} with datetime index + ohlcv.
    """
    try:
        from ..db.timescaledb_client import get_timescaledb_client
    except Exception:
        # fall back if pathing differs; edit as needed
        from timescaledb_client import get_timescaledb_client  # type: ignore

    client = get_timescaledb_client()
    if not client.ensure_connection():  # connect once; caller can close later
        raise RuntimeError("Cannot connect to TimescaleDB")

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = client.get_market_data(sym, timeframe)
        if df is None or df.empty:
            continue

        # Normalize to OHLCV indexed by datetime
        # Convert timestamp to datetime index
        ts_series = pd.to_datetime(df["ts"], errors="coerce")
        if ts_series.dt.tz is not None:
            ts_series = ts_series.dt.tz_convert(None)
        
        # Remove rows with invalid timestamps
        valid_timestamps = ~ts_series.isna()
        ts_series = ts_series[valid_timestamps]
        df_clean = df[valid_timestamps]
        
        # Convert Decimal objects to float using direct conversion
        d = pd.DataFrame(
            {
                "open": [float(x) for x in df_clean["open"]],
                "high": [float(x) for x in df_clean["high"]],
                "low": [float(x) for x in df_clean["low"]],
                "close": [float(x) for x in df_clean["close"]],
                "volume": [float(x) for x in df_clean["volume"]],
            },
            index=ts_series,
        )
        
        d = d.dropna()
        out[sym] = d

    client.disconnect()
    return out
