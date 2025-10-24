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
    Look for:
      (1)   LL → HH → HL
      (2)   LL → HH → HH → HL

    with any number of ordinary bars between the swings (we're matching by *swing order*, not bar adjacency).

    Monday rule: only return a match if last_price >= HL price (HL not broken on the new week).
    """
    # Indices of swing lows/highs by label
    # We’ll walk the sequence once from oldest → newest
    n = len(swings)
    for i in range(n):
        # start on a confirmed LL
        if not (swings[i].kind == "L" and swings[i].label == "LL"):
            continue

        # we need at least one HH after this LL
        hh1 = None
        hh2 = None
        hl_confirm = None

        # search forward
        for j in range(i + 1, n):
            s = swings[j]

            if hh1 is None and s.kind == "H" and s.label == "HH":
                hh1 = j
                continue

            # optional second HH (pattern 2)
            if hh1 is not None and hh2 is None and s.kind == "H" and s.label == "HH":
                hh2 = j
                continue

            # confirmation HL *after* at least one HH
            if hh1 is not None and s.kind == "L" and s.label == "HL":
                hl_confirm = j
                break

        if hh1 is not None and hl_confirm is not None:
            ll = swings[i]
            hl = swings[hl_confirm]

            # Monday check: don't accept if current price already broke the HL
            if last_price < hl.price:
                continue

            hhs = [swings[hh1]]
            if hh2 is not None:
                hhs.append(swings[hh2])

            return Match(
                symbol=symbol,
                ll_date=ll.ts, ll_price=ll.price,
                hh_dates=[h.ts for h in hhs],
                hh_prices=[h.price for h in hhs],
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
        from utils.timescaledb_client import get_timescaledb_client
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
        ts_series = pd.to_datetime(df["ts"])
        if ts_series.dt.tz is not None:
            ts_series = ts_series.dt.tz_convert(None)
        
        d = pd.DataFrame(
            {
                "open": pd.to_numeric(df["open"], errors="coerce"),
                "high": pd.to_numeric(df["high"], errors="coerce"),
                "low": pd.to_numeric(df["low"], errors="coerce"),
                "close": pd.to_numeric(df["close"], errors="coerce"),
                "volume": pd.to_numeric(df["volume"], errors="coerce"),
            },
            index=ts_series,
        ).dropna()
        out[sym] = d

    client.disconnect()
    return out
