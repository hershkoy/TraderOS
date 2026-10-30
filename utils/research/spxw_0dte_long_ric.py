"""Jason Rutschel 0DTE SPXW long reverse-iron-condor claim-check.

Transcript rules (Theta Profits interview):
  - Skip Monday
  - VIX gaps down vs prior close, then continues down >=0.5% from the open
    by 11:00 ET (1% preferred; cutoff 11:00)
  - Buy a ~$6 / 30-wide put debit spread + ~$3 / 30-wide call debit spread
  - Hold to cash settle (automated); discretionary 100% TP not modeled

VIX intraday: Theta index OHLC is STANDARD-only on this plan. Default trigger
uses Yahoo ^VIX 1h bars (about 730d lookback). Optional ``eod_low`` mode uses
the session low vs open as a timing-optimistic diagnostic for longer windows.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.research.spxw_0dte_iron_fly import (
    DEFAULT_MULT,
    FILL_MID,
    FILL_NATURAL,
    OPT_SYMBOL,
    _coerce_chain,
    _penny,
    _valid_quotes,
    chain_spot,
    compound_fixed_contracts,
    fetch_chain,
    summarize_pnls,
)

LOG = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

CLAIMED: Dict[str, Any] = {
    "source": "Jason Rutschel / Theta Profits interview (0DTE long RIC)",
    "window": "about 4+ years",
    "n": 338,
    "win_rate": 0.49,
    "avg_win": 1400.0,
    "avg_loss": 739.0,
    "cagr": 0.367,
    "max_dd": 0.15,
    "structure": "long 30-wide put~$6 + call~$3, VIX gap+down 0.5%, skip Mon, settle",
}

VIX_MODE_HOURLY = "hourly"
VIX_MODE_EOD_LOW = "eod_low"

DEFAULT_WING = 30.0
DEFAULT_PUT_DEBIT = 6.0
DEFAULT_CALL_DEBIT = 3.0
DEFAULT_VIX_DROP = 0.005  # 0.5% from open
DEFAULT_CUTOFF = dtime(11, 0)
DEFAULT_EOD_ENTRY = dtime(10, 30)
# Tue-Fri
DEFAULT_WEEKDAYS = (1, 2, 3, 4)
DEBIT_TOL = 3.0  # max distance from target debit to accept a spread


def load_yahoo_vix_hourly(start: date, end: date) -> pd.DataFrame:
    """Yahoo ^VIX 1h bars in America/New_York. Limited to ~730d by Yahoo."""
    import yfinance as yf

    # Yahoo rejects 1h requests whose start is older than ~730 calendar days.
    floor = date.today() - timedelta(days=720)
    req_start = max(start, floor)
    if req_start > end:
        LOG.warning("Yahoo VIX 1h window empty after 730d clamp (%s > %s)", req_start, end)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "date"])
    if req_start > start:
        LOG.info("Clamped Yahoo VIX 1h start %s -> %s (730d limit)", start, req_start)

    raw = yf.download(
        "^VIX",
        start=req_start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1h",
        progress=False,
        auto_adjust=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    idx = pd.DatetimeIndex(pd.to_datetime(raw.index))
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(ET)
    out = pd.DataFrame(
        {
            "timestamp": idx,
            "open": pd.to_numeric(raw["Open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(raw["High"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(raw["Low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw["Close"], errors="coerce").to_numpy(),
        }
    )
    out["date"] = out["timestamp"].dt.date
    return out.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)


def _session_open_et(ts: pd.Timestamp) -> bool:
    t = ts.tz_convert(ET).time() if ts.tzinfo else ts.time()
    return dtime(9, 30) <= t <= dtime(16, 0)


def find_vix_trigger_hourly(
    day: date,
    *,
    prior_close: float,
    session_open: float,
    hourly: pd.DataFrame,
    min_drop: float = DEFAULT_VIX_DROP,
    cutoff: dtime = DEFAULT_CUTOFF,
) -> Tuple[Optional[dtime], str]:
    """First RTH hour where VIX is still gapped down and down >=min_drop from open."""
    if not (math.isfinite(prior_close) and math.isfinite(session_open)):
        return None, "no_vix"
    if float(session_open) >= float(prior_close):
        return None, "no_gap"
    if hourly is None or hourly.empty:
        return None, "no_hourly"
    day_bars = hourly.loc[hourly["date"] == day].copy()
    if day_bars.empty:
        return None, "no_hourly"
    day_bars = day_bars.loc[day_bars["timestamp"].map(_session_open_et)].copy()
    if day_bars.empty:
        return None, "no_rth"
    target = float(session_open) * (1.0 - float(min_drop))
    for _, row in day_bars.sort_values("timestamp").iterrows():
        ts: pd.Timestamp = row["timestamp"]
        t = ts.tz_convert(ET).time() if ts.tzinfo else ts.time()
        if t > cutoff:
            break
        # Prefer low (touched) then close of the hour
        px = float(row["low"]) if pd.notna(row["low"]) else float(row["close"])
        if px <= target:
            # Snap chain pull to :00 or :30 inside the hour, not past cutoff
            minute = 0 if t.minute < 30 else 30
            entry = dtime(t.hour, minute)
            if entry < dtime(9, 45):
                entry = dtime(9, 45)
            if entry > cutoff:
                entry = cutoff
            return entry, "ok"
    return None, "no_drop"


def find_vix_trigger_eod_low(
    *,
    prior_close: float,
    session_open: float,
    session_low: float,
    min_drop: float = DEFAULT_VIX_DROP,
    entry: dtime = DEFAULT_EOD_ENTRY,
) -> Tuple[Optional[dtime], str]:
    """Optimistic proxy: gap + session low already 0.5% under open → enter at fixed clock."""
    if not all(math.isfinite(x) for x in (prior_close, session_open, session_low)):
        return None, "no_vix"
    if float(session_open) >= float(prior_close):
        return None, "no_gap"
    target = float(session_open) * (1.0 - float(min_drop))
    if float(session_low) > target:
        return None, "no_drop"
    return entry, "ok"


def _spread_debit(
    long_row: pd.Series,
    short_row: pd.Series,
    *,
    fill: str,
) -> float:
    if fill == FILL_NATURAL:
        return float(long_row["ask"]) - float(short_row["bid"])
    return 0.5 * (
        float(long_row["bid"]) + float(long_row["ask"]) - float(short_row["bid"]) - float(short_row["ask"])
    )


def _nearest_debit_spread(
    side: pd.DataFrame,
    *,
    right: str,
    wing: float,
    target_debit: float,
    fill: str,
    spot: float,
) -> Optional[Dict[str, Any]]:
    """Pick long/short strikes of width ``wing`` whose debit is nearest target."""
    if side.empty:
        return None
    work = side.sort_values("strike").reset_index(drop=True)
    strikes = work["strike"].to_numpy(dtype=float)
    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    for i, k in enumerate(strikes):
        if right == "P":
            long_k = float(k)
            short_k = long_k - float(wing)
            # keep put spreads near / below spot
            if long_k > spot + 50:
                continue
        else:
            long_k = float(k)
            short_k = long_k + float(wing)
            if long_k < spot - 50:
                continue
        long_hits = work.loc[(work["strike"] - long_k).abs() < 0.01]
        short_hits = work.loc[(work["strike"] - short_k).abs() < 0.01]
        if long_hits.empty or short_hits.empty:
            continue
        long_row = long_hits.iloc[0]
        short_row = short_hits.iloc[0]
        debit = _spread_debit(long_row, short_row, fill=fill)
        if debit <= 0 or debit >= float(wing):
            continue
        score = abs(debit - float(target_debit))
        if score < best_score:
            best_score = score
            best = {
                "long_strike": long_k,
                "short_strike": short_k,
                "debit": float(debit),
                "score": float(score),
                "long_bid": float(long_row["bid"]),
                "long_ask": float(long_row["ask"]),
                "short_bid": float(short_row["bid"]),
                "short_ask": float(short_row["ask"]),
            }
    if best is None or best["score"] > DEBIT_TOL:
        return None
    return best


def select_long_ric(
    chain: pd.DataFrame,
    *,
    wing: float = DEFAULT_WING,
    put_debit: float = DEFAULT_PUT_DEBIT,
    call_debit: float = DEFAULT_CALL_DEBIT,
    clock: dtime = DEFAULT_EOD_ENTRY,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Pick strikes by mid debit targets; report mid and natural package debits."""
    df = _valid_quotes(_coerce_chain(chain, clock))
    if df.empty or "right" not in df.columns:
        return None, "no_chain"
    spot = chain_spot(chain, clock)
    if spot is None:
        return None, "no_spot"
    puts = df.loc[df["right"] == "P"].copy()
    calls = df.loc[df["right"] == "C"].copy()
    put_sp = _nearest_debit_spread(
        puts,
        right="P",
        wing=wing,
        target_debit=put_debit,
        fill=FILL_MID,
        spot=float(spot),
    )
    call_sp = _nearest_debit_spread(
        calls,
        right="C",
        wing=wing,
        target_debit=call_debit,
        fill=FILL_MID,
        spot=float(spot),
    )
    if put_sp is None:
        return None, "no_put_spread"
    if call_sp is None:
        return None, "no_call_spread"

    def _nat(spread: Dict[str, Any]) -> float:
        return float(spread["long_ask"]) - float(spread["short_bid"])

    debit_mid = float(put_sp["debit"]) + float(call_sp["debit"])
    debit_nat = _nat(put_sp) + _nat(call_sp)
    if debit_mid <= 0:
        return None, "no_debit"
    return (
        {
            "spot": float(spot),
            "wing": float(wing),
            "put": put_sp,
            "call": call_sp,
            "debit_mid": debit_mid,
            "debit_natural": debit_nat,
            "put_debit_mid": float(put_sp["debit"]),
            "call_debit_mid": float(call_sp["debit"]),
            "put_debit_natural": _nat(put_sp),
            "call_debit_natural": _nat(call_sp),
        },
        "ok",
    )


def settle_long_ric(
    *,
    settle: float,
    put_long: float,
    put_short: float,
    call_long: float,
    call_short: float,
    debit: float,
    mult: float = DEFAULT_MULT,
) -> Dict[str, float]:
    wing_put = abs(float(put_long) - float(put_short))
    wing_call = abs(float(call_short) - float(call_long))
    put_val = max(0.0, min(wing_put, float(put_long) - float(settle)))
    call_val = max(0.0, min(wing_call, float(settle) - float(call_long)))
    pnl_pts = put_val + call_val - float(debit)
    return {
        "put_value": put_val,
        "call_value": call_val,
        "pnl_points": pnl_pts,
        "pnl": pnl_pts * float(mult),
    }


def simulate_day(
    chain: pd.DataFrame,
    *,
    settle: float,
    fill: str = FILL_MID,
    wing: float = DEFAULT_WING,
    put_debit: float = DEFAULT_PUT_DEBIT,
    call_debit: float = DEFAULT_CALL_DEBIT,
    clock: dtime = DEFAULT_EOD_ENTRY,
) -> Tuple[Optional[Dict[str, Any]], str]:
    picked, reason = select_long_ric(
        chain,
        wing=wing,
        put_debit=put_debit,
        call_debit=call_debit,
        clock=clock,
    )
    if picked is None:
        return None, reason
    debit = (
        float(picked["debit_natural"])
        if fill == FILL_NATURAL
        else float(picked["debit_mid"])
    )
    if debit <= 0:
        return None, "no_debit"
    settled = settle_long_ric(
        settle=float(settle),
        put_long=float(picked["put"]["long_strike"]),
        put_short=float(picked["put"]["short_strike"]),
        call_long=float(picked["call"]["long_strike"]),
        call_short=float(picked["call"]["short_strike"]),
        debit=debit,
    )
    put_debit_x = (
        picked["put_debit_natural"] if fill == FILL_NATURAL else picked["put_debit_mid"]
    )
    call_debit_x = (
        picked["call_debit_natural"] if fill == FILL_NATURAL else picked["call_debit_mid"]
    )
    return (
        {
            "fill": fill,
            "spot": picked["spot"],
            "settle": float(settle),
            "wing": picked["wing"],
            "put_long": picked["put"]["long_strike"],
            "put_short": picked["put"]["short_strike"],
            "call_long": picked["call"]["long_strike"],
            "call_short": picked["call"]["short_strike"],
            "put_debit": _penny(put_debit_x),
            "call_debit": _penny(call_debit_x),
            "debit": _penny(debit),
            "debit_mid": _penny(picked["debit_mid"]),
            "debit_natural": _penny(picked["debit_natural"]),
            "max_loss": _penny(debit * DEFAULT_MULT),
            "put_value": settled["put_value"],
            "call_value": settled["call_value"],
            "pnl": settled["pnl"],
            "won": settled["pnl"] > 0,
        },
        "ok",
    )


def eligible_sessions(
    expirations: Sequence[date],
    *,
    start: date,
    end: date,
    weekdays: Sequence[int] = DEFAULT_WEEKDAYS,
) -> List[date]:
    out: List[date] = []
    for d in expirations:
        if d < start or d > end:
            continue
        if int(d.weekday()) not in set(int(x) for x in weekdays):
            continue
        out.append(d)
    return out


def claimed_one_lot_pnl() -> float:
    n = int(CLAIMED["n"])
    wr = float(CLAIMED["win_rate"])
    winners = int(round(n * wr))
    losers = n - winners
    return winners * float(CLAIMED["avg_win"]) - losers * float(CLAIMED["avg_loss"])
