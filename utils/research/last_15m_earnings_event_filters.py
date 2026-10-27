"""Earnings / event skips on the last-15m H2 book.

Pre-registered overlays (occupancy not re-walked):

1. Flatten before a known print if it falls inside the next N sessions
   (1 / 5 / 10). Same-day AMC / next-session BMO that would exit at the
   entry mid are treated as entry skips.
2. Tight entry skip: AMC on the fill session or BMO next session.
3. Skip when overnight expected move from prior-session ATM IV exceeds the
   ATR k=2 hard-stop (missing IV keeps).
4. Biotech / pharma sleeve off via SIC 2830-2836 or sector label.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.realistic_purchaser import as_et, bar_mid
from utils.scanning.channel_touch_bought import (
    ATR_STOP_MULT,
    STOP_PCT_CEIL,
    STOP_PCT_FLOOR,
)

FLATTEN_SESSIONS: Tuple[int, ...] = (1, 5, 10)
BIOTECH_SIC_LO = 2830
BIOTECH_SIC_HI = 2836
BIOTECH_SECTOR_TOKENS: Tuple[str, ...] = (
    "BIOTECH",
    "BIOTECHNOLOGY",
    "PHARMA",
    "PHARMACEUTICAL",
    "PHARMACEUTICALS",
    "DRUG",
)
EXIT_EARNINGS_FLATTEN = "earnings_flatten"
EXIT_EARNINGS_SKIP = "earnings_entry_skip"


@dataclass(frozen=True)
class EarningsEvent:
    earn_date: date
    amc: bool


@dataclass
class FlattenResult:
    used: bool
    reason: str
    gain_pct: float
    sell_px: Optional[float] = None
    flatten_session: Optional[date] = None
    sessions_ahead: Optional[int] = None
    skipped: bool = False


def parse_session_date(raw: Any) -> Optional[date]:
    """Calendar session date. Date-only strings stay the calendar day (no UTC shift)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    if isinstance(raw, date) and not isinstance(raw, datetime):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            # YYYY-MM-DD or YYYY-MM-DD HH:MM... — date part is the session key.
            try:
                return date.fromisoformat(text[:10])
            except ValueError:
                pass
    try:
        t = pd.Timestamp(raw)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    if t.tzinfo is None:
        # Naive timestamps with a time component are treated as ET wall clocks
        # for buy_time/sell_time already stored as ET strings; pure midnights
        # stay the calendar date.
        if t.hour == 0 and t.minute == 0 and t.second == 0:
            return t.date()
        return t.date()
    return t.tz_convert("America/New_York").date()


def stop_pct_for_row(row: Mapping[str, Any]) -> float:
    """ATR k=2 distance as % of entry, clamped 1.5%-6%."""
    try:
        atr_pct = float(row.get("atr_1d_pct") if row.get("atr_1d_pct") is not None else row.get("atr_pct"))
    except (TypeError, ValueError):
        atr_pct = float("nan")
    if atr_pct == atr_pct and atr_pct > 0:
        raw = float(ATR_STOP_MULT) * atr_pct
        return float(max(100.0 * STOP_PCT_FLOOR, min(100.0 * STOP_PCT_CEIL, raw)))
    return float(100.0 * STOP_PCT_CEIL)


def overnight_em_pct(atm_iv: Any) -> Optional[float]:
    """1-session expected move % from annualized ATM IV (prior session)."""
    try:
        iv = float(atm_iv)
    except (TypeError, ValueError):
        return None
    if iv != iv or iv <= 0:
        return None
    return float(100.0 * iv * math.sqrt(1.0 / 252.0))


def is_biotech_sector(label: Any) -> bool:
    text = str(label or "").strip().upper()
    if not text:
        return False
    return any(tok in text for tok in BIOTECH_SECTOR_TOKENS)


def is_biotech_sic(sic: Any) -> bool:
    try:
        code = int(str(sic).strip()[:4])
    except (TypeError, ValueError):
        return False
    return BIOTECH_SIC_LO <= code <= BIOTECH_SIC_HI


def next_earnings_after_fill(
    events: Sequence[EarningsEvent],
    buy_day: date,
) -> Optional[EarningsEvent]:
    """Next print that can hit an open last-15m position.

    AMC on ``buy_day`` counts (report after the 15:45 fill). BMO on ``buy_day``
    already printed before the fill, so it is ignored.
    """
    for ev in sorted(events, key=lambda e: (e.earn_date, not e.amc)):
        if ev.earn_date < buy_day:
            continue
        if ev.earn_date == buy_day and not ev.amc:
            continue
        return ev
    return None


def flatten_session_for_event(ev: EarningsEvent) -> date:
    """Last RTH session whose 15:45 mid is still before the print."""
    if ev.amc:
        return ev.earn_date
    return ev.earn_date - timedelta(days=1)


def count_sessions_ahead(
    buy_day: date,
    flatten_day: date,
    session_days: Sequence[date],
) -> Optional[int]:
    """RTH sessions from buy to flatten inclusive of buy? Exclusive.

    ``0`` means flatten session is the fill session (AMC tonight).
    ``1`` means the next RTH session after buy is the flatten day.
    """
    if flatten_day < buy_day:
        return None
    if flatten_day == buy_day:
        return 0
    days = sorted({d for d in session_days if buy_day < d <= flatten_day})
    if not days and flatten_day > buy_day:
        # Fallback: calendar business days when 15m index is missing.
        return int(np.busday_count(buy_day, flatten_day))
    return int(len(days))


def last_rth_mid(
    by_day: Mapping[date, Sequence[dict]],
    session_day: date,
) -> Optional[Tuple[float, datetime]]:
    bars = list(by_day.get(session_day) or [])
    if not bars:
        return None
    bars = sorted(bars, key=lambda b: as_et(b["ts"]))
    last = bars[-1]
    try:
        hi = float(last["high"])
        lo = float(last["low"])
    except (TypeError, ValueError, KeyError):
        return None
    if hi != hi or lo != lo:
        return None
    ts = as_et(last["ts"])
    return float(bar_mid(hi, lo)), ts


def apply_flatten_before_earnings(
    row: Mapping[str, Any],
    events: Sequence[EarningsEvent],
    by_day: Mapping[date, Sequence[dict]],
    *,
    max_sessions: int,
) -> FlattenResult:
    """Exit last RTH mid before a print inside ``max_sessions``, else keep."""
    entry_px = float(row["buy_price"])
    orig_px = float(row["sell_price"])
    stored = row.get("gain_pct")
    try:
        orig_gain = float(stored) if stored is not None and stored == stored else (orig_px / entry_px - 1.0) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        orig_gain = (orig_px / entry_px - 1.0) * 100.0
    buy_day = parse_session_date(row.get("buy_time") or row.get("buy_date"))
    sell_day = parse_session_date(row.get("sell_time") or row.get("sell_date"))
    if buy_day is None:
        return FlattenResult(used=False, reason="no_buy_day", gain_pct=orig_gain)

    ev = next_earnings_after_fill(events, buy_day)
    if ev is None:
        return FlattenResult(used=False, reason="no_earnings", gain_pct=orig_gain)

    flat_day = flatten_session_for_event(ev)
    # BMO: earn_date - 1 calendar may be a weekend; snap to last available RTH <= that.
    if (not ev.amc) and flat_day not in by_day:
        prior = sorted(d for d in by_day.keys() if d < ev.earn_date)
        if prior:
            flat_day = prior[-1]

    session_days = sorted(by_day.keys()) if by_day else []
    ahead = count_sessions_ahead(buy_day, flat_day, session_days)
    if ahead is None:
        return FlattenResult(used=False, reason="bad_flatten_day", gain_pct=orig_gain)
    if ahead > int(max_sessions):
        return FlattenResult(
            used=False,
            reason="outside_window",
            gain_pct=orig_gain,
            flatten_session=flat_day,
            sessions_ahead=ahead,
        )

    if sell_day is not None and sell_day < flat_day:
        return FlattenResult(
            used=False,
            reason="already_out",
            gain_pct=orig_gain,
            flatten_session=flat_day,
            sessions_ahead=ahead,
        )

    # Same-session AMC: last mid is the entry print — do not hold the report.
    if ahead == 0:
        return FlattenResult(
            used=True,
            reason=EXIT_EARNINGS_SKIP,
            gain_pct=orig_gain,
            flatten_session=flat_day,
            sessions_ahead=ahead,
            skipped=True,
        )

    mid = last_rth_mid(by_day, flat_day)
    if mid is None:
        return FlattenResult(
            used=False,
            reason="no_15m_mid",
            gain_pct=orig_gain,
            flatten_session=flat_day,
            sessions_ahead=ahead,
        )
    exit_px, exit_ts = mid
    sell_ts = None
    raw_sell = row.get("sell_time") or row.get("sell_date")
    if raw_sell is not None and not (isinstance(raw_sell, float) and pd.isna(raw_sell)):
        try:
            sell_ts = as_et(raw_sell)
        except (TypeError, ValueError):
            sell_ts = None
    if sell_ts is not None and exit_ts >= sell_ts:
        return FlattenResult(
            used=False,
            reason="already_out",
            gain_pct=orig_gain,
            sell_px=orig_px,
            flatten_session=flat_day,
            sessions_ahead=ahead,
        )
    gain = (exit_px / entry_px - 1.0) * 100.0
    return FlattenResult(
        used=True,
        reason=EXIT_EARNINGS_FLATTEN,
        gain_pct=float(gain),
        sell_px=float(exit_px),
        flatten_session=flat_day,
        sessions_ahead=ahead,
        skipped=False,
    )


def entry_skip_earnings_mask(
    trades: pd.DataFrame,
    earnings_map: Mapping[str, Sequence[EarningsEvent]],
) -> pd.Series:
    """True = keep. Skip AMC on fill session or BMO on the next calendar session."""
    keep = []
    for _, row in trades.iterrows():
        sym = str(row["stock"]).upper()
        buy_day = parse_session_date(row.get("buy_time") or row.get("buy_date"))
        if buy_day is None:
            keep.append(True)
            continue
        events = list(earnings_map.get(sym) or [])
        skip = False
        for ev in events:
            if ev.earn_date == buy_day and ev.amc:
                skip = True
                break
            if (not ev.amc) and ev.earn_date > buy_day:
                # next session BMO: earn_date is the morning after buy_day
                gap = int(np.busday_count(buy_day, ev.earn_date))
                if gap == 1:
                    skip = True
                    break
        keep.append(not skip)
    return pd.Series(keep, index=trades.index, dtype=bool)


def em_gt_stop_keep_mask(
    trades: pd.DataFrame,
    *,
    iv_col: str = "name_atm_iv",
) -> pd.Series:
    """True = keep. Missing IV keeps. Skip when overnight EM% > ATR stop%."""
    keep = []
    for _, row in trades.iterrows():
        em = overnight_em_pct(row.get(iv_col))
        if em is None:
            keep.append(True)
            continue
        stop = stop_pct_for_row(row)
        keep.append(bool(em <= stop))
    return pd.Series(keep, index=trades.index, dtype=bool)


def biotech_keep_mask(
    trades: pd.DataFrame,
    *,
    sector_by_symbol: Mapping[str, str],
    sic_by_symbol: Mapping[str, str],
) -> pd.Series:
    """True = keep. Drop biotech/pharma by sector label or SIC 2830-2836."""
    keep = []
    for _, row in trades.iterrows():
        sym = str(row["stock"]).upper()
        if is_biotech_sector(sector_by_symbol.get(sym)):
            keep.append(False)
            continue
        if is_biotech_sic(sic_by_symbol.get(sym)):
            keep.append(False)
            continue
        keep.append(True)
    return pd.Series(keep, index=trades.index, dtype=bool)


def events_from_yahoo_frame(frame: Any) -> List[EarningsEvent]:
    """Parse yfinance get_earnings_dates frame into causal reported events."""
    if frame is None or getattr(frame, "empty", True):
        return []
    out: List[EarningsEvent] = []
    for ts, row in frame.iterrows():
        reported = row.get("Reported EPS") if hasattr(row, "get") else None
        if reported is None:
            continue
        try:
            if reported != reported:  # NaN
                continue
        except Exception:
            pass
        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("America/New_York")
        else:
            t = t.tz_convert("America/New_York")
        amc = int(t.hour) >= 16
        out.append(EarningsEvent(earn_date=t.date(), amc=amc))
    out.sort(key=lambda e: (e.earn_date, not e.amc))
    # Unique by date preferring AMC if both somehow appear.
    by_day: Dict[date, EarningsEvent] = {}
    for ev in out:
        prev = by_day.get(ev.earn_date)
        if prev is None or (ev.amc and not prev.amc):
            by_day[ev.earn_date] = ev
    return [by_day[d] for d in sorted(by_day)]


__all__ = [
    "BIOTECH_SECTOR_TOKENS",
    "BIOTECH_SIC_HI",
    "BIOTECH_SIC_LO",
    "EXIT_EARNINGS_FLATTEN",
    "EXIT_EARNINGS_SKIP",
    "EarningsEvent",
    "FLATTEN_SESSIONS",
    "FlattenResult",
    "apply_flatten_before_earnings",
    "biotech_keep_mask",
    "count_sessions_ahead",
    "em_gt_stop_keep_mask",
    "entry_skip_earnings_mask",
    "events_from_yahoo_frame",
    "flatten_session_for_event",
    "is_biotech_sector",
    "is_biotech_sic",
    "last_rth_mid",
    "next_earnings_after_fill",
    "overnight_em_pct",
    "parse_session_date",
    "stop_pct_for_row",
]
