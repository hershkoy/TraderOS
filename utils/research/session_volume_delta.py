"""Buy/sell volume split from candle geometry (Volume Delta, not POC profile).

Same close-in-range split as ``VolumeDelta`` in weekly_bigvol: buy estimate is
``volume * (close - low) / (high - low)``. Aggregate 15m bars to an RTH session.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

MIN_TICK = 1e-4


@dataclass(frozen=True)
class BarDelta:
    buy_volume: float
    sell_volume: float
    buy_pct: float
    sell_pct: float
    delta: float


@dataclass(frozen=True)
class SessionDelta:
    session_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float
    sell_volume: float
    buy_pct: float
    sell_pct: float
    last_bar_ts: Optional[datetime]


def bar_volume_delta(
    high: float,
    low: float,
    close: float,
    volume: float,
    *,
    min_tick: float = MIN_TICK,
) -> BarDelta:
    """Close-location split of one bar's volume into buy vs sell."""
    vol = max(float(volume), 0.0)
    hi = float(high)
    lo = float(low)
    cl = float(close)
    rng = max(hi - lo, float(min_tick))
    positional = max(cl - lo, 0.0)
    buy = vol * (positional / rng)
    if buy > vol:
        buy = vol
    sell = vol - buy
    total = buy + sell
    if total > 0:
        buy_pct = (buy / total) * 100.0
        sell_pct = 100.0 - buy_pct
    else:
        buy_pct = 0.0
        sell_pct = 0.0
    return BarDelta(
        buy_volume=float(buy),
        sell_volume=float(sell),
        buy_pct=float(buy_pct),
        sell_pct=float(sell_pct),
        delta=float(buy - sell),
    )


def _px(bar: Dict[str, Any], key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def session_volume_delta(
    bars: Sequence[Dict[str, Any]],
    *,
    min_tick: float = MIN_TICK,
) -> Optional[SessionDelta]:
    """Sum 15m buy/sell over one RTH session; OHLC is first/max/min/last."""
    if not bars:
        return None
    ordered = list(bars)
    first_o = None
    last_c = None
    last_ts = None
    hi = None
    lo = None
    vol_sum = 0.0
    buy_sum = 0.0
    sell_sum = 0.0
    sess_day: Optional[date] = None
    for bar in ordered:
        o = _px(bar, "open")
        h = _px(bar, "high")
        l = _px(bar, "low")
        c = _px(bar, "close")
        v = _px(bar, "volume")
        if v is None:
            v = 0.0
        ts = bar.get("ts")
        if sess_day is None and ts is not None:
            if isinstance(ts, datetime):
                sess_day = ts.date()
            else:
                try:
                    sess_day = ts.date()  # type: ignore[union-attr]
                except AttributeError:
                    sess_day = None
        if first_o is None and o is not None:
            first_o = o
        if ts is not None and isinstance(ts, datetime):
            last_ts = ts
        if c is not None:
            last_c = c
        if h is not None:
            hi = h if hi is None else max(hi, h)
        if l is not None:
            lo = l if lo is None else min(lo, l)
        if h is not None and l is not None and c is not None:
            d = bar_volume_delta(h, l, c, v, min_tick=min_tick)
            buy_sum += d.buy_volume
            sell_sum += d.sell_volume
        vol_sum += max(v, 0.0)
    if first_o is None or last_c is None or hi is None or lo is None or sess_day is None:
        return None
    total = buy_sum + sell_sum
    if total > 0:
        buy_pct = (buy_sum / total) * 100.0
        sell_pct = 100.0 - buy_pct
    else:
        buy_pct = 0.0
        sell_pct = 0.0
    return SessionDelta(
        session_date=sess_day,
        open=float(first_o),
        high=float(hi),
        low=float(lo),
        close=float(last_c),
        volume=float(vol_sum),
        buy_volume=float(buy_sum),
        sell_volume=float(sell_sum),
        buy_pct=float(buy_pct),
        sell_pct=float(sell_pct),
        last_bar_ts=last_ts if isinstance(last_ts, datetime) else None,
    )


def sessions_from_by_day(
    by_day: Dict[date, Sequence[Dict[str, Any]]],
    *,
    min_tick: float = MIN_TICK,
) -> List[SessionDelta]:
    """Chronological RTH sessions from ``index_rth_15m_by_session`` output."""
    out: List[SessionDelta] = []
    for day in sorted(by_day.keys()):
        sess = session_volume_delta(list(by_day.get(day) or []), min_tick=min_tick)
        if sess is not None:
            out.append(sess)
    return out


def consecutive_seller_sessions(
    sessions: Iterable[SessionDelta],
    *,
    fill_day: date,
    seller_pct_min: float = 55.0,
    n_needed: int = 2,
    until_day: Optional[date] = None,
) -> Optional[date]:
    """First date on which ``n_needed`` consecutive *post-fill* sessions are seller-heavy.

    Fill day is excluded (last-15m mid is contemporaneous with that session close).
    ``until_day`` (inclusive) caps the search at the original sell session.
    """
    after = [s for s in sessions if s.session_date > fill_day]
    if until_day is not None:
        after = [s for s in after if s.session_date <= until_day]
    if n_needed <= 0:
        return None
    run = 0
    for sess in after:
        if sess.sell_pct >= float(seller_pct_min):
            run += 1
            if run >= int(n_needed):
                return sess.session_date
        else:
            run = 0
    return None
