"""Evening Doji Star (bearish) on session OHLC.

Textbook three-candle reversal at the top of a rise: large bullish, gapped-up
doji (buyer exhaustion), strong bearish close into the first candle's body.

``require_gap=False`` still forbids a gap *down* into the doji (open2 >= close1)
because US stocks often print a star without a true overnight gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from utils.research.session_volume_delta import SessionDelta

DOJI_BODY_FRAC = 0.20
LARGE_BODY_FRAC = 0.50
BEAR_BODY_FRAC = 0.50


@dataclass(frozen=True)
class Candle:
    session_date: date
    open: float
    high: float
    low: float
    close: float


def _range(c: Candle) -> float:
    return float(c.high) - float(c.low)


def _body(c: Candle) -> float:
    return abs(float(c.close) - float(c.open))


def is_doji(c: Candle, *, body_frac_max: float = DOJI_BODY_FRAC) -> bool:
    rng = _range(c)
    if rng <= 0:
        return False
    return (_body(c) / rng) <= float(body_frac_max)


def is_large_bull(c: Candle, *, body_frac_min: float = LARGE_BODY_FRAC) -> bool:
    rng = _range(c)
    if rng <= 0 or float(c.close) <= float(c.open):
        return False
    return (_body(c) / rng) >= float(body_frac_min)


def is_strong_bear(c: Candle, *, body_frac_min: float = BEAR_BODY_FRAC) -> bool:
    rng = _range(c)
    if rng <= 0 or float(c.close) >= float(c.open):
        return False
    return (_body(c) / rng) >= float(body_frac_min)


def closes_into_prior_body(first: Candle, third: Candle) -> bool:
    """Third close is at or below the midpoint of the first bullish body, still in range."""
    lo = min(float(first.open), float(first.close))
    hi = max(float(first.open), float(first.close))
    mid = (float(first.open) + float(first.close)) / 2.0
    cl = float(third.close)
    return cl <= mid + 1e-12 and cl >= lo - 1e-12 and hi >= lo


def candle_from_session(sess: SessionDelta) -> Candle:
    return Candle(
        session_date=sess.session_date,
        open=float(sess.open),
        high=float(sess.high),
        low=float(sess.low),
        close=float(sess.close),
    )


def is_evening_doji_star(
    first: Candle,
    star: Candle,
    third: Candle,
    *,
    require_gap: bool = True,
    doji_body_frac: float = DOJI_BODY_FRAC,
    large_body_frac: float = LARGE_BODY_FRAC,
    bear_body_frac: float = BEAR_BODY_FRAC,
) -> bool:
    if not is_large_bull(first, body_frac_min=large_body_frac):
        return False
    if not is_doji(star, body_frac_max=doji_body_frac):
        return False
    if require_gap:
        if float(star.open) <= float(first.close) + 1e-12:
            return False
    elif float(star.open) < float(first.close) - 1e-12:
        return False
    if not is_strong_bear(third, body_frac_min=bear_body_frac):
        return False
    return closes_into_prior_body(first, third)


def find_evening_doji_star(
    sessions: Sequence[SessionDelta],
    *,
    fill_day: date,
    require_gap: bool = True,
    until_day: Optional[date] = None,
) -> Optional[date]:
    """Return candle-3 session date if a star completes after the fill session.

    Candle 1 may be the fill day (last-15m mid is known with that close). Candle 3
    must be strictly after ``fill_day`` so we do not exit on the fill session.
    ``until_day`` (inclusive) caps the search at the original sell session.
    """
    candles = [candle_from_session(s) for s in sessions]
    n = len(candles)
    for i in range(0, n - 2):
        c1, c2, c3 = candles[i], candles[i + 1], candles[i + 2]
        if c1.session_date < fill_day:
            continue
        if c3.session_date <= fill_day:
            continue
        if until_day is not None and c3.session_date > until_day:
            continue
        if is_evening_doji_star(c1, c2, c3, require_gap=require_gap):
            return c3.session_date
    return None
