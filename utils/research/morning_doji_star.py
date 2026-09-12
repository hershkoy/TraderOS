"""Morning Doji Star (bullish) plus hammer / dragonfly / bullish engulfing.

Textbook three-candle reversal at the bottom of a decline: large bearish,
gapped-down doji, strong bullish close into the first candle's body.

``require_gap=False`` still forbids a gap *up* into the doji (open2 <= close1)
because US stocks often print a star without a true overnight gap.
"""
from __future__ import annotations

from datetime import date
from typing import Optional, Sequence

from utils.research.evening_doji_star import (
    BEAR_BODY_FRAC,
    DOJI_BODY_FRAC,
    LARGE_BODY_FRAC,
    Candle,
    _body,
    _range,
    candle_from_session,
    is_doji,
)
from utils.research.session_volume_delta import SessionDelta

HAMMER_WICK_MULT = 2.0
HAMMER_UPPER_FRAC = 0.25
DRAGONFLY_UPPER_FRAC = 0.10


def is_large_bear(c: Candle, *, body_frac_min: float = LARGE_BODY_FRAC) -> bool:
    rng = _range(c)
    if rng <= 0 or float(c.close) >= float(c.open):
        return False
    return (_body(c) / rng) >= float(body_frac_min)


def is_strong_bull(c: Candle, *, body_frac_min: float = BEAR_BODY_FRAC) -> bool:
    rng = _range(c)
    if rng <= 0 or float(c.close) <= float(c.open):
        return False
    return (_body(c) / rng) >= float(body_frac_min)


def closes_into_prior_bear_body(first: Candle, third: Candle) -> bool:
    """Third close is at or above the midpoint of the first bearish body, still in range."""
    lo = min(float(first.open), float(first.close))
    hi = max(float(first.open), float(first.close))
    mid = (float(first.open) + float(first.close)) / 2.0
    cl = float(third.close)
    return cl >= mid - 1e-12 and cl <= hi + 1e-12 and hi >= lo


def is_morning_doji_star(
    first: Candle,
    star: Candle,
    third: Candle,
    *,
    require_gap: bool = True,
    doji_body_frac: float = DOJI_BODY_FRAC,
    large_body_frac: float = LARGE_BODY_FRAC,
    bull_body_frac: float = BEAR_BODY_FRAC,
) -> bool:
    if not is_large_bear(first, body_frac_min=large_body_frac):
        return False
    if not is_doji(star, body_frac_max=doji_body_frac):
        return False
    if require_gap:
        if float(star.open) >= float(first.close) - 1e-12:
            return False
    elif float(star.open) > float(first.close) + 1e-12:
        return False
    if not is_strong_bull(third, body_frac_min=bull_body_frac):
        return False
    return closes_into_prior_bear_body(first, third)


def find_morning_doji_star(
    sessions: Sequence[SessionDelta],
    *,
    start_day: date,
    end_day: date,
    require_gap: bool = True,
) -> Optional[date]:
    """Return candle-3 session date if a star completes in [start_day, end_day]."""
    candles = [candle_from_session(s) for s in sessions]
    n = len(candles)
    for i in range(0, n - 2):
        c1, c2, c3 = candles[i], candles[i + 1], candles[i + 2]
        if c3.session_date < start_day or c3.session_date > end_day:
            continue
        if is_morning_doji_star(c1, c2, c3, require_gap=require_gap):
            return c3.session_date
    return None


def is_bullish_engulfing(prior: Candle, curr: Candle) -> bool:
    if float(prior.close) >= float(prior.open) - 1e-12:
        return False
    if float(curr.close) <= float(curr.open) + 1e-12:
        return False
    return (
        float(curr.open) <= float(prior.close) + 1e-12
        and float(curr.close) >= float(prior.open) - 1e-12
    )


def is_hammer(
    c: Candle,
    *,
    wick_mult: float = HAMMER_WICK_MULT,
    upper_frac_max: float = HAMMER_UPPER_FRAC,
) -> bool:
    rng = _range(c)
    if rng <= 0:
        return False
    body = _body(c)
    upper = float(c.high) - max(float(c.open), float(c.close))
    lower = min(float(c.open), float(c.close)) - float(c.low)
    if upper > float(upper_frac_max) * rng + 1e-12:
        return False
    if body <= 0:
        return lower >= 0.6 * rng
    return lower >= float(wick_mult) * body - 1e-12


def is_dragonfly_doji(
    c: Candle,
    *,
    doji_body_frac: float = DOJI_BODY_FRAC,
    upper_frac_max: float = DRAGONFLY_UPPER_FRAC,
) -> bool:
    if not is_doji(c, body_frac_max=doji_body_frac):
        return False
    rng = _range(c)
    if rng <= 0:
        return False
    upper = float(c.high) - max(float(c.open), float(c.close))
    return upper <= float(upper_frac_max) * rng + 1e-12
