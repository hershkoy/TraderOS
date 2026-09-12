"""Morning Doji Star / hammer / engulfing fixtures."""
from __future__ import annotations

from datetime import date

from utils.research.evening_doji_star import Candle
from utils.research.morning_doji_star import (
    find_morning_doji_star,
    is_bullish_engulfing,
    is_dragonfly_doji,
    is_hammer,
    is_large_bear,
    is_morning_doji_star,
)
from utils.research.session_volume_delta import SessionDelta


def _c(day, o, h, l, c) -> Candle:
    return Candle(session_date=day, open=o, high=h, low=l, close=c)


def _s(day, o, h, l, c) -> SessionDelta:
    return SessionDelta(
        session_date=day,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=100.0,
        buy_volume=50.0,
        sell_volume=50.0,
        buy_pct=50.0,
        sell_pct=50.0,
        last_bar_ts=None,
    )


def test_large_bear_body_at_least_50pct():
    assert is_large_bear(_c(date(2024, 2, 9), 14.20, 14.30, 12.95, 13.00))
    assert not is_large_bear(_c(date(2024, 2, 9), 13.70, 14.20, 13.00, 13.50))


def test_morning_doji_star_textbook_gap():
    c1 = _c(date(2024, 2, 9), 14.20, 14.30, 12.95, 13.00)
    c2 = _c(date(2024, 2, 12), 12.90, 13.05, 12.80, 12.92)
    c3 = _c(date(2024, 2, 13), 13.10, 14.25, 13.05, 14.10)
    assert is_morning_doji_star(c1, c2, c3, require_gap=True)


def test_morning_doji_star_rejects_no_gap_when_required():
    c1 = _c(date(2024, 2, 9), 14.20, 14.30, 12.95, 13.00)
    c2_flat = _c(date(2024, 2, 12), 13.00, 13.10, 12.85, 13.02)
    c3 = _c(date(2024, 2, 13), 13.10, 14.25, 13.05, 14.10)
    assert is_morning_doji_star(c1, c2_flat, c3, require_gap=True) is False
    assert is_morning_doji_star(c1, c2_flat, c3, require_gap=False) is True


def test_find_morning_doji_star_window():
    sessions = [
        _s(date(2024, 2, 9), 14.20, 14.30, 12.95, 13.00),
        _s(date(2024, 2, 12), 12.90, 13.05, 12.80, 12.92),
        _s(date(2024, 2, 13), 13.10, 14.25, 13.05, 14.10),
    ]
    got = find_morning_doji_star(
        sessions, start_day=date(2024, 2, 9), end_day=date(2024, 2, 13), require_gap=True
    )
    assert got == date(2024, 2, 13)
    miss = find_morning_doji_star(
        sessions, start_day=date(2024, 2, 14), end_day=date(2024, 2, 20), require_gap=True
    )
    assert miss is None


def test_bullish_engulfing():
    prior = _c(date(2024, 2, 12), 14.00, 14.10, 13.20, 13.30)
    curr = _c(date(2024, 2, 13), 13.20, 14.20, 13.10, 14.10)
    assert is_bullish_engulfing(prior, curr)
    assert not is_bullish_engulfing(curr, prior)


def test_hammer_and_dragonfly():
    hammer = _c(date(2024, 2, 12), 13.80, 13.90, 13.00, 13.85)
    assert is_hammer(hammer)
    dragon = _c(date(2024, 2, 12), 13.88, 13.90, 13.00, 13.87)
    assert is_dragonfly_doji(dragon)
    tall = _c(date(2024, 2, 12), 13.20, 14.20, 13.10, 14.10)
    assert not is_hammer(tall)
    assert not is_dragonfly_doji(tall)
