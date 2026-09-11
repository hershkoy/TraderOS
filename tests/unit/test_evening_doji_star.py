"""Evening Doji Star fixtures (textbook vs near-miss)."""
from __future__ import annotations

from datetime import date

from utils.research.evening_doji_star import (
    Candle,
    find_evening_doji_star,
    is_doji,
    is_evening_doji_star,
    is_large_bull,
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


def test_doji_body_at_most_20pct():
    assert is_doji(_c(date(2024, 2, 12), 14.10, 14.20, 13.90, 14.12))
    assert not is_doji(_c(date(2024, 2, 12), 13.80, 14.20, 13.70, 14.15))


def test_large_bull_body_at_least_50pct():
    assert is_large_bull(_c(date(2024, 2, 9), 13.00, 14.20, 12.95, 14.10))
    assert not is_large_bull(_c(date(2024, 2, 9), 13.50, 14.20, 13.00, 13.70))


def test_evening_doji_star_textbook_gap():
    c1 = _c(date(2024, 2, 9), 13.00, 14.30, 12.95, 14.20)
    c2 = _c(date(2024, 2, 12), 14.30, 14.40, 14.15, 14.28)
    c3 = _c(date(2024, 2, 13), 14.20, 14.25, 13.40, 13.50)
    assert is_evening_doji_star(c1, c2, c3, require_gap=True)


def test_evening_doji_star_rejects_no_gap_when_required():
    c1 = _c(date(2024, 2, 9), 13.00, 14.30, 12.95, 14.20)
    c2_gap_down = _c(date(2024, 2, 12), 14.10, 14.30, 14.00, 14.12)
    c2_flat = _c(date(2024, 2, 12), 14.20, 14.35, 14.10, 14.22)
    c3 = _c(date(2024, 2, 13), 14.10, 14.15, 13.40, 13.50)
    assert is_evening_doji_star(c1, c2_gap_down, c3, require_gap=True) is False
    assert is_evening_doji_star(c1, c2_flat, c3, require_gap=True) is False
    assert is_evening_doji_star(c1, c2_flat, c3, require_gap=False) is True


def test_find_evening_doji_star_fill_may_be_candle1():
    fill = date(2024, 2, 9)
    sessions = [
        _s(date(2024, 2, 9), 13.00, 14.30, 12.95, 14.20),
        _s(date(2024, 2, 12), 14.30, 14.40, 14.15, 14.28),
        _s(date(2024, 2, 13), 14.20, 14.25, 13.40, 13.50),
    ]
    got = find_evening_doji_star(sessions, fill_day=fill, require_gap=True)
    assert got == date(2024, 2, 13)


def test_find_evening_doji_star_ignores_pre_fill():
    fill = date(2024, 2, 9)
    sessions = [
        _s(date(2024, 2, 7), 13.00, 14.30, 12.95, 14.20),
        _s(date(2024, 2, 8), 14.30, 14.40, 14.15, 14.28),
        _s(date(2024, 2, 9), 14.20, 14.25, 13.40, 13.50),
    ]
    got = find_evening_doji_star(sessions, fill_day=fill, require_gap=True)
    assert got is None
