"""Realistic 15m / next-open-mid sells after a last-RTH 15m buy."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from utils.research.realistic_exits import (
    EXIT_HARD_STOP,
    EXIT_TRAIL_STOP,
    REASON_FILLED,
    REASON_NO_ENTRY_BAR,
    REASON_NO_NEXT_BAR,
    atr_dollars,
    find_bar_index,
    flatten_rth_sessions,
    simulate_exit_15m_next_mid,
    simulate_exit_daily_close_next_open_mid,
)
from utils.research.realistic_purchaser import bar_mid
from utils.scanning.channel_touch_bought import hard_stop_price

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c}


def test_atr_dollars_from_stored_pct():
    assert atr_dollars(atr_pct=4.694, ref_px=6.41) == pytest.approx(0.3008854)


def test_wve_like_15m_next_mid_sells_bar_after_gap():
    """Buy last 15m mid 6.85. Next 09:30 gaps through the 6% stop; fill 09:45 mid."""
    entry = _bar(_et(2023, 12, 6, 15, 45), 6.80, 7.00, 6.70, 6.90)
    gap = _bar(_et(2023, 12, 7, 9, 30), 5.50, 5.80, 5.20, 5.40)
    nxt = _bar(_et(2023, 12, 7, 9, 45), 5.42, 5.60, 5.30, 5.50)
    later = _bar(_et(2023, 12, 8, 9, 30), 5.20, 5.40, 5.00, 5.10)
    bars = [entry, gap, nxt, later]
    entry_px = 6.85
    atr = atr_dollars(atr_pct=4.694, ref_px=6.41)
    hard = hard_stop_price(entry_px, atr_at_entry=atr)
    assert hard == pytest.approx(entry_px * (1.0 - 0.06), rel=1e-6)
    got = simulate_exit_15m_next_mid(
        bars, entry_ts=entry["ts"], entry_px=entry_px, atr_at_entry=atr
    )
    assert got.filled is True
    assert got.reason == REASON_FILLED
    assert got.exit_reason == EXIT_HARD_STOP
    assert got.sell_px == pytest.approx(bar_mid(5.60, 5.30))
    assert got.exec_bar_ts == nxt["ts"]
    assert got.decision_ts == gap["ts"]
    assert got.hard_stop == pytest.approx(hard)


def test_wve_like_daily_close_sells_next_open_mid():
    """Same gap day: decide at Dec 7 close, fill Dec 8 09:30 mid — not the 6.0254 clip."""
    entry = _bar(_et(2023, 12, 6, 15, 45), 6.80, 7.00, 6.70, 6.90)
    d7_0930 = _bar(_et(2023, 12, 7, 9, 30), 5.50, 5.80, 5.20, 5.40)
    d7_1545 = _bar(_et(2023, 12, 7, 15, 45), 5.45, 5.50, 5.30, 5.35)
    d8_0930 = _bar(_et(2023, 12, 8, 9, 30), 5.10, 5.30, 4.90, 5.00)
    d8_1545 = _bar(_et(2023, 12, 8, 15, 45), 5.00, 5.10, 4.95, 5.05)
    by_day = {
        entry["ts"].date(): [entry],
        d7_0930["ts"].date(): [d7_0930, d7_1545],
        d8_0930["ts"].date(): [d8_0930, d8_1545],
    }
    entry_px = 6.85
    atr = atr_dollars(atr_pct=4.694, ref_px=6.41)
    got = simulate_exit_daily_close_next_open_mid(
        by_day, entry_ts=entry["ts"], entry_px=entry_px, atr_at_entry=atr
    )
    assert got.filled is True
    assert got.exit_reason == EXIT_HARD_STOP
    assert got.sell_px == pytest.approx(bar_mid(5.30, 4.90))
    assert got.exec_bar_ts == d8_0930["ts"]
    assert got.decision_ts == d7_1545["ts"]
    assert got.sell_px != pytest.approx(6.0254)


def test_entry_bar_low_does_not_trigger():
    """15:45 wick through the stop is already in the past at fill time."""
    entry = _bar(_et(2023, 12, 6, 15, 45), 6.80, 7.00, 5.00, 6.90)
    nxt = _bar(_et(2023, 12, 7, 9, 30), 6.80, 6.90, 6.70, 6.85)
    later = _bar(_et(2023, 12, 7, 9, 45), 6.85, 6.95, 6.75, 6.90)
    hit = _bar(_et(2023, 12, 7, 10, 0), 6.40, 6.50, 6.00, 6.20)
    fill = _bar(_et(2023, 12, 7, 10, 15), 6.20, 6.30, 6.10, 6.22)
    bars = [entry, nxt, later, hit, fill]
    got = simulate_exit_15m_next_mid(
        bars, entry_ts=entry["ts"], entry_px=6.85, atr_at_entry=0.30
    )
    assert got.filled is True
    assert got.decision_ts == hit["ts"]
    assert got.exec_bar_ts == fill["ts"]


def test_trail_stop_uses_15m_peak_then_next_mid():
    entry = _bar(_et(2023, 12, 6, 15, 45), 10.00, 10.20, 9.90, 10.10)
    # Peak 12.00 -> 10% trail 10.80. Keep this bar's low above the trail.
    rally = _bar(_et(2023, 12, 7, 9, 30), 10.20, 12.00, 11.00, 11.80)
    dump = _bar(_et(2023, 12, 7, 9, 45), 11.70, 11.80, 10.50, 10.60)
    fill = _bar(_et(2023, 12, 7, 10, 0), 10.55, 10.70, 10.40, 10.50)
    got = simulate_exit_15m_next_mid(
        [entry, rally, dump, fill],
        entry_ts=entry["ts"],
        entry_px=10.00,
        atr_at_entry=0.20,
    )
    assert got.exit_reason == EXIT_TRAIL_STOP
    assert got.sell_px == pytest.approx(bar_mid(10.70, 10.40))
    assert got.peak_px == pytest.approx(12.00)


def test_last_bar_decision_without_next_is_skipped():
    entry = _bar(_et(2023, 12, 6, 15, 45), 6.80, 7.00, 6.70, 6.90)
    gap = _bar(_et(2023, 12, 7, 9, 30), 5.50, 5.80, 5.20, 5.40)
    got = simulate_exit_15m_next_mid(
        [entry, gap], entry_ts=entry["ts"], entry_px=6.85, atr_at_entry=0.30
    )
    assert got.filled is False
    assert got.reason == REASON_NO_NEXT_BAR


def test_flatten_and_find_bar():
    a = _bar(_et(2023, 12, 6, 15, 45), 1, 1, 1, 1)
    b = _bar(_et(2023, 12, 7, 9, 30), 1, 1, 1, 1)
    by_day = {a["ts"].date(): [a], b["ts"].date(): [b]}
    bars = flatten_rth_sessions(by_day)
    assert find_bar_index(bars, "2023-12-06 20:45") == 0
    assert find_bar_index(bars, a["ts"]) == 0


def test_missing_entry_bar():
    bars = [_bar(_et(2023, 12, 7, 9, 30), 1, 1, 1, 1)]
    got = simulate_exit_15m_next_mid(
        bars, entry_ts=_et(2023, 12, 6, 15, 45), entry_px=6.85
    )
    assert got.reason == REASON_NO_ENTRY_BAR
