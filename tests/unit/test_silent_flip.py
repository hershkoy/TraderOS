"""Silent Flip 15m open-range claim-check (no TimescaleDB)."""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from utils.research.silent_flip import (
    EXIT_EOD,
    EXIT_STOP,
    EXIT_TARGET,
    LEVEL_FLIP_HIGH,
    LEVEL_RANGE_LOW,
    SIDE_LONG,
    SIDE_SHORT,
    SilentFlipParams,
    WINDOW_FIRST_60,
    WINDOW_OPEN_ONLY,
    collect_symbol_trades,
    prior_flip_levels,
    scan_session,
    session_hl,
    simulate_exit,
)

ET = ZoneInfo("America/New_York")


def _bar(ts: datetime, o: float, h: float, l: float, c: float, v: float = 1000.0) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _rth(day: datetime, hm: tuple[int, int]) -> datetime:
    return datetime(day.year, day.month, day.day, hm[0], hm[1], tzinfo=ET)


def _session_bars(
    day: datetime,
    ohlc: list[tuple[float, float, float, float]],
    start: tuple[int, int] = (9, 30),
) -> list[dict]:
    bars = []
    ts = _rth(day, start)
    for o, h, l, c in ohlc:
        bars.append(_bar(ts, o, h, l, c))
        ts = ts + timedelta(minutes=15)
    return bars


def test_session_hl_is_max_high_min_low():
    day = datetime(2026, 8, 24)
    bars = _session_bars(
        day,
        [
            (10.0, 11.0, 9.8, 10.5),
            (10.5, 10.8, 9.5, 9.7),
            (9.7, 10.2, 9.6, 10.0),
        ],
    )
    assert session_hl(bars) == (11.0, 9.5)


def test_prior_flip_uses_most_recent_beyond_range():
    hls = [
        (12.0, 8.0),
        (11.5, 9.5),
        (10.0, 9.2),
        None,
    ]
    levels = prior_flip_levels(hls, 3, lookback=10)
    assert levels is not None
    assert levels["range_high"] == 10.0
    assert levels["range_low"] == 9.2
    assert levels["flip_high"] == 11.5
    assert levels["flip_low"] == 8.0


def test_long_range_low_fills_silent_high_and_targets_range_high():
    levels = {
        "range_high": 12.0,
        "range_low": 10.0,
        "flip_high": 13.0,
        "flip_low": 9.0,
    }
    day = datetime(2026, 8, 25)
    bars = _session_bars(
        day,
        [
            (10.40, 10.45, 9.98, 10.05),
            (10.08, 10.30, 10.06, 10.28),
            (10.29, 10.35, 10.20, 10.32),
            (10.32, 12.10, 10.30, 12.05),
        ],
    )
    rec = scan_session(bars, levels, params=SilentFlipParams(min_body_frac=0.3))
    assert rec is not None
    assert rec["side"] == SIDE_LONG
    assert rec["level_name"] == LEVEL_RANGE_LOW
    assert rec["buy_price"] == pytest.approx(10.30)
    assert rec["sell_price"] == pytest.approx(12.0)
    assert rec["exit_reason"] == EXIT_TARGET
    assert rec["target_px"] == pytest.approx(12.0)


def test_close_through_support_is_not_a_setup():
    levels = {
        "range_high": 12.0,
        "range_low": 10.0,
        "flip_high": 13.0,
        "flip_low": 9.0,
    }
    day = datetime(2026, 8, 25)
    bars = _session_bars(
        day,
        [
            (10.40, 10.45, 9.70, 9.80),
            (9.85, 10.20, 9.80, 10.18),
            (10.19, 10.40, 10.10, 10.30),
        ],
    )
    assert scan_session(bars, levels) is None


def test_same_color_silent_is_rejected():
    levels = {
        "range_high": 12.0,
        "range_low": 10.0,
        "flip_high": 13.0,
        "flip_low": 9.0,
    }
    day = datetime(2026, 8, 25)
    bars = _session_bars(
        day,
        [
            (10.40, 10.45, 9.98, 10.05),
            (10.04, 10.10, 9.99, 10.00),
            (10.01, 10.40, 10.00, 10.30),
        ],
    )
    assert scan_session(bars, levels) is None


def test_same_bar_stop_beats_target_on_entry_bar():
    bars = [
        _bar(_rth(datetime(2026, 8, 25), (10, 0)), 10.30, 12.20, 9.90, 11.00),
    ]
    rec = simulate_exit(
        bars,
        entry_i=0,
        entry_px=10.30,
        side=SIDE_LONG,
        stop_px=9.95,
        target_px=12.00,
    )
    assert rec is not None
    assert rec["exit_reason"] == EXIT_STOP
    assert rec["sell_px"] == pytest.approx(9.95)


def test_short_flip_high_targets_range_low():
    levels = {
        "range_high": 20.0,
        "range_low": 18.0,
        "flip_high": 21.0,
        "flip_low": 17.0,
    }
    day = datetime(2026, 8, 25)
    bars = _session_bars(
        day,
        [
            (20.60, 21.02, 20.50, 20.90),
            (20.88, 20.95, 20.70, 20.72),
            (20.71, 20.80, 20.60, 20.65),
            (20.64, 20.66, 17.90, 18.10),
        ],
    )
    rec = scan_session(bars, levels)
    assert rec is not None
    assert rec["side"] == SIDE_SHORT
    assert rec["level_name"] == LEVEL_FLIP_HIGH
    assert rec["buy_price"] == pytest.approx(20.70)
    assert rec["target_px"] == pytest.approx(18.0)
    assert rec["sell_price"] == pytest.approx(18.0)
    assert rec["exit_reason"] == EXIT_TARGET


def test_open_only_skips_0945_candle1():
    levels = {
        "range_high": 12.0,
        "range_low": 10.0,
        "flip_high": 13.0,
        "flip_low": 9.0,
    }
    day = datetime(2026, 8, 25)
    morning = _session_bars(
        day,
        [
            (10.50, 10.60, 10.40, 10.55),
            (10.40, 10.45, 9.98, 10.05),
            (10.08, 10.30, 10.06, 10.28),
            (10.29, 10.35, 10.20, 10.32),
            (10.32, 12.10, 10.30, 12.05),
        ],
    )
    first60 = scan_session(morning, levels, params=SilentFlipParams(window=WINDOW_FIRST_60))
    open_only = scan_session(morning, levels, params=SilentFlipParams(window=WINDOW_OPEN_ONLY))
    assert first60 is not None
    assert open_only is None


def test_collect_uses_prior_session_only():
    d0 = datetime(2026, 8, 24)
    d1 = datetime(2026, 8, 25)
    d2 = datetime(2026, 8, 26)
    prior = _session_bars(
        d0,
        [(10.0, 11.0, 9.0, 10.5)] * 12,
    )
    range_day = _session_bars(
        d1,
        [(10.2, 10.8, 10.0, 10.4)] * 12,
    )
    trade_day = _session_bars(
        d2,
        [
            (10.40, 10.45, 9.98, 10.05),
            (10.08, 10.30, 10.06, 10.28),
            (10.29, 10.35, 10.20, 10.32),
                (10.32, 10.50, 10.30, 10.45),
        ]
        + [(10.70, 10.75, 10.60, 10.65)] * 8,
    )
    by_day = {
        d0.date(): prior,
        d1.date(): range_day,
        d2.date(): trade_day,
    }
    trades = collect_symbol_trades(by_day)
    assert len(trades) == 1
    rec = trades[0]
    assert rec["range_high"] == pytest.approx(10.8)
    assert rec["range_low"] == pytest.approx(10.0)
    assert rec["flip_high"] == pytest.approx(11.0)
    assert rec["flip_low"] == pytest.approx(9.0)
    assert rec["exit_reason"] == EXIT_EOD
    assert rec["gain_pct"] > 0
