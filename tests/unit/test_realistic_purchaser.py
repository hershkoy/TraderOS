"""Unit tests for close-confirm 15m realistic purchaser (not wired to the runner yet)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from utils.research.realistic_purchaser import (
    DEFAULT_MAX_LOW_TO_MID_PCT,
    REASON_BAD_OHLC,
    REASON_BAD_SIGNAL,
    REASON_CHASE,
    REASON_END_OF_SESSION,
    REASON_FILLED,
    REASON_NO_NEXT_BAR,
    REASON_OVERNIGHT,
    REASON_WILD_RANGE,
    bar_mid,
    expected_exec_bar_start,
    is_last_rth_15m_bar,
    low_to_mid_pct,
    purchase_after_close_signal,
    purchase_at_signal_index,
    signal_time_from_bar,
)

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm) -> datetime:
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _bar(ts, o, h, l, c, v=1e5):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


# ZLAB 2025-06-10 09:30 ET IB 15m (HTML fill 38.75 was the signal-bar low).
_ZLAB_SIG = _bar(_et(2025, 6, 10, 9, 30), 39.15, 39.98, 38.75, 39.38)
_ZLAB_NEXT_QUIET = _bar(_et(2025, 6, 10, 9, 45), 39.40, 39.55, 39.30, 39.48)


def test_signal_clock_is_bar_close_not_period_start():
    start = _et(2025, 6, 10, 9, 30)
    assert signal_time_from_bar(start) == _et(2025, 6, 10, 9, 45)
    assert expected_exec_bar_start(start) == _et(2025, 6, 10, 9, 45)
    last = _et(2025, 6, 10, 15, 45)
    assert is_last_rth_15m_bar(last)
    assert signal_time_from_bar(last) == _et(2025, 6, 10, 16, 0)
    assert expected_exec_bar_start(last) is None


CASES = [
    pytest.param(
        "tight_continuation_fills_mid_not_low",
        _bar(_et(2025, 6, 10, 10, 0), 20.00, 20.10, 19.95, 20.05),
        _bar(_et(2025, 6, 10, 10, 15), 20.06, 20.12, 20.04, 20.10),
        {},
        True,
        REASON_FILLED,
        20.08,
        id="tight_continuation",
    ),
    pytest.param(
        "zlab_gap_through_does_not_buy_signal_low",
        _ZLAB_SIG,
        _ZLAB_NEXT_QUIET,
        {},
        True,
        REASON_FILLED,
        bar_mid(39.55, 39.30),
        id="zlab_next_mid",
    ),
    pytest.param(
        "wild_next_bar_low_to_mid_above_half_pct",
        _bar(_et(2025, 6, 10, 11, 0), 40.00, 40.20, 39.90, 40.10),
        _bar(_et(2025, 6, 10, 11, 15), 40.20, 41.20, 39.80, 41.00),
        {},
        False,
        REASON_WILD_RANGE,
        None,
        id="wild_cancel",
    ),
    pytest.param(
        "boundary_low_to_mid_equals_half_pct_fills",
        _bar(_et(2025, 6, 10, 11, 30), 100.00, 100.20, 99.90, 100.00),
        _bar(_et(2025, 6, 10, 11, 45), 100.00, 100.50, 99.50, 100.10),
        {},
        True,
        REASON_FILLED,
        100.00,
        id="boundary_equal_fills",
    ),
    pytest.param(
        "just_over_half_pct_cancels",
        _bar(_et(2025, 6, 10, 12, 0), 100.00, 100.20, 99.90, 100.00),
        _bar(_et(2025, 6, 10, 12, 15), 100.00, 100.51, 99.49, 100.10),
        {},
        False,
        REASON_WILD_RANGE,
        None,
        id="boundary_over_cancels",
    ),
    pytest.param(
        "quiet_gap_up_fills_mid_without_chase_cap",
        _bar(_et(2025, 6, 10, 13, 0), 100.00, 100.30, 99.90, 100.00),
        _bar(_et(2025, 6, 10, 13, 15), 101.00, 101.15, 100.95, 101.10),
        {},
        True,
        REASON_FILLED,
        101.05,
        id="quiet_chase_fills",
    ),
    pytest.param(
        "quiet_gap_up_cancels_when_chase_cap_on",
        _bar(_et(2025, 6, 10, 13, 0), 100.00, 100.30, 99.90, 100.00),
        _bar(_et(2025, 6, 10, 13, 15), 101.00, 101.15, 100.95, 101.10),
        {"max_chase_pct": 0.005},
        False,
        REASON_CHASE,
        None,
        id="quiet_chase_cap",
    ),
    pytest.param(
        "dump_fills_mid_not_the_low",
        _bar(_et(2025, 6, 10, 13, 30), 50.00, 50.20, 49.90, 50.10),
        _bar(_et(2025, 6, 10, 13, 45), 49.80, 49.90, 49.50, 49.70),
        {},
        True,
        REASON_FILLED,
        49.70,
        id="dump_mid",
    ),
    pytest.param(
        "doji_next_bar_fill_equals_mid",
        _bar(_et(2025, 6, 10, 14, 0), 30.00, 30.10, 29.95, 30.05),
        _bar(_et(2025, 6, 10, 14, 15), 30.05, 30.05, 30.05, 30.05),
        {},
        True,
        REASON_FILLED,
        30.05,
        id="doji",
    ),
    pytest.param(
        "last_rth_window_signal_at_1545_uses_1545_bar",
        _bar(_et(2025, 6, 10, 15, 30), 25.00, 25.10, 24.95, 25.04),
        _bar(_et(2025, 6, 10, 15, 45), 25.05, 25.12, 25.00, 25.08),
        {},
        True,
        REASON_FILLED,
        25.06,
        id="last_window_1545",
    ),
    pytest.param(
        "confirm_at_1600_no_window",
        _bar(_et(2025, 6, 10, 15, 45), 25.00, 25.20, 24.90, 25.10),
        None,
        {},
        False,
        REASON_END_OF_SESSION,
        None,
        id="eod_1600",
    ),
    pytest.param(
        "confirm_at_1600_does_not_roll_to_next_open",
        _bar(_et(2025, 6, 10, 15, 45), 25.00, 25.20, 24.90, 25.10),
        _bar(_et(2025, 6, 11, 9, 30), 25.20, 25.30, 25.10, 25.25),
        {},
        False,
        REASON_END_OF_SESSION,
        None,
        id="eod_no_overnight",
    ),
    pytest.param(
        "friday_1545_window_still_fills",
        _bar(_et(2025, 6, 13, 15, 30), 12.00, 12.08, 11.96, 12.02),
        _bar(_et(2025, 6, 13, 15, 45), 12.03, 12.10, 12.00, 12.06),
        {},
        True,
        REASON_FILLED,
        12.05,
        id="friday_last_window",
    ),
    pytest.param(
        "friday_1600_does_not_use_monday_open",
        _bar(_et(2025, 6, 13, 15, 45), 12.00, 12.20, 11.90, 12.10),
        _bar(_et(2025, 6, 16, 9, 30), 12.40, 12.50, 12.30, 12.45),
        {},
        False,
        REASON_END_OF_SESSION,
        None,
        id="friday_no_monday",
    ),
    pytest.param(
        "midday_skip_to_next_session_is_overnight",
        _bar(_et(2025, 6, 13, 15, 0), 12.00, 12.08, 11.96, 12.02),
        _bar(_et(2025, 6, 16, 9, 30), 12.40, 12.50, 12.30, 12.45),
        {},
        False,
        REASON_OVERNIGHT,
        None,
        id="overnight_gap",
    ),
    pytest.param(
        "intraday_gap_wrong_next_bar",
        _bar(_et(2025, 6, 10, 10, 0), 18.00, 18.10, 17.95, 18.05),
        _bar(_et(2025, 6, 10, 11, 0), 18.20, 18.30, 18.10, 18.25),
        {},
        False,
        REASON_NO_NEXT_BAR,
        None,
        id="intraday_gap",
    ),
    pytest.param(
        "missing_next_bar_midday",
        _bar(_et(2025, 6, 10, 10, 0), 18.00, 18.10, 17.95, 18.05),
        None,
        {},
        False,
        REASON_NO_NEXT_BAR,
        None,
        id="missing_next",
    ),
    pytest.param(
        "bad_exec_ohlc_high_below_low",
        _bar(_et(2025, 6, 10, 10, 0), 18.00, 18.10, 17.95, 18.05),
        _bar(_et(2025, 6, 10, 10, 15), 18.10, 18.00, 18.20, 18.05),
        {},
        False,
        REASON_BAD_OHLC,
        None,
        id="bad_ohlc",
    ),
    pytest.param(
        "first_rth_bar_signal_buys_0945_not_0930",
        _bar(_et(2025, 6, 10, 9, 30), 39.15, 39.98, 38.75, 39.38),
        _bar(_et(2025, 6, 10, 9, 45), 39.40, 39.50, 39.36, 39.44),
        {},
        True,
        REASON_FILLED,
        bar_mid(39.50, 39.36),
        id="open_session",
    ),
]


@pytest.mark.parametrize(
    "title,signal_bar,next_bar,kwargs,filled,reason,fill_px",
    CASES,
)
def test_purchase_cases(title, signal_bar, next_bar, kwargs, filled, reason, fill_px):
    got = purchase_after_close_signal(signal_bar, next_bar, **kwargs)
    assert got.filled is filled, title
    assert got.reason == reason, title
    if fill_px is None:
        assert got.fill_px is None, title
    else:
        assert got.fill_px == pytest.approx(fill_px, rel=1e-9, abs=1e-9), title
        assert got.fill_px != signal_bar["low"] or signal_bar["low"] == next_bar["low"] == next_bar["high"]


def test_zlab_fill_is_not_3875():
    got = purchase_after_close_signal(_ZLAB_SIG, _ZLAB_NEXT_QUIET)
    assert got.filled
    assert got.fill_px == pytest.approx(39.425)
    assert got.fill_px != pytest.approx(38.75)
    assert got.signal_time == _et(2025, 6, 10, 9, 45)
    assert got.signal_px == pytest.approx(39.38)
    assert got.low_to_mid_pct < DEFAULT_MAX_LOW_TO_MID_PCT


def test_low_to_mid_is_half_range_over_mid():
    assert low_to_mid_pct(100.5, 99.5) == pytest.approx(0.005)
    assert bar_mid(100.5, 99.5) == pytest.approx(100.0)


def test_disable_wild_gate_fills_rocket():
    sig = _bar(_et(2025, 6, 10, 11, 0), 40.00, 40.20, 39.90, 40.10)
    nxt = _bar(_et(2025, 6, 10, 11, 15), 40.20, 41.20, 39.80, 41.00)
    got = purchase_after_close_signal(sig, nxt, max_low_to_mid_pct=None)
    assert got.filled
    assert got.reason == REASON_FILLED
    assert got.fill_px == pytest.approx(bar_mid(41.20, 39.80))


def test_naive_utc_ib_stamp_maps_to_et_session():
    # 2025-06-10 13:30 UTC = 09:30 ET.
    sig = _bar(datetime(2025, 6, 10, 13, 30), 39.15, 39.98, 38.75, 39.38)
    nxt = _bar(datetime(2025, 6, 10, 13, 45), 39.40, 39.55, 39.30, 39.48)
    got = purchase_after_close_signal(sig, nxt, naive_tz="UTC")
    assert got.filled
    assert got.signal_time == _et(2025, 6, 10, 9, 45)


def test_purchase_at_signal_index_uses_next_row():
    idx = pd.DatetimeIndex(
        [_et(2025, 6, 10, 9, 30), _et(2025, 6, 10, 9, 45), _et(2025, 6, 10, 10, 0)]
    )
    df = pd.DataFrame(
        {
            "open": [39.15, 39.40, 39.50],
            "high": [39.98, 39.55, 39.60],
            "low": [38.75, 39.30, 39.40],
            "close": [39.38, 39.48, 39.55],
            "volume": [1e6, 2e5, 1.5e5],
        },
        index=idx,
    )
    got = purchase_at_signal_index(df, 0)
    assert got.filled
    assert got.fill_px == pytest.approx(bar_mid(39.55, 39.30))
    last = purchase_at_signal_index(df, 2)
    assert not last.filled
    assert last.reason == REASON_NO_NEXT_BAR


def test_weekend_signal_is_bad():
    sat = _bar(_et(2025, 6, 14, 10, 0), 10.0, 10.2, 9.9, 10.1)
    nxt = _bar(_et(2025, 6, 14, 10, 15), 10.1, 10.2, 10.0, 10.15)
    got = purchase_after_close_signal(sat, nxt)
    assert not got.filled
    assert got.reason == REASON_BAD_SIGNAL
