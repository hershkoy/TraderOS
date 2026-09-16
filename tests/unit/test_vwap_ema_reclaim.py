"""LOD -> 9 EMA / session VWAP reclaim (no TimescaleDB)."""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from utils.research.realistic_exits import REASON_FILLED
from utils.research.vwap_ema_reclaim import (
    EXIT_STOP,
    EXIT_TARGET,
    ReclaimParams,
    ema_next,
    first_reclaim_index,
    measured_move_levels,
    scan_session,
    session_vwap,
    simulate_measured_move_exit,
)

ET = ZoneInfo("America/New_York")


def _bar(ts: datetime, o: float, h: float, l: float, c: float, v: float = 1000.0) -> dict:
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def _session_start(day: datetime = datetime(2026, 8, 24, 9, 30, tzinfo=ET)) -> datetime:
    return day


def test_session_vwap_is_typical_price_volume():
    high = np.array([10.0, 12.0])
    low = np.array([8.0, 10.0])
    close = np.array([9.0, 11.0])
    vol = np.array([1.0, 3.0])
    vwap = session_vwap(high, low, close, vol)
    tp0 = (10 + 8 + 9) / 3.0
    tp1 = (12 + 10 + 11) / 3.0
    assert vwap[0] == pytest.approx(tp0)
    assert vwap[1] == pytest.approx((tp0 * 1 + tp1 * 3) / 4.0)


def test_measured_move_is_one_unit_target_one_third_stop():
    unit, stop, target = measured_move_levels(9.0, 12.0)
    assert unit == pytest.approx(3.0)
    assert stop == pytest.approx(11.0)
    assert target == pytest.approx(15.0)
    assert measured_move_levels(12.0, 12.0) is None


def test_first_reclaim_requires_lod_then_ema_cross():
    ema = np.array([10.0, 10.0, 9.5, 9.6, 9.8, 10.1])
    vwap = np.array([10.0, 10.0, 10.0, 9.9, 9.9, 10.0])
    low = np.array([9.9, 9.5, 9.0, 9.2, 9.4, 9.7])
    assert first_reclaim_index(ema, vwap, low) == 5


def test_first_reclaim_skips_new_lod_on_signal_bar():
    ema = np.array([10.0, 10.0, 9.5, 9.6, 9.8, 10.1])
    vwap = np.array([10.0, 10.0, 10.0, 9.9, 9.9, 10.0])
    low = np.array([9.9, 9.5, 9.0, 9.2, 9.4, 8.8])
    assert first_reclaim_index(ema, vwap, low) is None


def test_first_reclaim_skips_if_lod_already_above_vwap():
    ema = np.array([10.0, 10.1, 10.2])
    vwap = np.array([9.5, 9.6, 9.7])
    low = np.array([9.8, 9.9, 10.0])
    assert first_reclaim_index(ema, vwap, low) is None


def test_target_fills_next_15m_mid_not_bar_high():
    t0 = _session_start()
    entry = t0
    b0 = _bar(entry, 12.0, 12.2, 11.9, 12.0)
    b1 = _bar(t0 + timedelta(minutes=15), 12.1, 15.5, 12.0, 15.2)
    b2 = _bar(t0 + timedelta(minutes=30), 15.2, 15.4, 14.8, 15.0)
    got = simulate_measured_move_exit(
        [b0, b1, b2],
        entry_ts=entry,
        stop_px=11.0,
        target_px=15.0,
    )
    assert got.filled
    assert got.reason == REASON_FILLED
    assert got.exit_reason == EXIT_TARGET
    assert got.sell_px == pytest.approx((15.4 + 14.8) / 2.0)
    assert got.exec_bar_ts == b2["ts"]
    assert got.sell_px != pytest.approx(15.5)


def test_same_bar_stop_and_target_is_stop_then_next_mid():
    t0 = _session_start()
    b0 = _bar(t0, 12.0, 12.1, 11.9, 12.0)
    b1 = _bar(t0 + timedelta(minutes=15), 12.0, 16.0, 10.5, 11.0)
    b2 = _bar(t0 + timedelta(minutes=30), 11.0, 11.4, 10.8, 11.2)
    got = simulate_measured_move_exit(
        [b0, b1, b2],
        entry_ts=t0,
        stop_px=11.0,
        target_px=15.0,
    )
    assert got.exit_reason == EXIT_STOP
    assert got.sell_px == pytest.approx((11.4 + 10.8) / 2.0)


def test_last_bar_flatten_is_close_not_overnight():
    t0 = _session_start()
    b0 = _bar(t0, 12.0, 12.1, 11.9, 12.0)
    last = _bar(t0 + timedelta(minutes=15), 12.0, 12.3, 11.8, 12.2)
    got = simulate_measured_move_exit(
        [b0, last],
        entry_ts=t0,
        stop_px=11.0,
        target_px=15.0,
    )
    assert got.exit_reason == "eod"
    assert got.sell_px == pytest.approx(12.2)


def _bounce_session() -> list:
    """Dump to a LOD then grind so span-2 EMA crosses session VWAP."""
    t0 = _session_start()
    rows = [
        (100.0, 100.3, 99.7, 99.9, 1000.0),
        (99.8, 99.9, 97.0, 97.2, 4000.0),
        (97.3, 97.8, 97.1, 97.6, 1000.0),
        (97.7, 98.5, 97.5, 98.3, 1000.0),
        (98.4, 99.2, 98.2, 99.0, 1000.0),
        (99.1, 99.8, 98.9, 99.6, 1000.0),
        (99.7, 100.4, 99.5, 100.2, 1000.0),
        (100.2, 100.6, 100.0, 100.4, 800.0),
        (100.4, 100.7, 100.2, 100.5, 800.0),
        (100.5, 103.5, 100.4, 103.0, 800.0),
        (103.0, 103.2, 102.6, 102.8, 800.0),
    ]
    bars = []
    for i, (o, h, l, c, v) in enumerate(rows):
        bars.append(_bar(t0 + timedelta(minutes=15 * i), o, h, l, c, v))
    return bars


def test_scan_session_buys_next_mid_not_signal_close_or_lod():
    bars = _bounce_session()
    params = ReclaimParams(ema_span=2, fill_mode="next-mid")
    trade, ema = scan_session(bars, ema_prev=None, params=params, naive_tz="America/New_York")
    assert ema is not None
    assert trade is not None
    sig = trade["signal_ts"]
    buy = trade["buy_ts"]
    assert buy != sig
    assert trade["buy_price"] != pytest.approx(trade["lod"])
    assert trade["fill_mode"] == "next-mid"
    # Next-mid of the bar after the cross, not that bar's close.
    sig_i = next(i for i, b in enumerate(bars) if b["ts"] == sig)
    nxt = bars[sig_i + 1]
    assert buy == nxt["ts"]
    assert trade["buy_price"] == pytest.approx((nxt["high"] + nxt["low"]) / 2.0)
    assert trade["lod_age_bars"] >= 1
    assert trade["signal_bar_i"] == sig_i


def test_scan_session_signal_close_fills_cross_bar_close():
    bars = _bounce_session()
    params = ReclaimParams(ema_span=2, fill_mode="signal-close")
    trade, _ = scan_session(bars, ema_prev=None, params=params, naive_tz="America/New_York")
    assert trade is not None
    sig = trade["signal_ts"]
    assert trade["buy_ts"] == sig
    sig_bar = next(b for b in bars if b["ts"] == sig)
    assert trade["buy_price"] == pytest.approx(sig_bar["close"])


def test_late_session_signal_is_skipped():
    t0 = datetime(2026, 8, 24, 15, 15, tzinfo=ET)
    bars = [
        _bar(t0, 100.0, 100.2, 97.0, 97.5, 2000.0),
        _bar(t0 + timedelta(minutes=15), 97.6, 98.0, 97.4, 97.8, 1000.0),
        _bar(t0 + timedelta(minutes=30), 97.9, 100.5, 97.8, 100.2, 1000.0),
    ]
    params = ReclaimParams(
        ema_span=2,
        fill_mode="next-mid",
        last_entry_clock=time(15, 0),
    )
    trade, _ = scan_session(bars, ema_prev=90.0, params=params, naive_tz="America/New_York")
    assert trade is None


def test_ema_next_hugs_close():
    a = ema_next(None, 10.0, span=9)
    b = ema_next(a, 20.0, span=9)
    assert a == pytest.approx(10.0)
    assert 10.0 < b < 20.0
    assert b == pytest.approx(10.0 + (2.0 / 10.0) * (20.0 - 10.0))


def test_daily_top1_keeps_largest_unit_per_session():
    from utils.research.vwap_ema_reclaim import attach_unit_clock, take_daily_topn

    rows = pd.DataFrame(
        {
            "stock": ["AAA", "BBB", "AAA"],
            "unit": [1.0, 3.0, 2.0],
            "cross_px": [100.0, 100.0, 100.0],
            "signal_ts": [
                "2026-08-24 14:00:00+00:00",
                "2026-08-24 14:15:00+00:00",
                "2026-08-25 14:00:00+00:00",
            ],
            "gain_pct": [1.0, 2.0, 3.0],
        }
    )
    tagged = attach_unit_clock(rows)
    kept = take_daily_topn(tagged, n=1)
    assert list(kept["stock"]) == ["BBB", "AAA"]
    assert list(kept["unit"]) == [3.0, 2.0]


def test_morning_mask_is_before_11et():
    from utils.research.vwap_ema_reclaim import attach_unit_clock, mask_morning

    rows = pd.DataFrame(
        {
            "stock": ["A", "B"],
            "unit": [1.0, 1.0],
            "cross_px": [100.0, 100.0],
            "signal_ts": [
                "2026-08-24 14:30:00+00:00",  # 10:30 ET
                "2026-08-24 15:30:00+00:00",  # 11:30 ET
            ],
        }
    )
    tagged = attach_unit_clock(rows)
    m = mask_morning(tagged, before_hour=11.0)
    assert list(m) == [True, False]

