"""Causal Deepvue weekend scan helpers (no TimescaleDB)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from utils.research.realistic_exits import EXIT_HARD_STOP, REASON_FILLED
from utils.research.realistic_purchaser import purchase_next_session_open_mid
from utils.research.weekly_deepvue import (
    DeepvueParams,
    EXIT_MA_2CLOSE,
    occupancy_take,
    qqq_risk_on,
    rmv_ratio,
    simulate_deepvue_exit,
    week_end_mask,
    _pick_focus,
)

ET = ZoneInfo("America/New_York")


def _idx(*days):
    return pd.DatetimeIndex(pd.to_datetime(list(days)))


def test_week_end_mask_uses_thursday_when_friday_missing():
    idx = _idx("2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20")
    mask = week_end_mask(idx)
    assert list(mask) == [False, False, False, True]
    idx2 = _idx("2026-08-17", "2026-08-18", "2026-08-19")
    mask2 = week_end_mask(idx2)
    assert list(mask2) == [False, False, True]


def test_qqq_risk_on_needs_close_above_rising_ema():
    idx = pd.bdate_range("2026-01-02", periods=30)
    close = pd.Series(np.linspace(100.0, 130.0, len(idx)), index=idx)
    on = qqq_risk_on(close, span=21)
    assert bool(on.iloc[-1])
    down = pd.Series(np.linspace(130.0, 100.0, len(idx)), index=idx)
    off = qqq_risk_on(down, span=21)
    assert not bool(off.iloc[-1])


def test_rmv_low_when_range_contracts():
    idx = pd.bdate_range("2026-01-02", periods=20)
    high = pd.DataFrame({"AAA": np.linspace(12.0, 20.0, 20)}, index=idx)
    low = pd.DataFrame({"AAA": np.linspace(10.0, 18.0, 20)}, index=idx)
    high.iloc[-5:] = 20.2
    low.iloc[-5:] = 19.8
    rmv = rmv_ratio(high, low, short=5, long=15)
    assert float(rmv["AAA"].iloc[-1]) < 20.0


def test_pick_focus_caps_sector_and_daily_list():
    params = DeepvueParams(max_per_sector=2, weekly_focus=15, daily_focus=4)
    rows = []
    for i in range(8):
        rows.append(
            {
                "stock": "S%d" % i,
                "sector": "Software" if i < 5 else "Biotech",
                "rmv": float(i),
                "near_ma_pct": 0.01,
            }
        )
    picked = _pick_focus(rows, params)
    assert len(picked) == 4
    software = [r for r in picked if r["sector"] == "Software"]
    assert len(software) <= 2
    assert picked[0]["stock"] == "S0"


def test_next_open_mid_buy_is_monday_0930_after_friday_signal():
    friday = datetime(2026, 8, 21, tzinfo=ET)
    monday = datetime(2026, 8, 24, 9, 30, tzinfo=ET)
    bars = [
        {"ts": monday, "open": 10.0, "high": 10.4, "low": 9.8, "close": 10.2, "volume": 1e6},
        {
            "ts": datetime(2026, 8, 24, 9, 45, tzinfo=ET),
            "open": 10.2,
            "high": 10.5,
            "low": 10.1,
            "close": 10.3,
            "volume": 1e6,
        },
    ]
    got = purchase_next_session_open_mid(bars, signal_session_date=friday.date())
    assert got.filled
    assert got.fill_px == pytest.approx(10.1)
    assert as_et_floor(got.exec_bar_ts) == monday


def as_et_floor(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(ET).to_pydatetime().replace(second=0, microsecond=0)


def test_hard_stop_fills_next_15m_mid_not_same_bar_low():
    entry = datetime(2026, 8, 24, 9, 30, tzinfo=ET)
    b1 = {
        "ts": entry,
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.5,
        "volume": 1,
    }
    b2 = {
        "ts": datetime(2026, 8, 24, 9, 45, tzinfo=ET),
        "open": 100.0,
        "high": 100.2,
        "low": 96.0,
        "close": 96.5,
        "volume": 1,
    }
    b3 = {
        "ts": datetime(2026, 8, 24, 10, 0, tzinfo=ET),
        "open": 96.4,
        "high": 97.0,
        "low": 96.2,
        "close": 96.8,
        "volume": 1,
    }
    by_day = {entry.date(): [b1, b2, b3]}
    got = simulate_deepvue_exit(
        by_day,
        entry_ts=entry,
        entry_px=100.0,
        below_ema={},
        stop_pct=0.03,
    )
    assert got.filled
    assert got.reason == REASON_FILLED
    assert got.exit_reason == EXIT_HARD_STOP
    assert got.sell_px == pytest.approx(96.6)
    assert got.exec_bar_ts == b3["ts"]


def test_two_closes_below_ema_sells_next_15m_mid():
    d0 = datetime(2026, 8, 24, 9, 30, tzinfo=ET)
    d0_last = datetime(2026, 8, 24, 15, 45, tzinfo=ET)
    d1_open = datetime(2026, 8, 25, 9, 30, tzinfo=ET)
    d1_last = datetime(2026, 8, 25, 15, 45, tzinfo=ET)
    d2_open = datetime(2026, 8, 26, 9, 30, tzinfo=ET)

    def bar(ts, px):
        return {"ts": ts, "open": px, "high": px + 0.2, "low": px - 0.2, "close": px, "volume": 1}

    by_day = {
        d0.date(): [bar(d0, 100.0), bar(d0_last, 99.0)],
        d1_open.date(): [bar(d1_open, 98.5), bar(d1_last, 98.0)],
        d2_open.date(): [bar(d2_open, 97.5)],
    }
    below = {d0.date(): True, d1_open.date(): True}
    got = simulate_deepvue_exit(
        by_day,
        entry_ts=d0,
        entry_px=100.0,
        below_ema=below,
        stop_pct=0.50,
        ma_closes=2,
    )
    assert got.filled
    assert got.exit_reason == EXIT_MA_2CLOSE
    assert got.sell_px == pytest.approx(97.5)
    assert got.exec_bar_ts == d2_open


def test_occupancy_drops_fifth_name_and_same_symbol():
    t0 = datetime(2026, 8, 24, 9, 30, tzinfo=ET)
    t1 = datetime(2026, 8, 31, 9, 30, tzinfo=ET)
    far = datetime(2026, 10, 1, 9, 30, tzinfo=ET)
    trades = []
    for sym in ["A", "B", "C", "D", "E"]:
        trades.append({"stock": sym, "buy_ts": t0, "sell_ts": far})
    trades.append({"stock": "A", "buy_ts": t1, "sell_ts": far})
    kept = occupancy_take(trades, max_open=4)
    assert [k["stock"] for k in kept] == ["A", "B", "C", "D"]
