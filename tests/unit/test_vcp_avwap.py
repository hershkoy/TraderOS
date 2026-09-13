"""Causal VCP AVWAP helpers (no TimescaleDB)."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from utils.research.realistic_exits import EXIT_HARD_STOP, REASON_FILLED
from utils.research.vcp_avwap import (
    EXIT_WEEKLY_MA,
    VcpParams,
    arm_sessions,
    contractions_ok,
    find_vcp_breakout_fill,
    pullback_depths,
    simulate_vcp_exit,
    volume_dryup,
    zigzag_swings,
)

ET = ZoneInfo("America/New_York")


def test_zigzag_records_contracting_pullbacks():
    closes = [100.0, 110.0, 103.0, 109.0, 105.0, 108.0]
    swings = zigzag_swings(closes, pct=0.03)
    pbs = pullback_depths(swings)
    assert contractions_ok(pbs, min_contractions=2)
    assert abs(pbs[0]) > abs(pbs[1])


def test_volume_dryup_requires_later_window_lower():
    high = [100.0] * 20
    low = [10.0] * 20
    assert volume_dryup(high + low, window=20)
    assert not volume_dryup(low + high, window=20)
    assert volume_dryup([1.0] * 10, window=20)


def _vcp_daily(n_warm: int = 252) -> pd.DataFrame:
    """Flat warmup then a 40-session VCP with two >3% contracting pullbacks."""
    idx = pd.bdate_range("2018-01-02", periods=n_warm + 41)
    close = np.linspace(99.5, 100.0, n_warm).tolist()
    # Peak 120, pullback 108 (~10%), peak 118, pullback 112 (~5%), grind.
    base = (
        list(np.linspace(100.0, 120.0, 5))
        + list(np.linspace(118.0, 108.0, 5))
        + list(np.linspace(110.0, 118.0, 5))
        + list(np.linspace(117.0, 112.0, 5))
        + list(np.linspace(112.5, 116.0, 20))
    )
    close.extend(base)
    close.append(116.2)
    high = [c * 1.004 for c in close]
    high[n_warm + 4] = 120.5
    high[n_warm + 14] = 118.4
    low = [c * 0.996 for c in close]
    low[n_warm + 9] = 107.5
    low[n_warm + 19] = 111.5
    vol = [2_000_000.0] * n_warm + [2_000_000.0] * 20 + [800_000.0] * 20 + [900_000.0]
    open_px = close[:]
    return pd.DataFrame(
        {"open": open_px, "high": high, "low": low, "close": close, "volume": vol},
        index=idx[: len(close)],
    )


def test_arm_sessions_uses_prior_day_not_watch_day():
    daily = _vcp_daily()
    rows = arm_sessions(daily, VcpParams())
    assert rows, "expected at least one armed VCP near 52w"
    last = rows[-1]
    assert last["watch_date"] > last["signal_date"]
    assert last["pivot_price"] >= 118.0
    assert last["near_52w"] >= 0.92


def test_breakout_fill_is_signal_close_not_bar_low():
    pivot = 110.0
    avwap = 105.0
    prior = [100.0] * 40
    miss = {
        "ts": datetime(2026, 8, 24, 9, 30, tzinfo=ET),
        "open": 108.0,
        "high": 109.0,
        "low": 107.5,
        "close": 108.5,
        "volume": 50.0,
    }
    hit = {
        "ts": datetime(2026, 8, 24, 9, 45, tzinfo=ET),
        "open": 109.5,
        "high": 112.0,
        "low": 109.0,
        "close": 111.5,
        "volume": 400.0,
    }
    nxt = {
        "ts": datetime(2026, 8, 24, 10, 0, tzinfo=ET),
        "open": 111.6,
        "high": 113.0,
        "low": 111.0,
        "close": 112.0,
        "volume": 80.0,
    }
    got = find_vcp_breakout_fill(
        [miss, hit, nxt],
        pivot=pivot,
        avwap=avwap,
        rvol_min=1.5,
        prior_volumes=prior,
        rvol_window_bars=20,
        fill_mode="signal-close",
        naive_tz="America/New_York",
    )
    assert got.filled
    assert got.fill_px == pytest.approx(111.5)
    assert got.fill_px != pytest.approx(109.0)


def test_hard_stop_fills_next_15m_mid_not_same_bar_low():
    entry = datetime(2026, 8, 24, 9, 30, tzinfo=ET)
    b0 = {
        "ts": entry,
        "open": 111.5,
        "high": 112.0,
        "low": 111.0,
        "close": 111.5,
        "volume": 1,
    }
    b1 = {
        "ts": datetime(2026, 8, 24, 9, 45, tzinfo=ET),
        "open": 111.0,
        "high": 111.2,
        "low": 100.0,
        "close": 101.0,
        "volume": 1,
    }
    b2 = {
        "ts": datetime(2026, 8, 24, 10, 0, tzinfo=ET),
        "open": 101.0,
        "high": 103.0,
        "low": 100.5,
        "close": 102.0,
        "volume": 1,
    }
    by_day = {entry.date(): [b0, b1, b2]}
    exited = simulate_vcp_exit(
        by_day,
        entry_ts=entry,
        entry_px=111.5,
        atr_at_entry=2.0,
        atr_mult=1.0,
        weekly_below={},
        avwap_by_prior_session={},
    )
    assert exited.filled
    assert exited.reason == REASON_FILLED
    assert exited.exit_reason == EXIT_HARD_STOP
    assert exited.sell_px == pytest.approx(101.75)
    assert exited.exec_bar_ts == b2["ts"]


def test_weekly_ma_exit_fills_next_15m_mid():
    entry = datetime(2026, 8, 21, 15, 30, tzinfo=ET)
    last = {
        "ts": datetime(2026, 8, 21, 15, 45, tzinfo=ET),
        "open": 110.0,
        "high": 111.0,
        "low": 109.5,
        "close": 110.2,
        "volume": 1,
    }
    nxt = {
        "ts": datetime(2026, 8, 24, 9, 30, tzinfo=ET),
        "open": 109.0,
        "high": 110.0,
        "low": 108.0,
        "close": 109.5,
        "volume": 1,
    }
    entry_bar = {
        "ts": entry,
        "open": 111.0,
        "high": 111.5,
        "low": 110.5,
        "close": 111.0,
        "volume": 1,
    }
    by_day = {
        entry.date(): [entry_bar, last],
        nxt["ts"].date(): [nxt],
    }
    exited = simulate_vcp_exit(
        by_day,
        entry_ts=entry,
        entry_px=111.0,
        atr_at_entry=50.0,
        atr_mult=1.0,
        weekly_below={entry.date(): True},
        avwap_by_prior_session={},
    )
    assert exited.filled
    assert exited.exit_reason == EXIT_WEEKLY_MA
    assert exited.sell_px == pytest.approx(109.0)
    assert exited.exec_bar_ts == nxt["ts"]
