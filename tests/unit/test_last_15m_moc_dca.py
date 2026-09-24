"""Unit tests for 3-day MOC DCA on last-15m H2."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.last_15m_moc_dca import (
    EXIT_SCALEIN_STOP,
    REASON_NO_FILLS,
    REASON_OK,
    apply_moc_dca,
    resist_at_session,
)

ET = ZoneInfo("America/New_York")


def _et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET)


def _bar(ts, o, h, l, c):
    return {"ts": ts, "open": o, "high": h, "low": l, "close": c, "volume": 100.0}


def _daily(days_ohlc):
    idx = [pd.Timestamp(d) for d, _ in days_ohlc]
    rows = []
    for _d, ohlc in days_ohlc:
        rows.append(ohlc)
    return pd.DataFrame(rows, index=idx)


def _row(**kwargs):
    base = {
        "stock": "TEST",
        "buy_date": "2024-02-08",
        "buy_time": "2024-02-08 15:45",
        "buy_price": 10.10,
        "signal_x": 10.00,
        "slope_pct_per_bar": 0.0,
        "l1_price": 9.0,
        "channel_width": 1.0,
        "channel_pos": 1.1,
        "atr_1d_pct": 2.0,
        "atr_pct": 2.0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_resist_uses_signal_x_and_slope():
    row = _row(signal_x=10.0, slope_pct_per_bar=1.0, l1_price=10.0)
    assert resist_at_session(row, bars_after_fill=0) == 10.0
    # slope = 1% of l1 = 0.1 per bar
    assert abs(resist_at_session(row, bars_after_fill=2) - 10.2) < 1e-9


def test_three_moc_buys_avg_cost_and_allocated_pnl():
    days = [
        (date(2024, 2, 8), {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0}),
        (date(2024, 2, 9), {"open": 10.0, "high": 10.3, "low": 9.95, "close": 10.5}),
        (date(2024, 2, 12), {"open": 10.5, "high": 10.6, "low": 10.4, "close": 11.0}),
        (date(2024, 2, 13), {"open": 11.0, "high": 11.1, "low": 10.9, "close": 11.0}),
    ]
    panel = _daily(days)
    by_day = {
        date(2024, 2, 8): [_bar(_et(2024, 2, 8, 15, 45), 10.0, 10.2, 9.9, 10.0)],
        date(2024, 2, 9): [_bar(_et(2024, 2, 9, 15, 45), 10.0, 10.3, 9.95, 10.5)],
        date(2024, 2, 12): [_bar(_et(2024, 2, 12, 15, 45), 10.5, 10.6, 10.4, 11.0)],
        # Force trail: peak after entry, then dump through 10% trail from ~11.
        date(2024, 2, 13): [
            _bar(_et(2024, 2, 13, 9, 30), 11.0, 12.0, 11.0, 11.9),
            _bar(_et(2024, 2, 13, 9, 45), 11.9, 11.9, 10.5, 10.6),
            _bar(_et(2024, 2, 13, 10, 0), 10.6, 10.7, 10.5, 10.55),
        ],
    }
    got = apply_moc_dca(_row(), panel_1d=panel, by_day_15m=by_day, n_tranches=3)
    assert got.filled is True
    assert got.reason == REASON_OK
    assert got.n_tranches_filled == 3
    assert got.n_paused == 0
    # Equal dollar: avg of 10, 10.5, 11 = not share-weighted of prices alone
    # shares = (1/3)/10 + (1/3)/10.5 + (1/3)/11
    s = (1 / 3) / 10 + (1 / 3) / 10.5 + (1 / 3) / 11
    avg = 1.0 / s
    assert abs(got.avg_cost - avg) < 1e-6
    assert got.sell_px is not None
    assert got.gain_allocated_pct is not None


def test_pause_below_rail_skips_tranche():
    days = [
        (date(2024, 2, 8), {"open": 10.0, "high": 10.2, "low": 9.9, "close": 10.2}),
        # Below rail 10.0 -> pause, but low stays above ATR hard-stop (~9.79)
        (date(2024, 2, 9), {"open": 9.95, "high": 9.98, "low": 9.90, "close": 9.92}),
        (date(2024, 2, 12), {"open": 10.1, "high": 10.4, "low": 10.0, "close": 10.3}),
        (date(2024, 2, 13), {"open": 10.3, "high": 10.5, "low": 9.0, "close": 9.2}),
    ]
    panel = _daily(days)
    by_day = {
        date(2024, 2, 8): [_bar(_et(2024, 2, 8, 15, 45), 10.0, 10.2, 9.9, 10.2)],
        date(2024, 2, 9): [_bar(_et(2024, 2, 9, 15, 45), 9.95, 9.98, 9.90, 9.92)],
        date(2024, 2, 12): [_bar(_et(2024, 2, 12, 15, 45), 10.1, 10.4, 10.0, 10.3)],
        date(2024, 2, 13): [
            _bar(_et(2024, 2, 13, 9, 30), 10.3, 10.5, 10.2, 10.4),
            _bar(_et(2024, 2, 13, 9, 45), 10.4, 10.4, 9.0, 9.2),
            _bar(_et(2024, 2, 13, 10, 0), 9.2, 9.3, 9.0, 9.1),
        ],
    }
    got = apply_moc_dca(_row(), panel_1d=panel, by_day_15m=by_day, n_tranches=3)
    assert got.filled is True
    assert got.n_paused == 1
    assert got.n_tranches_filled == 2
    assert abs(got.capital_deployed - 2.0 / 3.0) < 1e-9
    # Allocated return uses full 1.0 notional (idle cash)
    assert got.gain_allocated_pct is not None
    assert got.gain_deployed_pct is not None
    assert abs(got.gain_allocated_pct - got.gain_deployed_pct * (2.0 / 3.0)) < 0.05


def test_scalein_stop_exits_before_later_tranches():
    # ATR 2% * k=2 = 4% stop from first fill 10 -> hard 9.6
    days = [
        (date(2024, 2, 8), {"open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0}),
        # Low tags stop; pause remaining
        (date(2024, 2, 9), {"open": 9.9, "high": 9.95, "low": 9.4, "close": 9.5}),
        (date(2024, 2, 12), {"open": 9.5, "high": 9.6, "low": 9.4, "close": 9.55}),
    ]
    panel = _daily(days)
    by_day = {
        date(2024, 2, 8): [_bar(_et(2024, 2, 8, 15, 45), 10.0, 10.1, 9.9, 10.0)],
        date(2024, 2, 9): [_bar(_et(2024, 2, 9, 15, 45), 9.9, 9.95, 9.4, 9.5)],
    }
    got = apply_moc_dca(
        _row(atr_1d_pct=2.0),
        panel_1d=panel,
        by_day_15m=by_day,
        n_tranches=3,
    )
    assert got.filled is True
    assert got.exit_reason == EXIT_SCALEIN_STOP
    assert got.n_tranches_filled == 1
    assert got.sell_px is not None
    assert got.sell_px < 10.0


def test_all_paused_is_no_fill():
    days = [
        (date(2024, 2, 8), {"open": 9.5, "high": 9.6, "low": 9.4, "close": 9.5}),
        (date(2024, 2, 9), {"open": 9.4, "high": 9.5, "low": 9.3, "close": 9.4}),
        (date(2024, 2, 12), {"open": 9.3, "high": 9.4, "low": 9.2, "close": 9.3}),
    ]
    panel = _daily(days)
    by_day = {
        date(2024, 2, 8): [_bar(_et(2024, 2, 8, 15, 45), 9.5, 9.6, 9.4, 9.5)],
        date(2024, 2, 9): [_bar(_et(2024, 2, 9, 15, 45), 9.4, 9.5, 9.3, 9.4)],
        date(2024, 2, 12): [_bar(_et(2024, 2, 12, 15, 45), 9.3, 9.4, 9.2, 9.3)],
    }
    got = apply_moc_dca(_row(signal_x=10.0), panel_1d=panel, by_day_15m=by_day)
    assert got.filled is False
    assert got.reason == REASON_NO_FILLS
    assert got.n_paused == 3
