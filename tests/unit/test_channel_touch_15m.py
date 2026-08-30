"""Unit tests for 15m channel-touch live helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.scanning.channel_touch_15m import (
    FROZEN_OVERSHOOT_MIN,
    LIVE_15M_DEFAULTS,
    armed_rows_for_symbol,
    attach_last_prices,
    format_15m_message,
    is_hot_proximity,
    lookback_start,
    passes_h5_stack,
    unique_symbol_day_ok,
    walk_h2_resist_asof,
)


def _rails(n: int = 80, width: float = 2.0, slope: float = 0.01, y0: float = 10.0):
    i = np.arange(n, dtype=float)
    support = y0 + slope * i
    resist = support + width
    return support, resist, width, slope, y0


def test_wait_12_does_not_fill_early_close_above_resist():
    n = 80
    h2 = 20
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    # During wait, close above resist is ignored.
    close[h2 + 5] = resist[h2 + 5] + 0.3
    high[h2 + 5] = close[h2 + 5] + 0.1
    low[h2 + 5] = resist[h2 + 5] - 0.05
    st = walk_h2_resist_asof(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2_idx=h2,
        as_of_i=h2 + 8,
        min_wait=12,
        error_pct=0.24,
    )
    assert st["status"] == "waiting"
    assert st["wait_bars"] == 8


def test_fill_on_first_close_above_resist_after_min_wait():
    n = 80
    h2 = 20
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    fill_i = h2 + 12
    close[fill_i] = resist[fill_i] + 0.25
    high[fill_i] = close[fill_i] + 0.1
    low[fill_i] = resist[fill_i] - 0.05
    st = walk_h2_resist_asof(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2_idx=h2,
        as_of_i=fill_i,
        min_wait=12,
        error_pct=0.24,
        slip=0.001,
    )
    assert st["status"] == "filled"
    assert st["fill_i"] == fill_i
    assert st["fill_px"] is not None
    assert st["overshoot"] is not None


def test_wide_overshoot_when_bar_is_already_through_rail():
    n = 80
    h2 = 20
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    fill_i = h2 + 12
    close[fill_i] = resist[fill_i] + 1.5
    high[fill_i] = close[fill_i] + 0.1
    low[fill_i] = resist[fill_i] + 0.8
    st = walk_h2_resist_asof(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2_idx=h2,
        as_of_i=fill_i,
        min_wait=12,
        error_pct=0.24,
        slip=0.001,
    )
    assert st["status"] == "filled"
    assert st["overshoot"] >= FROZEN_OVERSHOOT_MIN


def test_close_through_support_cancels():
    n = 80
    h2 = 20
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    close[h2 + 3] = support[h2 + 3] - 0.5
    high[h2 + 3] = support[h2 + 3] - 0.1
    low[h2 + 3] = close[h2 + 3] - 0.1
    st = walk_h2_resist_asof(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2_idx=h2,
        as_of_i=h2 + 10,
        min_wait=12,
        error_pct=0.24,
    )
    assert st["status"] == "cancelled"


def test_armed_after_wait_while_still_below_resist():
    n = 80
    h2 = 20
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    st = walk_h2_resist_asof(
        high,
        low,
        close,
        support_x0=0,
        support_y0=y0,
        support_slope=slope,
        width=width,
        h2_idx=h2,
        as_of_i=h2 + 20,
        min_wait=12,
        error_pct=0.24,
    )
    assert st["status"] == "armed"
    assert st["wait_ok"] if "wait_ok" in st else st["wait_bars"] >= 12
    assert st["dist_to_resist_pct"] < 0


def test_proximity_default_is_at_or_above_resist_not_approaching():
    assert is_hot_proximity(101.0, 100.0, below_pct=0.0) is True
    assert is_hot_proximity(100.0, 100.0, below_pct=0.0) is True
    assert is_hot_proximity(98.0, 100.0, below_pct=0.0) is False
    assert is_hot_proximity(98.0, 100.0, below_pct=3.0) is True


def test_h5_stack_requires_prior_vol_and_overshoot():
    row = {"wait_ok": True, "volume_rel_20": 2.1, "overshoot": 0.09}
    assert passes_h5_stack(row) is True
    assert passes_h5_stack({**row, "volume_rel_20": 1.5}) is False
    assert passes_h5_stack({**row, "overshoot": 0.01}) is False
    assert passes_h5_stack({**row, "wait_ok": False}) is False


def test_unique_symbol_day_blocks_second_fill():
    filled = ["AAA|2026-08-30"]
    assert unique_symbol_day_ok("AAA", "2026-08-30 15:45:00", filled) is False
    assert unique_symbol_day_ok("BBB", "2026-08-30 15:45:00", filled) is True
    assert unique_symbol_day_ok("AAA", "2026-08-29 15:45:00", filled) is True


def test_attach_last_prices_marks_hot():
    rows = [{"stock": "AAA", "resist": 50.0, "status": "armed"}]
    out = attach_last_prices(rows, {"AAA": 51.0}, below_pct=0.0)
    assert out[0]["hot"] is True
    assert out[0]["dist_live_pct"] > 0
    out2 = attach_last_prices(rows, {"AAA": 49.0}, below_pct=0.0)
    assert out2[0]["hot"] is False


def test_armed_rows_uses_injected_setup():
    n = 60
    idx = pd.date_range("2024-01-02 14:30", periods=n, freq="15min")
    support, resist, width, slope, y0 = _rails(n)
    close = support + 0.4
    high = close + 0.2
    low = close - 0.2
    vol = np.full(n, 1e5)
    vol[-2] = 3e5
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )
    h2 = 20
    setups = [
        {
            "support_x0": 0,
            "support_y0": y0,
            "support_slope": slope,
            "channel_width": width,
            "h2_idx": h2,
            "h2_date": idx[h2].strftime("%Y-%m-%d"),
            "start_date": idx[0].strftime("%Y-%m-%d"),
            "end_date": idx[h2].strftime("%Y-%m-%d"),
        }
    ]
    rows = armed_rows_for_symbol("TEST", df, setups=setups, min_wait=12, max_span_days=10.0)
    assert len(rows) == 1
    assert rows[0]["stock"] == "TEST"
    assert rows[0]["status"] == "armed"


def test_format_message_no_fills():
    msg = format_15m_message(
        as_of="2026-08-29 15:45:00",
        n_armed=12,
        n_hot=3,
        fills=pd.DataFrame(),
        n_universe=1478,
        stale_warning="IB 15m last bar 2025-12-02",
    )
    assert "No new 15m fills" in msg
    assert "WARNING:" in msg
    assert "vol>=2" in msg


def test_lookback_covers_requested_sessions():
    start = lookback_start(sessions=40, now=pd.Timestamp("2026-08-30"))
    assert (pd.Timestamp("2026-08-30") - pd.Timestamp(start)).days >= 40


def test_live_defaults_are_15m_h5_stack():
    d = LIVE_15M_DEFAULTS
    assert d["timeframe"] == "15m"
    assert d["provider"] == "IB"
    assert d["min_l3_wait_bars"] == 12
    assert d["max_channel_span_days"] == 10.0
    assert d["volume_rel_min"] == 2.0
    assert d["overshoot_min"] == FROZEN_OVERSHOOT_MIN
    assert d["proximity_below_pct"] == 0.0
