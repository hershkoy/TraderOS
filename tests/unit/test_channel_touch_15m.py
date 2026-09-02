"""Unit tests for 15m channel-touch live helpers."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from utils.scanning.channel_touch_15m import (
    FROZEN_OVERSHOOT_MIN,
    LIVE_15M_DEFAULTS,
    armed_rows_for_symbol,
    attach_last_prices,
    drop_incomplete_15m_bars,
    format_15m_message,
    format_hot_message,
    is_hot_proximity,
    is_prior_et_session,
    live_refresh_symbols,
    lookback_start,
    merge_rescanned_15m_rows,
    newly_hot_rows,
    notify_payloads,
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


def test_newly_hot_once_per_day():
    rows = [
        {"stock": "AAA", "hot": True, "hot_notified_on": None},
        {"stock": "BBB", "hot": True, "hot_notified_on": "2026-08-30"},
        {"stock": "CCC", "hot": False, "hot_notified_on": None},
    ]
    got = newly_hot_rows(rows, today="2026-08-30")
    assert [r["stock"] for r in got] == ["AAA"]
    again = newly_hot_rows(
        [{"stock": "BBB", "hot": True, "hot_notified_on": "2026-08-30"}],
        today="2026-08-31",
    )
    assert [r["stock"] for r in again] == ["BBB"]


def test_notify_payloads_gated_by_settings():
    fills = pd.DataFrame([{"stock": "AAA", "fill_px": 10.0, "resist": 9.9, "wait_bars": 12, "volume_rel_20": 2.2, "overshoot": 0.1}])
    hot = [{"stock": "BBB", "last_price": 11.0, "resist": 10.8, "dist_live_pct": 1.8, "wait_bars": 14, "volume_rel_20": 2.4}]
    none = notify_payloads(
        fills=fills,
        newly_hot=hot,
        settings={"telegram_on_fill": False, "telegram_on_hot": False},
        as_of="2026-08-30 15:45:00",
        n_armed=4,
        n_hot=1,
    )
    assert none == []
    fill_only = notify_payloads(
        fills=fills,
        newly_hot=hot,
        settings={"telegram_on_fill": True, "telegram_on_hot": False},
        as_of="2026-08-30 15:45:00",
        n_armed=4,
        n_hot=1,
    )
    assert len(fill_only) == 1
    assert "AAA" in fill_only[0]
    assert "BUY NOW" in fill_only[0]
    hot_ignored = notify_payloads(
        fills=pd.DataFrame(),
        newly_hot=hot,
        settings={"telegram_on_fill": True, "telegram_on_hot": True},
        as_of="2026-08-30 15:45:00",
        n_armed=4,
        n_hot=1,
    )
    assert hot_ignored == []
    msg = format_hot_message(as_of="t", newly_hot=hot, n_armed=1, n_hot=1)
    assert "BBB" in msg


def test_drop_incomplete_15m_bar_keeps_closed_period():
    now = datetime(2026, 9, 2, 18, 30, 5, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-09-02 18:00:00Z", "2026-09-02 18:15:00Z", "2026-09-02 18:30:00Z"]
            ),
            "close": [1.0, 2.0, 3.0],
        }
    )
    got = drop_incomplete_15m_bars(df, now=now)
    stamps = list(pd.to_datetime(got["timestamp"], utc=True))
    assert stamps == [
        pd.Timestamp("2026-09-02 18:00:00Z"),
        pd.Timestamp("2026-09-02 18:15:00Z"),
    ]


def test_live_refresh_symbols_hot_first():
    rows = [
        {"stock": "WAIT", "timeframe": "15m", "status": "waiting", "hot": False, "resist": 10, "last_price": 9.0, "dist_live_pct": -10},
        {"stock": "ARM", "timeframe": "15m", "status": "armed", "hot": False, "resist": 10, "last_price": 9.4, "dist_live_pct": -6},
        {"stock": "NEAR", "timeframe": "15m", "status": "armed", "hot": False, "resist": 10, "last_price": 9.96, "dist_live_pct": -0.4},
        {"stock": "HOT", "timeframe": "15m", "status": "armed", "hot": True, "resist": 10, "last_price": 10.1, "dist_live_pct": 1.0},
        {"stock": "DAILY", "timeframe": "1d", "status": "armed", "hot": True, "resist": 10, "last_price": 11.0},
    ]
    got = live_refresh_symbols(rows, below_pct=0.5)
    assert got[:2] == ["NEAR", "HOT"]
    assert "ARM" in got and "WAIT" in got
    assert "DAILY" not in got


def test_merge_rescanned_replaces_only_scanned_15m():
    existing = [
        {"stock": "AAA", "timeframe": "15m", "status": "armed", "wait_bars": 12},
        {"stock": "BBB", "timeframe": "15m", "status": "waiting", "wait_bars": 4},
        {"stock": "AAA", "timeframe": "1d", "status": "armed", "wait_bars": 8},
    ]
    new_rows = [{"stock": "AAA", "status": "filled", "wait_bars": 16, "as_of": "2026-09-02 18:15:00"}]
    got = merge_rescanned_15m_rows(existing, new_rows, scanned=["AAA"])
    tf_status = {(r["stock"], r.get("timeframe", "15m")): r["status"] for r in got}
    assert tf_status[("AAA", "15m")] == "filled"
    assert tf_status[("BBB", "15m")] == "waiting"
    assert tf_status[("AAA", "1d")] == "armed"


def test_prior_et_session_uses_new_york_date():
    # 2026-09-01 19:45 UTC = 15:45 EDT Sep 1
    assert is_prior_et_session(
        "2026-09-01 19:45:00",
        now=datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc),
    )
    assert not is_prior_et_session(
        "2026-09-02 18:15:00",
        now=datetime(2026, 9, 2, 18, 30, tzinfo=timezone.utc),
    )

