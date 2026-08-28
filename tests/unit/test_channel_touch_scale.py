"""Unit tests for 15m channel-touch scale helpers."""
from __future__ import annotations

from types import SimpleNamespace

from utils.research.channel_touch_scale import (
    BARS_PER_RTH_SESSION,
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
    PRESET_15M,
    VOL_SCALE,
    apply_daily_long_history_defaults,
    overlay_preset,
    rs_bar_lookbacks,
    scale_pct_points,
)


def test_rs_lookbacks_are_session_counts():
    assert rs_bar_lookbacks(1) == (63, 126)
    assert rs_bar_lookbacks(26) == (63 * 26, 126 * 26)


def test_vol_scale_shrinks_daily_pct():
    assert abs(scale_pct_points(1.2) - (1.2 / VOL_SCALE)) < 1e-12
    assert PRESET_15M["trail_pct"] == round(0.10 / VOL_SCALE, 3)
    assert PRESET_15M["stop_pct_ceil"] == round(0.06 / VOL_SCALE, 3)


def test_preset_15m_is_intraday_not_daily_bars():
    assert PRESET_15M["timeframe"] == "15m"
    assert PRESET_15M["provider"] == "IB"
    assert PRESET_15M["pivot_len"] == 8
    assert PRESET_15M["pivot_len"] < 15
    assert PRESET_15M["window_bars"] == BARS_PER_RTH_SESSION * 15
    assert PRESET_15M["window_step_bars"] == BARS_PER_RTH_SESSION * 5
    assert PRESET_15M["bars_per_session"] == 26
    assert PRESET_15M["max_channel_span_days"] == 10.0


def test_overlay_skips_explicit_cli():
    ns = SimpleNamespace(pivot_len=15, trail_pct=0.10)
    overlay_preset(ns, {"pivot_len": 8, "trail_pct": 0.02}, ["--pivot-len", "15"])
    assert ns.pivot_len == 15
    assert ns.trail_pct == 0.02


def test_apply_daily_long_history_defaults():
    ns = SimpleNamespace(preset="", timeframe="1d", no_window_scan=False, window_bars=0, window_step_bars=0)
    apply_daily_long_history_defaults(ns)
    assert ns.window_bars == DAILY_WINDOW_BARS
    assert ns.window_step_bars == DAILY_WINDOW_STEP_BARS

    ns2 = SimpleNamespace(preset="15m", timeframe="15m", no_window_scan=False, window_bars=0, window_step_bars=0)
    apply_daily_long_history_defaults(ns2)
    assert ns2.window_bars == 0

    ns3 = SimpleNamespace(preset="", timeframe="1d", no_window_scan=True, window_bars=0, window_step_bars=0)
    apply_daily_long_history_defaults(ns3)
    assert ns3.window_bars == 0
