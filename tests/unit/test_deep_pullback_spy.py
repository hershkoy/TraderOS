"""Unit tests for Deep Pullback SPY feature helpers."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.scanning.deep_pullback_spy import (
    ET,
    _features,
    evaluate_live,
    in_moc_alert_window,
    moc_entry_target,
)


def _synthetic_uptrend(n: int = 250, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.008, size=n)
    close = 100.0 * np.cumprod(1.0 + rets)
    high = close * 1.005
    low = close * 0.995
    ts = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": high, "low": low, "close": close, "volume": 1e6}
    )


def test_features_full_signal_on_forced_plunge():
    df = _synthetic_uptrend(260)
    closes = df["close"].to_numpy(dtype=float)
    sma = float(np.mean(closes[-200:]))
    df.loc[df.index[-2], "close"] = max(sma * 1.05, closes[-2])
    df.loc[df.index[-1], "close"] = max(sma * 1.01, float(df.iloc[-2]["close"]) * 0.97)
    today = float(df.iloc[-1]["close"])
    for i in range(-15, -1):
        df.loc[df.index[i], "close"] = today * 1.02
    df.loc[df.index[-1], "close"] = today
    df["high"] = df["close"] * 1.005

    feat = _features(df["close"].to_numpy(float), df["high"].to_numpy(float))
    assert feat["ok"]
    assert feat["cond_sma"]
    assert feat["cond_low"]
    assert feat["cond_ret"]
    assert feat["full_signal"]


def test_moc_entry_target_caps():
    df = _synthetic_uptrend(260)
    tgt = moc_entry_target(df["close"].to_numpy(float))
    assert tgt["ok"]
    assert tgt["moc_max"] == min(tgt["low_cap"], tgt["ret_cap"])
    assert tgt["sma_floor"] < tgt["prior_close"]


def test_in_moc_window_last_15m():
    # Wednesday 15:45 ET = 15m to close
    now = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    assert in_moc_alert_window(now, window_min=15) is True
    now2 = datetime(2026, 9, 16, 15, 44, tzinfo=ET)
    assert in_moc_alert_window(now2, window_min=15) is False
    now3 = datetime(2026, 9, 16, 16, 1, tzinfo=ET)
    assert in_moc_alert_window(now3, window_min=15) is False


def test_in_moc_buy_now_last_2m():
    from utils.scanning.deep_pullback_spy import in_moc_buy_now_window

    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 58, tzinfo=ET)) is True
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 57, tzinfo=ET)) is False
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 50, tzinfo=ET)) is False


def test_approach_alert_only_in_moc_window_with_target():
    df = _synthetic_uptrend(260)
    closes = df["close"].to_numpy(dtype=float)
    sma = float(np.mean(closes[-200:]))
    # Flat recent window so 15d low is well defined; inject one worse prior day
    # so today's mild dip is near_low + sma but not full (ret not worst).
    floor = max(sma * 1.08, float(closes[-2]))
    for i in range(-20, -1):
        df.loc[df.index[i], "close"] = floor
        df.loc[df.index[i], "high"] = floor * 1.01
    # Prior session was a sharper drop than today's developing print will be
    df.loc[df.index[-2], "close"] = floor * 0.97
    df.loc[df.index[-2], "high"] = floor * 0.98
    df.loc[df.index[-1], "close"] = floor
    df.loc[df.index[-1], "high"] = floor * 1.01
    last = floor * 0.997  # within 0.5% of recent lows, above SMA
    assert last > sma

    early = datetime(2026, 9, 16, 14, 0, tzinfo=ET)
    row_e, alerts_e = evaluate_live(df, last_price=last, prev_row=None, now=early)
    assert row_e.get("status") == "waiting"
    assert not any(a.get("kind") == "approach" for a in alerts_e)

    # 15:44 is outside the 15m approaching window
    almost = datetime(2026, 9, 16, 15, 44, tzinfo=ET)
    _, alerts_a = evaluate_live(df, last_price=last, prev_row=None, now=almost)
    assert not any(a.get("kind") == "approach" for a in alerts_a)

    late = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    row_l, alerts_l = evaluate_live(df, last_price=last, prev_row=None, now=late)
    approach = [a for a in alerts_l if a.get("kind") == "approach"]
    assert len(approach) == 1
    assert "closes at or below" in approach[0]["text"]
    assert "send MOC buy" in approach[0]["text"]
    assert row_l.get("hot_notified_on", "").endswith(":approach")
    assert row_l.get("resist") is not None


def test_evaluate_live_armed_buy_now_only_last_2m():
    df = _synthetic_uptrend(260)
    closes = df["close"].to_numpy(dtype=float)
    sma = float(np.mean(closes[-200:]))
    floor = max(sma * 1.05, float(closes[-1]))
    for i in range(-20, 0):
        df.loc[df.index[i], "close"] = floor * 1.02
        df.loc[df.index[i], "high"] = floor * 1.03
    last = floor * 0.97
    assert last > sma

    # Armed on /hot at 15:50; approaching warn (not buy-now) while in 15m window
    mid = datetime(2026, 9, 16, 15, 50, tzinfo=ET)
    row_m, alerts_m = evaluate_live(df, last_price=last, prev_row=None, now=mid)
    assert row_m.get("status") == "armed"
    assert row_m.get("hot") is True
    assert not any(a.get("kind") == "arm" for a in alerts_m)
    assert any(a.get("kind") == "approach" and "15:58" in a.get("text", "") for a in alerts_m)

    late = datetime(2026, 9, 16, 15, 58, tzinfo=ET)
    row, alerts = evaluate_live(
        df,
        last_price=last,
        prev_row={"hot_notified_on": row_m.get("hot_notified_on")},
        now=late,
    )
    assert row.get("status") == "armed"
    assert row.get("hot") is True
    assert any(a.get("kind") == "arm" and "MOC buy now" in a.get("text", "") for a in alerts)


def test_evaluate_live_exit_when_in_position():
    df = _synthetic_uptrend(260)
    prev = {
        "status": "filled",
        "fill_px": float(df.iloc[-5]["close"]),
        "h2_time": df.iloc[-5]["ts"],
    }
    prior_high = float(df.iloc[-1]["high"])
    row, alerts = evaluate_live(
        df, last_price=prior_high * 1.01, prev_row=prev, now=datetime(2026, 9, 16, 15, 50, tzinfo=ET)
    )
    assert row.get("status") == "exit_signal"
    assert any(a.get("kind") == "exit" for a in alerts)
