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
    in_moc_buy_now_window,
    moc_entry_target,
    moc_price_buy_ok,
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


def _near_low_setup(df: pd.DataFrame) -> float:
    """Mild near-low (approaching, not full). Returns last price."""
    closes = df["close"].to_numpy(dtype=float)
    sma = float(np.mean(closes[-200:]))
    floor = max(sma * 1.08, float(closes[-2]))
    for i in range(-20, -1):
        df.loc[df.index[i], "close"] = floor
        df.loc[df.index[i], "high"] = floor * 1.01
    df.loc[df.index[-2], "close"] = floor * 0.97
    df.loc[df.index[-2], "high"] = floor * 0.98
    df.loc[df.index[-1], "close"] = floor
    df.loc[df.index[-1], "high"] = floor * 1.01
    last = floor * 0.997
    assert last > sma
    return last


def _full_plunge_setup(df: pd.DataFrame) -> float:
    closes = df["close"].to_numpy(dtype=float)
    sma = float(np.mean(closes[-200:]))
    floor = max(sma * 1.05, float(closes[-1]))
    for i in range(-20, 0):
        df.loc[df.index[i], "close"] = floor * 1.02
        df.loc[df.index[i], "high"] = floor * 1.03
    last = floor * 0.97
    assert last > sma
    return last


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
    now = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    assert in_moc_alert_window(now, window_min=15) is True
    now2 = datetime(2026, 9, 16, 15, 44, tzinfo=ET)
    assert in_moc_alert_window(now2, window_min=15) is False
    now3 = datetime(2026, 9, 16, 16, 1, tzinfo=ET)
    assert in_moc_alert_window(now3, window_min=15) is False


def test_in_moc_buy_now_from_1557():
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 57, tzinfo=ET)) is True
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 58, tzinfo=ET)) is True
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 56, tzinfo=ET)) is False
    assert in_moc_buy_now_window(datetime(2026, 9, 16, 15, 50, tzinfo=ET)) is False


def test_approach_alert_only_in_moc_window_with_target():
    df = _synthetic_uptrend(260)
    last = _near_low_setup(df)

    early = datetime(2026, 9, 16, 14, 0, tzinfo=ET)
    row_e, alerts_e = evaluate_live(df, last_price=last, prev_row=None, now=early)
    assert row_e.get("status") == "waiting"
    assert not any(a.get("kind") == "approach" for a in alerts_e)

    almost = datetime(2026, 9, 16, 15, 44, tzinfo=ET)
    _, alerts_a = evaluate_live(df, last_price=last, prev_row=None, now=almost)
    assert not any(a.get("kind") == "approach" for a in alerts_a)

    late = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    row_l, alerts_l = evaluate_live(df, last_price=last, prev_row=None, now=late)
    approach = [a for a in alerts_l if a.get("kind") == "approach"]
    assert len(approach) == 1
    assert "stays at or below" in approach[0]["text"]
    assert "15:57" in approach[0]["text"]
    assert row_l.get("hot_notified_on", "").endswith(":approach")
    assert row_l.get("resist") is not None
    assert "wait_ok" in row_l  # green/red from current vs target


def test_sticky_watch_tracks_current_vs_target_green_red():
    df = _synthetic_uptrend(260)
    last = _near_low_setup(df)
    t0 = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    row0, alerts0 = evaluate_live(df, last_price=last, prev_row=None, now=t0)
    assert any(a.get("kind") == "approach" for a in alerts0)
    target = float(row0["resist"])
    prev = {"hot_notified_on": row0["hot_notified_on"], "status": row0["status"]}

    # Above target -> red (wait_ok False), row still sticky
    high = target * 1.02
    row_hi, _ = evaluate_live(
        df, last_price=high, prev_row=prev, now=datetime(2026, 9, 16, 15, 46, tzinfo=ET)
    )
    assert row_hi.get("status") == "waiting"
    assert row_hi.get("wait_ok") is False
    assert float(row_hi["last_price"]) == high
    assert not moc_price_buy_ok(high, row_hi.get("_target") or {"ok": False})

    # Back below target -> green
    low = target * 0.99
    row_lo, _ = evaluate_live(
        df,
        last_price=low,
        prev_row={"hot_notified_on": prev["hot_notified_on"], "status": "waiting"},
        now=datetime(2026, 9, 16, 15, 47, tzinfo=ET),
    )
    assert row_lo.get("wait_ok") is True
    assert float(row_lo["last_price"]) == low


def test_buy_now_at_1557_only_if_below_target():
    df = _synthetic_uptrend(260)
    last = _full_plunge_setup(df)
    mid = datetime(2026, 9, 16, 15, 50, tzinfo=ET)
    row_m, alerts_m = evaluate_live(df, last_price=last, prev_row=None, now=mid)
    assert any(a.get("kind") == "approach" for a in alerts_m)
    assert not any(a.get("kind") == "arm" for a in alerts_m)
    target = float(row_m["resist"])
    prev = {"hot_notified_on": row_m["hot_notified_on"]}

    # 15:56 still not buy-now window
    row_56, alerts_56 = evaluate_live(
        df, last_price=last, prev_row=prev, now=datetime(2026, 9, 16, 15, 56, tzinfo=ET)
    )
    assert not any(a.get("kind") == "arm" for a in alerts_56)

    # 15:57 above target -> no buy now
    row_red, alerts_red = evaluate_live(
        df,
        last_price=target * 1.01,
        prev_row=prev,
        now=datetime(2026, 9, 16, 15, 57, tzinfo=ET),
    )
    assert row_red.get("wait_ok") is False
    assert not any(a.get("kind") == "arm" for a in alerts_red)

    # 15:57 below target -> BUY NOW
    row_ok, alerts_ok = evaluate_live(
        df,
        last_price=min(last, target * 0.99),
        prev_row=prev,
        now=datetime(2026, 9, 16, 15, 57, tzinfo=ET),
    )
    assert row_ok.get("wait_ok") is True
    assert any(a.get("kind") == "arm" and "BUY NOW" in a.get("text", "") for a in alerts_ok)


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
