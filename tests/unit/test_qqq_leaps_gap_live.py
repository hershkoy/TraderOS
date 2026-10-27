"""Unit tests for QQQ LEAPS gap live evaluate helpers."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.scanning.qqq_leaps_gap_live import (
    ET,
    classify_gap,
    evaluate_live,
    gap_pct,
    levels_from_completed,
)


def _daily_above_sma(n: int = 160, last_close: float = 500.0) -> pd.DataFrame:
    """Monotone uptrend so prior close is above SMA100."""
    close = np.linspace(last_close * 0.7, last_close, n)
    ts = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "ts": ts,
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": 1e6,
        }
    )


def test_gap_pct_and_levels():
    assert abs(gap_pct(99.0, 100.0) + 1.0) < 1e-9
    closes = np.linspace(400, 500, 120)
    lv = levels_from_completed(closes)
    assert lv["ok"]
    assert lv["prior_close"] == 500.0
    assert abs(lv["strat1_target"] - 495.0) < 1e-6
    assert abs(lv["strat2_target"] - 490.0) < 1e-6


def test_classify_soft_and_full():
    soft = classify_gap(-0.75, above_sma=True)
    assert soft["approaching"]
    assert soft["near_s1"]
    assert not soft["full_signal"]
    full = classify_gap(-1.2, above_sma=True)
    assert full["full_s1"]
    assert full["full_signal"]
    assert full["active_mode"] == "strat1"
    s2 = classify_gap(-2.1, above_sma=False)
    assert s2["full_s2"]
    assert s2["active_mode"] == "strat2"
    assert s2["contracts"] == 2


def test_evaluate_approach_premarket():
    df = _daily_above_sma(160, last_close=500.0)
    now = datetime(2025, 6, 3, 9, 0, tzinfo=ET)  # Tuesday premarket
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2025-06-02 20:00", tz="UTC")
    prior = float(df.iloc[-1]["close"])
    last = prior * (1.0 - 0.008)  # soft near -1%
    row, alerts = evaluate_live(df, last_price=last, session_open=None, now=now)
    assert row
    assert row["status"] == "waiting"
    assert row["hot"]
    assert any(a["kind"] == "approach" for a in alerts)


def test_evaluate_approach_once_then_silent():
    df = _daily_above_sma(160, last_close=500.0)
    now = datetime(2025, 6, 3, 9, 0, tzinfo=ET)
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2025-06-02 20:00", tz="UTC")
    prior = float(df.iloc[-1]["close"])
    last = prior * (1.0 - 0.008)
    row, alerts = evaluate_live(df, last_price=last, session_open=None, now=now)
    assert len([a for a in alerts if a["kind"] == "approach"]) == 1
    assert row["hot_notified_on"] == "2025-06-03:approach"
    later = datetime(2025, 6, 3, 9, 1, tzinfo=ET)
    _, again = evaluate_live(
        df,
        last_price=last,
        session_open=None,
        now=later,
        prev_row={"hot_notified_on": row["hot_notified_on"], "status": "waiting"},
    )
    assert again == []


def test_legacy_date_stamp_does_not_repeat_approach():
    """DATE column used to store YYYY-MM-DD and the suffix was lost."""
    df = _daily_above_sma(160, last_close=500.0)
    now = datetime(2025, 6, 3, 8, 5, tzinfo=ET)
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2025-06-02 20:00", tz="UTC")
    prior = float(df.iloc[-1]["close"])
    last = prior * (1.0 - 0.0136)  # already through -1%
    row, alerts = evaluate_live(
        df,
        last_price=last,
        session_open=None,
        now=now,
        prev_row={"hot_notified_on": "2025-06-03", "status": "waiting"},
    )
    assert alerts == []
    assert row["hot_notified_on"] == "2025-06-03:approach"


def test_preopen_full_gap_says_already_there():
    df = _daily_above_sma(160, last_close=713.44)
    now = datetime(2026, 10, 23, 8, 5, tzinfo=ET)
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2026-10-22 20:00", tz="UTC")
    _, alerts = evaluate_live(df, last_price=703.75, session_open=None, now=now)
    assert len(alerts) == 1
    text = alerts[0]["text"]
    assert "already at gap_est=" in text
    assert "before the open" in text
    assert "approaching" not in text


def test_evaluate_arm_after_open():
    df = _daily_above_sma(160, last_close=500.0)
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2025-06-02 20:00", tz="UTC")
    prior = float(df.iloc[-1]["close"])
    now = datetime(2025, 6, 3, 9, 45, tzinfo=ET)
    open_px = prior * 0.985  # -1.5% gap -> strat1
    row, alerts = evaluate_live(
        df,
        last_price=open_px,
        session_open=open_px,
        now=now,
        prev_row={"hot_notified_on": "2025-06-03:approach"},
    )
    assert row["status"] == "armed"
    assert any(a["kind"] == "arm" for a in alerts)
    assert row["wait_ok"]
    assert row["hot_notified_on"] == "2025-06-03:arm"
    _, again = evaluate_live(
        df,
        last_price=open_px,
        session_open=open_px,
        now=datetime(2025, 6, 3, 9, 46, tzinfo=ET),
        prev_row={"hot_notified_on": row["hot_notified_on"], "status": "armed"},
    )
    assert again == []
