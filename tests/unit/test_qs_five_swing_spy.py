"""Unit tests for QS Five-Swing SPY live MOC helpers."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.scanning.qs_five_swing_spy import (
    ET,
    build_developing_ohlc,
    evaluate_live,
    moc_close_target,
    moc_price_buy_ok,
)
from utils.research.qs_five_swing import prepare_frame


def _synth_daily(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.008, size=n)
    close = 400.0 * np.cumprod(1.0 + rets)
    high = close * 1.01
    low = close * 0.99
    # End on a Wednesday so weekday tests are stable
    ts = pd.date_range("2024-01-02", periods=n, freq="B", tz="UTC")
    return pd.DataFrame(
        {"ts": ts, "open": close, "high": high, "low": low, "close": close, "volume": 1e7}
    )


def test_build_developing_ohlc_from_last():
    dev = build_developing_ohlc(prior_close=100.0, last=99.0)
    assert dev["close"] == 99.0
    assert dev["high"] == 100.0
    assert dev["low"] == 99.0


def test_s3_close_ceiling_and_buy_ok():
    df = _synth_daily(60)
    # Force a clear S3: today close under prior 5 lows
    floor = float(df["low"].iloc[-6:-1].min())
    last = floor * 0.99
    df.loc[df.index[-1], "close"] = last
    df.loc[df.index[-1], "low"] = last
    df.loc[df.index[-1], "high"] = floor * 1.01
    d = prepare_frame(df)
    i = len(d) - 1
    dev = {
        "open": float(d.iloc[i - 1]["close"]),
        "high": float(d.iloc[i]["high"]),
        "low": float(d.iloc[i]["low"]),
        "close": last,
    }
    tgt = moc_close_target(d, i, dev, "S3")
    assert tgt is not None
    assert last < tgt
    assert moc_price_buy_ok(last, tgt, sid="S3", signal=True) is True
    assert moc_price_buy_ok(tgt * 1.01, tgt, sid="S3", signal=True) is False


def test_approach_then_buy_now_at_1557():
    df = _synth_daily(80)
    # Build a strong S3 plunge on last bar
    prior_lows = df["low"].iloc[-6:-1]
    floor = float(prior_lows.min())
    last = floor * 0.985
    df.loc[df.index[-1], "open"] = floor
    df.loc[df.index[-1], "high"] = floor * 1.005
    df.loc[df.index[-1], "low"] = last
    df.loc[df.index[-1], "close"] = last
    # Stamp last bar as 2026-09-16 (Wed) so clocks align
    df.loc[df.index[-1], "ts"] = pd.Timestamp("2026-09-16 20:00:00", tz="UTC")

    t_approach = datetime(2026, 9, 16, 15, 45, tzinfo=ET)
    row0, alerts0 = evaluate_live(df, last_price=last, prev_row=None, now=t_approach)
    assert row0
    assert any(a.get("kind") == "approach" for a in alerts0)
    assert row0.get("hot_notified_on", "").endswith(":approach")
    assert row0.get("wait_ok") is True
    target = float(row0["resist"])
    prev = {"hot_notified_on": row0["hot_notified_on"], "status": row0["status"]}

    # Above target -> red, sticky
    row_hi, _ = evaluate_live(
        df,
        last_price=target * 1.02,
        prev_row=prev,
        now=datetime(2026, 9, 16, 15, 50, tzinfo=ET),
    )
    assert row_hi.get("status") in ("waiting", "armed")
    assert row_hi.get("wait_ok") is False

    # 15:56 no buy-now
    _, a56 = evaluate_live(
        df, last_price=last, prev_row=prev, now=datetime(2026, 9, 16, 15, 56, tzinfo=ET)
    )
    assert not any(a.get("kind") == "arm" for a in a56)

    # 15:57 below target -> BUY NOW
    row_ok, a57 = evaluate_live(
        df, last_price=last, prev_row=prev, now=datetime(2026, 9, 16, 15, 57, tzinfo=ET)
    )
    assert row_ok.get("wait_ok") is True
    assert any(a.get("kind") == "arm" and "BUY NOW" in a.get("text", "") for a in a57)

    # 15:57 above target -> no BUY NOW
    _, a_red = evaluate_live(
        df,
        last_price=target * 1.02,
        prev_row=prev,
        now=datetime(2026, 9, 16, 15, 57, tzinfo=ET),
    )
    assert not any(a.get("kind") == "arm" for a in a_red)
