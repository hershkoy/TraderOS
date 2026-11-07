"""Unit tests for Patrick Nil P/B daily claim-check helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.research.patrick_nil_pb import (
    BOOK_FAKE,
    BOOK_RANGE,
    HTF_BULL,
    NilParams,
    collect_symbol_trades,
    count_boundary_touches,
    detect_impulse_at,
    weekly_htf_bias,
)


def _atr_arr(n: int, val: float = 1.0) -> np.ndarray:
    return np.full(n, float(val), dtype=float)


def test_detect_impulse_p_and_b():
    n = 30
    close = np.linspace(100.0, 100.0, n)
    high = close + 0.5
    low = close - 0.5
    atr = _atr_arr(n, 1.0)
    # Strong up impulse over last 10 bars
    for i in range(10, 21):
        close[i] = 100.0 + (i - 10) * 0.4
        high[i] = close[i] + 0.2
        low[i] = close[i] - 0.2
    params = NilParams(impulse_lookback=10, impulse_atr_mult=1.5, near_extreme_frac=0.25)
    assert detect_impulse_at(close, high, low, atr, 20, params) == "P"

    close2 = np.linspace(100.0, 100.0, n)
    high2 = close2 + 0.5
    low2 = close2 - 0.5
    for i in range(10, 21):
        close2[i] = 100.0 - (i - 10) * 0.4
        high2[i] = close2[i] + 0.2
        low2[i] = close2[i] - 0.2
    assert detect_impulse_at(close2, high2, low2, atr, 20, params) == "B"


def test_count_boundary_touches_three():
    high = np.array([10.0, 12.0, 11.0, 12.05, 10.5, 10.0, 11.0], dtype=float)
    low = np.array([9.0, 10.0, 10.0, 11.0, 10.0, 9.95, 10.2], dtype=float)
    # range 10..12
    n = count_boundary_touches(high, low, 0, 6, 12.0, 10.0, touch_tol_pct=0.15)
    assert n >= 3


def test_fake_breakout_next_open_fill():
    """P impulse -> range with 3 touches -> upside fake -> short next open."""
    rows = []
    px = 100.0
    # Warmup flat
    for i in range(40):
        rows.append(
            {
                "ts": pd.Timestamp("2019-01-02") + pd.Timedelta(days=i),
                "open": px,
                "high": px + 0.3,
                "low": px - 0.3,
                "close": px,
                "volume": 1e6,
            }
        )
    # Up impulse ~10 bars (+2.0 total vs ATR~1)
    for i in range(10):
        px = 100.0 + (i + 1) * 0.25
        rows.append(
            {
                "ts": pd.Timestamp("2019-02-11") + pd.Timedelta(days=i),
                "open": px - 0.1,
                "high": px + 0.15,
                "low": px - 0.15,
                "close": px,
                "volume": 1e6,
            }
        )
    # Pause bar (confirms impulse end) then horizontal range around base.
    base = px
    rows.append(
        {
            "ts": pd.Timestamp("2019-02-21"),
            "open": base,
            "high": base + 0.15,
            "low": base - 0.15,
            "close": base - 0.05,
            "volume": 1e6,
        }
    )
    rh, rl = base + 0.5, base - 0.5
    seq = [
        (rh, (rh + rl) / 2),  # touch high
        (rl + 0.1, rl),  # touch low
        (rh, rl + 0.2),  # touch high
        ((rh + rl) / 2, (rh + rl) / 2),
        ((rh + rl) / 2, (rh + rl) / 2),
        (rh - 0.05, rl + 0.1),
    ]
    for i, (hi, lo) in enumerate(seq):
        mid = 0.5 * (hi + lo)
        rows.append(
            {
                "ts": pd.Timestamp("2019-02-22") + pd.Timedelta(days=i),
                "open": mid,
                "high": hi,
                "low": lo,
                "close": mid,
                "volume": 1e6,
            }
        )
    # Breakout bar close above range high
    rows.append(
        {
            "ts": pd.Timestamp("2019-02-27"),
            "open": rh,
            "high": rh + 0.8,
            "low": rh - 0.1,
            "close": rh + 0.6,
            "volume": 1e6,
        }
    )
    # Fake: close back inside
    rows.append(
        {
            "ts": pd.Timestamp("2019-02-28"),
            "open": rh + 0.2,
            "high": rh + 0.3,
            "low": rl + 0.2,
            "close": (rh + rl) / 2,
            "volume": 1e6,
        }
    )
    # Entry day open + path to target/stop
    entry_open = (rh + rl) / 2
    rows.append(
        {
            "ts": pd.Timestamp("2019-03-01"),
            "open": entry_open,
            "high": rh + 0.2,
            "low": rl - 0.1,
            "close": rl,
            "volume": 1e6,
        }
    )
    # Extra bars
    for i in range(5):
        rows.append(
            {
                "ts": pd.Timestamp("2019-03-02") + pd.Timedelta(days=i),
                "open": rl,
                "high": rl + 0.2,
                "low": rl - 0.2,
                "close": rl,
                "volume": 1e6,
            }
        )

    df = pd.DataFrame(rows)
    params = NilParams(
        impulse_lookback=10,
        impulse_atr_mult=1.0,
        min_range_bars=5,
        max_range_bars=20,
        max_range_atr=5.0,
        min_touches=3,
        touch_tol_pct=0.5,
        atr_len=5,
        weekly_sma=3,
        htf_neutral_pct=5.0,
    )
    trades = collect_symbol_trades(
        df, symbol="TEST", cap_bucket="large", params=params, books=(BOOK_FAKE,)
    )
    assert trades, "expected at least one fake-breakout trade"
    tr = trades[0]
    assert tr["book"] == BOOK_FAKE
    assert tr["side"] == "short"
    # Causal next-open fill after a completed signal bar (not same-bar).
    assert pd.Timestamp(tr["buy_date"]) > pd.Timestamp("2019-02-21")
    assert float(tr["buy_price"]) > 0
    # Stop beyond failed breakout extreme / above range high
    assert float(tr["stop_px"]) >= rh


def test_range_fade_next_open_and_stop_wins_same_bar():
    rows = []
    px = 50.0
    for i in range(40):
        rows.append(
            {
                "ts": pd.Timestamp("2018-06-01") + pd.Timedelta(days=i),
                "open": px,
                "high": px + 0.2,
                "low": px - 0.2,
                "close": px,
                "volume": 1e6,
            }
        )
    # Down impulse (B)
    for i in range(10):
        px = 50.0 - (i + 1) * 0.3
        rows.append(
            {
                "ts": pd.Timestamp("2018-07-11") + pd.Timedelta(days=i),
                "open": px + 0.1,
                "high": px + 0.15,
                "low": px - 0.15,
                "close": px,
                "volume": 1e6,
            }
        )
    base = px
    rh, rl = base + 0.6, base - 0.6
    for i, (hi, lo, cl) in enumerate(
        [
            (rh, rl + 0.2, (rh + rl) / 2),
            (rh - 0.1, rl, (rh + rl) / 2),
            (rh, rl + 0.1, (rh + rl) / 2),
            (rh - 0.05, rl + 0.05, (rh + rl) / 2),
            (rh - 0.1, rl + 0.1, (rh + rl) / 2),
            (rh, rl, rl),  # touch low on valid bar
        ]
    ):
        rows.append(
            {
                "ts": pd.Timestamp("2018-07-21") + pd.Timedelta(days=i),
                "open": cl,
                "high": hi,
                "low": lo,
                "close": cl,
                "volume": 1e6,
            }
        )
    # Next open fill for long fade; same bar tags stop and target -> stop wins
    rows.append(
        {
            "ts": pd.Timestamp("2018-07-27"),
            "open": rl + 0.05,
            "high": rh + 0.5,
            "low": rl - 0.5,
            "close": rh,
            "volume": 1e6,
        }
    )
    for i in range(3):
        rows.append(
            {
                "ts": pd.Timestamp("2018-07-28") + pd.Timedelta(days=i),
                "open": rh,
                "high": rh + 0.1,
                "low": rh - 0.1,
                "close": rh,
                "volume": 1e6,
            }
        )
    df = pd.DataFrame(rows)
    params = NilParams(
        impulse_lookback=10,
        impulse_atr_mult=1.0,
        min_range_bars=5,
        max_range_atr=5.0,
        min_touches=3,
        touch_tol_pct=0.5,
        atr_len=5,
        weekly_sma=3,
        htf_neutral_pct=5.0,
    )
    trades = collect_symbol_trades(
        df, symbol="TEST", cap_bucket="mid", params=params, books=(BOOK_RANGE,)
    )
    assert trades, "expected range-fade trade"
    tr = trades[0]
    assert tr["book"] == BOOK_RANGE
    assert tr["side"] == "long"
    # Next-open after a range-low touch; structure stop below entry.
    assert pd.Timestamp(tr["buy_date"]) >= pd.Timestamp("2018-07-21")
    assert float(tr["buy_price"]) > float(tr["stop_px"])
    assert float(tr["target_px"]) > float(tr["buy_price"])
    assert tr["exit_reason"] in ("stop", "target", "time_stop")

    # Same-bar stop+target resolution: stop wins.
    from utils.research.patrick_nil_pb import _resolve_exit

    ex = _resolve_exit(
        "long",
        0,
        10.0,
        9.0,
        11.0,
        np.array([10.0]),
        np.array([11.5]),
        np.array([8.5]),
        np.array([10.5]),
        pd.DatetimeIndex(["2020-01-02"]),
        40,
    )
    assert ex["exit_reason"] == "stop"
    assert ex["sell_price"] == 9.0


def test_weekly_htf_bias_length():
    n = 400
    ts = pd.bdate_range("2018-01-01", periods=n)
    close = np.linspace(100.0, 150.0, n)
    bias = weekly_htf_bias(ts, close, sma_len=10, neutral_pct=0.5)
    assert len(bias) == n
    # Uptrend should eventually mark bull after SMA warms
    assert HTF_BULL in set(bias.tolist())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
