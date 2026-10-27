"""Unit tests for causal L2 bounce entry (buy at L2+pivot_len, no H2 wait)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import trades_for_symbol  # noqa: E402
from find_ascending_channels import (  # noqa: E402
    find_l2_bounce_setups,
)


def _l1_h1_l2_frame(n: int = 120, pivot_len: int = 3):
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    support = 10.0 + 0.03 * np.arange(n)
    close = support + 1.0
    high = close + 0.4
    low = close - 0.4
    l1, h1, l2 = 20, 32, 44
    low[l1] = support[l1]
    high[l1] = support[l1] + 0.3
    close[l1] = support[l1] + 0.15
    high[h1] = support[h1] + 2.0
    low[h1] = support[h1] + 1.4
    close[h1] = support[h1] + 1.8
    low[l2] = support[l2]
    high[l2] = support[l2] + 0.3
    close[l2] = support[l2] + 0.15
    # Keep post-L2 closes above support so confirm is valid.
    for i in range(l2 + 1, min(n, l2 + pivot_len + 5)):
        close[i] = support[i] + 0.8
        high[i] = close[i] + 0.3
        low[i] = support[i] + 0.2
    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": 1e5},
        index=idx,
    )
    return df, idx, l1, h1, l2, support


def test_find_l2_bounce_arms_at_confirm_not_h2():
    df, idx, l1, h1, l2, _support = _l1_h1_l2_frame()
    setups = find_l2_bounce_setups(
        df,
        pivot_len=3,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_intervening_rally_pct=2.0,
        min_total_rise_pct=2.0,
    )
    assert setups, "expected an L2 bounce setup"
    ch = next(c for c in setups if int(c["l2_idx"]) == l2)
    assert int(ch["l1_idx"]) == l1
    assert int(ch["confirm_idx"]) == l2 + 3
    assert float(ch["channel_width"]) > 0


def test_l2_touch_buys_at_confirm_close():
    df, idx, _l1, _h1, l2, _support = _l1_h1_l2_frame()
    rows = trades_for_symbol(
        "TEST",
        df,
        entry_mode="l2_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_total_rise_pct=2.0,
    )
    assert rows, "expected an L2 bounce trade"
    trade = rows[0]
    assert int(trade["touch_num"]) == 2
    assert trade["entry_mode"] == "l2_touch"
    buy = pd.Timestamp(trade["buy_date"])
    confirm = idx[l2 + 3]
    assert buy == pd.Timestamp(confirm).normalize() or buy.strftime("%Y-%m-%d") == confirm.strftime(
        "%Y-%m-%d"
    )


def test_l2_touch_skips_if_support_breaks_before_confirm():
    df, idx, _l1, _h1, l2, support = _l1_h1_l2_frame()
    # Break support on the bar after L2 so confirm is aborted.
    i = l2 + 1
    df.iloc[i, df.columns.get_loc("close")] = float(support[i]) * 0.90
    df.iloc[i, df.columns.get_loc("low")] = float(support[i]) * 0.85
    df.iloc[i, df.columns.get_loc("high")] = float(support[i]) * 0.95
    rows = trades_for_symbol(
        "TEST",
        df,
        entry_mode="l2_touch",
        pivot_len=3,
        entry_features=False,
        squeeze_adaptive=False,
        stop_pct=0.03,
        trail_pct=0.10,
        min_bars_apart=8,
        max_low_pivots=16,
        error_pct=2.0,
        min_rally_pct=2.0,
        min_total_rise_pct=2.0,
    )
    buys = [r for r in rows if int(r.get("touch_num", 0)) == 2]
    # Either no trade, or none whose touch is this broken L2.
    for r in buys:
        assert pd.Timestamp(r["touch_date"]).strftime("%Y-%m-%d") != idx[l2].strftime("%Y-%m-%d")
