"""Causal last-15m H2 vs L3 regime switch helpers."""
from __future__ import annotations

import pandas as pd
import pytest

from utils.research.last_15m_regime_switch import (
    combine_h2_l3,
    h2_on_at,
    keep_unless_tlt_below_sma,
    keep_unless_uup_above_sma,
    rolling_h2_regime,
    sma,
)


def _h2_row(sell, gain, buy="2019-01-01 19:45"):
    return {
        "stock": "AAA",
        "buy_time": buy,
        "sell_time": sell,
        "gain_pct": gain,
        "buy_date": str(sell)[:10],
    }


def test_rolling_wr_stays_on_until_window_then_trips_off():
    rows = []
    for i in range(20):
        rows.append(_h2_row("2021-06-%02d 20:00" % (i + 1), 1.0 if i < 10 else -1.0))
    # WR20 = 0.50 still ON. Next 10 losses roll WR down.
    for i in range(10):
        rows.append(_h2_row("2021-07-%02d 20:00" % (i + 1), -1.0))
    df = pd.DataFrame(rows)
    reg = rolling_h2_regime(df, window=20, off_wr=0.28, on_wr=0.32)
    assert bool(reg.iloc[19]["h2_on"])
    # After 20 closed: 10w/10l WR=0.50. After 10 more losses window is 10w/20l wait
    # window last 20: originally 10w+10l then 10l -> 10w from first 10 of window
    # closed 21-30 are losses; window at 30 is trades 11-30 = 10l (from 11-20) + 10l = WR 0
    assert bool(reg.iloc[-1]["h2_on"]) is False
    assert float(reg.iloc[-1]["roll_wr"]) < 0.28


def test_paper_wr_switch_back_without_live_fills():
    rows = [_h2_row("2021-01-%02d 20:00" % (i + 1), -1.0) for i in range(20)]
    rows += [_h2_row("2021-02-%02d 20:00" % (i + 1), 1.0) for i in range(20)]
    df = pd.DataFrame(rows)
    reg = rolling_h2_regime(df, window=20, off_wr=0.28, on_wr=0.32)
    assert bool(reg.iloc[19]["h2_on"]) is False
    assert bool(reg.iloc[-1]["h2_on"]) is True


def test_h2_on_at_uses_only_prior_closes():
    rows = [_h2_row("2022-01-10 20:00", -1.0) for _ in range(20)]
    df = pd.DataFrame(rows)
    reg = rolling_h2_regime(df, window=20, off_wr=0.28, on_wr=0.32)
    # Fill at the same timestamp as the 20th close must not see that close.
    before = h2_on_at(["2022-01-10 19:59"], reg)
    at = h2_on_at(["2022-01-10 20:00"], reg)
    after = h2_on_at(["2022-01-10 20:01"], reg)
    assert bool(before.iloc[0]) is True
    assert bool(at.iloc[0]) is True
    assert bool(after.iloc[0]) is False


def test_combine_takes_l3_only_when_h2_off():
    h2 = pd.DataFrame(
        [
            {"stock": "H", "buy_date": "2022-01-03", "gain_pct": -2.0},
            {"stock": "H2", "buy_date": "2022-06-01", "gain_pct": 5.0},
        ]
    )
    l3 = pd.DataFrame(
        [
            {"stock": "L", "buy_date": "2022-01-03", "gain_pct": 0.2},
            {"stock": "L2", "buy_date": "2022-06-01", "gain_pct": 0.1},
        ]
    )
    h2_on = pd.Series([False, True], index=h2.index)
    l3_on = pd.Series([True, False], index=l3.index)
    out = combine_h2_l3(h2, l3, h2_on, l3_on)
    assert set(zip(out["stock"], out["sleeve"])) == {("H2", "last15m_h2"), ("L", "l3_wait12")}


def test_tlt_below_sma_skips_and_missing_keeps():
    trades = pd.DataFrame(
        {
            "tlt_px": [90.0, 110.0, None],
            "tlt_sma": [100.0, 100.0, 100.0],
        }
    )
    keep = keep_unless_tlt_below_sma(trades)
    assert list(keep.astype(int)) == [0, 1, 1]


def test_uup_above_sma_skips_and_missing_keeps():
    trades = pd.DataFrame(
        {
            "uup_px": [30.0, 20.0, None],
            "uup_sma": [25.0, 25.0, 25.0],
        }
    )
    keep = keep_unless_uup_above_sma(trades)
    assert list(keep.astype(int)) == [0, 1, 1]


def test_sma_needs_full_window():
    s = pd.Series([1.0, 2.0, 3.0, 4.0])
    ma = sma(s, 3)
    assert pd.isna(ma.iloc[1])
    assert ma.iloc[2] == pytest.approx(2.0)


def test_combine_h2_alt_short_sleeve():
    from utils.research.h2_l3_blend_occupancy import SLEEVE_SHORT
    from utils.research.last_15m_regime_switch import combine_h2_alt

    h2 = pd.DataFrame(
        [
            {"stock": "H", "buy_date": "2022-01-03", "gain_pct": -2.0},
            {"stock": "H2", "buy_date": "2022-06-01", "gain_pct": 5.0},
        ]
    )
    short = pd.DataFrame(
        [
            {"stock": "S", "buy_date": "2022-01-03", "gain_pct": 0.3},
            {"stock": "S2", "buy_date": "2022-06-01", "gain_pct": 0.1},
        ]
    )
    out = combine_h2_alt(
        h2,
        short,
        pd.Series([False, True], index=h2.index),
        pd.Series([True, False], index=short.index),
        alt_sleeve=SLEEVE_SHORT,
    )
    assert set(zip(out["stock"], out["sleeve"])) == {("H2", "last15m_h2"), ("S", SLEEVE_SHORT)}
