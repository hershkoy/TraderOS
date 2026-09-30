"""Unit tests for post-earnings put credit spread claim-check helpers."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from utils.research.post_earnings_pcs import (
    EarnEvent,
    book_stats,
    entry_session_date,
    iv_on_day,
    normalize_daily,
    pick_short_strike,
    post_earn_hist_pass,
    pre_earnings_close,
    put_vertical_mark,
    simulate_pcs,
    spread_width_for_price,
)


def _synth_daily(
    start: date,
    n: int,
    start_px: float = 100.0,
    drift: float = 0.001,
) -> pd.DataFrame:
    rows = []
    px = float(start_px)
    for i in range(n):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        o = px
        c = px * (1.0 + drift)
        h = max(o, c) * 1.005
        l = min(o, c) * 0.995
        rows.append(
            {
                "ts": pd.Timestamp(d),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1_000_000,
            }
        )
        px = c
    return pd.DataFrame(rows)


def test_spread_width_for_price():
    assert spread_width_for_price(40) == 2.5
    assert spread_width_for_price(100) == 5.0
    assert spread_width_for_price(300) == 10.0


def test_iv_on_day_decays_to_base():
    assert iv_on_day(0.40, 0.20, 0, 10) == pytest.approx(0.40)
    assert iv_on_day(0.40, 0.20, 10, 10) == pytest.approx(0.20)
    mid = iv_on_day(0.40, 0.20, 5, 10)
    assert 0.20 < mid < 0.40


def test_entry_session_amc_bmo():
    df = normalize_daily(_synth_daily(date(2024, 1, 2), 20))
    # Pick a Wednesday in the series
    days = [ts.date() for ts in df.index]
    wed = next(d for d in days if d.weekday() == 2)
    idx = days.index(wed)
    assert idx + 1 < len(days)
    nxt = days[idx + 1]
    amc = EarnEvent(earn_date=wed, session="amc")
    bmo = EarnEvent(earn_date=wed, session="bmo")
    assert entry_session_date(df, amc) == nxt
    assert entry_session_date(df, bmo) == wed


def test_pre_earnings_close_amc_uses_earn_day_close():
    df = normalize_daily(_synth_daily(date(2024, 1, 2), 20, start_px=100.0))
    days = [ts.date() for ts in df.index]
    wed = next(d for d in days if d.weekday() == 2)
    ev = EarnEvent(earn_date=wed, session="amc")
    pre = pre_earnings_close(df, ev)
    assert pre == pytest.approx(float(df.loc[pd.Timestamp(wed), "close"]))


def test_put_vertical_mark_positive_for_otm():
    m = put_vertical_mark(100.0, 95.0, 90.0, 37 / 365.25, 0.05, 0.30)
    assert m > 0
    # Wider OTM should be cheaper credit
    m2 = put_vertical_mark(100.0, 90.0, 85.0, 37 / 365.25, 0.05, 0.30)
    assert m2 < m


def test_pick_short_strike_below_pre():
    k = pick_short_strike(110.0, 100.0, 37 / 365.25, 0.05, 0.35, target_delta=-0.20)
    assert k is not None
    assert k <= 100.0
    assert k < 110.0


def test_simulate_pcs_fills_on_gap_up():
    # Strong uptrend so gap-up earnings + IV crush likely hits TP
    df = normalize_daily(_synth_daily(date(2023, 1, 3), 260, start_px=100.0, drift=0.004))
    days = [ts.date() for ts in df.index]
    # Mid series Wednesday as AMC earnings
    mid = days[len(days) // 2]
    while mid.weekday() != 2:
        mid = days[days.index(mid) + 1]
    # Force a gap up on next session by bumping that close
    nxt = entry_session_date(df, EarnEvent(mid, "amc"))
    assert nxt is not None
    pre = float(df.loc[pd.Timestamp(mid), "close"])
    df.loc[pd.Timestamp(nxt), "close"] = pre * 1.04
    df.loc[pd.Timestamp(nxt), "open"] = pre * 1.03
    df.loc[pd.Timestamp(nxt), "high"] = pre * 1.05
    df.loc[pd.Timestamp(nxt), "low"] = pre * 1.02
    # Keep drifting up after
    loc = int(df.index.get_loc(pd.Timestamp(nxt)))
    for j in range(loc + 1, min(loc + 30, len(df))):
        px = float(df.iloc[j - 1]["close"]) * 1.003
        df.iloc[j, df.columns.get_loc("close")] = px
        df.iloc[j, df.columns.get_loc("open")] = px
        df.iloc[j, df.columns.get_loc("high")] = px * 1.002
        df.iloc[j, df.columns.get_loc("low")] = px * 0.998

    res = simulate_pcs(
        df,
        EarnEvent(mid, "amc"),
        symbol="TEST",
        iv_elev=1.5,
        iv_crush_days=8,
        min_credit_frac=0.05,  # relax for synthetic
    )
    assert res.filled, res.reason
    assert res.row is not None
    assert res.row["gap_pct"] > 0
    assert res.row["credit"] > 0
    assert res.row["max_risk"] > 0


def test_simulate_pcs_skips_big_drop():
    df = normalize_daily(_synth_daily(date(2023, 1, 3), 120, start_px=100.0))
    days = [ts.date() for ts in df.index]
    mid = days[40]
    nxt = entry_session_date(df, EarnEvent(mid, "amc"))
    pre = float(df.loc[pd.Timestamp(mid), "close"])
    df.loc[pd.Timestamp(nxt), "close"] = pre * 0.85
    res = simulate_pcs(df, EarnEvent(mid, "amc"), symbol="TEST")
    assert not res.filled
    assert res.reason == "skip_drop"


def test_book_stats_empty():
    st = book_stats(pd.DataFrame())
    assert st["n"] == 0


def test_hist_pass_no_history_allows():
    df = normalize_daily(_synth_daily(date(2023, 1, 3), 80))
    ok, scored, n_ok = post_earn_hist_pass(df, [EarnEvent(date(2023, 2, 1), "amc")], 0)
    assert ok is True
    assert scored == 0
