"""Unit tests for QQQ LEAPS gap claim-check helpers."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from utils.research.qqq_leaps_gap import (
    book_stats,
    count_signals,
    prepare_qqq_daily,
    round_qqq_strike,
    run_book,
    signal_mask,
    strike_for_call_delta,
)


def _synth_qqq(n: int = 250, start: date = date(2020, 1, 2), seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    px = 200.0
    day = start
    for i in range(n):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        ret = float(rng.normal(0.0008, 0.012))
        o = px * (1.0 + float(rng.normal(0.0, 0.004)))
        c = px * (1.0 + ret)
        h = max(o, c) * 1.005
        l = min(o, c) * 0.995
        if i in (120, 150, 180):
            o = px * 0.985
            c = o * 1.01
            h = max(o, c) * 1.002
            l = min(o, c) * 0.998
        if i == 160:
            o = px * 0.975
            c = o * 1.02
            h = max(o, c) * 1.002
            l = min(o, c) * 0.998
        rows.append(
            {
                "ts": pd.Timestamp(day, tz="UTC"),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1_000_000,
            }
        )
        px = c
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def test_round_qqq_strike():
    assert round_qqq_strike(252.4) == 252.0
    assert round_qqq_strike(457.0) in (455.0, 460.0)


def test_strike_for_call_delta_near_target():
    from utils.options.greeks import calculate_greeks

    S, T, r, sig, q = 400.0, 1.0, 0.05, 0.22, 0.005
    K = strike_for_call_delta(S, T, r, sig, q, target_delta=0.60)
    d = calculate_greeks(S, K, T, r, sig, "C", q=q)["delta"]
    assert 0.55 <= d <= 0.65


def test_prepare_and_signals():
    raw = _synth_qqq()
    d = prepare_qqq_daily(raw)
    assert "gap_pct" in d.columns
    assert d["sma100"].notna().sum() > 0
    m1 = signal_mask(d, mode="strat1", gap_pct=1.0, sma_rule="prior")
    m2 = signal_mask(d, mode="strat2", gap_pct=2.0, sma_rule="prior")
    assert int(m2.sum()) >= 1
    assert int(m1.sum()) >= 1


def test_run_book_smoke():
    raw = _synth_qqq(n=300)
    d = prepare_qqq_daily(raw)
    start = d["session_date"].iloc[110]
    end = d["session_date"].iloc[-1]
    book = run_book(
        d,
        mode="strat1",
        gap_pct=1.0,
        contracts=1,
        start=start,
        end=end,
        max_open=3,
        iv_elev=1.0,
        iv_crush_days=1,
    )
    assert book.signals_raw >= 1
    st = book_stats(book.trades)
    assert st["n"] >= 0
    counts = count_signals(d, start=start, end=end)
    assert counts["strat2"] >= 0
