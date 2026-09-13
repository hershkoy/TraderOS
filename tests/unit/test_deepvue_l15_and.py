"""Causal Deepvue x last-15m week-join helpers (no TimescaleDB)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.deepvue_l15_and import (
    in_focus_trade_week,
    keep_deepvue_if_last15m_trade_week,
    keep_last15m_ever_focused,
    keep_last15m_in_focus_trade_week,
    keep_last15m_on_signal_day,
    overlap_counts,
)


def test_trade_week_excludes_signal_friday_and_day_8():
    friday = date(2026, 8, 14)
    assert in_focus_trade_week(friday, friday) is False
    assert in_focus_trade_week(date(2026, 8, 17), friday) is True
    assert in_focus_trade_week(date(2026, 8, 21), friday) is True
    assert in_focus_trade_week(date(2026, 8, 22), friday) is False
    assert in_focus_trade_week(date(2026, 8, 13), friday) is False


def test_last15m_keep_week_and_signal_day():
    signals = pd.DataFrame(
        [
            {"stock": "aaa", "signal_date": "2026-08-14"},
            {"stock": "BBB", "signal_date": "2026-08-14"},
        ]
    )
    trades = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2026-08-14", "buy_time": "2026-08-14 19:45:00"},
            {"stock": "AAA", "buy_date": "2026-08-17", "buy_time": "2026-08-17 19:45:00"},
            {"stock": "ZZZ", "buy_date": "2026-08-17", "buy_time": "2026-08-17 19:45:00"},
        ]
    )
    week = keep_last15m_in_focus_trade_week(trades, signals)
    day = keep_last15m_on_signal_day(trades, signals)
    ever = keep_last15m_ever_focused(trades, signals)
    assert list(week) == [False, True, False]
    assert list(day) == [True, False, False]
    assert list(ever) == [True, True, False]


def test_deepvue_keep_when_last15m_prints_that_week():
    deepvue = pd.DataFrame(
        [
            {"stock": "AAA", "signal_date": "2026-08-14", "gain_pct": 2.0},
            {"stock": "BBB", "signal_date": "2026-08-14", "gain_pct": -3.0},
        ]
    )
    last15m = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2026-08-18", "buy_time": "2026-08-18 19:45:00"},
        ]
    )
    keep = keep_deepvue_if_last15m_trade_week(deepvue, last15m)
    assert list(keep) == [True, False]


def test_overlap_counts_shape():
    signals = pd.DataFrame([{"stock": "AAA", "signal_date": "2026-08-14"}])
    last15m = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2026-08-17", "buy_time": "2026-08-17 19:45:00"},
            {"stock": "BBB", "buy_date": "2026-08-17", "buy_time": "2026-08-17 19:45:00"},
        ]
    )
    deepvue = pd.DataFrame([{"stock": "AAA", "signal_date": "2026-08-14", "gain_pct": 1.0}])
    counts = overlap_counts(last15m, signals, deepvue)
    assert counts["n_last15m"] == 2
    assert counts["n_l15_and_week"] == 1
    assert counts["n_signals_with_l15_week"] == 1
    assert counts["n_stock_overlap"] == 1
