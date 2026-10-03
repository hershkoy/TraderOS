"""Unit tests for cashflow_four_filters (no network / light DB mocks)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from utils.research.cashflow_four_filters import (
    book_stats,
    build_trades,
    dedupe_symbol_year,
)


def test_dedupe_symbol_year_keeps_earliest():
    df = pd.DataFrame(
        [
            {"symbol": "AAA", "filed": date(2020, 3, 1), "x": 1},
            {"symbol": "AAA", "filed": date(2020, 8, 1), "x": 2},
            {"symbol": "AAA", "filed": date(2021, 2, 1), "x": 3},
            {"symbol": "BBB", "filed": date(2020, 5, 1), "x": 4},
        ]
    )
    out = dedupe_symbol_year(df)
    assert len(out) == 3
    aaa20 = out[(out.symbol == "AAA") & (out.filed == date(2020, 3, 1))]
    assert len(aaa20) == 1


def test_build_trades_5y():
    idx = pd.date_range("2018-01-02", periods=2000, freq="B")
    px = pd.Series(range(100, 100 + len(idx)), index=[d.date() for d in idx])
    closes = {"AAA": px, "SPY": px * 1.1}
    screen = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "filed": date(2018, 6, 15),
                "period_end": date(2018, 3, 31),
                "fp": "Q1",
                "ocf_to_debt": 0.5,
                "capex_to_ocf": 0.2,
                "net_buyback_pct": 0.03,
                "fcf_roa": 0.15,
                "mkt_cap": 1e9,
                "sic": "3571",
            }
        ]
    )
    trades = build_trades(screen, closes, hold_years=5, require_full_hold=True, spy=closes["SPY"])
    assert len(trades) == 1
    assert trades.iloc[0]["ret"] > 0
    assert trades.iloc[0]["spy_ret"] is not None


def test_book_stats_empty():
    assert book_stats(pd.DataFrame())["n"] == 0
