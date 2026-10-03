"""Unit tests for cashflow_four_filters_v3 ranking / CapEx exception."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from utils.research.cashflow_four_filters_v3 import (
    build_eligible_pool,
    mom_12_1,
    portfolio_annual_rebalance,
    select_topn_by_year,
)


def _price_series(start: date, n: int = 500, start_px: float = 50.0, drift: float = 0.05):
    idx = [start + timedelta(days=i) for i in range(n)]
    return pd.Series([start_px + i * drift for i in range(n)], index=idx)


def test_mom_12_1_skips_recent_month():
    start = date(2020, 1, 1)
    s = _price_series(start, n=450, drift=0.1)
    asof = date(2021, 2, 1)
    m = mom_12_1(s, asof, skip_days=21)
    assert m is not None and m > 0


def test_capex_exception_allows_high_reinvest():
    filed = date(2023, 3, 1)
    spy = _price_series(date(2021, 1, 1), n=900, start_px=100.0, drift=0.02)
    aaa = _price_series(date(2021, 1, 1), n=900, start_px=40.0, drift=0.08)  # beats spy
    closes = {"AAA": aaa, "SPY": spy}
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "1",
                "period_end": date(2022, 12, 31),
                "filed": filed,
                "fy": 2022,
                "fp": "Q4",
                "ocf": 300.0,
                "capex": 200.0,
                "fcf": 100.0,
                "total_debt": 50.0,
                "assets": 800.0,
                "shares_out": 10.0,
                "net_buybacks_ttm": 20.0,
                "capex_to_ocf": 0.67,  # > 35%
                "dividends_ttm": 5.0,
                "op_income_ttm": 250.0,
                "ni_ttm": 200.0,
                "cash": 20.0,
            }
        ]
    )
    # cash_op = 250 - (200-300) = 350; roa = 350/800 = 0.4375 > 0.15
    out = build_eligible_pool(df, closes, spy, fcf_yield_min=0.01, sh_yield_min=0.001, cash_op_roa_min=0.05)
    assert len(out) == 1
    assert bool(out.iloc[0]["capex_exception"]) is True


def test_select_topn_by_year():
    rows = []
    for i in range(40):
        rows.append(
            {
                "symbol": f"S{i}",
                "filed": date(2022, 2, 1 + (i % 20)),
                "fcf_yield": 0.05 + i * 0.001,
                "cash_op_roa": 0.06 + (40 - i) * 0.001,
                "rel_mom": 0.01 * i,
                "mom_12_1": 0.01 * i,
            }
        )
    df = pd.DataFrame(rows)
    top = select_topn_by_year(df, top_n=10)
    assert len(top) == 10
    assert "composite" in top.columns


def test_portfolio_cagr_excludes_thin_cohorts():
    """Single-name +100% year must not drive headline CAGR when min_names=10."""
    start = date(2019, 1, 1)
    n = 2000  # through ~2024 so 2020 and 2022 1y holds both clear
    closes = {
        "AEO": _price_series(start, n=n, start_px=10.0, drift=0.05),
        "SPY": _price_series(start, n=n, start_px=100.0, drift=0.02),
    }
    selected_rows = [
        {"symbol": "AEO", "filed": date(2020, 9, 9)},
    ]
    for i in range(12):
        sym = f"S{i}"
        closes[sym] = _price_series(start, n=n, start_px=20.0, drift=0.01)
        selected_rows.append({"symbol": sym, "filed": date(2022, 3, 1 + i)})
    selected = pd.DataFrame(selected_rows)
    years, stats = portfolio_annual_rebalance(
        selected, closes, benchmarks={"SPY": closes["SPY"]}, min_names=10
    )
    assert int(stats["n_years_all"]) >= 2
    assert int(stats["n_years"]) == 1  # only 2022 thick
    assert stats.get("cagr_all_years_pct") is not None
    assert stats["cagr_pct"] < stats["cagr_all_years_pct"]
