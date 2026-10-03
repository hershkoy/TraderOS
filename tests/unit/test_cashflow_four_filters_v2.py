"""Unit tests for cashflow_four_filters_v2 helpers."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from utils.research.cashflow_four_filters_v2 import apply_v2_screen, mom_12m


def test_mom_12m_positive():
    start = date(2020, 1, 2)
    idx = [start + timedelta(days=i) for i in range(400)]
    # linear climb
    s = pd.Series([10.0 + i * 0.05 for i in range(400)], index=idx)
    m = mom_12m(s, date(2021, 1, 5), lookback_days=365)
    assert m is not None and m > 0


def test_apply_v2_screen_filters():
    filed = date(2022, 3, 15)
    idx = [filed - timedelta(days=400 - i) for i in range(450)]
    px = pd.Series([50.0 + i * 0.1 for i in range(450)], index=idx)
    closes = {"AAA": px}
    df = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "1",
                "period_end": date(2021, 12, 31),
                "filed": filed,
                "fy": 2021,
                "fp": "Q4",
                "ocf": 200.0,
                "capex": 40.0,
                "fcf": 160.0,
                "total_debt": 100.0,
                "assets": 1000.0,
                "shares_out": 10.0,
                "net_buybacks_ttm": 5.0,
                "capex_to_ocf": 0.2,
                "dividends_ttm": 10.0,
                "op_income_ttm": 180.0,
                "ni_ttm": 150.0,
                "cash": 50.0,
                "exclude_sector": False,
            }
        ]
    )
    out = apply_v2_screen(df, closes, fcf_yield_min=0.01, sh_yield_min=0.005, cash_op_roa_min=0.05)
    assert len(out) == 1
    assert out.iloc[0]["fcf_yield"] > 0
    assert out.iloc[0]["sh_yield"] > 0
