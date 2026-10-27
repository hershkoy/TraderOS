"""Unit tests for channel-touch asof-filed fundamentals join."""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.channel_touch_fundamentals import attach_fundamentals_asof


def _fund_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "cik": "1",
                "period_end": date(2021, 12, 31),
                "filed": date(2022, 2, 15),
                "fy": 2021,
                "fp": "Q4",
                "ocf": 100.0,
                "capex": 20.0,
                "fcf": 80.0,
                "total_debt": 200.0,
                "assets": 500.0,
                "shares_out": 10.0,
                "buybacks_ttm": 50.0,
                "issuance_ttm": 0.0,
                "net_buybacks_ttm": 50.0,
                "mkt_cap": 1000.0,
                "ocf_to_debt": 0.5,
                "capex_to_ocf": 0.2,
                "net_buyback_pct": 0.05,
                "fcf_roa": 0.16,
                "sic": "7370",
                "sic_desc": "tech",
                "exclude_sector": False,
                "debt_zero": False,
            },
            {
                "symbol": "AAA",
                "cik": "1",
                "period_end": date(2022, 3, 31),
                "filed": date(2022, 5, 10),
                "fy": 2022,
                "fp": "Q1",
                "ocf": 10.0,
                "capex": 5.0,
                "fcf": 5.0,
                "total_debt": 200.0,
                "assets": 500.0,
                "shares_out": 10.0,
                "buybacks_ttm": 0.0,
                "issuance_ttm": 0.0,
                "net_buybacks_ttm": 0.0,
                "mkt_cap": 1000.0,
                "ocf_to_debt": 0.05,
                "capex_to_ocf": 0.5,
                "net_buyback_pct": 0.0,
                "fcf_roa": 0.01,
                "sic": "7370",
                "sic_desc": "tech",
                "exclude_sector": False,
                "debt_zero": False,
            },
        ]
    )


def test_asof_uses_filed_not_period_end():
    trades = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2022-03-01", "gain_pct_net": 1.0},
            {"stock": "AAA", "buy_date": "2022-06-01", "gain_pct_net": -1.0},
            {"stock": "BBB", "buy_date": "2022-06-01", "gain_pct_net": 0.5},
        ]
    )
    out = attach_fundamentals_asof(trades, _fund_frame())
    assert int(out.loc[0, "has_fund"]) == 1
    # Before second filing: first row (passes screen)
    assert int(out.loc[0, "fund_passes_screen"]) == 1
    assert abs(float(out.loc[0, "ocf_to_debt"]) - 0.5) < 1e-9
    # After second filing: weaker row
    assert int(out.loc[1, "has_fund"]) == 1
    assert int(out.loc[1, "fund_passes_screen"]) == 0
    assert abs(float(out.loc[1, "ocf_to_debt"]) - 0.05) < 1e-9
    # Missing symbol
    assert int(out.loc[2, "has_fund"]) == 0
    assert int(out.loc[2, "fund_passes_screen"]) == 0


def test_before_first_filing_has_no_fund():
    trades = pd.DataFrame([{"stock": "AAA", "buy_date": "2022-01-01", "gain_pct_net": 0.0}])
    out = attach_fundamentals_asof(trades, _fund_frame())
    assert int(out.loc[0, "has_fund"]) == 0
