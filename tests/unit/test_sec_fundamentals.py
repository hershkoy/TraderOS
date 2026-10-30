"""Unit tests for SEC fundamentals parser (no network)."""
from __future__ import annotations

from datetime import date

import pytest

from utils.research.sec_fundamentals import (
    FundamentalRow,
    attach_mkt_caps,
    build_fundamental_rows,
    compute_ratios,
    is_excluded_sic,
    passes_four_filters,
    single_period_flows,
    ttm_sum,
)


def _obs(end: str, fp: str, val: float, filed: str, fy: int = 2024) -> dict:
    return {"end": end, "fp": fp, "val": val, "filed": filed, "fy": fy, "form": "10-Q"}


def _facts_payload() -> dict:
    """Minimal companyfacts-like payload with 5 quarters of CF/BS data."""
    q = [
        ("2023-03-31", "Q1", "2023-05-01", 2023),
        ("2023-06-30", "Q2", "2023-08-01", 2023),
        ("2023-09-30", "Q3", "2023-11-01", 2023),
        ("2023-12-31", "Q4", "2024-02-01", 2023),
        ("2024-03-31", "Q1", "2024-05-01", 2024),
    ]
    ocf_vals = [100, 110, 120, 130, 140]
    capex_vals = [20, 22, 24, 26, 28]  # < 35% of OCF
    buy_vals = [50, 50, 50, 50, 60]
    iss_vals = [0, 0, 0, 0, 0]
    assets_vals = [1000, 1050, 1100, 1150, 1200]
    debt_lt = [200, 200, 180, 180, 160]
    debt_cur = [50, 50, 50, 40, 40]
    shares = [100, 100, 98, 98, 96]

    def series(vals):
        return {
            "label": "x",
            "description": "",
            "units": {
                "USD": [
                    _obs(end, fp, v, filed, fy)
                    for (end, fp, filed, fy), v in zip(q, vals)
                ]
            },
        }

    def shares_series(vals):
        return {
            "label": "shares",
            "description": "",
            "units": {
                "shares": [
                    _obs(end, fp, v, filed, fy)
                    for (end, fp, filed, fy), v in zip(q, vals)
                ]
            },
        }

    return {
        "entityName": "Test Co",
        "cik": 123,
        "facts": {
            "us-gaap": {
                "NetCashProvidedByUsedInOperatingActivities": series(ocf_vals),
                "PaymentsToAcquirePropertyPlantAndEquipment": series(capex_vals),
                "PaymentsForRepurchaseOfCommonStock": series(buy_vals),
                "ProceedsFromIssuanceOfCommonStock": series(iss_vals),
                "Assets": series(assets_vals),
                "LongTermDebt": series(debt_lt),
                "DebtCurrent": series(debt_cur),
                "CommonStockSharesOutstanding": shares_series(shares),
            }
        },
    }


class TestSicExclude:
    def test_aapl_not_excluded(self):
        assert is_excluded_sic("3571") is False  # electronic computers

    def test_jpm_financials(self):
        assert is_excluded_sic("6021") is True

    def test_xom_energy(self):
        assert is_excluded_sic("1311") is True

    def test_nee_utilities(self):
        assert is_excluded_sic("4911") is True

    def test_amt_real_estate(self):
        assert is_excluded_sic("6798") is True

    def test_materials_mining(self):
        assert is_excluded_sic("1040") is True

    def test_none(self):
        assert is_excluded_sic(None) is False


class TestTtmSum:
    def test_needs_four(self):
        vals = [
            (date(2023, 3, 31), 10.0),
            (date(2023, 6, 30), 20.0),
            (date(2023, 9, 30), 30.0),
        ]
        assert ttm_sum(vals, date(2023, 9, 30), 4) is None

    def test_sums_last_four(self):
        vals = [
            (date(2023, 3, 31), 10.0),
            (date(2023, 6, 30), 20.0),
            (date(2023, 9, 30), 30.0),
            (date(2023, 12, 31), 40.0),
            (date(2024, 3, 31), 50.0),
        ]
        assert ttm_sum(vals, date(2024, 3, 31), 4) == 140.0
        assert ttm_sum(vals, date(2023, 12, 31), 4) == 100.0


class TestBuildRows:
    def test_quarterly_ttm_and_ratios(self):
        facts = _facts_payload()
        rows = build_fundamental_rows("TEST", "123", facts, sic="3571")
        q_rows = [r for r in rows if r.fp.startswith("Q")]
        assert len(q_rows) == 5

        # First 3 quarters lack full TTM
        assert q_rows[0].ocf is None
        assert q_rows[3].ocf == pytest.approx(460.0)  # 100+110+120+130
        last = q_rows[4]
        assert last.ocf == pytest.approx(500.0)  # 110+120+130+140
        assert last.capex == pytest.approx(100.0)  # 22+24+26+28
        assert last.total_debt == pytest.approx(200.0)  # 160+40
        assert last.assets == pytest.approx(1200.0)
        assert last.buybacks_ttm == pytest.approx(210.0)  # 50*3+60
        assert last.exclude_sector is False
        assert last.tag_ocf == "NetCashProvidedByUsedInOperatingActivities"
        assert last.fcf == pytest.approx(400.0)
        assert last.fcf_roa == pytest.approx(400.0 / 1200.0)
        assert last.capex_to_ocf == pytest.approx(0.2)
        assert last.ocf_to_debt == pytest.approx(500.0 / 200.0)

    def test_excluded_sic_flag(self):
        facts = _facts_payload()
        rows = build_fundamental_rows("XOM", "34088", facts, sic="1311")
        assert all(r.exclude_sector for r in rows)

    def test_debt_zero_pass(self):
        row = FundamentalRow(
            symbol="X",
            cik="1",
            period_end=date(2024, 1, 1),
            filed=date(2024, 2, 1),
            fy=2023,
            fp="Q4",
            ocf=100.0,
            total_debt=0.0,
        )
        compute_ratios(row)
        assert row.debt_zero is True
        assert row.ocf_to_debt == float("inf")
        flags = passes_four_filters(row)
        assert flags["f1_ocf_debt"] is True


class TestMktCap:
    def test_attach_asof_filed(self):
        row = FundamentalRow(
            symbol="T",
            cik="1",
            period_end=date(2024, 3, 31),
            filed=date(2024, 5, 1),
            fy=2024,
            fp="Q1",
            shares_out=100.0,
            ocf=500.0,
            capex=100.0,
            assets=1000.0,
            net_buybacks_ttm=50.0,
        )
        closes = {
            date(2024, 4, 30): 10.0,
            date(2024, 5, 1): 12.0,
            date(2024, 5, 2): 13.0,
        }
        attach_mkt_caps([row], closes)
        assert row.mkt_cap == pytest.approx(1200.0)
        assert row.net_buyback_pct == pytest.approx(50.0 / 1200.0)

    def test_prior_session_fallback(self):
        row = FundamentalRow(
            symbol="T",
            cik="1",
            period_end=date(2024, 3, 31),
            filed=date(2024, 5, 1),
            fy=2024,
            fp="Q1",
            shares_out=50.0,
        )
        closes = {date(2024, 4, 30): 20.0}
        attach_mkt_caps([row], closes)
        assert row.mkt_cap == pytest.approx(1000.0)


def _flow(end, fp, val, start, filed="2024-02-15", fy=2023):
    return {
        "end": end,
        "start": start,
        "fp": fp,
        "val": val,
        "filed": filed,
        "fy": fy,
        "form": "10-Q",
    }


class TestYtdDifference:
    def test_year_to_date_becomes_single_quarters(self):
        rows = [
            _flow("2023-03-31", "Q1", 100, "2023-01-01"),
            _flow("2023-06-30", "Q2", 250, "2023-01-01"),
            _flow("2023-09-30", "Q3", 400, "2023-01-01"),
            _flow("2023-12-31", "FY", 600, "2023-01-01", filed="2024-02-15"),
        ]
        got = single_period_flows(rows)
        assert got[("2023-03-31", "Q1")] == pytest.approx(100)
        assert got[("2023-06-30", "Q2")] == pytest.approx(150)
        assert got[("2023-09-30", "Q3")] == pytest.approx(150)
        assert got[("2023-12-31", "Q4")] == pytest.approx(200)
        assert got[("2023-12-31", "FY")] == pytest.approx(600)

    def test_discrete_quarter_wins_over_ytd(self):
        rows = [
            _flow("2023-03-31", "Q1", 100, "2023-01-01"),
            _flow("2023-06-30", "Q2", 250, "2023-01-01"),
            _flow("2023-06-30", "Q2", 40, "2023-04-01"),
        ]
        got = single_period_flows(rows)
        assert got[("2023-06-30", "Q2")] == pytest.approx(40)


class TestStatementFields:
    def test_ttm_margins_and_balance_sheet(self):
        facts = _facts_payload()
        gaap = facts["facts"]["us-gaap"]
        # Five discrete quarters. Durations are one quarter, so no YTD diff.
        bounds = [
            ("2023-01-01", "2023-03-31"),
            ("2023-04-01", "2023-06-30"),
            ("2023-07-01", "2023-09-30"),
            ("2023-10-01", "2023-12-31"),
            ("2024-01-01", "2024-03-31"),
        ]
        q = [
            ("2023-03-31", "Q1", "2023-05-01", 2023),
            ("2023-06-30", "Q2", "2023-08-01", 2023),
            ("2023-09-30", "Q3", "2023-11-01", 2023),
            ("2023-12-31", "Q4", "2024-02-01", 2023),
            ("2024-03-31", "Q1", "2024-05-01", 2024),
        ]

        def series(vals, unit="USD"):
            return {
                "units": {
                    unit: [
                        {
                            "end": end,
                            "start": start,
                            "fp": fp,
                            "val": v,
                            "filed": filed,
                            "fy": fy,
                            "form": "10-Q",
                        }
                        for (start, end), (end, fp, filed, fy), v in zip(bounds, q, vals)
                    ]
                }
            }

        gaap["Revenues"] = series([100, 100, 100, 100, 100])
        gaap["GrossProfit"] = series([50, 50, 50, 50, 50])
        gaap["OperatingIncomeLoss"] = series([30, 30, 30, 30, 30])
        gaap["NetIncomeLoss"] = series([20, 20, 20, 20, 20])
        gaap["IncomeLossFromContinuingOperationsBeforeIncomeTaxes"] = series(
            [25, 25, 25, 25, 25]
        )
        gaap["InterestExpense"] = series([4, 4, 4, 4, 4])
        gaap["InterestIncome"] = series([1, 1, 1, 1, 1])
        gaap["DepreciationAndAmortization"] = series([5, 5, 5, 5, 5])
        gaap["AssetsCurrent"] = series([300, 300, 300, 300, 300])
        gaap["LiabilitiesCurrent"] = series([100, 100, 100, 100, 100])
        gaap["StockholdersEquity"] = series([400, 400, 400, 400, 400])
        gaap["WeightedAverageNumberOfDilutedSharesOutstanding"] = series(
            [10, 10, 10, 10, 10], unit="shares"
        )
        gaap["EarningsPerShareDiluted"] = series(
            [2, 2, 2, 2, 2], unit="USD/shares"
        )

        rows = build_fundamental_rows("TEST", "123", facts, sic="3571")
        last = [r for r in rows if r.fp == "Q1" and r.period_end == date(2024, 3, 31)][0]
        assert last.revenue == pytest.approx(400)
        assert last.gross_profit == pytest.approx(200)
        assert last.operating_income == pytest.approx(120)
        assert last.net_income == pytest.approx(80)
        assert last.diluted_eps == pytest.approx(8)
        assert last.diluted_shares == pytest.approx(10)
        assert last.gross_margin == pytest.approx(0.5)
        assert last.operating_margin == pytest.approx(0.3)
        assert last.net_margin == pytest.approx(0.2)
        assert last.current_ratio == pytest.approx(3)
        assert last.roe == pytest.approx(80 / 400)
        # EBITDA = operating income + depreciation = 120 + 20
        assert last.ebitda == pytest.approx(140)
        assert last.debt_to_ebitda == pytest.approx(200 / 140)
        assert last.net_interest == pytest.approx(12)  # (4-1) * 4
        assert last.interest_to_ocf == pytest.approx(12 / last.ocf)
        assert last.filed == date(2024, 5, 1)
