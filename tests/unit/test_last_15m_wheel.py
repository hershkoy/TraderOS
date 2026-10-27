"""Unit tests for last-15m one-cycle wheel helpers."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pytest

from utils.research.last_15m_wheel import (
    EXIT_CALL_ASSIGN,
    EXIT_PUT_OTM,
    EXIT_SHARE_LIQ,
    cash_secured_pct,
    cycle_pnl_per_share,
    occupancy_fcfs_wheel,
    pick_atm_contract,
    select_contract,
    simulate_one_cycle_wheel,
    wheel_slot_equity,
    wheel_slot_equity_mtm,
    wheel_unrealized_cs_frac,
)


class FakeSource:
    def __init__(
        self,
        chains: Dict[Tuple[str, date, str], pd.DataFrame],
        quotes: Dict[Tuple[str, date, float, str, date], Dict[str, Any]],
    ) -> None:
        self.chains = chains
        self.quotes = quotes

    def chain_on_day(
        self,
        symbol: str,
        day: date,
        right: str,
        dte_lo: int,
        dte_hi: int,
        *,
        spot_hint: Optional[float] = None,
    ) -> pd.DataFrame:
        del dte_lo, dte_hi, spot_hint
        right_c = "C" if str(right).upper().startswith("C") else "P"
        df = self.chains.get((str(symbol).upper(), day, right_c))
        if df is None:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        return df.copy()

    def quote_on_day(
        self,
        symbol: str,
        expiration: date,
        strike: float,
        right: str,
        day: date,
    ) -> Optional[Dict[str, Any]]:
        right_c = "C" if str(right).upper().startswith("C") else "P"
        return self.quotes.get(
            (str(symbol).upper(), expiration, float(strike), right_c, day)
        )


def _panel(days: List[date], closes: List[float]) -> pd.DataFrame:
    idx = pd.to_datetime(days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1},
        index=idx,
    )


def test_cycle_pnl_otm_put():
    pnl = cycle_pnl_per_share(put_credit=1.5, put_strike=100.0, assigned=False)
    assert pnl == pytest.approx(1.5)
    assert cash_secured_pct(pnl, 100.0) == pytest.approx(1.5)


def test_cycle_pnl_assign_call_away():
    # Put credit 2, assigned at 100, call credit 1.2, called at 105
    # pnl = 2 - 100 + 1.2 + 105 = 8.2
    pnl = cycle_pnl_per_share(
        put_credit=2.0,
        put_strike=100.0,
        assigned=True,
        call_credit=1.2,
        call_strike=105.0,
        called_away=True,
    )
    assert pnl == pytest.approx(8.2)
    assert cash_secured_pct(pnl, 100.0) == pytest.approx(8.2)


def test_cycle_pnl_assign_liquidate():
    # Put 2, K=100, call 1, liquidate at 97 -> 2 - 100 + 1 + 97 = 0
    pnl = cycle_pnl_per_share(
        put_credit=2.0,
        put_strike=100.0,
        assigned=True,
        call_credit=1.0,
        call_strike=105.0,
        called_away=False,
        share_exit_px=97.0,
    )
    assert pnl == pytest.approx(0.0)


def test_pick_atm_prefers_otm_put():
    day = date(2024, 1, 2)
    exp = day + timedelta(days=30)
    chain = pd.DataFrame(
        [
            {"expiration": exp, "strike": 95.0, "bid": 1.0, "ask": 1.2, "delta": -0.25},
            {"expiration": exp, "strike": 100.0, "bid": 2.0, "ask": 2.2, "delta": -0.50},
            {"expiration": exp, "strike": 105.0, "bid": 4.0, "ask": 4.2, "delta": -0.70},
        ]
    )
    got = pick_atm_contract(chain, day=day, spot=100.0, target_dte=30, right="P")
    assert got is not None
    assert got["strike"] == 100.0


def test_select_contract_uses_delta_when_present():
    day = date(2024, 1, 2)
    exp = day + timedelta(days=30)
    chain = pd.DataFrame(
        [
            {"expiration": exp, "strike": 95.0, "bid": 1.0, "ask": 1.2, "delta": -0.20},
            {"expiration": exp, "strike": 90.0, "bid": 2.0, "ask": 2.2, "delta": -0.31},
            {"expiration": exp, "strike": 85.0, "bid": 3.0, "ask": 3.2, "delta": -0.45},
        ]
    )
    got = select_contract(
        chain, day=day, spot=100.0, target_dte=30, target_delta=0.30, right="P"
    )
    assert got is not None
    assert got["strike"] == 90.0


def test_simulate_put_expire_otm():
    buy = date(2024, 1, 2)
    put_exp = buy + timedelta(days=30)
    days = [buy + timedelta(days=i) for i in range(0, 40)]
    # Spot stays above put strike
    closes = [100.0 + (0.1 if i else 0) for i in range(len(days))]
    panel = _panel(days, closes)
    put_chain = pd.DataFrame(
        [
            {
                "expiration": put_exp,
                "strike": 95.0,
                "bid": 1.0,
                "ask": 1.2,
                "delta": -0.30,
            }
        ]
    )
    src = FakeSource(
        chains={("AAA", buy, "P"): put_chain},
        quotes={
            ("AAA", put_exp, 95.0, "P", buy): {
                "bid": 1.0,
                "ask": 1.2,
                "mid": 1.1,
                "source": "fake",
            }
        },
    )
    row = pd.Series(
        {"stock": "AAA", "buy_date": buy.isoformat(), "buy_price": 100.0, "gain_pct": 2.0}
    )
    got = simulate_one_cycle_wheel(row, panel, src, mark_mode="mid")
    assert got.filled
    assert got.exit_reason == EXIT_PUT_OTM
    assert got.assigned is False
    assert got.gain_cash_secured_pct == pytest.approx(1.1 / 95.0 * 100.0)


def test_simulate_assign_then_call_away():
    buy = date(2024, 1, 2)
    put_exp = buy + timedelta(days=30)
    call_exp = put_exp + timedelta(days=30)
    # Build business-day-ish calendar (include weekends; _spot_on handles)
    days = []
    d = buy
    while d <= call_exp + timedelta(days=2):
        days.append(d)
        d += timedelta(days=1)
    closes = []
    for day in days:
        if day < put_exp:
            closes.append(100.0)
        elif day == put_exp:
            closes.append(90.0)  # ITM assign
        elif day < call_exp:
            closes.append(100.0)
        else:
            closes.append(110.0)  # above call strike -> called
    panel = _panel(days, closes)

    put_chain = pd.DataFrame(
        [
            {
                "expiration": put_exp,
                "strike": 95.0,
                "bid": 2.0,
                "ask": 2.4,
                "delta": -0.30,
            }
        ]
    )
    call_chain = pd.DataFrame(
        [
            {
                "expiration": call_exp,
                "strike": 105.0,
                "bid": 1.0,
                "ask": 1.2,
                "delta": 0.30,
            }
        ]
    )
    quotes = {
        ("AAA", put_exp, 95.0, "P", buy): {
            "bid": 2.0,
            "ask": 2.4,
            "mid": 2.2,
            "source": "fake",
        },
        ("AAA", call_exp, 105.0, "C", put_exp): {
            "bid": 1.0,
            "ask": 1.2,
            "mid": 1.1,
            "source": "fake",
        },
    }
    # If put_exp is weekend, open call on next weekday — seed a few days
    for slack in range(0, 4):
        day = put_exp + timedelta(days=slack)
        quotes[("AAA", call_exp, 105.0, "C", day)] = {
            "bid": 1.0,
            "ask": 1.2,
            "mid": 1.1,
            "source": "fake",
        }

    chains = {("AAA", buy, "P"): put_chain}
    for slack in range(0, 4):
        day = put_exp + timedelta(days=slack)
        chains[("AAA", day, "C")] = call_chain

    src = FakeSource(chains=chains, quotes=quotes)
    row = pd.Series(
        {"stock": "AAA", "buy_date": buy.isoformat(), "buy_price": 100.0, "gain_pct": -5.0}
    )
    got = simulate_one_cycle_wheel(row, panel, src, mark_mode="mid")
    assert got.filled
    assert got.assigned
    assert got.called_away
    assert got.exit_reason == EXIT_CALL_ASSIGN
    # pnl = 2.2 - 95 + 1.1 + 105 = 13.3; cs% = 13.3/95*100
    assert got.pnl_per_share == pytest.approx(13.3)
    assert got.gain_cash_secured_pct == pytest.approx(13.3 / 95.0 * 100.0)


def test_simulate_assign_then_liquidate():
    buy = date(2024, 1, 2)
    put_exp = buy + timedelta(days=30)
    call_exp = put_exp + timedelta(days=30)
    days = []
    d = buy
    while d <= call_exp + timedelta(days=2):
        days.append(d)
        d += timedelta(days=1)
    closes = []
    for day in days:
        if day < put_exp:
            closes.append(100.0)
        elif day == put_exp:
            closes.append(90.0)
        else:
            closes.append(100.0)  # below call 105 -> liquidate
    panel = _panel(days, closes)

    put_chain = pd.DataFrame(
        [
            {
                "expiration": put_exp,
                "strike": 95.0,
                "bid": 2.0,
                "ask": 2.4,
                "delta": -0.30,
            }
        ]
    )
    call_chain = pd.DataFrame(
        [
            {
                "expiration": call_exp,
                "strike": 105.0,
                "bid": 1.0,
                "ask": 1.2,
                "delta": 0.30,
            }
        ]
    )
    quotes = {
        ("AAA", put_exp, 95.0, "P", buy): {
            "bid": 2.0,
            "ask": 2.4,
            "mid": 2.2,
            "source": "fake",
        },
    }
    for slack in range(0, 4):
        day = put_exp + timedelta(days=slack)
        quotes[("AAA", call_exp, 105.0, "C", day)] = {
            "bid": 1.0,
            "ask": 1.2,
            "mid": 1.1,
            "source": "fake",
        }
    chains = {("AAA", buy, "P"): put_chain}
    for slack in range(0, 4):
        day = put_exp + timedelta(days=slack)
        chains[("AAA", day, "C")] = call_chain

    src = FakeSource(chains=chains, quotes=quotes)
    row = pd.Series(
        {"stock": "AAA", "buy_date": buy.isoformat(), "buy_price": 100.0, "gain_pct": -1.0}
    )
    got = simulate_one_cycle_wheel(row, panel, src, mark_mode="mid")
    assert got.filled
    assert got.assigned
    assert not got.called_away
    assert got.exit_reason == EXIT_SHARE_LIQ
    # pnl = 2.2 - 95 + 1.1 + 100 = 8.3
    assert got.pnl_per_share == pytest.approx(8.3)


def test_filter_wheel_by_liquidity_half_and_adv():
    from utils.research.last_15m_wheel import filter_wheel_by_liquidity

    mid = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "put_entry_date": "2022-01-10",
                "put_expiry": "2022-02-18",
                "put_strike": 50.0,
                "put_credit": 1.0,
                "adv_20": 30e6,
                "wheel_ok": True,
                "buy_date": "2022-01-10",
                "wheel_exit_date": "2022-02-18",
                "gain_pct": 2.0,
            },
            {
                "stock": "BBB",
                "put_entry_date": "2022-01-11",
                "put_expiry": "2022-02-18",
                "put_strike": 20.0,
                "put_credit": 1.0,
                "adv_20": 5e6,
                "wheel_ok": True,
                "buy_date": "2022-01-11",
                "wheel_exit_date": "2022-02-18",
                "gain_pct": 1.0,
            },
            {
                "stock": "CCC",
                "put_entry_date": "2022-01-12",
                "put_expiry": "2022-02-18",
                "put_strike": 40.0,
                "put_credit": 2.0,
                "adv_20": 40e6,
                "wheel_ok": True,
                "buy_date": "2022-01-12",
                "wheel_exit_date": "2022-02-18",
                "gain_pct": 3.0,
            },
        ]
    )
    # AAA: half 10% of mid; BBB: half 40%; CCC: half 5%
    ab = mid.copy()
    ab["put_credit"] = [0.90, 0.60, 1.90]
    ab["gain_pct"] = [1.8, 0.5, 2.8]
    m2, a2, meta = filter_wheel_by_liquidity(
        mid, ab, min_adv=25e6, max_half_pct_of_mid=15.0
    )
    assert meta["n_matched"] == 3
    assert meta["n_kept"] == 2  # AAA + CCC
    assert set(m2["stock"]) == {"AAA", "CCC"}
    assert list(a2["put_credit"]) == [0.90, 1.90]


def test_occupancy_fcfs_respects_max_open():
    trades = [
        {
            "symbol": "AAA",
            "entry_date": date(2022, 1, 3),
            "exit_date": date(2022, 2, 3),
            "gain_pct": 1.0,
        },
        {
            "symbol": "BBB",
            "entry_date": date(2022, 1, 4),
            "exit_date": date(2022, 2, 4),
            "gain_pct": 2.0,
        },
        {
            "symbol": "CCC",
            "entry_date": date(2022, 1, 5),
            "exit_date": date(2022, 2, 5),
            "gain_pct": 3.0,
        },
    ]
    kept, meta = occupancy_fcfs_wheel(trades, max_open=1, unique_symbol=True)
    assert meta["n_kept"] == 1
    assert meta["n_skipped"] == 2
    assert kept[0]["symbol"] == "AAA"
    kept2, meta2 = occupancy_fcfs_wheel(trades, max_open=2, unique_symbol=True)
    assert meta2["n_kept"] == 2
    assert {k["symbol"] for k in kept2} == {"AAA", "BBB"}


def test_wheel_slot_equity_compounds_exit():
    trades = [
        {
            "symbol": "AAA",
            "entry_date": date(2022, 1, 3),
            "exit_date": date(2022, 1, 10),
            "gain_pct": 10.0,
        }
    ]
    cal = pd.bdate_range("2022-01-03", "2022-01-14")
    eq, inv = wheel_slot_equity(trades, cal, max_open=1)
    assert float(eq.iloc[0]) == pytest.approx(1.0)
    # Fully invested one slot; exit returns +10%
    assert float(eq.iloc[-1]) == pytest.approx(1.10)
    assert float(inv.max()) == pytest.approx(1.0)


def test_wheel_unrealized_cs_frac_put_and_assign():
    # Short put: credit 2, mark 1, strike 100 -> +1%
    assert wheel_unrealized_cs_frac(
        date(2022, 1, 5),
        put_strike=100.0,
        put_credit=2.0,
        put_expiry=date(2022, 2, 18),
        put_mark=1.0,
        assigned=False,
        spot=105.0,
    ) == pytest.approx(0.01)
    # OTM expiry: credit kept
    assert wheel_unrealized_cs_frac(
        date(2022, 2, 18),
        put_strike=100.0,
        put_credit=2.0,
        put_expiry=date(2022, 2, 18),
        put_mark=0.5,
        assigned=False,
        spot=105.0,
    ) == pytest.approx(0.02)
    # Assigned + short call: put_cred - K + spot + call_cred - call_mark
    # 2 - 100 + 95 + 1.5 - 0.5 = -2 -> -2%
    assert wheel_unrealized_cs_frac(
        date(2022, 3, 1),
        put_strike=100.0,
        put_credit=2.0,
        put_expiry=date(2022, 2, 18),
        put_mark=None,
        assigned=True,
        spot=95.0,
        call_credit=1.5,
        call_strike=100.0,
        call_mark=0.5,
        call_active=True,
    ) == pytest.approx(-0.02)


def test_wheel_slot_equity_mtm_marks_underwater_put():
    trades = [
        {
            "symbol": "AAA",
            "entry_date": date(2022, 1, 3),
            "exit_date": date(2022, 1, 10),
            "gain_pct": 2.0,  # final: credit kept / 100
            "put_strike": 100.0,
            "put_credit": 2.0,
            "put_expiry": date(2022, 1, 10),
            "put_entry_date": date(2022, 1, 3),
            "assigned": False,
        }
    ]
    cal = pd.bdate_range("2022-01-03", "2022-01-12")
    # Mid-hold mark jumps to 5 (underwater vs 2 credit) -> cs frac = (2-5)/100 = -0.03
    mark_paths = {
        0: {
            "put": {
                date(2022, 1, 4): 5.0,
                date(2022, 1, 5): 5.0,
                date(2022, 1, 6): 5.0,
                date(2022, 1, 7): 5.0,
            },
            "call": {},
        }
    }
    eq_mtm, inv, meta = wheel_slot_equity_mtm(
        trades, cal, max_open=1, mark_paths=mark_paths
    )
    assert float(eq_mtm.iloc[0]) == pytest.approx(1.0)
    # First open day is 1/3 — no mark map -> flat to credit mark fallback = 0 unrealized?
    # put_mark None -> intrinsic with no spot -> mark=put_credit -> frac 0
    # On 1/4 underwater: 0.97
    assert float(eq_mtm.loc[pd.Timestamp("2022-01-04")]) == pytest.approx(0.97)
    # Exit 1/10 realizes +2%
    assert float(eq_mtm.iloc[-1]) == pytest.approx(1.02)
    assert meta["mark_hits"] >= 1

