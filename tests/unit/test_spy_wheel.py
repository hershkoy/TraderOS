"""Unit tests for SPY wheel / covered-strangle helpers and state machine."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

from utils.research.spy_wheel import (
    ClosedLeg,
    ShortLeg,
    entry_credit,
    exit_debit,
    pick_contract,
    run_book,
    target_call_contracts,
)


class FakeSource:
    """Deterministic chains/paths for unit tests."""

    def __init__(
        self,
        chains: Dict[Tuple[date, str], pd.DataFrame],
        paths: Dict[Tuple[date, float, str], pd.DataFrame],
    ) -> None:
        self.chains = chains
        self.paths = paths

    def chain_on_day(
        self, day: date, right: str, dte_lo: int, dte_hi: int
    ) -> pd.DataFrame:
        right_c = "C" if str(right).upper().startswith("C") else "P"
        df = self.chains.get((day, right_c))
        if df is None or df.empty:
            return pd.DataFrame(
                columns=["expiration", "strike", "bid", "ask", "delta"]
            )
        out = df.copy()
        out["dte"] = out["expiration"].map(lambda e: (e - day).days)
        return out[
            (out["dte"] >= int(dte_lo)) & (out["dte"] <= int(dte_hi))
        ].drop(columns=["dte"], errors="ignore")

    def quote_path(
        self,
        expiration: date,
        strike: float,
        right: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        right_c = "C" if str(right).upper().startswith("C") else "P"
        df = self.paths.get((expiration, float(strike), right_c))
        if df is None or df.empty:
            return pd.DataFrame(columns=["session_date", "bid", "ask", "mid"])
        out = df.copy()
        out = out[
            (out["session_date"] >= start) & (out["session_date"] <= end)
        ]
        if "mid" not in out.columns:
            out["mid"] = (out["bid"] + out["ask"]) / 2.0
        return out


def _spots(days: List[date], prices: List[float]) -> pd.DataFrame:
    idx = pd.to_datetime(days)
    return pd.DataFrame({"close": prices}, index=idx)


def test_target_call_contracts_50pct():
    assert target_call_contracts(200, 0.5) == 1
    assert target_call_contracts(300, 0.5) == 1
    assert target_call_contracts(400, 0.5) == 2
    assert target_call_contracts(100, 0.5) == 0
    assert target_call_contracts(200, 1.0) == 2


def test_pick_contract_nearest_delta():
    day = date(2020, 1, 2)
    exp = day + timedelta(days=30)
    chain = pd.DataFrame(
        [
            {"expiration": exp, "strike": 95.0, "bid": 1.0, "ask": 1.2, "delta": -0.20},
            {"expiration": exp, "strike": 90.0, "bid": 2.0, "ask": 2.2, "delta": -0.31},
            {"expiration": exp, "strike": 85.0, "bid": 3.0, "ask": 3.2, "delta": -0.45},
        ]
    )
    got = pick_contract(chain, day=day, target_dte=30, target_delta=0.30)
    assert got is not None
    assert got["strike"] == 90.0
    assert abs(got["delta"]) == pytest.approx(0.31)


def test_entry_exit_marks():
    q = {"bid": 1.0, "ask": 1.4, "mid": 1.2}
    assert entry_credit(q, "mid") == 1.2
    assert entry_credit(q, "ask_bid") == 1.0
    assert exit_debit(1.0, 1.4, "mid") == pytest.approx(1.2)
    assert exit_debit(1.0, 1.4, "ask_bid") == pytest.approx(1.4)


def test_wheel_put_assignment_then_call():
    """Sell put, expire ITM -> shares; sell call, expire OTM."""
    d0 = date(2020, 6, 1)
    days = [
        d0 + timedelta(days=i)
        for i in range(80)
        if (d0 + timedelta(days=i)).weekday() < 5
    ]
    prices = []
    for i, _ in enumerate(days):
        prices.append(100.0 - min(i, 20) * 0.5 if i < 20 else 90.0)
    spots = _spots(days, prices)

    put_exp = days[15]
    call_exp = days[35]
    put_chain_day = days[0]
    chains = {
        (put_chain_day, "P"): pd.DataFrame(
            [
                {
                    "expiration": put_exp,
                    "strike": 98.0,
                    "bid": 2.0,
                    "ask": 2.2,
                    "delta": -0.30,
                }
            ]
        ),
    }
    assign_day = put_exp
    next_days = [d for d in days if d > assign_day]
    call_open = next_days[0]
    chains[(call_open, "C")] = pd.DataFrame(
        [
            {
                "expiration": call_exp,
                "strike": 95.0,
                "bid": 1.5,
                "ask": 1.7,
                "delta": 0.30,
            }
        ]
    )

    put_path_rows = []
    for d in days:
        if d <= put_exp:
            mid = max(98.0 - prices[days.index(d)], 0.1) if d == put_exp else 2.0
            put_path_rows.append(
                {"session_date": d, "bid": mid - 0.05, "ask": mid + 0.05, "mid": mid}
            )
    call_path_rows = []
    for d in days:
        if call_open <= d <= call_exp:
            call_path_rows.append(
                {"session_date": d, "bid": 0.4, "ask": 0.6, "mid": 0.5}
            )

    paths = {
        (put_exp, 98.0, "P"): pd.DataFrame(put_path_rows),
        (call_exp, 95.0, "C"): pd.DataFrame(call_path_rows),
    }
    src = FakeSource(chains, paths)
    book = run_book(
        mode="wheel",
        start=days[0],
        end=days[-1],
        source=src,
        mark_mode="mid",
        target_dte=20,
        target_delta=0.30,
        dte_tol=20,
        tp_frac=None,
        spots=spots,
        min_credit=0.05,
    )
    assert any(e.get("event") == "assign_put" for e in book.events)
    assert book.equity.iloc[-1] > 0


def test_tp_closes_short():
    d0 = date(2021, 1, 4)
    days = [d0 + timedelta(days=i) for i in range(40) if (d0 + timedelta(days=i)).weekday() < 5]
    prices = [100.0] * len(days)
    spots = _spots(days, prices)
    exp = days[20]
    chains = {
        (days[0], "P"): pd.DataFrame(
            [
                {
                    "expiration": exp,
                    "strike": 95.0,
                    "bid": 2.0,
                    "ask": 2.0,
                    "delta": -0.30,
                }
            ]
        )
    }
    path_rows = []
    for i, d in enumerate(days):
        if d > exp:
            break
        # After a few days, premium collapses to 0.4 (<= 25% of 2.0)
        mid = 2.0 if i < 3 else 0.4
        path_rows.append(
            {"session_date": d, "bid": mid, "ask": mid, "mid": mid}
        )
    paths = {(exp, 95.0, "P"): pd.DataFrame(path_rows)}
    src = FakeSource(chains, paths)
    book = run_book(
        mode="wheel",
        start=days[0],
        end=days[-1],
        source=src,
        mark_mode="mid",
        target_dte=20,
        target_delta=0.30,
        dte_tol=25,
        tp_frac=0.75,
        spots=spots,
    )
    assert any(c.reason == "tp" for c in book.closed)


def test_sma_put_gate_blocks_then_allows():
    # Build sessions: price below SMA then jumps above; put strike cash-feasible.
    d0 = date(2019, 1, 2)
    days: List[date] = []
    d = d0
    while len(days) < 230:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    prices = [90.0 if i < 210 else 120.0 for i in range(len(days))]
    spots = _spots(days, prices)

    chains: Dict[Tuple[date, str], pd.DataFrame] = {}
    paths: Dict[Tuple[date, float, str], pd.DataFrame] = {}
    for day in days[200:]:
        exp_p = day + timedelta(days=20)
        chains[(day, "P")] = pd.DataFrame(
            [
                {
                    "expiration": exp_p,
                    "strike": 85.0,
                    "bid": 1.5,
                    "ask": 1.7,
                    "delta": -0.30,
                }
            ]
        )
        key_p = (exp_p, 85.0, "P")
        if key_p not in paths:
            paths[key_p] = pd.DataFrame(
                [
                    {
                        "session_date": d2,
                        "bid": 1.5,
                        "ask": 1.7,
                        "mid": 1.6,
                    }
                    for d2 in days
                    if day <= d2 <= exp_p
                ]
            )
        exp_c = day + timedelta(days=20)
        chains[(day, "C")] = pd.DataFrame(
            [
                {
                    "expiration": exp_c,
                    "strike": 125.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "delta": 0.30,
                }
            ]
        )
        key_c = (exp_c, 125.0, "C")
        if key_c not in paths:
            paths[key_c] = pd.DataFrame(
                [
                    {
                        "session_date": d2,
                        "bid": 1.0,
                        "ask": 1.2,
                        "mid": 1.1,
                    }
                    for d2 in days
                    if day <= d2 <= exp_c
                ]
            )

    src = FakeSource(chains, paths)
    start = days[200]
    book = run_book(
        mode="cs",
        start=start,
        end=days[-1],
        source=src,
        mark_mode="mid",
        target_dte=20,
        target_delta=0.30,
        dte_tol=15,
        tp_frac=0.75,
        cover_frac=0.5,
        use_sma_put_gate=True,
        spots=spots,
    )
    put_opens = [
        e for e in book.events if e.get("event") == "open" and e.get("right") == "P"
    ]
    assert len(put_opens) >= 1
    first_put = date.fromisoformat(put_opens[0]["date"])
    # Prior close > SMA only after the jump (flat 90 keeps close <= SMA)
    assert first_put >= days[211]


def test_cover_frac_one_call_on_200_shares():
    assert target_call_contracts(200, 0.5) == 1


def test_unit_lot_put_opens_when_cash_below_strike():
    """After premium-only years, cash can be < strike*100; still open 1 lot."""
    d0 = date(2015, 1, 5)
    days = [
        d0 + timedelta(days=i)
        for i in range(40)
        if (d0 + timedelta(days=i)).weekday() < 5
    ]
    # Spot ~200 needs ~20k collateral; start NAV only 10k via low synthetic open
    # run_book sets cash = spot0*100, so use spot path that jumps after entry day.
    prices = [100.0] + [200.0] * (len(days) - 1)
    spots = _spots(days, prices)
    exp = days[18]
    chains = {
        (days[0], "P"): pd.DataFrame(
            [
                {
                    "expiration": exp,
                    "strike": 95.0,
                    "bid": 1.0,
                    "ask": 1.2,
                    "delta": -0.30,
                }
            ]
        ),
        (days[1], "P"): pd.DataFrame(
            [
                {
                    "expiration": days[1] + timedelta(days=25),
                    "strike": 190.0,
                    "bid": 3.0,
                    "ask": 3.2,
                    "delta": -0.30,
                }
            ]
        ),
    }
    # Expire first put OTM on exp with spot already 200? days[0] put strike 95
    # simpler: only open on day0, then on day1 after we need high strike
    paths = {
        (exp, 95.0, "P"): pd.DataFrame(
            [
                {
                    "session_date": d,
                    "bid": 0.05,
                    "ask": 0.10,
                    "mid": 0.075,
                }
                for d in days
                if d <= exp
            ]
        ),
        (days[1] + timedelta(days=25), 190.0, "P"): pd.DataFrame(
            [
                {
                    "session_date": d,
                    "bid": 2.0,
                    "ask": 2.2,
                    "mid": 2.1,
                }
                for d in days
                if d >= days[1]
            ]
        ),
    }
    # Force TP on first put quickly so we re-open at high spot with cash ~10k < 19k
    src = FakeSource(chains, paths)
    book = run_book(
        mode="wheel",
        start=days[0],
        end=days[-1],
        source=src,
        mark_mode="mid",
        target_dte=25,
        target_delta=0.30,
        dte_tol=20,
        tp_frac=0.75,
        spots=spots,
    )
    put_opens = [
        e for e in book.events if e.get("event") == "open" and e.get("right") == "P"
    ]
    assert len(put_opens) >= 2
    assert any(float(e["strike"]) >= 180 for e in put_opens)
