"""Unit tests for Vanna Rally trigger + contract pick + overnight book."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pytest

from utils.research.vanna_rally import (
    buy_mark,
    find_signal_days,
    pick_otm_call,
    sell_mark,
    simulate_book,
    size_contracts,
)


class FakeSource:
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
        return out[(out["dte"] >= int(dte_lo)) & (out["dte"] <= int(dte_hi))].drop(
            columns=["dte"], errors="ignore"
        )

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
        out = out[(out["session_date"] >= start) & (out["session_date"] <= end)]
        if "mid" not in out.columns:
            out["mid"] = (out["bid"] + out["ask"]) / 2.0
        return out


def test_find_signal_days_requires_all_three():
    idx = pd.to_datetime(
        ["2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"]
    )
    # Day 3: VIX gaps down, drops 1% after open, SPY rallies 0.5% from open
    vix = pd.DataFrame(
        {
            "open": [20.0, 19.0, 18.0, 18.5],
            "high": [21.0, 19.5, 18.2, 19.0],
            "low": [19.5, 18.8, 17.5, 18.0],  # day3 low 17.5 = -2.8% from 18 open
            "close": [20.5, 19.2, 17.8, 18.2],
        },
        index=idx,
    )
    spy = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 101.0],
            "high": [100.5, 100.2, 100.6, 101.5],  # day3 high +0.6%
            "low": [99.5, 99.8, 99.9, 100.5],
            "close": [100.0, 100.1, 100.4, 101.2],
        },
        index=idx,
    )
    sig = find_signal_days(
        vix, spy, start=date(2018, 1, 2), end=date(2018, 1, 5)
    )
    assert list(sig["session"]) == [date(2018, 1, 4)]


def test_find_signal_days_rejects_vix_up_open():
    idx = pd.to_datetime(["2018-01-02", "2018-01-03"])
    vix = pd.DataFrame(
        {
            "open": [20.0, 21.0],  # gaps up
            "high": [21.0, 21.5],
            "low": [19.0, 20.0],
            "close": [20.5, 20.2],
        },
        index=idx,
    )
    spy = pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [100.5, 101.0],
            "low": [99.5, 99.8],
            "close": [100.0, 100.8],
        },
        index=idx,
    )
    sig = find_signal_days(
        vix, spy, start=date(2018, 1, 2), end=date(2018, 1, 3)
    )
    assert sig.empty


def test_pick_otm_call_nearest_target():
    day = date(2018, 6, 1)
    exp = day + timedelta(days=8)
    chain = pd.DataFrame(
        [
            {"expiration": exp, "strike": 100.0, "bid": 2.0, "ask": 2.2, "delta": 0.5},
            {"expiration": exp, "strike": 102.5, "bid": 0.9, "ask": 1.1, "delta": 0.3},
            {"expiration": exp, "strike": 103.0, "bid": 0.8, "ask": 1.0, "delta": 0.28},
            {"expiration": exp, "strike": 105.0, "bid": 0.4, "ask": 0.6, "delta": 0.15},
        ]
    )
    pick = pick_otm_call(chain, day=day, spot=100.0, otm=0.025)
    assert pick is not None
    assert pick["strike"] == 102.5
    assert pick["dte"] == 8


def test_marks_and_sizing():
    assert buy_mark(1.0, 1.2, "mid") == pytest.approx(1.1)
    assert buy_mark(1.0, 1.2, "ask_bid") == pytest.approx(1.2)
    assert buy_mark(1.0, 1.2, "mid", leg_slip=0.05) == pytest.approx(1.15)
    assert sell_mark(1.0, 1.2, "mid") == pytest.approx(1.1)
    assert sell_mark(1.0, 1.2, "ask_bid") == pytest.approx(1.0)
    # $100k * 2.5% / ($1.10 * 100) = 22
    assert size_contracts(100_000, 1.10, 0.025) == 22


def test_simulate_book_overnight_tp():
    d0 = date(2018, 6, 1)
    d1 = date(2018, 6, 4)
    d2 = date(2018, 6, 5)
    exp = d0 + timedelta(days=9)
    chains = {
        (d0, "C"): pd.DataFrame(
            [
                {
                    "expiration": exp,
                    "strike": 102.5,
                    "bid": 0.95,
                    "ask": 1.05,
                    "delta": 0.3,
                }
            ]
        )
    }
    paths = {
        (exp, 102.5, "C"): pd.DataFrame(
            [
                {"session_date": d0, "bid": 0.95, "ask": 1.05},
                {"session_date": d1, "bid": 8.5, "ask": 9.0},  # > 8x mid 1.0
            ]
        )
    }
    src = FakeSource(chains, paths)
    idx = pd.to_datetime([d0])
    signals = pd.DataFrame(
        {
            "open_vix": [18.0],
            "prev_close": [19.0],
            "low_vix": [17.5],
            "open_spy": [100.0],
            "high_spy": [100.6],
            "close_spy": [100.4],
        },
        index=idx,
    )
    book = simulate_book(
        source=src,
        signals=signals,
        sessions=[d0, d1, d2],
        spot_by_day={d0: 100.4, d1: 101.0, d2: 101.2},
        start=d0,
        end=d2,
        mark_mode="mid",
        tp_mult=8.0,
        risk_frac=0.025,
        start_equity=100_000.0,
        tbill_apy=0.0,
        comm_per_contract=0.0,
    )
    assert len(book.trades) == 1
    t = book.trades[0]
    assert t.exit_reason == "tp"
    assert t.exit_px == pytest.approx(8.0)  # 8x entry mid 1.0
    assert t.pnl is not None and t.pnl > 0
