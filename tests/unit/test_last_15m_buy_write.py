"""Unit tests for last-15m buy-write helpers."""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pytest

from utils.research.last_15m_buy_write import (
    EXIT_CALL_ASSIGN,
    EXIT_SHARE_LIQ,
    buy_write_pnl_per_share,
    simulate_buy_write,
    stock_notional_pct,
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


def _panel(days, closes) -> pd.DataFrame:
    idx = pd.to_datetime(days)
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes, "volume": 1},
        index=idx,
    )


def test_buy_write_pnl_called_away():
    # buy 100, credit 2, called at 105 -> 5 + 2 = 7
    pnl = buy_write_pnl_per_share(
        entry_px=100.0,
        call_credit=2.0,
        called_away=True,
        call_strike=105.0,
    )
    assert pnl == pytest.approx(7.0)
    assert stock_notional_pct(pnl, 100.0) == pytest.approx(7.0)


def test_buy_write_pnl_liquidate_with_btc():
    # buy 100, credit 2, BTC 0.5, sell stock 98 -> -100+2-0.5+98 = -0.5
    pnl = buy_write_pnl_per_share(
        entry_px=100.0,
        call_credit=2.0,
        called_away=False,
        share_exit_px=98.0,
        call_close_debit=0.5,
    )
    assert pnl == pytest.approx(-0.5)


def test_simulate_buy_write_call_assign():
    entry = date(2024, 1, 2)
    exp = date(2024, 2, 2)  # Friday
    days = sorted({entry, exp})
    closes = [100.0, 110.0]
    panel = _panel(days, closes)
    chain = pd.DataFrame(
        [
            {
                "expiration": exp,
                "strike": 105.0,
                "bid": 1.8,
                "ask": 2.2,
                "delta": 0.30,
                "dte": 30,
            }
        ]
    )
    src = FakeSource(
        chains={("AAA", entry, "C"): chain},
        quotes={
            ("AAA", exp, 105.0, "C", entry): {
                "bid": 1.8,
                "ask": 2.2,
                "mid": 2.0,
                "source": "fake",
            }
        },
    )
    row = pd.Series(
        {
            "stock": "AAA",
            "buy_date": entry,
            "buy_price": 100.0,
            "gain_pct": 1.0,
        }
    )
    got = simulate_buy_write(row, panel, src, mark_mode="mid")
    assert got.filled
    assert got.exit_reason == EXIT_CALL_ASSIGN
    assert got.gain_pct == pytest.approx(7.0)  # (105-100)+2


def test_apply_credit_slip_reduces_gain():
    from utils.research.last_15m_buy_write import apply_credit_slip

    row = {
        "bw_ok": True,
        "buy_price": 100.0,
        "call_credit": 2.0,
        "bw_called_away": True,
        "call_strike": 105.0,
    }
    # called: (-100+2+105)/100 = 7%; slip 0.5 -> (-100+1.5+105)/100 = 6.5%
    assert apply_credit_slip(row, slip=0.0) == pytest.approx(7.0)
    assert apply_credit_slip(row, slip=0.5) == pytest.approx(6.5)


def test_simulate_buy_write_bs_fallback():
    entry = date(2024, 1, 2)
    exp = date(2024, 2, 2)
    panel = _panel([entry, exp], [100.0, 98.0])
    row = pd.Series(
        {
            "stock": "CCC",
            "buy_date": entry,
            "buy_price": 100.0,
            "gain_pct": -2.0,
            "atr_1d_pct": 2.5,
        }
    )
    from utils.research.last_15m_buy_write import simulate_buy_write_bs

    got = simulate_buy_write_bs(row, panel, mark_mode="mid")
    assert got.filled
    assert got.call is not None
    assert got.call.source == "bs"
    assert got.gain_pct is not None


def test_simulate_buy_write_liquidate():
    entry = date(2024, 1, 2)
    exp = entry + timedelta(days=28)
    days = [entry, exp]
    panel = _panel(days, [100.0, 95.0])
    chain = pd.DataFrame(
        [
            {
                "expiration": exp,
                "strike": 105.0,
                "bid": 1.8,
                "ask": 2.2,
                "delta": 0.30,
                "dte": 28,
            }
        ]
    )
    src = FakeSource(
        chains={("BBB", entry, "C"): chain},
        quotes={
            ("BBB", exp, 105.0, "C", entry): {
                "bid": 1.8,
                "ask": 2.2,
                "mid": 2.0,
                "source": "fake",
            }
        },
    )
    row = pd.Series(
        {"stock": "BBB", "buy_date": entry, "buy_price": 100.0, "gain_pct": -5.0}
    )
    got = simulate_buy_write(row, panel, src, mark_mode="mid")
    assert got.filled
    assert got.exit_reason == EXIT_SHARE_LIQ
    # (95-100)+2 = -3
    assert got.gain_pct == pytest.approx(-3.0)
