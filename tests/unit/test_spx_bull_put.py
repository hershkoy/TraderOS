"""Unit tests for the 45 DTE bull put vertical (no Theta)."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.spx_bull_put import (
    book_stats,
    pick_vertical_strikes,
    run_bull_put_book,
    trade_pnl,
    vertical_credit,
    vertical_debit,
    walk_spread_exit,
)


SHORT_Q = {"bid": 10.0, "ask": 10.2, "mid": 10.1}
LONG_Q = {"bid": 6.9, "ask": 7.1, "mid": 7.0}


def _picker(**kwargs):
    return (
        {
            "short_k": 4700.0,
            "long_k": 4675.0,
            "width": 25.0,
            "delta": 0.22,
            "iv": 0.18,
            "short_q": dict(SHORT_Q),
            "long_q": dict(LONG_Q),
        },
        "ok",
    )


def _spots() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-08", "2024-03-15")
    return pd.DataFrame(
        {"open": 5000.0, "high": 5010.0, "low": 4990.0, "close": 5000.0},
        index=idx,
    )


def _lookup_factory(marks):
    def _lookup(exp, strike, day):
        hit = marks.get((exp, float(strike), day))
        return None if hit is None else dict(hit)

    return _lookup


def test_pick_quotes_do_not_clobber_dividend_yield():
    def quote_fn(_k):
        return {"bid": 10.0, "ask": 10.4, "mid": 10.2}

    picked, reason = pick_vertical_strikes(
        spot=5000.0,
        strikes=[4700.0, 4725.0, 4750.0, 4775.0, 4800.0, 5000.0],
        dte=45,
        width=25.0,
        quote_fn=quote_fn,
    )
    assert reason in ("ok", "no_delta", "no_credit", "no_wing")
    assert picked is None or picked["short_k"] < 5000


def test_vertical_mid_slip_and_natural():
    credit = vertical_credit(SHORT_Q, LONG_Q, "mid", 0.05)
    debit = vertical_debit(
        {"bid": 3.9, "ask": 4.1, "mid": 4.0},
        {"bid": 2.9, "ask": 3.1, "mid": 3.0},
        "mid",
        0.05,
    )
    assert abs(credit - 3.05) < 1e-9
    assert abs(debit - 1.05) < 1e-9
    natural = vertical_credit(SHORT_Q, LONG_Q, "ask_bid", 0.0)
    assert abs(natural - (10.0 - 7.1)) < 1e-9
    natural_debit = vertical_debit(SHORT_Q, LONG_Q, "ask_bid", 0.0)
    assert abs(natural_debit - (10.2 - 6.9)) < 1e-9


def test_trade_pnl_charges_four_legs():
    pnl, ror, risk = trade_pnl(3.05, 1.05, 25.0)
    assert abs(pnl - 197.40) < 1e-9
    assert abs(risk - 21.95) < 1e-9
    assert abs(ror - (197.40 / 2195.0 * 100.0)) < 1e-9


def test_walk_prefers_stop_then_target_then_time():
    exp = date(2024, 2, 23)
    days = [date(2024, 1, 12), date(2024, 1, 18), date(2024, 2, 2)]

    sl = walk_spread_exit(
        session_days=days,
        debit_on=lambda d: {date(2024, 1, 12): 8.0}.get(d),
        expiration=exp,
        credit=3.05,
        use_stop=True,
    )
    assert sl is not None and sl[2] == "sl" and sl[0] == date(2024, 1, 12)

    tp = walk_spread_exit(
        session_days=days,
        debit_on=lambda d: {date(2024, 1, 18): 1.05}.get(d),
        expiration=exp,
        credit=3.05,
    )
    assert tp is not None and tp[2] == "tp"

    timed = walk_spread_exit(
        session_days=days,
        debit_on=lambda d: {date(2024, 2, 2): 2.5}.get(d),
        expiration=exp,
        credit=3.05,
    )
    assert timed is not None and timed[2] == "time_dte" and timed[0] == date(2024, 2, 2)

    # Same 21 DTE session, but the close is already through the stop.
    stopped_late = walk_spread_exit(
        session_days=[date(2024, 2, 2)],
        debit_on=lambda d: 8.0,
        expiration=exp,
        credit=3.05,
        use_stop=True,
    )
    assert stopped_late is not None and stopped_late[2] == "sl"


def test_book_tp_then_next_monday_and_busy_week():
    exp1 = date(2024, 2, 23)
    exp2 = date(2024, 3, 8)
    close_q_short = {"bid": 3.9, "ask": 4.1, "mid": 4.0}
    close_q_long = {"bid": 2.9, "ask": 3.1, "mid": 3.0}
    marks = {
        (exp1, 4700.0, date(2024, 1, 18)): close_q_short,
        (exp1, 4675.0, date(2024, 1, 18)): close_q_long,
        (exp2, 4700.0, date(2024, 1, 23)): close_q_short,
        (exp2, 4675.0, date(2024, 1, 23)): close_q_long,
    }
    book = run_bull_put_book(
        start=date(2024, 1, 8),
        end=date(2024, 3, 15),
        spots=_spots(),
        exps=[exp1, exp2],
        strike_cache={exp1: [4675.0, 4700.0, 5000.0], exp2: [4675.0, 4700.0, 5000.0]},
        picker=_picker,
        quote_lookup=_lookup_factory(marks),
        package_slip=0.05,
        max_concurrent=1,
    )
    kept = [t for t in book.trades if not t.skipped]
    assert [t.entry_date for t in kept] == [date(2024, 1, 8), date(2024, 1, 22)]
    assert [t.exit_reason for t in kept] == ["tp", "tp"]
    assert abs(kept[0].pnl - 197.40) < 1e-6
    reasons = [s["reason"] for s in book.skipped]
    assert "max_concurrent" in reasons
    stats = book_stats(kept)
    assert stats["n"] == 2
    assert stats["pf"] is None or stats["pf"] > 1


def test_time_stop_without_debit_stop():
    exp = date(2024, 2, 23)
    marks = {
        (exp, 4700.0, date(2024, 1, 12)): {"bid": 19.8, "ask": 20.2, "mid": 20.0},
        (exp, 4675.0, date(2024, 1, 12)): {"bid": 0.9, "ask": 1.1, "mid": 1.0},
        (exp, 4700.0, date(2024, 2, 2)): {"bid": 5.4, "ask": 5.6, "mid": 5.5},
        (exp, 4675.0, date(2024, 2, 2)): {"bid": 2.9, "ask": 3.1, "mid": 3.0},
    }
    book = run_bull_put_book(
        start=date(2024, 1, 8),
        end=date(2024, 2, 8),
        spots=_spots(),
        exps=[exp],
        strike_cache={exp: [4675.0, 4700.0, 5000.0]},
        picker=_picker,
        quote_lookup=_lookup_factory(marks),
        package_slip=0.05,
        use_stop=False,
        max_entries=1,
    )
    kept = [t for t in book.trades if not t.skipped]
    assert len(kept) == 1
    assert kept[0].exit_reason == "time_dte"
    assert kept[0].exit_date == date(2024, 2, 2)
