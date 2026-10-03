"""Unit tests for multi-name earnings expected-move helpers (no Theta)."""
from __future__ import annotations

from datetime import date

from utils.research.earnings_em import (
    STRUCT_IC,
    STRUCT_PCS,
    STRUCT_SKIP,
    attach_trailing,
    bmo_sessions,
    earn_sessions,
    pick_expiry_for_report,
    pick_structure,
    trailing_stats,
    unadjusted_price,
)


def test_yahoo_splits_scale_only_sessions_before_the_event():
    splits = [(date(2024, 6, 10), 10.0), (date(2020, 7, 20), 4.0)]
    # 2024-05-22 is before the 10-for-1 and after the 4-for-1.
    assert abs(unadjusted_price(date(2024, 5, 22), 94.95, splits) - 949.5) < 1e-9
    assert abs(unadjusted_price(date(2024, 8, 28), 125.61, splits) - 125.61) < 1e-9
    # Before both splits: 10 * 4.
    assert abs(unadjusted_price(date(2019, 1, 2), 5.0, splits) - 200.0) < 1e-9
    assert unadjusted_price(date(2024, 5, 22), 100.0, []) == 100.0


def test_bmo_enters_the_prior_session_and_exits_the_report_open():
    days = [date(2026, 1, 28), date(2026, 1, 29), date(2026, 1, 30)]
    assert bmo_sessions(date(2026, 1, 29), days) == (date(2026, 1, 28), date(2026, 1, 29))
    assert earn_sessions(date(2026, 1, 29), days, amc=False) == (
        date(2026, 1, 28),
        date(2026, 1, 29),
    )
    assert earn_sessions(date(2026, 1, 29), days, amc=True) == (
        date(2026, 1, 29),
        date(2026, 1, 30),
    )


def test_bmo_expiry_can_be_the_report_day_amc_cannot():
    exps = [date(2026, 1, 29), date(2026, 1, 30), date(2026, 2, 6)]
    assert pick_expiry_for_report(exps, date(2026, 1, 29), amc=False) == date(2026, 1, 29)
    assert pick_expiry_for_report(exps, date(2026, 1, 29), amc=True) == date(2026, 1, 30)


def _row(sym, day, inside, down, green=True):
    return {
        "symbol": sym,
        "earn_date": day,
        "inside_em": inside,
        "broke_down": down,
        "opened_green": green,
        "broke_up": False,
    }


def test_trailing_stats_ignore_the_event_being_classified():
    prior = [
        _row("AAPL", "2023-01-01", True, False),
        _row("AAPL", "2023-04-01", True, False),
        _row("AAPL", "2023-07-01", True, False),
        _row("AAPL", "2023-10-01", True, False),
        _row("AAPL", "2024-01-01", True, False),
        _row("AAPL", "2024-04-01", False, True),
    ]
    short = trailing_stats(prior[:5], min_n=6)
    assert short["ready"] is False
    ready = trailing_stats(prior, min_n=6)
    assert ready["ready"] is True
    assert ready["n"] == 6
    assert ready["inside"] == 5
    assert ready["broke_down"] == 1


def test_structure_pick_is_pcs_only_when_the_downside_never_broke():
    clean = {"ready": True, "inside_frac": 0.83, "broke_down": 0}
    mixed = {"ready": True, "inside_frac": 0.83, "broke_down": 1}
    leaky = {"ready": True, "inside_frac": 0.50, "broke_down": 0}
    assert pick_structure(clean) == STRUCT_PCS
    assert pick_structure(mixed) == STRUCT_IC
    assert pick_structure(leaky) == STRUCT_SKIP
    assert pick_structure({"ready": False, "n": 3}) == STRUCT_SKIP


def test_walk_forward_assigns_the_next_event_from_prior_reports_only():
    rows = [
        _row("MSFT", "2023-01-24", True, False),
        _row("MSFT", "2023-04-25", True, False),
        _row("MSFT", "2023-07-25", True, False),
        _row("MSFT", "2023-10-24", True, False),
        _row("MSFT", "2024-01-30", True, False),
        _row("MSFT", "2024-04-25", True, False),
        _row("MSFT", "2024-07-23", False, True),
        _row("AAPL", "2024-07-01", True, False),
    ]
    tagged = attach_trailing(rows, min_n=6, window=12)
    by_day = {(r["symbol"], r["earn_date"]): r for r in tagged}
    # First six MSFT events are not ready.
    assert by_day[("MSFT", "2023-01-24")]["wf_structure"] == STRUCT_SKIP
    assert by_day[("MSFT", "2024-04-25")]["wf_structure"] == STRUCT_SKIP
    # Seventh MSFT uses the six clean reports -> PCS, not the down-gap on itself.
    seventh = by_day[("MSFT", "2024-07-23")]
    assert seventh["trail_n"] == 6
    assert seventh["trail_broke_down"] == 0
    assert seventh["wf_structure"] == STRUCT_PCS
    # AAPL has only one row.
    assert by_day[("AAPL", "2024-07-01")]["wf_structure"] == STRUCT_SKIP
