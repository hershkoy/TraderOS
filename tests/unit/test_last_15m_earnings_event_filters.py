"""Unit tests for last-15m earnings / biotech event filter helpers."""
from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.last_15m_earnings_event_filters import (
    EarningsEvent,
    apply_flatten_before_earnings,
    biotech_keep_mask,
    count_sessions_ahead,
    em_gt_stop_keep_mask,
    entry_skip_earnings_mask,
    flatten_session_for_event,
    is_biotech_sector,
    is_biotech_sic,
    next_earnings_after_fill,
    overnight_em_pct,
    stop_pct_for_row,
)

ET = ZoneInfo("America/New_York")


def _bar(day: date, hour: int, minute: int, mid: float) -> dict:
    ts = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)
    return {"ts": ts, "high": mid + 0.05, "low": mid - 0.05, "open": mid, "close": mid}


def test_overnight_em_and_stop_pct():
    em = overnight_em_pct(0.252)
    assert em is not None
    assert abs(em - 100.0 * 0.252 * (1.0 / 252.0) ** 0.5) < 1e-9
    assert overnight_em_pct(None) is None
    assert abs(stop_pct_for_row({"atr_1d_pct": 2.0}) - 4.0) < 1e-9
    assert abs(stop_pct_for_row({"atr_1d_pct": 0.1}) - 1.5) < 1e-9
    assert abs(stop_pct_for_row({"atr_1d_pct": 10.0}) - 6.0) < 1e-9


def test_biotech_flags():
    assert is_biotech_sic("2834")
    assert is_biotech_sic(2836)
    assert not is_biotech_sic("7370")
    assert is_biotech_sector("Biotechnology")
    assert is_biotech_sector("Health Care Pharmaceuticals")
    assert not is_biotech_sector("Information Technology")


def test_next_earnings_ignores_same_day_bmo():
    events = [
        EarningsEvent(earn_date=date(2024, 2, 9), amc=False),
        EarningsEvent(earn_date=date(2024, 2, 9), amc=True),
    ]
    got = next_earnings_after_fill(events, date(2024, 2, 9))
    assert got is not None and got.amc
    assert flatten_session_for_event(got) == date(2024, 2, 9)


def test_entry_skip_amc_today_and_bmo_next():
    trades = pd.DataFrame(
        [
            {"stock": "AAA", "buy_date": "2024-02-09", "gain_pct": 1.0},
            {"stock": "BBB", "buy_date": "2024-02-09", "gain_pct": 1.0},
            {"stock": "CCC", "buy_date": "2024-02-09", "gain_pct": 1.0},
        ]
    )
    emap = {
        "AAA": [EarningsEvent(earn_date=date(2024, 2, 9), amc=True)],
        "BBB": [EarningsEvent(earn_date=date(2024, 2, 12), amc=False)],  # Mon after Fri
        "CCC": [EarningsEvent(earn_date=date(2024, 3, 1), amc=True)],
    }
    keep = entry_skip_earnings_mask(trades, emap)
    assert list(keep) == [False, False, True]


def test_em_gt_stop_missing_keeps():
    trades = pd.DataFrame(
        [
            {"stock": "A", "name_atm_iv": 2.0, "atr_1d_pct": 1.0, "gain_pct": -5.0},
            {"stock": "B", "name_atm_iv": float("nan"), "atr_1d_pct": 1.0, "gain_pct": 1.0},
            {"stock": "C", "name_atm_iv": 0.10, "atr_1d_pct": 3.0, "gain_pct": 1.0},
        ]
    )
    keep = em_gt_stop_keep_mask(trades)
    # A: EM ~ 12.6% > stop 2% -> drop; B missing keep; C EM small keep
    assert list(keep) == [False, True, True]


def test_biotech_keep_mask():
    trades = pd.DataFrame(
        [
            {"stock": "MRNA", "gain_pct": 1.0},
            {"stock": "AAPL", "gain_pct": 1.0},
            {"stock": "PFE", "gain_pct": 1.0},
        ]
    )
    keep = biotech_keep_mask(
        trades,
        sector_by_symbol={"MRNA": "Biotechnology", "AAPL": "Technology"},
        sic_by_symbol={"PFE": "2834", "AAPL": "3571"},
    )
    assert list(keep) == [False, True, False]


def test_flatten_amc_same_day_skips_entry():
    buy = date(2024, 2, 9)
    by_day = {
        buy: [_bar(buy, 15, 45, 14.17)],
        date(2024, 2, 12): [_bar(date(2024, 2, 12), 15, 45, 14.0)],
    }
    row = {
        "buy_price": 14.17,
        "sell_price": 11.15,
        "gain_pct": -21.3,
        "buy_date": "2024-02-09",
        "sell_date": "2024-02-21",
        "buy_time": datetime(2024, 2, 9, 15, 45, tzinfo=ET),
        "sell_time": datetime(2024, 2, 21, 9, 45, tzinfo=ET),
    }
    events = [EarningsEvent(earn_date=buy, amc=True)]
    got = apply_flatten_before_earnings(row, events, by_day, max_sessions=1)
    assert got.used and got.skipped
    assert got.sessions_ahead == 0


def test_flatten_later_print_exits_last_mid():
    buy = date(2024, 2, 9)
    flat = date(2024, 2, 20)
    by_day = {
        buy: [_bar(buy, 15, 45, 14.17)],
        date(2024, 2, 12): [_bar(date(2024, 2, 12), 15, 45, 14.5)],
        flat: [_bar(flat, 15, 30, 14.0), _bar(flat, 15, 45, 13.90)],
    }
    row = {
        "buy_price": 14.17,
        "sell_price": 11.15,
        "gain_pct": -21.3,
        "buy_date": "2024-02-09",
        "sell_date": "2024-02-21",
        "buy_time": datetime(2024, 2, 9, 15, 45, tzinfo=ET),
        "sell_time": datetime(2024, 2, 21, 9, 45, tzinfo=ET),
    }
    events = [EarningsEvent(earn_date=flat, amc=True)]
    ahead = count_sessions_ahead(buy, flat, sorted(by_day.keys()))
    assert ahead == 2
    got = apply_flatten_before_earnings(row, events, by_day, max_sessions=10)
    assert got.used and not got.skipped
    assert got.sell_px is not None and abs(got.sell_px - 13.90) < 1e-9
    assert got.gain_pct < 0
