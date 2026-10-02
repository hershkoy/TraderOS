"""Unit tests for Forward Factor math + v2 helpers."""
from __future__ import annotations

from datetime import date

from utils.research.forward_factor import (
    _calendar_debit_credit,
    forward_factor,
    forward_vol,
    has_earnings_in_window,
    is_friday_expiry,
    pick_expiry,
)


def test_forward_vol_video_example():
    iv1, iv2 = 0.45, 0.35
    t1, t2 = 30 / 365.0, 60 / 365.0
    fv = forward_vol(iv1, iv2, t1, t2)
    assert fv is not None
    assert abs(fv - 0.2062) < 0.002
    ff = forward_factor(iv1, fv)
    assert ff is not None
    assert abs(ff - 1.18) < 0.03


def test_forward_vol_rejects_crossed_variance():
    assert forward_vol(0.50, 0.10, 30 / 365.0, 60 / 365.0) is None


def test_forward_factor_zero_when_flat():
    iv = 0.20
    t1, t2 = 60 / 365.0, 90 / 365.0
    fv = forward_vol(iv, iv, t1, t2)
    assert fv is not None
    assert abs(fv - iv) < 1e-9
    assert abs(forward_factor(iv, fv)) < 1e-9


def test_pick_expiry_friday_only():
    day = date(2024, 1, 8)  # Monday
    exps = [
        date(2024, 2, 2),  # Friday ~25d
        date(2024, 2, 5),  # Monday
        date(2024, 2, 9),  # Friday ~32d
        date(2024, 3, 8),  # Friday ~60d
    ]
    front = pick_expiry(exps, day, 30, 10, friday_only=True, monthly_only=False)
    assert front == date(2024, 2, 9)
    assert is_friday_expiry(front)


def test_mid_slip_entry_exit_haircut():
    front = {"bid": 1.0, "ask": 1.20, "mid": 1.10, "spread": 0.20}
    back = {"bid": 3.0, "ask": 3.20, "mid": 3.10, "spread": 0.20}
    entry = _calendar_debit_credit(front, back, mark_mode="mid_slip", slip_frac=0.10, side="entry")
    exit_ = _calendar_debit_credit(front, back, mark_mode="mid_slip", slip_frac=0.10, side="exit")
    assert abs(entry - (2.0 + 0.04)) < 1e-9  # mid 2.0 + 0.1*(0.2+0.2)
    assert abs(exit_ - (2.0 - 0.04)) < 1e-9


def test_earnings_window_skips_equity():
    emap = {"AAPL": [date(2024, 5, 2)]}
    assert has_earnings_in_window("AAPL", date(2024, 4, 1), date(2024, 5, 17), emap)
    assert not has_earnings_in_window("AAPL", date(2024, 6, 1), date(2024, 7, 1), emap)
    assert not has_earnings_in_window("SPY", date(2024, 4, 1), date(2024, 5, 17), emap)
