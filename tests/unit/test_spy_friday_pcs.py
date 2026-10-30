"""Unit tests for SPY Friday 7 DTE put credit spread helpers."""
from __future__ import annotations

from datetime import date

import pandas as pd

from utils.research.spy_friday_pcs import (
    friday_entry_days,
    intrinsic_spread_value,
    pick_pcs,
    vertical_credit,
    vertical_debit,
)


def test_friday_entry_days_skips_closed_friday():
    # 2021-12-24 Fri open, 2021-12-31 Fri open; invent a closed Friday
    sessions = [
        date(2021, 12, 23),
        date(2021, 12, 27),  # Mon after Christmas Fri closed
        date(2021, 12, 28),
        date(2021, 12, 29),
        date(2021, 12, 30),
        date(2021, 12, 31),
    ]
    # Include Christmas week: Fri Dec 24 is NOT in sessions -> roll to Mon 27
    entries = friday_entry_days(sessions, date(2021, 12, 20), date(2021, 12, 31))
    assert date(2021, 12, 27) in entries
    assert date(2021, 12, 31) in entries


def test_vertical_mid_and_natural():
    # short 1.00/1.20, long 0.40/0.50
    assert abs(vertical_credit(1.0, 1.2, 0.4, 0.5, "mid") - 0.65) < 1e-9
    assert abs(vertical_credit(1.0, 1.2, 0.4, 0.5, "ask_bid") - 0.50) < 1e-9
    assert abs(vertical_debit(1.0, 1.2, 0.4, 0.5, "mid") - 0.65) < 1e-9
    assert abs(vertical_debit(1.0, 1.2, 0.4, 0.5, "ask_bid") - 0.80) < 1e-9


def test_vertical_package_slip_clamped_to_nbbo():
    # mid credit 0.65, natural 0.50; slip 0.05 -> 0.60; slip 0.50 clamps to 0.50
    assert abs(vertical_credit(1.0, 1.2, 0.4, 0.5, "mid", package_slip=0.05) - 0.60) < 1e-9
    assert abs(vertical_credit(1.0, 1.2, 0.4, 0.5, "mid", package_slip=0.50) - 0.50) < 1e-9
    # mid debit 0.65, natural 0.80; slip 0.05 -> 0.70; slip 0.50 clamps to 0.80
    assert abs(vertical_debit(1.0, 1.2, 0.4, 0.5, "mid", package_slip=0.05) - 0.70) < 1e-9
    assert abs(vertical_debit(1.0, 1.2, 0.4, 0.5, "mid", package_slip=0.50) - 0.80) < 1e-9


def test_intrinsic_spread_value():
    # spot 100, short 95 / long 90 -> both OTM
    assert intrinsic_spread_value(100, 95, 90) == 0.0
    # spot 92 -> short ITM 3, long OTM -> value 3
    assert intrinsic_spread_value(92, 95, 90) == 3.0
    # spot 85 -> both ITM, value = width 5
    assert intrinsic_spread_value(85, 95, 90) == 5.0


def test_pick_pcs_nearest_delta_and_wing():
    day = date(2024, 1, 5)  # Friday
    exp = date(2024, 1, 12)
    rows = []
    for k, d, bid, ask in [
        (470.0, -0.55, 2.0, 2.2),
        (465.0, -0.40, 1.4, 1.6),
        (460.0, -0.28, 0.9, 1.0),
        (460.0, -0.28, 0.9, 1.0),  # duplicate ok
        (455.0, -0.18, 0.4, 0.5),
    ]:
        rows.append(
            {
                "expiration": exp,
                "strike": k,
                "bid": bid,
                "ask": ask,
                "delta": d,
            }
        )
    # wing for 465 short is 460
    chain = pd.DataFrame(rows)
    got, reason = pick_pcs(
        chain, day=day, spot=480.0, target_delta=0.40, width=5.0, mark_mode="mid"
    )
    assert reason == "ok"
    assert got is not None
    assert got["short_k"] == 465.0
    assert got["long_k"] == 460.0
    assert got["dte"] == 7
    assert abs(got["credit"] - 0.55) < 1e-9  # mid 1.5 - mid 0.95
