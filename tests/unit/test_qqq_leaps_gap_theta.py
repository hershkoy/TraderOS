"""Unit tests for QQQ LEAPS Theta helpers (no live Terminal required)."""
from __future__ import annotations

from datetime import date

from utils.research.qqq_leaps_gap_theta import leaps_cycle_candidates


def test_leaps_cycle_candidates_prefers_cycle_months():
    day = date(2022, 5, 5)
    exps = [
        date(2023, 5, 15),
        date(2023, 6, 16),
        date(2023, 3, 17),
        date(2023, 7, 21),
        date(2023, 9, 15),
    ]
    got = leaps_cycle_candidates(exps, day)
    assert date(2023, 5, 15) not in got
    assert date(2023, 7, 21) not in got
    assert got[0] == date(2023, 6, 16)
