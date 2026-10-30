"""Unit tests for SPXW iron-fly L5 live state (no IB/Theta)."""
from __future__ import annotations

from datetime import date

from utils.scanning.spxw_iron_fly_l5 import (
    _decide_from_stored,
    _empty_state,
    decide_today,
)


def test_active_takes_when_few_losses():
    state = _empty_state()
    state["outcomes"] = [
        {"date": "2026-01-02", "taken": True, "pnl": 100.0},
        {"date": "2026-01-03", "taken": True, "pnl": -50.0},
    ]
    d = decide_today(state, date(2026, 1, 6))
    assert d["take"] is True
    assert d["machine"] == "active"


def test_sit_after_five_losses():
    state = _empty_state()
    state["outcomes"] = [
        {"date": f"2026-01-{i:02d}", "taken": True, "pnl": -100.0}
        for i in range(2, 7)
    ]
    d = _decide_from_stored(state["outcomes"], "2026-01-07", state)
    assert d["take"] is False
    assert d["machine"] == "sit"
    assert d["sit_left"] == 3


def test_probe_after_sit_exhausted():
    state = _empty_state()
    outcomes = [
        {"date": f"2026-01-{i:02d}", "taken": True, "pnl": -100.0}
        for i in range(2, 7)
    ]
    # three sit skips
    outcomes += [
        {"date": "2026-01-07", "taken": False, "pnl": None},
        {"date": "2026-01-08", "taken": False, "pnl": None},
        {"date": "2026-01-09", "taken": False, "pnl": None},
    ]
    d = _decide_from_stored(outcomes, "2026-01-10", state)
    assert d["take"] is True
    assert d["machine"] == "probe"
