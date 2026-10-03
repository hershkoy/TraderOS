"""NVDA earnings expected-move premium sale (Options With Ravish).

Video: https://youtu.be/FtX1JshVLvg (2026-02-24)

Shared helpers live in ``utils.research.earnings_em``. This module keeps the
NVDA claim constants and the split-adjusted price used by the original runner.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict

from utils.research.earnings_em import (  # noqa: F401
    COMM_PER_LEG,
    EXAMPLE_WIDTH,
    MODES,
    MULT,
    WIDTH_PCT,
    amc_sessions,
    book_stats,
    buy_price,
    call_intrinsic,
    classify_open,
    defined_risk,
    fetch_clock_nbbo,
    held_iron_condor,
    held_short_put,
    is_amc,
    mode_key,
    nearest_strike,
    package_pnl,
    parse_earn_date,
    pick_expiry_after,
    put_intrinsic,
    sell_price,
    spread_intrinsic,
    strike_just_above,
    strike_just_below,
    synthetic_forward,
    target_wing,
    usable_quote,
    wing_strike,
)
from utils.research.earnings_em import unadjusted_price as unadjusted_price_with_splits

CLAIMED: Dict[str, Any] = {
    "source": "Options With Ravish — Best Earnings Options Trading Strategy (FtX1JshVLvg)",
    "symbol": "NVDA",
    "window": "12 earnings, about 2023-02-22 through 2025-11-19",
    "short_put_wr": 1.0,
    "pcs_wr": 1.0,
    "iron_condor_wr": 0.75,
    "opened_flat_or_higher": "11 of 12",
    "inside_expected_move": "9 of 12",
    "broke_downside": "0 of 12",
    "entry": "15:45 ET on an after-close report",
    "exit": "next session open",
    "example": "spot ~190, EM 5.7% ($10.8), short 177.5 put, 5-wide wings, Feb 27 expiry",
}

# Yahoo close is split-adjusted. Theta option strikes are not.
# NVDA 10-for-1 on 2024-06-10. No earnings fall across that week.
NVDA_SPLIT = date(2024, 6, 10)
NVDA_SPLIT_FACTOR = 10.0
NVDA_SPLITS = ((NVDA_SPLIT, NVDA_SPLIT_FACTOR),)


def unadjusted_price(session: date, adjusted: float) -> float:
    """Map a split-adjusted NVDA print onto the strike scale of that session."""
    return unadjusted_price_with_splits(session, adjusted, NVDA_SPLITS)
