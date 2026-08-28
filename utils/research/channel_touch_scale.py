"""15m scale of daily channel-touch keepers.

Detector v1 is unchanged (bar-indexed Edwards/Magee). This module only supplies
timeframe-scaled defaults so a 15m hunt is not a naive ``--timeframe 15m`` flip.

Scale rules:
  - Pivots are *shorter* (intraday fractals), not 15 daily bars expressed in 15m.
  - Percent rallies / stops / trails shrink by sqrt(bars per RTH session) (vol).
  - RS 63/126 stay session counts; multiply by bars-per-session for lookback bars.
  - ``find_channels`` only sees the last ``max_low_pivots`` swings, so 15m history
    uses a sliding window (15 sessions, step 5) rather than raising that cap.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

BARS_PER_RTH_SESSION = 26  # 6.5h * 4
VOL_SCALE = math.sqrt(BARS_PER_RTH_SESSION)  # ~5.099

RS_SESSION_LOOKBACKS = (63, 126)

# Daily long-history: find_channels v1 only sees last max_low_pivots swings unless windowed.
DAILY_WINDOW_BARS = 504   # ~2y trading days
DAILY_WINDOW_STEP_BARS = 252  # ~1y step


def rs_bar_lookbacks(bars_per_session: int = BARS_PER_RTH_SESSION) -> tuple[int, int]:
    """Bar counts for 63-session and 126-session RS vs SPY."""
    b = max(1, int(bars_per_session))
    return (63 * b, 126 * b)


def scale_pct_points(daily_pct: float, vol_scale: float = VOL_SCALE) -> float:
    """Shrink a daily percent-of-price threshold to 15m vol."""
    return float(daily_pct) / float(vol_scale)


def cli_flag_present(argv: Sequence[str], dest: str) -> bool:
    flag = "--" + dest.replace("_", "-")
    return any(a == flag or a.startswith(flag + "=") for a in argv)


def overlay_preset(ns: Any, preset: Mapping[str, Any], argv: Sequence[str]) -> None:
    """Copy preset keys onto ns unless the user passed the matching CLI flag."""
    for key, val in preset.items():
        if cli_flag_present(argv, key):
            continue
        setattr(ns, key, val)


def apply_daily_long_history_defaults(ns: Any) -> None:
    """Use sliding window scan on daily backtests (v1 only sees last max_low_pivots)."""
    if (getattr(ns, "preset", "") or "").strip() == "15m":
        return
    if str(getattr(ns, "timeframe", "1d")) != "1d":
        return
    if getattr(ns, "no_window_scan", False):
        return
    if int(getattr(ns, "window_bars", 0) or 0) > 0:
        return
    ns.window_bars = DAILY_WINDOW_BARS
    if not int(getattr(ns, "window_step_bars", 0) or 0):
        ns.window_step_bars = DAILY_WINDOW_STEP_BARS


# Daily keepers (for documentation / tests). 15m percents = daily / VOL_SCALE.
PRESET_15M: dict[str, Any] = {
    "timeframe": "15m",
    "provider": "IB",
    "start": "2018-11-01",
    "end": "2025-12-02",
    "pivot_len": 8,
    "min_bars_apart": 8,
    "error_pct": round(scale_pct_points(1.2), 2),
    "min_rally_pct": round(scale_pct_points(4.0), 2),
    "min_pullback_pct": round(scale_pct_points(3.0), 2),
    "min_total_rise_pct": round(scale_pct_points(3.0), 2),
    "flat_pct": round(0.04 / BARS_PER_RTH_SESSION, 4),
    "max_low_pivots": 24,
    "window_bars": BARS_PER_RTH_SESSION * 15,
    "window_step_bars": BARS_PER_RTH_SESSION * 5,
    "trail_pct": round(0.10 / VOL_SCALE, 3),
    "trail_pct_wide": round(0.18 / VOL_SCALE, 3),
    "stop_pct": round(0.03 / VOL_SCALE, 3),
    "stop_pct_floor": round(0.015 / VOL_SCALE, 3),
    "stop_pct_ceil": round(0.06 / VOL_SCALE, 3),
    "atr_stop_mult": 2.0,
    "squeeze_adaptive": True,
    "squeeze_lookback": 100 * BARS_PER_RTH_SESSION,
    "adv_lookback": 20 * BARS_PER_RTH_SESSION,
    "max_channel_span_days": 10.0,
    "require_in_channel": True,
    "max_entries_per_day": 1,
    "friction_pct": 0.10,
    "chunk_size": 10,
    "min_bars": 2000,
    "bars_per_session": BARS_PER_RTH_SESSION,
    "include_time": True,
}
