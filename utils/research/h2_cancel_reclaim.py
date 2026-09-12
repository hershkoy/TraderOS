"""H2 setups that cancel on a support close, then later break the same rails.

TARS Nov-2023: windowed H2 2023-11-15 cancelled 2023-12-01 (close 15.85 vs
support 16.22, ~0.11x width), next session recovered, then close-above-resist
2023-12-27. Last-15m that day opened above the rail.

Pre-registered:
  cancel = first close < support * (1 - error_pct/100) after H2, before a
           resist-break fill (same rule as the H2 walker).
  shallow = cancel-bar close undershoot / width <= 0.25 (TARS was 0.11).
  recovered = next bar close back at/above that bar's support.
  later breakout = first close > resist * (1 + error_pct/100) after cancel,
                   still inside max_wait from H2 and min_wait from H2.
Occupancy is not re-walked. Fills that already printed before the cancel are
a different setup (June-2023 TARS throwover), not this sleeve.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

OUTCOME_FILLED = "filled_first"
OUTCOME_CANCELLED = "cancelled"
OUTCOME_EXPIRED = "expired"
OUTCOME_BAD = "bad"

ERROR_PCT = 1.2
MIN_WAIT = 6
MAX_WAIT = 252
SHALLOW_WIDTH = 0.25
SPAN_DAYS = 365.0


def _as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def rail_at(y0: float, x0: int, slope: float, i: int) -> float:
    return float(y0) + float(slope) * (int(i) - int(x0))


def close_broke_support(
    close: float,
    support: float,
    *,
    error_pct: float = ERROR_PCT,
) -> bool:
    if not np.isfinite(close) or not np.isfinite(support) or support <= 0:
        return False
    return float(close) < float(support) * (1.0 - float(error_pct) / 100.0)


def close_broke_resist(
    close: float,
    resist: float,
    *,
    error_pct: float = ERROR_PCT,
) -> bool:
    if not np.isfinite(close) or not np.isfinite(resist) or resist <= 0:
        return False
    return float(close) > float(resist) * (1.0 + float(error_pct) / 100.0)


def walk_h2_cancel_reclaim(
    close: np.ndarray,
    low: np.ndarray,
    *,
    h2_i: int,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    error_pct: float = ERROR_PCT,
    min_wait: int = MIN_WAIT,
    max_wait: int = MAX_WAIT,
    shallow_width: float = SHALLOW_WIDTH,
) -> Dict[str, Any]:
    """First-event walk: fill vs cancel vs expire, then later resist-break."""
    nan = {
        "outcome": OUTCOME_BAD,
        "fill_i": None,
        "cancel_i": None,
        "cancel_undershoot_close_width": float("nan"),
        "cancel_undershoot_low_width": float("nan"),
        "shallow": False,
        "recovered_next": False,
        "tars_like": False,
        "reclaim_i": None,
        "reclaim_within_wait": False,
        "wait_bars_cancel": None,
        "wait_bars_reclaim": None,
    }
    n = len(close)
    if n == 0 or len(low) != n:
        return nan
    if not np.isfinite(width) or width <= 0:
        return nan
    if not np.isfinite(support_y0) or not np.isfinite(support_slope):
        return nan
    h2 = int(h2_i)
    if h2 < 0 or h2 >= n - 1:
        return nan
    wait_n = max(1, int(max_wait))
    min_w = max(1, int(min_wait))
    end = min(n, h2 + 1 + wait_n)
    fill_i: Optional[int] = None
    cancel_i: Optional[int] = None
    for i in range(h2 + 1, end):
        sup = rail_at(support_y0, support_x0, support_slope, i)
        res = sup + float(width)
        c = float(close[i])
        if close_broke_support(c, sup, error_pct=error_pct):
            cancel_i = int(i)
            break
        if i >= h2 + min_w and close_broke_resist(c, res, error_pct=error_pct):
            fill_i = int(i)
            break
    if fill_i is not None:
        return {
            **nan,
            "outcome": OUTCOME_FILLED,
            "fill_i": fill_i,
            "wait_bars_cancel": None,
        }
    if cancel_i is None:
        return {**nan, "outcome": OUTCOME_EXPIRED}

    sup_c = rail_at(support_y0, support_x0, support_slope, cancel_i)
    c_px = float(close[cancel_i])
    lo_px = float(low[cancel_i])
    under_c = (sup_c - c_px) / float(width) if np.isfinite(c_px) else float("nan")
    under_l = (sup_c - lo_px) / float(width) if np.isfinite(lo_px) else float("nan")
    shallow = bool(np.isfinite(under_c) and 0.0 < under_c <= float(shallow_width))
    recovered = False
    nxt = cancel_i + 1
    if nxt < n:
        sup_n = rail_at(support_y0, support_x0, support_slope, nxt)
        recovered = bool(np.isfinite(close[nxt]) and float(close[nxt]) >= sup_n)
    reclaim_i: Optional[int] = None
    for i in range(cancel_i + 1, end):
        if i < h2 + min_w:
            continue
        sup = rail_at(support_y0, support_x0, support_slope, i)
        res = sup + float(width)
        if close_broke_resist(float(close[i]), res, error_pct=error_pct):
            reclaim_i = int(i)
            break
    return {
        "outcome": OUTCOME_CANCELLED,
        "fill_i": None,
        "cancel_i": cancel_i,
        "cancel_undershoot_close_width": float(under_c),
        "cancel_undershoot_low_width": float(under_l),
        "shallow": shallow,
        "recovered_next": recovered,
        "tars_like": bool(shallow and recovered),
        "reclaim_i": reclaim_i,
        "reclaim_within_wait": reclaim_i is not None,
        "wait_bars_cancel": int(cancel_i - h2),
        "wait_bars_reclaim": None if reclaim_i is None else int(reclaim_i - h2),
    }


def wilder_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    length: int = 14,
) -> np.ndarray:
    n = len(close)
    atr = np.full(n, np.nan, dtype=float)
    if n == 0 or len(high) != n or len(low) != n or n < int(length):
        return atr
    tr = np.empty(n, dtype=float)
    tr[0] = float(high[0]) - float(low[0])
    prev = close[:-1]
    tr[1:] = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - prev), np.abs(low[1:] - prev)),
    )
    atr[int(length) - 1] = float(np.nanmean(tr[: int(length)]))
    alpha = 1.0 / float(length)
    for i in range(int(length), n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def setup_row_ok(setup: Mapping[str, Any], *, max_span_days: float = SPAN_DAYS) -> bool:
    h2 = str(setup.get("h2_date") or "")
    l1 = str(setup.get("start_date") or "")
    if not h2 or not l1:
        return False
    try:
        span = (np.datetime64(h2) - np.datetime64(l1)).astype("timedelta64[D]").astype(int)
    except (TypeError, ValueError):
        return False
    if span < 0 or span > float(max_span_days):
        return False
    width = _as_float(setup.get("channel_width"))
    y0 = _as_float(setup.get("support_y0"))
    slope = _as_float(setup.get("support_slope"))
    h2_i = setup.get("h2_idx")
    x0 = setup.get("support_x0")
    if width is None or width <= 0 or y0 is None or slope is None:
        return False
    if h2_i is None or x0 is None:
        return False
    return True


def count_bucket(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    n = len(rows)
    cancelled = [r for r in rows if r.get("outcome") == OUTCOME_CANCELLED]
    shallow = [r for r in cancelled if r.get("shallow")]
    tars = [r for r in cancelled if r.get("tars_like")]
    recov = [r for r in cancelled if r.get("recovered_next")]

    def _reclaim(part: Sequence[Mapping[str, Any]]) -> int:
        return int(sum(1 for r in part if r.get("reclaim_within_wait")))

    return {
        "n_setups": n,
        "n_filled_first": int(sum(1 for r in rows if r.get("outcome") == OUTCOME_FILLED)),
        "n_expired": int(sum(1 for r in rows if r.get("outcome") == OUTCOME_EXPIRED)),
        "n_cancelled": len(cancelled),
        "n_cancel_reclaim": _reclaim(cancelled),
        "n_shallow": len(shallow),
        "n_shallow_reclaim": _reclaim(shallow),
        "n_recovered_next": len(recov),
        "n_recovered_reclaim": _reclaim(recov),
        "n_tars_like": len(tars),
        "n_tars_like_reclaim": _reclaim(tars),
    }
