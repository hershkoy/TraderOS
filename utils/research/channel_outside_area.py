"""Channel parallelogram vs undershoot-below-support area (L1 through buy).

The yellow region on a 3-touch chart is price that traded *below* the support
rail. Channel area is the parallel band (constant width) on a bar-index axis
from L1 through the buy bar, inclusive.

Ratio = sum(max(0, support - low)) / (width * n_bars).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SKIP_NO_PANEL = "no_panel"
SKIP_NO_L1 = "no_l1"
SKIP_NO_BUY = "no_buy"
SKIP_BAD_WINDOW = "bad_window"
SKIP_BAD_WIDTH = "bad_width"
SKIP_BAD_RAIL = "bad_rail"


def stamp_date(value: Any) -> Optional[date]:
    """Wall date of a stored stamp. Do not convert UTC midnight to New York."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        t = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(t):
        return None
    return date(int(t.year), int(t.month), int(t.day))


def date_to_first_index(index: Sequence[Any]) -> Dict[date, int]:
    out: Dict[date, int] = {}
    for i, ts in enumerate(index):
        d = stamp_date(ts)
        if d is None or d in out:
            continue
        out[d] = int(i)
    return out


def _as_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def support_slope_from_row(
    row: Mapping[str, Any],
    *,
    l1_i: int,
    l2_i: Optional[int],
) -> Optional[float]:
    """Price per bar. Prefer L1/L2 prices on this panel; else slope_pct * L1."""
    l1_px = _as_float(row.get("l1_price"))
    l2_px = _as_float(row.get("l2_price"))
    if (
        l1_px is not None
        and l2_px is not None
        and l2_i is not None
        and int(l2_i) != int(l1_i)
    ):
        return (l2_px - l1_px) / float(int(l2_i) - int(l1_i))
    slope_pct = _as_float(row.get("slope_pct_per_bar"))
    if l1_px is None or slope_pct is None or l1_px <= 0:
        return None
    return (slope_pct / 100.0) * l1_px


def channel_outside_areas(
    low: np.ndarray,
    high: np.ndarray,
    close: np.ndarray,
    *,
    support_y0: float,
    support_x0: int,
    support_slope: float,
    width: float,
    start_i: int,
    end_i: int,
) -> Dict[str, float]:
    """Riemann sums on bar-index x price. Invalid bars drop from both sums."""
    nan = {
        "n_bars": float("nan"),
        "n_valid": float("nan"),
        "channel_area": float("nan"),
        "outside_below_low_area": float("nan"),
        "outside_below_close_area": float("nan"),
        "outside_above_high_area": float("nan"),
        "outside_below_low_ratio": float("nan"),
        "outside_below_close_ratio": float("nan"),
        "outside_above_high_ratio": float("nan"),
        "max_undershoot_width": float("nan"),
        "n_bars_below_low": float("nan"),
        "n_bars_below_close": float("nan"),
    }
    if not np.isfinite(width) or width <= 0:
        return nan
    if not np.isfinite(support_y0) or not np.isfinite(support_slope):
        return nan
    n = len(low)
    if n == 0 or len(high) != n or len(close) != n:
        return nan
    lo = max(0, int(start_i))
    hi = min(n - 1, int(end_i))
    if hi < lo:
        return nan
    xs = np.arange(lo, hi + 1, dtype=float)
    support = float(support_y0) + float(support_slope) * (xs - float(support_x0))
    resist = support + float(width)
    lows = np.asarray(low[lo : hi + 1], dtype=float)
    highs = np.asarray(high[lo : hi + 1], dtype=float)
    closes = np.asarray(close[lo : hi + 1], dtype=float)
    ok = (
        np.isfinite(support)
        & np.isfinite(resist)
        & np.isfinite(lows)
        & np.isfinite(highs)
        & np.isfinite(closes)
    )
    n_valid = int(ok.sum())
    if n_valid <= 0:
        return nan
    support_ok = support[ok]
    resist_ok = resist[ok]
    lows_ok = lows[ok]
    highs_ok = highs[ok]
    closes_ok = closes[ok]
    below_low = np.maximum(0.0, support_ok - lows_ok)
    below_close = np.maximum(0.0, support_ok - closes_ok)
    above_high = np.maximum(0.0, highs_ok - resist_ok)
    channel_area = float(width) * float(n_valid)
    max_under = float(np.max(below_low / float(width))) if n_valid else float("nan")
    return {
        "n_bars": float(hi - lo + 1),
        "n_valid": float(n_valid),
        "channel_area": channel_area,
        "outside_below_low_area": float(below_low.sum()),
        "outside_below_close_area": float(below_close.sum()),
        "outside_above_high_area": float(above_high.sum()),
        "outside_below_low_ratio": float(below_low.sum() / channel_area),
        "outside_below_close_ratio": float(below_close.sum() / channel_area),
        "outside_above_high_ratio": float(above_high.sum() / channel_area),
        "max_undershoot_width": max_under,
        "n_bars_below_low": float((below_low > 0.0).sum()),
        "n_bars_below_close": float((below_close > 0.0).sum()),
    }


def areas_for_trade(
    row: Mapping[str, Any],
    df: pd.DataFrame,
    *,
    date_index: Optional[Dict[date, int]] = None,
) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
    """Compute outside-area stats for one trade on a daily panel."""
    if df is None or df.empty:
        return None, SKIP_NO_PANEL
    if "low" not in df.columns or "high" not in df.columns or "close" not in df.columns:
        return None, SKIP_NO_PANEL
    idx_map = date_index if date_index is not None else date_to_first_index(df.index)
    l1_d = stamp_date(row.get("l1_time") if row.get("l1_time") is not None else row.get("channel_start"))
    buy_d = stamp_date(row.get("buy_date"))
    l2_d = stamp_date(row.get("l2_time"))
    if l1_d is None or l1_d not in idx_map:
        return None, SKIP_NO_L1
    if buy_d is None or buy_d not in idx_map:
        return None, SKIP_NO_BUY
    l1_i = idx_map[l1_d]
    buy_i = idx_map[buy_d]
    if buy_i < l1_i:
        return None, SKIP_BAD_WINDOW
    width = _as_float(row.get("channel_width"))
    y0 = _as_float(row.get("l1_price"))
    if width is None or width <= 0 or y0 is None:
        return None, SKIP_BAD_WIDTH
    l2_i = idx_map.get(l2_d) if l2_d is not None else None
    slope = support_slope_from_row(row, l1_i=l1_i, l2_i=l2_i)
    if slope is None:
        return None, SKIP_BAD_RAIL
    low = df["low"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    stats = channel_outside_areas(
        low,
        high,
        close,
        support_y0=y0,
        support_x0=l1_i,
        support_slope=slope,
        width=width,
        start_i=l1_i,
        end_i=buy_i,
    )
    if not np.isfinite(stats["outside_below_low_ratio"]):
        return None, SKIP_BAD_WINDOW
    stats["l1_i"] = float(l1_i)
    stats["buy_i"] = float(buy_i)
    stats["support_slope"] = float(slope)
    return stats, None
