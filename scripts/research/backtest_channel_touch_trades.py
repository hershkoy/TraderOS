#!/usr/bin/env python3
"""
Backtest: long ascending-channel bottom touches (>= Nth touch).

Rules:
  - Detect classical ascending channels (same as find_ascending_channels)
  - Enter long on each bottom touch number >= entry_touch (default 3)
  - Entry at close of pivot-confirmation bar (touch_index + pivot_len)
  - Exit: 3% hard stop OR 10% trailing stop from peak (whichever is higher)
  - Optional --resist-arm-trail: hard stop only until upper rail is tagged, then trail from peak
  - One open position per symbol (skip new entries while in a trade)

Edge filters (optional):
  - 20d average dollar volume (ADV) floor
  - ATR%% of price floor (volatility)
  - Same-day relative-strength ranking vs SPY (keep top-N entries per buy_date)
  - Round-trip friction scenarios (slippage + commission) on expectancy

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --symbols GLD
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --all-symbols --workers 4 --load-workers 8 --edge-improve
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_trades.py --preset 15m --symbols GLD,QQQ,AAPL
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from find_ascending_channels import (
    _pick_symbols,
    find_channels,
    find_channels_windowed,
    find_h2_l3_setups,
    find_h2_l3_setups_windowed,
    list_symbols_fast,
)
from utils.data.ohlcv_loader import load_ohlcv_many
from utils.research.channel_touch_entry_features import (
    FEATURE_COLS,
    completed_asof,
    enrich_spy_entry_features,
    max_beyond_width,
    snapshot_stock_features,
    stock_entry_feature_series,
)
from utils.research.realistic_purchaser import (
    DEFAULT_FILL_MODE,
    DEFAULT_MAX_LOW_TO_MID_PCT,
    FILL_MODE_OPEN_CROSS,
    FILL_MODE_SIGNAL_CLOSE,
    exec_fill_15m_after_signal,
    exec_fill_daily_with_15m,
    normalize_fill_mode,
    purchase_open_cross_15m,
)
from utils.research.channel_touch_scale import PRESET_15M, apply_daily_long_history_defaults, overlay_preset
from utils.research.report_paths import dated_outdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_channel_touch_trades")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return tr.astype(float)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int = 14) -> np.ndarray:
    tr = _true_range(high, low, close)
    atr = np.full(len(close), np.nan, dtype=float)
    if len(close) < length:
        return atr
    atr[length - 1] = float(np.nanmean(tr[:length]))
    alpha = 1.0 / float(length)
    for i in range(length, len(close)):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _adv_20(close: np.ndarray, volume: np.ndarray, i: int, lookback: int = 20) -> float:
    if i < 0 or lookback <= 0:
        return float("nan")
    start = max(0, i - lookback + 1)
    c = close[start : i + 1]
    v = volume[start : i + 1]
    dv = c * v
    if len(dv) == 0 or not np.isfinite(dv).any():
        return float("nan")
    return float(np.nanmean(dv))


def _line_at(y0: float, x0: int, slope: float, x: int) -> float:
    return float(y0 + slope * (x - x0))


def iso_utc_ms(ts: object) -> Tuple[str, int]:
    """Naive stamps are UTC (IB/Alpaca OHLCV). Returns ISO-8601 Z + unix ms."""
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC")
    else:
        t = t.tz_localize("UTC")
    ms = int(round(t.timestamp() * 1000))
    return t.strftime("%Y-%m-%dT%H:%M:%SZ"), ms


def channel_rail_fields(
    ch: dict,
    dates,
    low: np.ndarray,
    *,
    include_time: bool = False,
) -> Dict[str, Any]:
    """L1/L2/H2 timestamps + absolute width for CTF JSON paste.

    ``include_time`` is unused for the ISO stamp (always emits HH:MM:SSZ);
    kept so callers can pass the backtest clock flag without a mismatch.
    """
    del include_time
    out: Dict[str, Any] = {}
    n = len(dates)
    l1 = ch.get("l1_idx", ch.get("support_x0"))
    l2 = ch.get("l2_idx")
    if l2 is None:
        idxs = ch.get("touch_indices") or []
        if len(idxs) >= 2:
            l2 = idxs[-1]
    h2 = ch.get("h2_idx")
    y1 = ch.get("support_y0")
    slope = ch.get("support_slope")
    width = ch.get("channel_width")

    def _at(idx: object) -> Optional[int]:
        if idx is None:
            return None
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return None
        return i if 0 <= i < n else None

    i1 = _at(l1)
    i2 = _at(l2)
    ih = _at(h2)
    if i1 is not None and y1 is not None and np.isfinite(float(y1)):
        iso, ms = iso_utc_ms(dates[i1])
        out["l1_time"] = iso
        out["l1_ms"] = ms
        out["l1_price"] = round(float(y1), 6)
    if i2 is not None:
        iso, ms = iso_utc_ms(dates[i2])
        out["l2_time"] = iso
        out["l2_ms"] = ms
        px = float(low[i2]) if i2 < len(low) and np.isfinite(low[i2]) else float("nan")
        if (not np.isfinite(px)) and i1 is not None and y1 is not None and slope is not None:
            px = _line_at(float(y1), int(i1), float(slope), int(i2))
        if np.isfinite(px):
            out["l2_price"] = round(float(px), 6)
    if ih is not None:
        iso, ms = iso_utc_ms(dates[ih])
        out["h2_time"] = iso
        out["h2_ms"] = ms
    elif ch.get("h2_date") or ch.get("end_date"):
        iso, ms = iso_utc_ms(ch.get("h2_date") or ch.get("end_date"))
        out["h2_time"] = iso
        out["h2_ms"] = ms
    if width is not None and np.isfinite(float(width)) and float(width) > 0:
        out["channel_width"] = round(float(width), 6)
    return out


def _support_tagged(bar_low: float, bar_high: float, support: float, error_pct: float) -> bool:
    """True if the bar's range intersects support (wick can tag mid-day)."""
    if not np.isfinite(bar_low) or not np.isfinite(bar_high) or not np.isfinite(support) or support <= 0:
        return False
    tol = float(error_pct) / 100.0
    return bar_low <= support * (1.0 + tol) and bar_high >= support * (1.0 - tol)


def _l3_rail_touch(
    bar_high: float,
    bar_low: float,
    bar_close: float,
    support: float,
    error_pct: float,
) -> bool:
    """Support tag from above: traded at/above the rail, wick tags, close not broken.

    A gap-through entirely under the line (WTFC 2019-07-16) is not a touch.
    """
    if not _support_tagged(bar_low, bar_high, support, error_pct):
        return False
    if not np.isfinite(bar_close):
        return False
    tol = float(error_pct) / 100.0
    if bar_high < support:
        return False
    if bar_close < support * (1.0 - tol):
        return False
    return True


def _limit_fill_at_support(support: float, bar_low: float, bar_high: float, slip_pct: float) -> Optional[float]:
    """Limit buy at support plus slippage, clipped to the bar's range.

    Never fills below the rail: if the bar never trades at/above support, skip.
    """
    if not np.isfinite(support) or support <= 0:
        return None
    if not np.isfinite(bar_low) or not np.isfinite(bar_high) or bar_high < bar_low:
        return None
    if float(bar_high) + 1e-12 < float(support):
        return None
    raw = support * (1.0 + max(0.0, float(slip_pct)))
    fill = min(float(bar_high), max(float(bar_low), raw))
    if not np.isfinite(fill) or fill <= 0:
        return None
    if fill + 1e-12 < float(support):
        return None
    return float(fill)


def _line_index_at(i: int, daily_i: Optional[np.ndarray]) -> int:
    if daily_i is None:
        return int(i)
    if i < 0 or i >= len(daily_i):
        return int(i)
    di = int(daily_i[i])
    return di if di >= 0 else int(i)


def _reentry_or_breakout_fill(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    start_i: int,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    h2: int,
    n: int,
    error_pct: float,
    slip: float,
    wait: int,
    daily_i: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, float, bool]]:
    """First fill at/above support after ``start_i``: reclaim through support or resist-break.

    Returns ``(i, fill, is_breakout)``. Never fills below the rail.
    """
    tol = float(error_pct) / 100.0
    wait_n = max(1, int(wait))
    h2_i = int(h2)
    begin = max(int(start_i), 0)
    for i in range(begin, int(n)):
        line_i = _line_index_at(i, daily_i)
        if daily_i is None:
            if i > h2_i + wait_n:
                break
        else:
            if line_i > h2_i + wait_n:
                break
        if line_i < 0:
            continue
        sup = _line_at(support_y0, support_x0, support_slope, line_i)
        if not np.isfinite(sup) or sup <= 0:
            continue
        resist = float(sup) + float(width or 0.0)
        if np.isfinite(resist) and resist > 0 and float(close[i]) > resist * (1.0 + tol):
            fill = _limit_fill_at_support(resist, float(low[i]), float(high[i]), slip)
            if fill is not None:
                return (int(i), float(fill), True)
        prev_below = False
        if i > 0:
            prev_line = _line_index_at(i - 1, daily_i)
            sup_prev = _line_at(support_y0, support_x0, support_slope, prev_line)
            if np.isfinite(sup_prev) and sup_prev > 0:
                prev_below = float(close[i - 1]) < float(sup_prev) * (1.0 - tol)
        this_below = float(close[i]) < float(sup) * (1.0 - tol)
        if prev_below and (not this_below) and float(high[i]) + 1e-12 >= float(sup) * (1.0 - tol):
            fill = _limit_fill_at_support(sup, float(low[i]), float(high[i]), slip)
            if fill is not None:
                return (int(i), float(fill), False)
    return None


def _ensure_fill_not_below_support(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    fill_i: int,
    fill_px: Optional[float],
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    h2: int,
    n: int,
    error_pct: float,
    slip: float,
    wait: int,
    daily_i: Optional[np.ndarray] = None,
) -> Optional[Tuple[int, float, bool]]:
    """Keep a fill at/above support; if it is below, defer to re-entry or resist-break."""
    if fill_px is None or not np.isfinite(float(fill_px)) or float(fill_px) <= 0:
        return None
    line_i = _line_index_at(int(fill_i), daily_i)
    sup = _line_at(support_y0, support_x0, support_slope, line_i)
    if not np.isfinite(sup) or sup <= 0:
        return (int(fill_i), float(fill_px), False)
    if float(fill_px) + 1e-12 >= float(sup):
        return (int(fill_i), float(fill_px), False)
    return _reentry_or_breakout_fill(
        high,
        low,
        close,
        start_i=int(fill_i),
        support_x0=support_x0,
        support_y0=support_y0,
        support_slope=support_slope,
        width=width,
        h2=h2,
        n=n,
        error_pct=error_pct,
        slip=slip,
        wait=wait,
        daily_i=daily_i,
    )


def _close_broke_support(
    close: np.ndarray,
    i: int,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    error_pct: float,
) -> bool:
    """True if bar ``i`` closed through support (beyond error_pct)."""
    if i < 0 or i >= len(close):
        return False
    sup = _line_at(support_y0, support_x0, support_slope, i)
    if not np.isfinite(sup) or sup <= 0:
        return False
    return float(close[i]) < float(sup) * (1.0 - float(error_pct) / 100.0)


def _shakeout_rebuy_fill(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    l3_i: int,
    n: int,
    error_pct: float,
    slip: float,
    shakeout_bars: int,
) -> Optional[Tuple[int, float]]:
    """After an L3 fill, first from-above reclaim following a close below support.

    Window is ``shakeout_bars`` trading bars after the L3 bar (not including it).
    Close above resistance cancels (breakout, not a shakeout). Does not emit a
    fill on the break bar itself. Caller skips the fill if the first trade is
    still open (rebuy, not hold-through).
    """
    max_n = int(shakeout_bars)
    if max_n <= 0 or l3_i < 0 or l3_i >= n - 1:
        return None
    broke = False
    for i in range(int(l3_i) + 1, min(n, int(l3_i) + 1 + max_n)):
        sup = _line_at(support_y0, support_x0, support_slope, i)
        resist = float(sup) + float(width or 0.0)
        if np.isfinite(resist) and resist > 0 and float(close[i]) > resist * (
            1.0 + error_pct / 100.0
        ):
            return None
        if _close_broke_support(
            close,
            i - 1,
            support_x0=support_x0,
            support_y0=support_y0,
            support_slope=support_slope,
            error_pct=error_pct,
        ):
            broke = True
        if _close_broke_support(
            close,
            i,
            support_x0=support_x0,
            support_y0=support_y0,
            support_slope=support_slope,
            error_pct=error_pct,
        ):
            broke = True
            continue
        if not broke:
            continue
        if not _l3_rail_touch(float(high[i]), float(low[i]), float(close[i]), sup, error_pct):
            continue
        fill = _limit_fill_at_support(sup, float(low[i]), float(high[i]), slip)
        if fill is None:
            return None
        return (int(i), float(fill))
    return None


def _shakeout_breakout_fill(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    first_i: int,
    h2: int,
    n: int,
    error_pct: float,
    slip: float,
    wait: int,
    min_inside_bars: int = 1,
) -> Optional[Tuple[int, float, int]]:
    """After a first H2 resist-break, wait for a shakeout then the next breakout.

    Shakeout = closes back inside the channel (not a resist-break close, not
    through support). Then fill the next close above resistance at the rail
    (+slip). Support close-through cancels. Consecutive bars still above
    resistance are the same breakout and do not refill. Window is remaining
    ``wait`` bars after H2. Returns ``(i, fill, inside_bars)``.
    """
    wait_n = max(1, int(wait))
    need = max(1, int(min_inside_bars))
    if first_i < 0 or first_i >= n - 1:
        return None
    tol = float(error_pct) / 100.0
    inside_bars = 0
    h2_i = int(h2)
    end = min(int(n), h2_i + 1 + wait_n)
    for i in range(int(first_i) + 1, end):
        if _close_broke_support(
            close,
            i - 1,
            support_x0=support_x0,
            support_y0=support_y0,
            support_slope=support_slope,
            error_pct=error_pct,
        ):
            return None
        if _close_broke_support(
            close,
            i,
            support_x0=support_x0,
            support_y0=support_y0,
            support_slope=support_slope,
            error_pct=error_pct,
        ):
            return None
        sup = _line_at(support_y0, support_x0, support_slope, i)
        resist = float(sup) + float(width or 0.0)
        if not np.isfinite(resist) or resist <= 0:
            continue
        brk = float(close[i]) > resist * (1.0 + tol)
        if not brk:
            inside_bars += 1
            continue
        if inside_bars < need:
            continue
        fill = _limit_fill_at_support(resist, float(low[i]), float(high[i]), slip)
        if fill is None:
            return None
        return (int(i), float(fill), int(inside_bars))
    return None


def _h2_rail_tag_fills(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    h2: int,
    n: int,
    error_pct: float,
    slip: float,
    wait: int,
    min_wait: int,
    entry_touch: int = 3,
    leave_width_frac: float = 0.20,
    shakeout_rebuy_bars: int = 0,
    h2_resist_break: bool = False,
    shakeout_breakout: bool = False,
    shakeout_breakout_min_inside: int = 1,
) -> List[Tuple[int, float, int, bool, bool]]:
    """From-above support tags after H2. First tag is L3 (touch_num=3).

    ``entry_touch`` 3 fills the first tag; 4 skips L3 and fills L4 after price
    leaves the rail and waits ``min_wait`` bars (abort if tagged early; do not
    retarget a later dip). Before L3, a high above H2 cancels (invalidated
    pullback). After L3, a new high is H3 and is allowed. Close through
    support or close above resistance still cancels. No future bars are used.

    ``shakeout_rebuy_bars``: after an L3 fill, if price closes through support
    and reclaims from above within N bars, emit a second fill (touch_num=3).
    Off by default. Not mixed with entry_touch=4. Rebuy vs hold-through is
    enforced by one-position-per-symbol in ``trades_for_symbol``.

    ``h2_resist_break``: instead of cancelling on a close above resistance after
    H2, emit a fill at the rail (breakout continuation). Support break still
    cancels. Off by default.

    ``shakeout_breakout``: after a first resist-break fill, if price closes back
    inside the channel then closes above resistance again (support still
    holds), emit one extra resist-break fill. Not the L3 support-reclaim
    rebuy. Occupancy skips the extra fill while the first trade is open.
    """
    want = max(3, int(entry_touch))
    wait_n = max(1, int(wait))
    min_w = max(1, int(min_wait))
    leave_frac = max(0.0, float(leave_width_frac))
    h2_px = float(high[h2]) if 0 <= h2 < n else float("nan")
    out: List[Tuple[int, float, int, bool, bool]] = []
    touch_count = 2
    arm_i = int(h2)
    need_leave = False
    shake_n = max(0, int(shakeout_rebuy_bars))
    take_break = bool(h2_resist_break) and want <= 3
    take_sbo = bool(shakeout_breakout) and take_break
    sbo_min = max(1, int(shakeout_breakout_min_inside))
    for i in range(int(h2) + 1, min(n, int(h2) + 1 + wait_n)):
        sup = _line_at(support_y0, support_x0, support_slope, i)
        resist = float(sup) + float(width or 0.0)
        if i > 0:
            sup_prev = _line_at(support_y0, support_x0, support_slope, i - 1)
            if float(close[i - 1]) < sup_prev * (1.0 - error_pct / 100.0):
                break
        if np.isfinite(resist) and resist > 0 and float(close[i]) > resist * (1.0 + error_pct / 100.0):
            if take_break and i >= int(h2) + min_w:
                fill = _limit_fill_at_support(resist, float(low[i]), float(high[i]), slip)
                if fill is not None:
                    out.append((i, float(fill), 3, False, True))
                    if take_sbo:
                        extra = _shakeout_breakout_fill(
                            high,
                            low,
                            close,
                            support_x0=support_x0,
                            support_y0=support_y0,
                            support_slope=support_slope,
                            width=width,
                            first_i=int(i),
                            h2=int(h2),
                            n=n,
                            error_pct=error_pct,
                            slip=slip,
                            wait=wait_n,
                            min_inside_bars=sbo_min,
                        )
                        if extra is not None:
                            out.append(
                                (
                                    int(extra[0]),
                                    float(extra[1]),
                                    3,
                                    False,
                                    True,
                                    True,
                                    int(extra[2]),
                                )
                            )
                break
            if take_break:
                continue
            break
        if touch_count < 3 and np.isfinite(h2_px) and float(high[i]) > h2_px:
            if not take_break:
                break
        touched = _l3_rail_touch(float(high[i]), float(low[i]), float(close[i]), sup, error_pct)
        broke = float(close[i]) < sup * (1.0 - error_pct / 100.0)
        if need_leave:
            leave_lvl = float(sup) + leave_frac * max(float(width or 0.0), 0.0)
            if np.isfinite(float(close[i])) and float(close[i]) > leave_lvl:
                need_leave = False
            elif broke:
                break
            continue
        if i < arm_i + min_w:
            if touched or broke:
                break
            continue
        if touched:
            fill = _limit_fill_at_support(sup, float(low[i]), float(high[i]), slip)
            touch_count += 1
            if fill is not None and touch_count >= want:
                out.append((i, float(fill), int(touch_count), False, False))
                if shake_n > 0 and int(touch_count) == 3:
                    extra = _shakeout_rebuy_fill(
                        high,
                        low,
                        close,
                        support_x0=support_x0,
                        support_y0=support_y0,
                        support_slope=support_slope,
                        width=width,
                        l3_i=int(i),
                        n=n,
                        error_pct=error_pct,
                        slip=slip,
                        shakeout_bars=shake_n,
                    )
                    if extra is not None:
                        out.append((int(extra[0]), float(extra[1]), 3, True, False))
                break
            if fill is None:
                break
            need_leave = True
            arm_i = int(i)
            continue
        if broke:
            break
    return out


def _session_date(ts: object) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("US/Eastern")
    return pd.Timestamp(t.date())


def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index()


def _map_15m_to_daily_i(m15_index: pd.DatetimeIndex, daily_dates: pd.DatetimeIndex) -> np.ndarray:
    lookup = {_session_date(d): i for i, d in enumerate(daily_dates)}
    out = np.full(len(m15_index), -1, dtype=int)
    for j, ts in enumerate(m15_index):
        di = lookup.get(_session_date(ts))
        if di is not None:
            out[j] = int(di)
    return out


def _h2_rail_tag_fills_on_15m(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    daily_i: np.ndarray,
    *,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    width: float,
    h2: int,
    error_pct: float,
    slip: float,
    wait_daily: int,
    min_wait_daily: int,
    entry_touch: int = 3,
    leave_width_frac: float = 0.20,
    h2_high: float = float("nan"),
) -> List[Tuple[int, float, int]]:
    """From-above support tags on 15m bars using daily rail indices.

    ``wait_daily`` / ``min_wait_daily`` are daily sessions after H2. Support is
    evaluated at each 15m bar's daily session index (constant within the day).
    """
    want = max(3, int(entry_touch))
    wait_n = max(1, int(wait_daily))
    min_w = max(1, int(min_wait_daily))
    leave_frac = max(0.0, float(leave_width_frac))
    h2_px = float(h2_high)
    n = len(high)
    out: List[Tuple[int, float, int]] = []
    touch_count = 2
    arm_di = int(h2)
    need_leave = False
    for j in range(n):
        di = int(daily_i[j])
        if di < 0 or di <= int(h2):
            continue
        if di > int(h2) + wait_n:
            break
        sup = _line_at(support_y0, support_x0, support_slope, di)
        resist = float(sup) + float(width or 0.0)
        if j > 0:
            di_prev = int(daily_i[j - 1])
            if di_prev >= 0:
                sup_prev = _line_at(support_y0, support_x0, support_slope, di_prev)
                if float(close[j - 1]) < sup_prev * (1.0 - error_pct / 100.0):
                    break
        if np.isfinite(resist) and resist > 0 and float(close[j]) > resist * (1.0 + error_pct / 100.0):
            break
        if touch_count < 3 and np.isfinite(h2_px) and float(high[j]) > h2_px:
            break
        touched = _l3_rail_touch(float(high[j]), float(low[j]), float(close[j]), sup, error_pct)
        broke = float(close[j]) < sup * (1.0 - error_pct / 100.0)
        if need_leave:
            leave_lvl = float(sup) + leave_frac * max(float(width or 0.0), 0.0)
            if np.isfinite(float(close[j])) and float(close[j]) > leave_lvl:
                need_leave = False
            elif broke:
                break
            continue
        if di < arm_di + min_w:
            if touched or broke:
                break
            continue
        if touched:
            fill = _limit_fill_at_support(sup, float(low[j]), float(high[j]), slip)
            touch_count += 1
            if fill is not None and touch_count >= want:
                out.append((j, float(fill), int(touch_count)))
                break
            if fill is None:
                break
            need_leave = True
            arm_di = int(di)
            continue
        if broke:
            break
    return out


def _hard_stop_price(
    entry_px: float,
    *,
    stop_pct: float,
    atr_at_entry: Optional[float] = None,
    atr_stop_mult: Optional[float] = None,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
) -> float:
    """Fixed %% stop, or ATR-scaled distance clamped to [floor, ceil] of price."""
    if (
        atr_stop_mult is not None
        and atr_at_entry is not None
        and np.isfinite(atr_at_entry)
        and atr_at_entry > 0
        and entry_px > 0
    ):
        dist = float(atr_stop_mult) * float(atr_at_entry)
        dist = max(entry_px * stop_pct_floor, min(entry_px * stop_pct_ceil, dist))
        return entry_px - dist
    return entry_px * (1.0 - float(stop_pct))


PEAK_TRAIL_MODES = ("off", "fixed", "time_decay", "gain_tighten")


def _normalize_peak_trail_mode(
    mode: Optional[str] = None,
    *,
    resist_arm_trail: bool = False,
) -> str:
    m = (mode or "off").strip().lower().replace("-", "_")
    if m in ("", "none", "off"):
        m = "off"
    if bool(resist_arm_trail) and m == "off":
        return "fixed"
    if m not in PEAK_TRAIL_MODES:
        raise ValueError(f"Unknown peak_trail_mode={mode!r}; expected one of {PEAK_TRAIL_MODES}")
    return m


def _peak_trail_width(
    *,
    mode: str,
    trail_pct: float,
    bars_held: int,
    peak: float,
    entry_px: float,
    trail_floor: float = 0.01,
    trail_decay_per_bar: float = 0.0002,
    trail_tighten_per_pct: float = 0.0033,
) -> float:
    """Trail width as a fraction of price (e.g. 0.04 = 4%)."""
    w0 = float(trail_pct)
    if mode == "fixed" or mode == "off":
        return w0
    floor = max(0.0, float(trail_floor))
    if mode == "time_decay":
        return max(floor, w0 - max(0, int(bars_held)) * float(trail_decay_per_bar))
    if mode == "gain_tighten":
        gain_pp = 0.0
        if entry_px > 0 and np.isfinite(peak) and peak > 0:
            gain_pp = max(0.0, (float(peak) / float(entry_px) - 1.0) * 100.0)
        steps = int(np.floor(gain_pp))
        return max(floor, w0 - steps * float(trail_tighten_per_pct))
    return w0


def _simulate_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    entry_i: int,
    *,
    stop_pct: float,
    trail_pct: float,
    trail_pct_wide: Optional[float] = None,
    squeeze_mom: Optional[np.ndarray] = None,
    squeeze_pctile: float = 75.0,
    squeeze_lookback: int = 100,
    atr_at_entry: Optional[float] = None,
    atr_stop_mult: Optional[float] = None,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    support_x0: Optional[int] = None,
    support_y0: Optional[float] = None,
    support_slope: Optional[float] = None,
    channel_width: Optional[float] = None,
    resist_exit: bool = False,
    resist_arm_trail: bool = False,
    peak_trail_mode: str = "off",
    trail_floor: float = 0.01,
    trail_decay_per_bar: float = 0.0002,
    trail_tighten_per_pct: float = 0.0033,
    trail_pct_tight: Optional[float] = None,
    squeeze_fade_tighten: bool = False,
    max_hold_days: Optional[int] = None,
    include_time: bool = False,
    entry_px: Optional[float] = None,
    skip_entry_bar_stop: bool = False,
) -> Optional[dict]:
    """Long from entry_i; default fill is close, or ``entry_px`` for a limit at support.

    If trail_pct_wide and squeeze_mom are provided, widen the trail when TTM
    Squeeze momentum is positive, non-decreasing, and strong vs its recent
    distribution (LazyBear lime-green / strong-up regime).

    ``peak_trail_mode`` (or legacy ``resist_arm_trail`` -> fixed): from-entry peak
    trail with no resist gate / no squeeze. Width modes: fixed, time_decay,
    gain_tighten. Exit when low hits stop; fill at stop. Reason ``peak_trail``.
    """
    n = len(close)
    if entry_i < 0 or entry_i >= n - 1:
        return None
    if entry_px is None:
        fill = float(close[entry_i])
    else:
        fill = float(entry_px)
    if not np.isfinite(fill) or fill <= 0:
        return None
    entry_px = fill

    pt_mode = _normalize_peak_trail_mode(peak_trail_mode, resist_arm_trail=resist_arm_trail)
    use_peak_trail = pt_mode != "off"

    hard_stop = _hard_stop_price(
        entry_px,
        stop_pct=stop_pct,
        atr_at_entry=None if use_peak_trail else atr_at_entry,
        atr_stop_mult=None if use_peak_trail else atr_stop_mult,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )
    peak = entry_px
    stop_level = hard_stop
    exit_i = n - 1
    exit_px = float(close[exit_i])
    exit_reason = "eod"
    used_wide = False
    wide = None if use_peak_trail else (
        float(trail_pct_wide) if trail_pct_wide is not None else None
    )
    tight = None if use_peak_trail else (
        float(trail_pct_tight) if trail_pct_tight is not None else None
    )
    have_line = (
        support_x0 is not None
        and support_y0 is not None
        and support_slope is not None
        and channel_width is not None
        and np.isfinite(support_y0)
        and np.isfinite(support_slope)
        and np.isfinite(channel_width)
    )

    lo0 = float(low[entry_i]) if entry_i < n else float("nan")
    if (
        not skip_entry_bar_stop
        and np.isfinite(lo0)
        and lo0 <= hard_stop < entry_px
    ):
        hold = 0
        gain_pct = (hard_stop / entry_px - 1.0) * 100.0
        ts_buy = dates[entry_i]
        ts_sell = dates[entry_i]
        out = {
            "buy_date": ts_buy.strftime("%Y-%m-%d"),
            "sell_date": ts_sell.strftime("%Y-%m-%d"),
            "buy_price": round(entry_px, 4),
            "sell_price": round(float(hard_stop), 4),
            "gain_pct": round(gain_pct, 2),
            "hold_days": hold,
            "exit_reason": "hard_stop",
            "peak_price": round(float(entry_px), 4),
            "trail_wide_used": False,
            "hard_stop_price": round(float(hard_stop), 4),
            "entry_i": int(entry_i),
            "exit_i": int(entry_i),
        }
        if include_time:
            out["buy_time"] = ts_buy.strftime("%Y-%m-%d %H:%M")
            out["sell_time"] = ts_sell.strftime("%Y-%m-%d %H:%M")
            out["hold_bars"] = hold
        return out

    for i in range(entry_i + 1, n):
        hi = float(high[i])
        lo = float(low[i])
        cl = float(close[i])
        if np.isfinite(hi):
            peak = max(peak, hi)

        if max_hold_days is not None and (i - entry_i) >= int(max_hold_days):
            exit_i = i
            exit_px = float(cl) if np.isfinite(cl) else float(close[i])
            exit_reason = "time_stop"
            break

        resist = float("nan")
        if have_line and np.isfinite(hi):
            resist = _line_at(float(support_y0), int(support_x0), float(support_slope), i) + float(
                channel_width
            )

        if resist_exit and not use_peak_trail and np.isfinite(resist) and hi >= resist:
            exit_i = i
            exit_px = float(resist)
            exit_reason = "resist_exit"
            break

        if use_peak_trail:
            bars_held = int(i - entry_i)
            trail_w = _peak_trail_width(
                mode=pt_mode,
                trail_pct=float(trail_pct),
                bars_held=bars_held,
                peak=float(peak),
                entry_px=float(entry_px),
                trail_floor=float(trail_floor),
                trail_decay_per_bar=float(trail_decay_per_bar),
                trail_tighten_per_pct=float(trail_tighten_per_pct),
            )
            trail_stop = peak * (1.0 - trail_w)
            stop_level = max(float(stop_level), float(hard_stop), float(trail_stop))
            if np.isfinite(lo) and lo <= stop_level:
                exit_i = i
                exit_px = float(stop_level)
                if abs(stop_level - hard_stop) < 1e-9 and trail_stop <= hard_stop + 1e-9:
                    exit_reason = "hard_stop"
                else:
                    exit_reason = "peak_trail"
                break
            exit_i = i
            exit_px = float(cl) if np.isfinite(cl) else float(close[i])
            exit_reason = "eod"
            continue

        trail_use = float(trail_pct)
        wide_now = False
        fade_now = False
        if squeeze_mom is not None and i < len(squeeze_mom):
            mom = float(squeeze_mom[i])
            mom_prev = float(squeeze_mom[i - 1]) if i > 0 else float("nan")
            if wide is not None and np.isfinite(mom) and mom > 0 and (
                not np.isfinite(mom_prev) or mom >= mom_prev
            ):
                start = max(0, i - int(squeeze_lookback) + 1)
                window = squeeze_mom[start : i + 1]
                window = window[np.isfinite(window)]
                if len(window) >= 20:
                    thr = float(np.nanpercentile(window, float(squeeze_pctile)))
                    if mom >= thr:
                        trail_use = wide
                        wide_now = True
            if squeeze_fade_tighten and tight is not None and not wide_now and np.isfinite(mom):
                start = max(0, i - int(squeeze_lookback) + 1)
                window = squeeze_mom[start : i + 1]
                window = window[np.isfinite(window)]
                below_med = False
                if len(window) >= 20:
                    med = float(np.nanpercentile(window, 50.0))
                    below_med = mom < med
                fading = (np.isfinite(mom_prev) and mom < mom_prev) or below_med
                if fading:
                    trail_use = tight
                    fade_now = True

        trail_stop = peak * (1.0 - trail_use)
        stop_level = max(hard_stop, trail_stop)
        trail_active = stop_level > hard_stop + 1e-9
        if np.isfinite(lo) and lo <= stop_level:
            exit_i = i
            exit_px = float(stop_level)
            if not trail_active:
                exit_reason = "hard_stop"
            elif wide_now:
                exit_reason = "trail_stop_wide"
                used_wide = True
            elif fade_now:
                exit_reason = "trail_stop_tight"
            else:
                exit_reason = "trail_stop"
            break
        exit_i = i
        exit_px = float(cl) if np.isfinite(cl) else float(close[i])
        exit_reason = "eod"

    hold = int(exit_i - entry_i)
    gain_pct = (exit_px / entry_px - 1.0) * 100.0
    ts_buy = dates[entry_i]
    ts_sell = dates[exit_i]
    out = {
        "buy_date": ts_buy.strftime("%Y-%m-%d"),
        "sell_date": ts_sell.strftime("%Y-%m-%d"),
        "buy_price": round(entry_px, 4),
        "sell_price": round(exit_px, 4),
        "gain_pct": round(gain_pct, 2),
        "hold_days": hold,
        "exit_reason": exit_reason,
        "peak_price": round(float(peak), 4),
        "trail_wide_used": bool(used_wide),
        "hard_stop_price": round(float(hard_stop), 4),
        "entry_i": int(entry_i),
        "exit_i": int(exit_i),
    }
    if include_time:
        out["buy_time"] = ts_buy.strftime("%Y-%m-%d %H:%M")
        out["sell_time"] = ts_sell.strftime("%Y-%m-%d %H:%M")
        out["hold_bars"] = hold
    return out


def _resolve_entry_i(
    *,
    t_idx: int,
    pivot_len: int,
    entry_mode: str,
    close: np.ndarray,
    n: int,
    support_x0: int,
    support_y0: float,
    support_slope: float,
    max_wait: Optional[int] = None,
) -> Optional[int]:
    mode = (entry_mode or "pivot").lower().strip()
    if mode == "pivot":
        entry_i = int(t_idx) + int(pivot_len)
        if entry_i < 0 or entry_i >= n:
            return None
        return entry_i
    if mode == "reclaim":
        wait = int(max_wait) if max_wait is not None else max(5, int(pivot_len) * 2)
        for i in range(int(t_idx) + 1, min(n, int(t_idx) + 1 + wait)):
            sup = _line_at(support_y0, support_x0, support_slope, i)
            px = float(close[i])
            if np.isfinite(px) and np.isfinite(sup) and px > sup:
                return i
        return None
    raise ValueError(f"Unknown entry_mode={entry_mode!r}")


def trades_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    entry_touch: int = 3,
    stop_pct: float = 0.03,
    trail_pct: float = 0.10,
    trail_pct_wide: Optional[float] = None,
    squeeze_adaptive: bool = False,
    squeeze_pctile: float = 75.0,
    squeeze_lookback: int = 100,
    pivot_len: int = 15,
    entry_mode: str = "pivot",
    atr_stop_mult: Optional[float] = None,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    resist_exit: bool = False,
    resist_arm_trail: bool = False,
    peak_trail_mode: str = "off",
    trail_floor: float = 0.01,
    trail_decay_per_bar: float = 0.0002,
    trail_tighten_per_pct: float = 0.0033,
    trail_pct_tight: Optional[float] = None,
    squeeze_fade_tighten: bool = False,
    max_hold_days: Optional[int] = None,
    adv_lookback: int = 20,
    atr_len: int = 14,
    window_bars: Optional[int] = None,
    window_step_bars: Optional[int] = None,
    include_time: bool = False,
    entry_features: bool = True,
    entry_slip_pct: float = 0.001,
    max_l3_wait_bars: int = 252,
    min_l3_wait_bars: int = 1,
    shakeout_rebuy_bars: int = 0,
    h2_resist_break: bool = False,
    h2_resist_break_only: bool = False,
    shakeout_breakout: bool = False,
    shakeout_breakout_min_inside: int = 1,
    shakeout_breakout_hard_stop: bool = False,
    df_15m: Optional[pd.DataFrame] = None,
    intraday_fill: str = "",
    feature_asof_prior_bar: bool = False,
    realistic_fill: bool = False,
    realistic_fill_mode: str = DEFAULT_FILL_MODE,
    max_low_to_mid_pct: Optional[float] = DEFAULT_MAX_LOW_TO_MID_PCT,
    max_chase_pct: Optional[float] = None,
    touch_error_pct: Optional[float] = None,
    **channel_kwargs,
) -> List[dict]:
    if df is None or df.empty:
        return []
    out = _normalize_ohlcv_frame(df)

    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    volume = (
        out["volume"].to_numpy(dtype=float)
        if "volume" in out.columns
        else np.full(len(out), np.nan, dtype=float)
    )
    atr = _atr(high, low, close, length=atr_len)
    dates = out.index
    n = len(out)

    mode = (entry_mode or "pivot").lower().strip()
    hybrid = mode == "l3_touch" and (intraday_fill or "").strip().lower() == "15m"
    if hybrid:
        if df_15m is None or df_15m.empty:
            return []
        m15 = _normalize_ohlcv_frame(df_15m)
        include_time = True
        feature_asof_prior_bar = True
    else:
        m15 = None

    squeeze_mom = None
    wide = None
    need_squeeze = squeeze_adaptive or squeeze_fade_tighten or bool(entry_features)
    feat_frame = m15 if hybrid else out
    if need_squeeze:
        from indicators.ttm_squeeze import calculate_squeeze_momentum

        mom = calculate_squeeze_momentum(feat_frame, lengthKC=20, use_logging=False)
        squeeze_mom = mom.to_numpy(dtype=float)
        if squeeze_adaptive and trail_pct_wide is not None:
            wide = float(trail_pct_wide)

    feat_series = (
        stock_entry_feature_series(feat_frame, squeeze_mom=squeeze_mom) if entry_features else None
    )
    if "min_rally_pct" in channel_kwargs:
        channel_kwargs.setdefault(
            "min_intervening_rally_pct", channel_kwargs.pop("min_rally_pct")
        )
    if "min_pullback_pct" in channel_kwargs:
        channel_kwargs.setdefault(
            "min_intervening_pullback_pct", channel_kwargs.pop("min_pullback_pct")
        )
    error_pct = float(channel_kwargs.get("error_pct", 1.2))
    # Tag/break tolerance for fills. Separate from detector pivot fitting (error_pct).
    tag_error_pct = (
        float(touch_error_pct) if touch_error_pct is not None else float(error_pct)
    )
    use_realistic = bool(realistic_fill)
    fill_mode = normalize_fill_mode(realistic_fill_mode)
    is_15m_bars = bool(hybrid or include_time)

    pending: List[tuple] = []
    if mode == "l3_touch":
        setups = (
            find_h2_l3_setups_windowed(
                out,
                window_bars=int(window_bars),
                step_bars=int(window_step_bars or window_bars),
                pivot_len=pivot_len,
                **channel_kwargs,
            )
            if window_bars and int(window_bars) > 0
            else find_h2_l3_setups(out, pivot_len=pivot_len, **channel_kwargs)
        )
        wait = max(1, int(max_l3_wait_bars))
        min_wait = max(1, int(min_l3_wait_bars))
        slip = float(entry_slip_pct)
        want_touch = max(3, int(entry_touch))
        if hybrid:
            m15_high = m15["high"].to_numpy(dtype=float)
            m15_low = m15["low"].to_numpy(dtype=float)
            m15_close = m15["close"].to_numpy(dtype=float)
            m15_vol = (
                m15["volume"].to_numpy(dtype=float)
                if "volume" in m15.columns
                else np.full(len(m15), np.nan, dtype=float)
            )
            m15_dates = m15.index
            daily_i_map = _map_15m_to_daily_i(m15_dates, dates)
            for ch in setups:
                sx0 = int(ch["support_x0"])
                sy0 = float(ch["support_y0"])
                sslope = float(ch["support_slope"])
                h2 = int(ch.get("h2_idx", -1))
                if h2 < 0:
                    continue
                tags = _h2_rail_tag_fills_on_15m(
                    m15_high,
                    m15_low,
                    m15_close,
                    daily_i_map,
                    support_x0=sx0,
                    support_y0=sy0,
                    support_slope=sslope,
                    width=float(ch.get("channel_width") or 0.0),
                    h2=h2,
                    error_pct=tag_error_pct,
                    slip=slip,
                    wait_daily=wait,
                    min_wait_daily=min_wait,
                    entry_touch=want_touch,
                    h2_high=float(high[h2]) if 0 <= h2 < n else float("nan"),
                )
                for j, fill, tnum in tags:
                    signal_j = int(j)
                    fill_px = float(fill)
                    if use_realistic:
                        adj = exec_fill_15m_after_signal(
                            m15,
                            signal_j,
                            max_low_to_mid_pct=max_low_to_mid_pct,
                            max_chase_pct=max_chase_pct,
                            fill_mode=fill_mode,
                        )
                        if adj is None:
                            continue
                        j, fill_px = adj
                    pending.append(
                        (
                            ch,
                            int(j),
                            float(fill_px),
                            tnum,
                            signal_j,
                            int(daily_i_map[signal_j]),
                            False,
                            False,
                        )
                    )
        else:
            for ch in setups:
                sx0 = int(ch["support_x0"])
                sy0 = float(ch["support_y0"])
                sslope = float(ch["support_slope"])
                h2 = int(ch.get("h2_idx", -1))
                if h2 < 0:
                    continue
                tags = _h2_rail_tag_fills(
                    high,
                    low,
                    close,
                    support_x0=sx0,
                    support_y0=sy0,
                    support_slope=sslope,
                    width=float(ch.get("channel_width") or 0.0),
                    h2=h2,
                    n=n,
                    error_pct=tag_error_pct,
                    slip=slip,
                    wait=wait,
                    min_wait=min_wait,
                    entry_touch=want_touch,
                    shakeout_rebuy_bars=int(shakeout_rebuy_bars),
                    h2_resist_break=bool(h2_resist_break),
                    shakeout_breakout=bool(shakeout_breakout),
                    shakeout_breakout_min_inside=int(shakeout_breakout_min_inside),
                )
                for tag in tags:
                    i, fill, tnum = int(tag[0]), tag[1], int(tag[2])
                    is_sh = bool(tag[3]) if len(tag) > 3 else False
                    is_brk = bool(tag[4]) if len(tag) > 4 else False
                    is_sbo = bool(tag[5]) if len(tag) > 5 else False
                    inside_n = int(tag[6]) if len(tag) > 6 else 0
                    if bool(h2_resist_break_only) and not is_brk:
                        continue
                    signal_i = int(i)
                    fill_px = float(fill) if fill is not None else None
                    fill_time = None
                    if use_realistic:
                        if is_15m_bars:
                            adj = exec_fill_15m_after_signal(
                                out,
                                signal_i,
                                max_low_to_mid_pct=max_low_to_mid_pct,
                                max_chase_pct=max_chase_pct,
                                fill_mode=fill_mode,
                            )
                            if adj is None:
                                continue
                            i, fill_px = adj
                        elif fill_mode == FILL_MODE_OPEN_CROSS:
                            resist_now = _line_at(
                                float(ch["support_y0"]),
                                int(ch["support_x0"]),
                                float(ch["support_slope"]),
                                signal_i,
                            ) + float(ch.get("channel_width") or 0.0)
                            got = purchase_open_cross_15m(
                                float(resist_now),
                                df_15m,
                                session_date=pd.Timestamp(dates[signal_i]).strftime("%Y-%m-%d"),
                            )
                            if not got.filled or got.fill_px is None:
                                continue
                            fill_px = float(got.fill_px)
                            fill_time = got.exec_bar_ts
                        else:
                            adj_px = exec_fill_daily_with_15m(
                                float(fill),
                                df_15m,
                                pd.Timestamp(dates[signal_i]).strftime("%Y-%m-%d"),
                                max_low_to_mid_pct=max_low_to_mid_pct,
                                max_chase_pct=max_chase_pct,
                                fill_mode=fill_mode,
                            )
                            if adj_px is None:
                                continue
                            fill_px = adj_px
                    pending.append(
                        (ch, i, fill_px, tnum, signal_i, signal_i, is_sh, is_brk, is_sbo, inside_n, fill_time)
                    )
    else:
        channels = (
            find_channels_windowed(
                out,
                window_bars=int(window_bars),
                step_bars=int(window_step_bars or window_bars),
                pivot_len=pivot_len,
                **channel_kwargs,
            )
            if window_bars and int(window_bars) > 0
            else find_channels(out, pivot_len=pivot_len, **channel_kwargs)
        )
        for ch in channels:
            touch_idxs: List[int] = list(ch.get("touch_indices") or [])
            if len(touch_idxs) < entry_touch:
                continue
            sx0 = int(ch["support_x0"])
            sy0 = float(ch["support_y0"])
            sslope = float(ch["support_slope"])
            for touch_num, t_idx in enumerate(touch_idxs, start=1):
                if touch_num < entry_touch:
                    continue
                entry_i = _resolve_entry_i(
                    t_idx=int(t_idx),
                    pivot_len=int(ch.get("pivot_len", pivot_len)),
                    entry_mode=entry_mode,
                    close=close,
                    n=n,
                    support_x0=sx0,
                    support_y0=sy0,
                    support_slope=sslope,
                )
                if entry_i is None:
                    continue
                pending.append((ch, int(entry_i), None, int(touch_num), int(t_idx), int(entry_i), False, False))

    pending.sort(key=lambda t: (int(t[1]), int(t[0].get("h2_idx", 0))))
    trades: List[dict] = []
    busy_until = -1
    last_exit_reason: Optional[str] = None
    if hybrid:
        sim_high = m15_high
        sim_low = m15_low
        sim_close = m15_close
        sim_dates = m15_dates
        sim_n = len(m15)
        hold_max = None
        if max_hold_days is not None:
            hold_max = int(max_hold_days) * 26
    else:
        sim_high = high
        sim_low = low
        sim_close = close
        sim_dates = dates
        sim_n = n
        hold_max = max_hold_days
        m15_high = high
        m15_low = low
        m15_close = close
        m15_dates = dates
        daily_i_map = None

    for item in pending:
        ch, entry_i, fill_px, touch_num, t_idx, daily_entry_i = item[:6]
        is_shakeout = bool(item[6]) if len(item) > 6 else False
        is_resist_break = bool(item[7]) if len(item) > 7 else False
        is_sbo = bool(item[8]) if len(item) > 8 else False
        inside_n = int(item[9]) if len(item) > 9 else 0
        fill_time = item[10] if len(item) > 10 else None
        if entry_i is None or entry_i <= busy_until or entry_i >= sim_n:
            continue
        if is_sbo and bool(shakeout_breakout_hard_stop):
            if last_exit_reason != "hard_stop":
                continue
        sx0 = int(ch["support_x0"])
        sy0 = float(ch["support_y0"])
        sslope = float(ch["support_slope"])
        width = float(ch["channel_width"])
        orig_entry_i = int(entry_i)
        if fill_px is not None:
            lifted = _ensure_fill_not_below_support(
                sim_high,
                sim_low,
                sim_close,
                fill_i=int(entry_i),
                fill_px=float(fill_px),
                support_x0=sx0,
                support_y0=sy0,
                support_slope=sslope,
                width=width,
                h2=int(ch.get("h2_idx", -1)),
                n=sim_n,
                error_pct=tag_error_pct,
                slip=float(entry_slip_pct),
                wait=max(1, int(max_l3_wait_bars)),
                daily_i=daily_i_map if hybrid else None,
            )
            if lifted is None:
                continue
            entry_i, fill_px, lifted_brk = lifted
            if lifted_brk:
                is_resist_break = True
            if entry_i is None or entry_i <= busy_until or entry_i >= sim_n:
                continue
        deferred_channel = int(entry_i) != orig_entry_i
        if hybrid and daily_i_map is not None and 0 <= int(entry_i) < len(daily_i_map):
            mapped = int(daily_i_map[int(entry_i)])
            line_i = mapped if mapped >= 0 else int(daily_entry_i)
        else:
            line_i = int(entry_i)
        feat_src = int(t_idx) if use_realistic else int(entry_i)
        atr_i_idx = max(0, feat_src - 1) if (hybrid or feature_asof_prior_bar or use_realistic) else feat_src
        if hybrid:
            atr_i = float(atr[atr_i_idx]) if atr_i_idx < len(atr) else float("nan")
        else:
            atr_src_i = max(0, feat_src - 1) if feature_asof_prior_bar else feat_src
            if use_realistic:
                atr_src_i = max(0, int(t_idx) - 1) if feature_asof_prior_bar else int(t_idx)
            atr_i = float(atr[atr_src_i]) if atr_src_i < len(atr) else float("nan")
        sim = _simulate_trade(
            sim_high,
            sim_low,
            sim_close,
            sim_dates,
            entry_i,
            stop_pct=stop_pct,
            trail_pct=trail_pct,
            trail_pct_wide=wide,
            squeeze_mom=squeeze_mom,
            squeeze_pctile=squeeze_pctile,
            squeeze_lookback=squeeze_lookback,
            atr_at_entry=atr_i if np.isfinite(atr_i) else None,
            atr_stop_mult=atr_stop_mult,
            stop_pct_floor=stop_pct_floor,
            stop_pct_ceil=stop_pct_ceil,
            support_x0=sx0,
            support_y0=sy0,
            support_slope=sslope,
            channel_width=width,
            resist_exit=resist_exit and not hybrid,
            resist_arm_trail=bool(resist_arm_trail),
            peak_trail_mode=str(peak_trail_mode or "off"),
            trail_floor=float(trail_floor),
            trail_decay_per_bar=float(trail_decay_per_bar),
            trail_tighten_per_pct=float(trail_tighten_per_pct),
            trail_pct_tight=trail_pct_tight,
            squeeze_fade_tighten=squeeze_fade_tighten,
            max_hold_days=hold_max,
            include_time=include_time,
            entry_px=fill_px,
            skip_entry_bar_stop=bool(
                (use_realistic and not is_15m_bars)
                or deferred_channel
                or (use_realistic and is_15m_bars and fill_mode == FILL_MODE_SIGNAL_CLOSE)
            ),
        )
        if sim is None:
            continue
        entry_px = float(sim["buy_price"])
        atr_pct = (atr_i / entry_px * 100.0) if entry_px > 0 and np.isfinite(atr_i) else float("nan")
        if hybrid:
            adv = _adv_20(close, volume, atr_i_idx, lookback=adv_lookback)
        else:
            adv_i = max(0, feat_src - 1) if feature_asof_prior_bar else feat_src
            adv = _adv_20(close, volume, adv_i, lookback=adv_lookback)
        support_at = _line_at(sy0, sx0, sslope, line_i)
        resist_at = support_at + width
        channel_pos = (
            (entry_px - support_at) / width
            if width > 0 and np.isfinite(support_at)
            else float("nan")
        )
        room_to_resist_pct = (
            (resist_at - entry_px) / entry_px * 100.0
            if entry_px > 0 and np.isfinite(resist_at)
            else float("nan")
        )
        if hybrid:
            beyond_prior = (
                max_beyond_width(high, sy0, sx0, sslope, width, sx0, max(sx0, line_i - 1))
                if line_i > sx0
                else 0.0
            )
            fill_over = float("nan")
            if width > 0 and np.isfinite(resist_at):
                fill_over = (float(sim_high[entry_i]) - float(resist_at)) / float(width)
            beyond = max(
                float(beyond_prior) if np.isfinite(beyond_prior) else 0.0,
                float(fill_over) if np.isfinite(fill_over) else 0.0,
            )
        else:
            beyond = max_beyond_width(high, sy0, sx0, sslope, width, sx0, entry_i)
        buy_ts = pd.Timestamp(sim_dates[entry_i])
        try:
            ch_start_ts = pd.Timestamp(ch["start_date"])
            ch_end_ts = pd.Timestamp(ch.get("h2_date") or ch["end_date"])
            span_days = int((ch_end_ts - ch_start_ts).days)
            age_days = int((buy_ts - ch_start_ts).days)
        except Exception:
            span_days = None
            age_days = None
        h2_idx = int(ch.get("h2_idx", t_idx))
        wait_bars = int(t_idx - h2_idx) if (hybrid or use_realistic) else int(entry_i - h2_idx)
        feat_i = (int(t_idx) - 1 if feature_asof_prior_bar else int(t_idx)) if use_realistic else (
            entry_i - 1 if feature_asof_prior_bar else entry_i
        )
        feat_snap = snapshot_stock_features(feat_series, feat_i) if feat_series is not None else {}
        feat_asof = None
        if feat_series is not None and 0 <= feat_i < len(sim_dates):
            feat_asof = pd.Timestamp(sim_dates[feat_i]).strftime("%Y-%m-%d %H:%M")
        touch_ts = pd.Timestamp(sim_dates[t_idx]) if 0 <= t_idx < len(sim_dates) else buy_ts
        touch_px = float(sim_low[t_idx]) if 0 <= t_idx < len(sim_low) else float("nan")
        trades.append(
            {
                "stock": symbol.upper(),
                "channel_start": ch["start_date"],
                "channel_end": ch.get("h2_date") or ch["end_date"],
                "touch_num": touch_num,
                "touch_date": touch_ts.strftime("%Y-%m-%d"),
                "touch_price": round(touch_px, 4) if np.isfinite(touch_px) else None,
                **(
                    {"touch_time": touch_ts.strftime("%Y-%m-%d %H:%M")}
                    if include_time
                    else {}
                ),
                **{k: v for k, v in sim.items() if k not in ("entry_i", "exit_i")},
                "entry_i": sim["entry_i"],
                "exit_i": sim["exit_i"],
                "adv_20": round(adv, 2) if np.isfinite(adv) else None,
                "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
                **channel_rail_fields(ch, dates, low, include_time=include_time or is_15m_bars),
                "slope_pct_per_bar": ch.get("slope_pct_per_bar"),
                "channel_width_pct": ch.get("channel_width_pct"),
                "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
                "room_to_resist_pct": (
                    round(float(room_to_resist_pct), 3) if np.isfinite(room_to_resist_pct) else None
                ),
                "bars_span": ch.get("bars_span"),
                "entry_mode": entry_mode,
                "shakeout_rebuy": bool(is_shakeout),
                "resist_break": bool(is_resist_break),
                "shakeout_breakout": bool(is_sbo),
                "shakeout_inside_bars": int(inside_n) if is_sbo else None,
                "parent_exit_reason": last_exit_reason if is_sbo else None,
                "wait_bars": wait_bars,
                "max_beyond_width": round(float(beyond), 4) if np.isfinite(beyond) else None,
                "channel_span_days": span_days,
                "channel_age_at_buy_days": age_days,
                "dow": int(buy_ts.dayofweek) if pd.notna(buy_ts) else None,
                "month": int(buy_ts.month) if pd.notna(buy_ts) else None,
                **({"feature_asof": feat_asof} if feat_asof else {}),
                **feat_snap,
            }
        )
        if fill_time is not None:
            ts_fill = pd.Timestamp(fill_time)
            if ts_fill.tzinfo is not None:
                ts_fill = ts_fill.tz_convert("UTC")
            else:
                ts_fill = ts_fill.tz_localize("UTC")
            trades[-1]["buy_time"] = ts_fill.strftime("%Y-%m-%d %H:%M")
        busy_until = sim["exit_i"]
        last_exit_reason = str(sim.get("exit_reason") or "")
    return trades


def _worker_symbol_trades(payload: dict) -> List[dict]:
    """ProcessPool worker: run trades for one symbol from a pickled OHLCV frame."""
    return trades_for_symbol(
        payload["symbol"],
        payload["df"],
        entry_touch=payload["entry_touch"],
        stop_pct=payload["stop_pct"],
        trail_pct=payload["trail_pct"],
        trail_pct_wide=payload.get("trail_pct_wide"),
        squeeze_adaptive=bool(payload.get("squeeze_adaptive", False)),
        squeeze_pctile=float(payload.get("squeeze_pctile", 75.0)),
        squeeze_lookback=int(payload.get("squeeze_lookback", 100)),
        pivot_len=payload["pivot_len"],
        entry_mode=str(payload.get("entry_mode", "pivot")),
        atr_stop_mult=payload.get("atr_stop_mult"),
        stop_pct_floor=float(payload.get("stop_pct_floor", 0.015)),
        stop_pct_ceil=float(payload.get("stop_pct_ceil", 0.06)),
        resist_exit=bool(payload.get("resist_exit", False)),
        resist_arm_trail=bool(payload.get("resist_arm_trail", False)),
        peak_trail_mode=str(payload.get("peak_trail_mode") or "off"),
        trail_floor=float(payload.get("trail_floor", 0.01)),
        trail_decay_per_bar=float(payload.get("trail_decay_per_bar", 0.0002)),
        trail_tighten_per_pct=float(payload.get("trail_tighten_per_pct", 0.0033)),
        trail_pct_tight=payload.get("trail_pct_tight"),
        squeeze_fade_tighten=bool(payload.get("squeeze_fade_tighten", False)),
        max_hold_days=payload.get("max_hold_days"),
        adv_lookback=int(payload.get("adv_lookback", 20)),
        window_bars=payload.get("window_bars"),
        window_step_bars=payload.get("window_step_bars"),
        include_time=bool(payload.get("include_time", False)),
        entry_features=bool(payload.get("entry_features", True)),
        entry_slip_pct=float(payload.get("entry_slip_pct", 0.001)),
        max_l3_wait_bars=int(payload.get("max_l3_wait_bars", 252)),
        min_l3_wait_bars=int(payload.get("min_l3_wait_bars", 1)),
        shakeout_rebuy_bars=int(payload.get("shakeout_rebuy_bars", 0)),
        h2_resist_break=bool(payload.get("h2_resist_break", False)),
        h2_resist_break_only=bool(payload.get("h2_resist_break_only", False)),
        shakeout_breakout=bool(payload.get("shakeout_breakout", False)),
        shakeout_breakout_min_inside=int(payload.get("shakeout_breakout_min_inside", 1)),
        shakeout_breakout_hard_stop=bool(payload.get("shakeout_breakout_hard_stop", False)),
        df_15m=payload.get("df_15m"),
        intraday_fill=str(payload.get("intraday_fill") or ""),
        feature_asof_prior_bar=bool(payload.get("feature_asof_prior_bar", False)),
        realistic_fill=bool(payload.get("realistic_fill", False)),
        realistic_fill_mode=str(payload.get("realistic_fill_mode") or DEFAULT_FILL_MODE),
        max_low_to_mid_pct=payload.get("max_low_to_mid_pct", DEFAULT_MAX_LOW_TO_MID_PCT),
        max_chase_pct=payload.get("max_chase_pct"),
        touch_error_pct=payload.get("touch_error_pct"),
        **(payload.get("channel_kwargs") or {}),
    )


def _summarize(trades: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    if trades.empty:
        return {
            "n_trades": 0,
            "n_symbols": 0,
            "win_rate_pct": None,
            "avg_gain_pct": None,
            "median_gain_pct": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "avg_hold_days": None,
        }
    g = trades[gain_col].astype(float)
    wins = trades[g > 0]
    losses = trades[g <= 0]
    avg_win = float(wins[gain_col].mean()) if len(wins) else 0.0
    avg_loss = float(losses[gain_col].mean()) if len(losses) else 0.0
    wr = len(wins) / len(trades) if len(trades) else 0.0
    expectancy = wr * avg_win + (1.0 - wr) * avg_loss
    gp = float(wins[gain_col].sum()) if len(wins) else 0.0
    gl = float((-losses[gain_col]).sum()) if len(losses) else 0.0
    pf = (gp / gl) if gl > 1e-12 else float("inf") if gp > 0 else 0.0
    out = {
        "n_trades": int(len(trades)),
        "n_symbols": int(trades["stock"].nunique()) if "stock" in trades.columns else 0,
        "win_rate_pct": round(wr * 100.0, 2),
        "avg_gain_pct": round(float(g.mean()), 2),
        "median_gain_pct": round(float(g.median()), 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "expectancy_pct": round(expectancy, 2),
        "profit_factor": round(pf, 3) if np.isfinite(pf) else None,
        "avg_hold_days": round(float(trades["hold_days"].mean()), 1) if "hold_days" in trades.columns else None,
    }
    if "exit_reason" in trades.columns:
        out["hard_stop_exits"] = int((trades["exit_reason"] == "hard_stop").sum())
        out["trail_stop_exits"] = int((trades["exit_reason"] == "trail_stop").sum())
        out["trail_stop_wide_exits"] = int((trades["exit_reason"] == "trail_stop_wide").sum())
        out["trail_stop_tight_exits"] = int((trades["exit_reason"] == "trail_stop_tight").sum())
        out["resist_exits"] = int((trades["exit_reason"] == "resist_exit").sum())
        out["resist_arm_trail_exits"] = int((trades["exit_reason"] == "resist_arm_trail").sum())
        out["peak_trail_exits"] = int((trades["exit_reason"] == "peak_trail").sum())
        out["time_stop_exits"] = int((trades["exit_reason"] == "time_stop").sum())
        out["eod_exits"] = int((trades["exit_reason"] == "eod").sum())
    return out


def _return_lookback(close: pd.Series, asof: pd.Timestamp, lookback: int) -> float:
    """Simple return from lookback bars before asof to asof (inclusive end)."""
    if close is None or close.empty or lookback <= 0:
        return float("nan")
    s = close.sort_index()
    if s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_convert(None)
    # last available bar on/before asof
    hist = s.loc[:asof]
    if len(hist) < lookback + 1:
        return float("nan")
    end_px = float(hist.iloc[-1])
    start_px = float(hist.iloc[-(lookback + 1)])
    if not np.isfinite(end_px) or not np.isfinite(start_px) or start_px <= 0:
        return float("nan")
    return end_px / start_px - 1.0


def enrich_rs(
    trades: pd.DataFrame,
    panels: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    *,
    lookbacks: Sequence[int] = (63, 126),
    bars_per_session: int = 1,
) -> pd.DataFrame:
    """Attach relative strength vs SPY at each buy timestamp.

    ``lookbacks`` are session counts (63/126). ``bars_per_session`` converts them
    to bar lookbacks (1 on daily, 26 on RTH 15m).
    """
    if trades.empty:
        return trades
    out = trades.copy()
    spy = spy_df["close"].astype(float).copy()
    if not isinstance(spy.index, pd.DatetimeIndex):
        spy.index = pd.DatetimeIndex(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)

    close_cache: Dict[str, pd.Series] = {}
    series_is_daily = int(bars_per_session) <= 1
    for lb in lookbacks:
        col = f"rs_spy_{lb}d"
        vals: List[float] = []
        for _, row in out.iterrows():
            sym = str(row["stock"]).upper()
            asof = completed_asof(row, series_is_daily=series_is_daily)
            if asof is None:
                vals.append(float("nan"))
                continue
            if sym not in close_cache:
                df = panels.get(sym)
                if df is None or df.empty:
                    close_cache[sym] = pd.Series(dtype=float)
                else:
                    c = df["close"].astype(float).copy()
                    if not isinstance(c.index, pd.DatetimeIndex):
                        c.index = pd.DatetimeIndex(c.index)
                    if c.index.tz is not None:
                        c.index = c.index.tz_convert(None)
                    close_cache[sym] = c
            stock_ret = _return_lookback(close_cache[sym], asof, int(lb) * max(1, int(bars_per_session)))
            spy_ret = _return_lookback(spy, asof, int(lb) * max(1, int(bars_per_session)))
            if np.isfinite(stock_ret) and np.isfinite(spy_ret):
                vals.append(round((stock_ret - spy_ret) * 100.0, 3))
            else:
                vals.append(float("nan"))
        out[col] = vals
    return out


def filter_trades(
    trades: pd.DataFrame,
    *,
    min_adv: Optional[float] = None,
    min_atr_pct: Optional[float] = None,
    max_channel_pos: Optional[float] = None,
    min_width_pct: Optional[float] = None,
    max_width_pct: Optional[float] = None,
    min_slope_pct: Optional[float] = None,
    max_slope_pct: Optional[float] = None,
    require_spy_above_sma: bool = False,
    max_channel_span_days: Optional[float] = None,
    max_channel_age_days: Optional[float] = None,
    require_in_channel: bool = False,
    max_beyond_width: Optional[float] = None,
    max_rsi: Optional[float] = None,
    min_close_loc: Optional[float] = None,
) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades
    # Derive span/age when dates exist (quality gates; detector unchanged)
    if (
        max_channel_span_days is not None or max_channel_age_days is not None
    ) and {"channel_start", "channel_end", "buy_date"}.issubset(out.columns):
        if "channel_span_days" not in out.columns or "channel_age_at_buy_days" not in out.columns:
            cs = pd.to_datetime(out["channel_start"], errors="coerce")
            ce = pd.to_datetime(out["channel_end"], errors="coerce")
            bd = pd.to_datetime(out["buy_date"], errors="coerce")
            out = out.copy()
            out["channel_span_days"] = (ce - cs).dt.days
            out["channel_age_at_buy_days"] = (bd - cs).dt.days
    m = pd.Series(True, index=out.index)
    if min_adv is not None:
        m &= out["adv_20"].fillna(0) >= float(min_adv)
    if min_atr_pct is not None:
        m &= out["atr_pct"].fillna(0) >= float(min_atr_pct)
    if max_channel_pos is not None and "channel_pos" in out.columns:
        m &= out["channel_pos"].fillna(999) <= float(max_channel_pos)
    if require_in_channel and "channel_pos" in out.columns:
        pos = out["channel_pos"]
        in_ch = (pos.fillna(999) <= 1.0) & (pos.fillna(-999) >= 0.0)
        if "resist_break" in out.columns:
            brk = out["resist_break"].fillna(False)
            if brk.dtype == object:
                brk = brk.astype(str).str.lower().isin(("true", "1", "yes"))
            else:
                brk = brk.astype(bool)
            in_ch = in_ch | brk
        m &= in_ch
    if min_width_pct is not None and "channel_width_pct" in out.columns:
        m &= out["channel_width_pct"].fillna(-1) >= float(min_width_pct)
    if max_width_pct is not None and "channel_width_pct" in out.columns:
        m &= out["channel_width_pct"].fillna(1e9) <= float(max_width_pct)
    if min_slope_pct is not None and "slope_pct_per_bar" in out.columns:
        m &= out["slope_pct_per_bar"].fillna(-1) >= float(min_slope_pct)
    if max_slope_pct is not None and "slope_pct_per_bar" in out.columns:
        m &= out["slope_pct_per_bar"].fillna(1e9) <= float(max_slope_pct)
    if require_spy_above_sma and "spy_above_sma50" in out.columns:
        m &= out["spy_above_sma50"].fillna(False).astype(bool)
    if max_channel_span_days is not None and "channel_span_days" in out.columns:
        m &= out["channel_span_days"].fillna(1e9) <= float(max_channel_span_days)
    if max_channel_age_days is not None and "channel_age_at_buy_days" in out.columns:
        m &= out["channel_age_at_buy_days"].fillna(1e9) <= float(max_channel_age_days)
    if max_beyond_width is not None and "max_beyond_width" in out.columns:
        m &= out["max_beyond_width"].fillna(999) <= float(max_beyond_width)
    if max_rsi is not None and "rsi_14" in out.columns:
        m &= out["rsi_14"].fillna(999) <= float(max_rsi)
    if min_close_loc is not None and "close_loc" in out.columns:
        m &= out["close_loc"].fillna(-1) >= float(min_close_loc)
    return out.loc[m].copy()


def select_same_day_rs(
    trades: pd.DataFrame,
    *,
    rs_col: str = "rs_spy_126d",
    max_per_day: int = 1,
) -> pd.DataFrame:
    """Keep top-N trades per buy_date ranked by relative strength vs SPY."""
    if trades.empty or max_per_day <= 0:
        return trades
    if rs_col not in trades.columns:
        return trades
    ranked = trades.sort_values(
        ["buy_date", rs_col],
        ascending=[True, False],
        na_position="last",
    )
    kept = ranked.groupby("buy_date", sort=False).head(int(max_per_day))
    return kept.reset_index(drop=True)


def keep_one_per_symbol_day(
    trades: pd.DataFrame,
    *,
    symbol_col: str = "stock",
    date_col: str = "buy_date",
    time_col: str = "buy_time",
) -> pd.DataFrame:
    """Keep the earliest fill per symbol per calendar day.

    Does not cap how many *names* fire that day. Use ``select_same_day_rs``
    only when you want a cross-symbol capacity cap.
    """
    if trades.empty or symbol_col not in trades.columns or date_col not in trades.columns:
        return trades
    out = trades.copy()
    out["_day"] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    sort_cols = ["_day", symbol_col]
    extra = ["_day"]
    if time_col in out.columns:
        out["_t"] = pd.to_datetime(out[time_col], errors="coerce")
        sort_cols.append("_t")
        extra.append("_t")
    out = out.sort_values(sort_cols, kind="mergesort")
    kept = out.groupby(["_day", symbol_col], sort=False, as_index=False).head(1)
    return kept.drop(columns=extra).reset_index(drop=True)


def apply_friction(trades: pd.DataFrame, round_trip_pct: float) -> pd.DataFrame:
    """Deduct round-trip friction (slippage + commission) from gross gain_pct."""
    out = trades.copy()
    out["gain_pct_net"] = out["gain_pct"].astype(float) - float(round_trip_pct)
    return out


def edge_scenarios(
    trades: pd.DataFrame,
    *,
    friction_pcts: Sequence[float] = (0.10, 0.25),
) -> pd.DataFrame:
    """Compare liquidity/vol/RS filters and net expectancy after friction."""
    scenarios: List[Tuple[str, pd.DataFrame]] = [
        ("baseline", trades),
        ("adv>=5M", filter_trades(trades, min_adv=5_000_000)),
        ("adv>=20M", filter_trades(trades, min_adv=20_000_000)),
        ("atr>=1.5%", filter_trades(trades, min_atr_pct=1.5)),
        ("atr>=2.0%", filter_trades(trades, min_atr_pct=2.0)),
        ("adv20M+atr1.5", filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5)),
        (
            "adv20M+atr1.5+rs_top1",
            select_same_day_rs(
                filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5),
                rs_col="rs_spy_126d",
                max_per_day=1,
            ),
        ),
        (
            "adv20M+atr1.5+rs_top3",
            select_same_day_rs(
                filter_trades(trades, min_adv=20_000_000, min_atr_pct=1.5),
                rs_col="rs_spy_126d",
                max_per_day=3,
            ),
        ),
        (
            "rs_top1_only",
            select_same_day_rs(trades, rs_col="rs_spy_126d", max_per_day=1),
        ),
    ]

    rows: List[dict] = []
    for name, subset in scenarios:
        gross = _summarize(subset, gain_col="gain_pct")
        row = {
            "scenario": name,
            "n_trades": gross["n_trades"],
            "n_symbols": gross["n_symbols"],
            "win_rate_pct": gross["win_rate_pct"],
            "avg_win_pct": gross["avg_win_pct"],
            "avg_loss_pct": gross["avg_loss_pct"],
            "expectancy_gross_pct": gross["expectancy_pct"],
            "profit_factor": gross["profit_factor"],
            "avg_hold_days": gross["avg_hold_days"],
        }
        for fr in friction_pcts:
            net_df = apply_friction(subset, fr)
            net = _summarize(net_df, gain_col="gain_pct_net")
            key = f"expectancy_net_{fr:.2f}pct"
            row[key] = net["expectancy_pct"]
            row[f"net_above_1_{fr:.2f}"] = (
                None if net["expectancy_pct"] is None else bool(net["expectancy_pct"] >= 1.0)
            )
        rows.append(row)
    return pd.DataFrame(rows)



def summarize_by_year(
    trades: pd.DataFrame,
    *,
    gain_col: str = "gain_pct",
    buckets: Sequence[Tuple[str, str, str]] = YEAR_BUCKETS,
) -> pd.DataFrame:
    rows: List[dict] = []
    if trades.empty or "buy_date" not in trades.columns:
        return pd.DataFrame(rows)
    bd = pd.to_datetime(trades["buy_date"])
    for name, start, end in buckets:
        m = (bd >= pd.Timestamp(start)) & (bd <= pd.Timestamp(end))
        s = _summarize(trades.loc[m], gain_col=gain_col)
        rows.append(
            {
                "bucket": name,
                "n_trades": s["n_trades"],
                "win_rate_pct": s["win_rate_pct"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "avg_win_pct": s["avg_win_pct"],
                "avg_loss_pct": s["avg_loss_pct"],
                "hard_stop_exits": s.get("hard_stop_exits"),
            }
        )
    full = _summarize(trades, gain_col=gain_col)
    rows.append(
        {
            "bucket": "FULL",
            "n_trades": full["n_trades"],
            "win_rate_pct": full["win_rate_pct"],
            "expectancy_pct": full["expectancy_pct"],
            "profit_factor": full["profit_factor"],
            "avg_win_pct": full["avg_win_pct"],
            "avg_loss_pct": full["avg_loss_pct"],
            "hard_stop_exits": full.get("hard_stop_exits"),
        }
    )
    return pd.DataFrame(rows)


def enrich_spy_regime(trades: pd.DataFrame, spy_df: pd.DataFrame, *, sma_len: int = 50) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    spy = spy_df["close"].astype(float).copy()
    if not isinstance(spy.index, pd.DatetimeIndex):
        spy.index = pd.DatetimeIndex(spy.index)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    spy = spy.sort_index()
    sma = spy.rolling(int(sma_len), min_periods=int(sma_len)).mean()
    above: List[Optional[bool]] = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=True)
        if asof is None:
            asof = pd.Timestamp(row["buy_date"])
        hist_c = spy.loc[:asof]
        hist_s = sma.loc[:asof]
        if hist_c.empty or hist_s.empty or not np.isfinite(float(hist_s.iloc[-1])):
            above.append(None)
            continue
        above.append(bool(float(hist_c.iloc[-1]) > float(hist_s.iloc[-1])))
    out["spy_above_sma50"] = above
    return out


def _scan_trades(
    panels: Dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    workers: int,
    base: dict,
    panels_15m: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    payloads = []
    extra_15 = panels_15m or {}
    for sym in symbols:
        if sym not in panels or sym == "SPY":
            continue
        payload = {"symbol": sym, "df": panels[sym], **base}
        if extra_15:
            payload["df_15m"] = extra_15.get(sym)
        payloads.append(payload)
    all_trades: List[dict] = []
    if workers and workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = [pool.submit(_worker_symbol_trades, p) for p in payloads]
            for fut in as_completed(futs):
                all_trades.extend(fut.result())
    else:
        for p in payloads:
            all_trades.extend(_worker_symbol_trades(p))
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def _finalize_report_trades(
    trades: pd.DataFrame,
    *,
    max_entries_per_day: int = 1,
    friction_pct: float = 0.25,
    geometry: bool = False,
    spy_regime: bool = False,
) -> Tuple[pd.DataFrame, str]:
    out = trades
    if geometry:
        out = filter_trades(
            out,
            max_channel_pos=0.40,
            min_width_pct=3.0,
            max_width_pct=35.0,
            min_slope_pct=0.02,
            max_slope_pct=0.50,
        )
    if spy_regime:
        out = filter_trades(out, require_spy_above_sma=True)
    if max_entries_per_day and max_entries_per_day > 0:
        out = select_same_day_rs(out, rs_col="rs_spy_126d", max_per_day=int(max_entries_per_day))
    gain_col = "gain_pct"
    if friction_pct and friction_pct > 0:
        out = apply_friction(out, friction_pct)
        gain_col = "gain_pct_net"
    return out, gain_col


def run_edge_v2(
    panels: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    symbols: Sequence[str],
    *,
    workers: int,
    friction_pct: float = 0.25,
    max_entries_per_day: int = 1,
    squeeze_pctile: float = 75.0,
    squeeze_lookback: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Run H0 baseline + H1-H5 scenarios on one loaded panel set."""
    base_common = {
        "entry_touch": 3,
        "trail_pct": 0.10,
        "trail_pct_wide": 0.18,
        "squeeze_adaptive": True,
        "squeeze_pctile": squeeze_pctile,
        "squeeze_lookback": squeeze_lookback,
        "stop_pct": 0.03,
        "pivot_len": 15,
        "entry_mode": "pivot",
        "atr_stop_mult": None,
        "stop_pct_floor": 0.015,
        "stop_pct_ceil": 0.06,
        "resist_exit": False,
        "trail_pct_tight": None,
        "squeeze_fade_tighten": False,
        "max_hold_days": None,
    }
    scan_specs: List[Tuple[str, dict]] = [
        ("H0_baseline", {}),
        ("H1_atr_stop_k1.0", {"atr_stop_mult": 1.0}),
        ("H1_atr_stop_k1.5", {"atr_stop_mult": 1.5}),
        ("H1_atr_stop_k2.0", {"atr_stop_mult": 2.0}),
        ("H2_pivot_len_5", {"pivot_len": 5}),
        ("H2_pivot_len_10", {"pivot_len": 10}),
        ("H2_entry_reclaim", {"entry_mode": "reclaim"}),
        (
            "H5_struct_exits",
            {
                "resist_exit": True,
                "squeeze_fade_tighten": True,
                "trail_pct_tight": 0.07,
                "max_hold_days": 50,
            },
        ),
        ("H1k1.5+H2_pivot10", {"atr_stop_mult": 1.5, "pivot_len": 10}),
        (
            "H1k1.5+H5_struct",
            {
                "atr_stop_mult": 1.5,
                "resist_exit": True,
                "squeeze_fade_tighten": True,
                "trail_pct_tight": 0.07,
                "max_hold_days": 50,
            },
        ),
    ]
    raw_by_name: Dict[str, pd.DataFrame] = {}
    for name, overrides in scan_specs:
        cfg = dict(base_common)
        cfg.update(overrides)
        t1 = time.perf_counter()
        logger.info("edge-v2 scan %s ...", name)
        raw = _scan_trades(panels, symbols=symbols, workers=workers, base=cfg)
        if raw.empty:
            raw_by_name[name] = raw
            logger.info("edge-v2 scan %s -> 0 trades (%.1fs)", name, time.perf_counter() - t1)
            continue
        raw = enrich_rs(raw, panels, spy_df, lookbacks=(63, 126))
        raw = enrich_spy_regime(raw, spy_df, sma_len=50)
        raw_by_name[name] = raw
        logger.info("edge-v2 scan %s -> %d raw trades (%.1fs)", name, len(raw), time.perf_counter() - t1)

    post_specs: List[Tuple[str, str, bool, bool]] = [
        ("H3_geometry", "H0_baseline", True, False),
        ("H4_spy_sma50", "H0_baseline", False, True),
        ("H3+H4", "H0_baseline", True, True),
        ("H1k1.5+H3", "H1_atr_stop_k1.5", True, False),
        ("H1k1.5+H4", "H1_atr_stop_k1.5", False, True),
    ]
    summary_rows: List[dict] = []
    year_frames: List[pd.DataFrame] = []

    def _add(name: str, raw: Optional[pd.DataFrame], geometry: bool, spy_regime: bool) -> None:
        if raw is None or raw.empty:
            summary_rows.append({"scenario": name, "n_trades": 0, "expectancy_pct": None, "profit_factor": None})
            return
        final, gain_col = _finalize_report_trades(
            raw,
            max_entries_per_day=max_entries_per_day,
            friction_pct=friction_pct,
            geometry=geometry,
            spy_regime=spy_regime,
        )
        s = _summarize(final, gain_col=gain_col)
        summary_rows.append(
            {
                "scenario": name,
                "n_trades": s["n_trades"],
                "n_symbols": s["n_symbols"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "win_rate_pct": s["win_rate_pct"],
                "avg_win_pct": s["avg_win_pct"],
                "avg_loss_pct": s["avg_loss_pct"],
                "avg_hold_days": s["avg_hold_days"],
                "hard_stop_exits": s.get("hard_stop_exits"),
                "trail_stop_exits": s.get("trail_stop_exits"),
                "trail_stop_wide_exits": s.get("trail_stop_wide_exits"),
                "trail_stop_tight_exits": s.get("trail_stop_tight_exits"),
                "resist_exits": s.get("resist_exits"),
                "time_stop_exits": s.get("time_stop_exits"),
                "eod_exits": s.get("eod_exits"),
            }
        )
        ydf = summarize_by_year(final, gain_col=gain_col)
        if not ydf.empty:
            ydf = ydf.copy()
            ydf.insert(0, "scenario", name)
            year_frames.append(ydf)

    for name, _ in scan_specs:
        _add(name, raw_by_name.get(name), geometry=False, spy_regime=False)
    for name, src, geo, regime in post_specs:
        _add(name, raw_by_name.get(src), geometry=geo, spy_regime=regime)

    summary_df = pd.DataFrame(summary_rows)
    year_df = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
    return summary_df, year_df, raw_by_name


def _drop_top_n_stats(trades: pd.DataFrame, *, gain_col: str, n: int = 10) -> dict:
    if trades is None or trades.empty or gain_col not in trades.columns:
        return {"n_trades": 0, "expectancy_pct": None, "profit_factor": None}
    ranked = trades.sort_values(gain_col, ascending=False)
    kept = ranked.iloc[int(n) :] if len(ranked) > int(n) else ranked.iloc[0:0]
    return _summarize(kept, gain_col=gain_col)


def run_peak_trail_sweep(
    panels: Dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    symbols: Sequence[str],
    *,
    workers: int,
    friction_pct: float = 0.10,
    max_entries_per_day: int = 1,
    require_in_channel: bool = True,
    max_channel_span_days: Optional[float] = 10.0,
    bars_per_session: int = 26,
    channel_kwargs: Optional[dict] = None,
    scan_base: Optional[dict] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Keeper + five peak-trail exit variants on one loaded 15m panel set."""
    ch_kw = dict(channel_kwargs or {})
    base = {
        "entry_touch": 3,
        "entry_mode": "l3_touch",
        "min_l3_wait_bars": 12,
        "max_l3_wait_bars": 252,
        "shakeout_rebuy_bars": 0,
        "h2_resist_break": False,
        "h2_resist_break_only": False,
        "shakeout_breakout": False,
        "shakeout_breakout_min_inside": 1,
        "shakeout_breakout_hard_stop": False,
        "realistic_fill": True,
        "realistic_fill_mode": FILL_MODE_SIGNAL_CLOSE,
        "touch_error_pct": 0.0,
        "include_time": True,
        "entry_features": True,
        "feature_asof_prior_bar": True,
        "entry_slip_pct": 0.001,
        "pivot_len": int(PRESET_15M["pivot_len"]),
        "window_bars": int(PRESET_15M["window_bars"]),
        "window_step_bars": int(PRESET_15M["window_step_bars"]),
        "adv_lookback": int(PRESET_15M["adv_lookback"]),
        "squeeze_pctile": 75.0,
        "squeeze_lookback": int(PRESET_15M["squeeze_lookback"]),
        "channel_kwargs": ch_kw,
    }
    if scan_base:
        base.update(scan_base)

    keeper = {
        "stop_pct": float(PRESET_15M["stop_pct"]),
        "trail_pct": float(PRESET_15M["trail_pct"]),
        "trail_pct_wide": float(PRESET_15M["trail_pct_wide"]),
        "squeeze_adaptive": True,
        "atr_stop_mult": float(PRESET_15M["atr_stop_mult"]),
        "stop_pct_floor": float(PRESET_15M["stop_pct_floor"]),
        "stop_pct_ceil": float(PRESET_15M["stop_pct_ceil"]),
        "resist_exit": False,
        "resist_arm_trail": False,
        "peak_trail_mode": "off",
        "trail_floor": 0.01,
        "trail_decay_per_bar": 0.0002,
        "trail_tighten_per_pct": 0.0033,
    }
    peak_common = {
        "squeeze_adaptive": False,
        "atr_stop_mult": None,
        "resist_exit": False,
        "resist_arm_trail": False,
        "trail_pct_wide": None,
        "trail_pct_tight": None,
        "squeeze_fade_tighten": False,
        "stop_pct_floor": 0.015,
        "stop_pct_ceil": 0.06,
        "trail_floor": 0.01,
        "trail_decay_per_bar": 0.0002,
        "trail_tighten_per_pct": 0.0033,
    }
    scan_specs: List[Tuple[str, dict]] = [
        ("Keeper", dict(keeper)),
        (
            "F125",
            {
                **peak_common,
                "peak_trail_mode": "fixed",
                "stop_pct": 0.0125,
                "trail_pct": 0.0125,
            },
        ),
        (
            "F200",
            {
                **peak_common,
                "peak_trail_mode": "fixed",
                "stop_pct": 0.02,
                "trail_pct": 0.02,
            },
        ),
        (
            "F300",
            {
                **peak_common,
                "peak_trail_mode": "fixed",
                "stop_pct": 0.03,
                "trail_pct": 0.03,
            },
        ),
        (
            "T4d",
            {
                **peak_common,
                "peak_trail_mode": "time_decay",
                "stop_pct": 0.04,
                "trail_pct": 0.04,
                "trail_decay_per_bar": 0.0002,
            },
        ),
        (
            "G4t",
            {
                **peak_common,
                "peak_trail_mode": "gain_tighten",
                "stop_pct": 0.04,
                "trail_pct": 0.04,
                "trail_tighten_per_pct": 0.0033,
            },
        ),
    ]

    raw_by_name: Dict[str, pd.DataFrame] = {}
    summary_rows: List[dict] = []
    year_frames: List[pd.DataFrame] = []
    rs_lookbacks = (21, 63, 126)

    for name, overrides in scan_specs:
        cfg = dict(base)
        cfg.update(overrides)
        t1 = time.perf_counter()
        logger.info("peak-trail sweep %s ...", name)
        raw = _scan_trades(panels, symbols=symbols, workers=workers, base=cfg)
        if raw.empty:
            raw_by_name[name] = raw
            summary_rows.append(
                {
                    "scenario": name,
                    "n_trades": 0,
                    "expectancy_pct": None,
                    "profit_factor": None,
                }
            )
            logger.info("peak-trail sweep %s -> 0 trades (%.1fs)", name, time.perf_counter() - t1)
            continue
        raw = enrich_rs(
            raw,
            panels,
            spy_df,
            lookbacks=rs_lookbacks,
            bars_per_session=int(bars_per_session),
        )
        raw_by_name[name] = raw
        filtered = filter_trades(
            raw,
            require_in_channel=bool(require_in_channel),
            max_channel_span_days=max_channel_span_days,
        )
        if max_entries_per_day and max_entries_per_day > 0:
            filtered = select_same_day_rs(
                filtered, rs_col="rs_spy_126d", max_per_day=int(max_entries_per_day)
            )
        gain_col = "gain_pct"
        if friction_pct and friction_pct > 0:
            filtered = apply_friction(filtered, friction_pct)
            gain_col = "gain_pct_net"
        s = _summarize(filtered, gain_col=gain_col)
        drop10 = _drop_top_n_stats(filtered, gain_col=gain_col, n=10)
        no_eod = filtered
        if "exit_reason" in filtered.columns:
            no_eod = filtered[filtered["exit_reason"] != "eod"]
        s_no_eod = _summarize(no_eod, gain_col=gain_col)
        summary_rows.append(
            {
                "scenario": name,
                "n_trades": s["n_trades"],
                "n_symbols": s["n_symbols"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "median_gain_pct": s.get("median_gain_pct"),
                "win_rate_pct": s["win_rate_pct"],
                "avg_hold_days": s["avg_hold_days"],
                "eod_exits": s.get("eod_exits"),
                "peak_trail_exits": s.get("peak_trail_exits"),
                "hard_stop_exits": s.get("hard_stop_exits"),
                "trail_stop_exits": s.get("trail_stop_exits"),
                "trail_stop_wide_exits": s.get("trail_stop_wide_exits"),
                "no_eod_n": s_no_eod["n_trades"],
                "no_eod_E": s_no_eod["expectancy_pct"],
                "no_eod_PF": s_no_eod["profit_factor"],
                "drop_top10_E": drop10.get("expectancy_pct"),
                "drop_top10_PF": drop10.get("profit_factor"),
            }
        )
        ydf = summarize_by_year(filtered, gain_col=gain_col)
        if not ydf.empty:
            ydf = ydf.copy()
            ydf.insert(0, "scenario", name)
            year_frames.append(ydf)
        logger.info(
            "peak-trail sweep %s -> raw=%d final=%d E=%s PF=%s (%.1fs)",
            name,
            len(raw),
            s["n_trades"],
            s["expectancy_pct"],
            s["profit_factor"],
            time.perf_counter() - t1,
        )

    summary_df = pd.DataFrame(summary_rows)
    year_df = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
    return summary_df, year_df, raw_by_name


REPORT_COLS = [
    "stock",
    "channel_start",
    "channel_end",
    "l1_time",
    "l1_ms",
    "l1_price",
    "l2_time",
    "l2_ms",
    "l2_price",
    "h2_time",
    "h2_ms",
    "channel_width",
    "touch_num",
    "touch_date",
    "touch_time",
    "touch_price",
    "buy_date",
    "buy_time",
    "sell_date",
    "sell_time",
    "buy_price",
    "sell_price",
    "gain_pct",
    "hold_days",
    "hold_bars",
    "exit_reason",
    "peak_price",
    "trail_wide_used",
    "adv_20",
    "atr_pct",
    "rs_spy_63d",
    "rs_spy_126d",
    "slope_pct_per_bar",
    "channel_width_pct",
    "channel_pos",
    "room_to_resist_pct",
    "bars_span",
    "entry_mode",
    "shakeout_rebuy",
    "resist_break",
    "shakeout_breakout",
    "shakeout_inside_bars",
    "parent_exit_reason",
    "wait_bars",
    "rs_spy_21d",
    "max_beyond_width",
    "channel_span_days",
    "channel_age_at_buy_days",
]


def _export_trade_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in REPORT_COLS if c in df.columns]
    extra = [c for c in list(FEATURE_COLS) + ["feature_asof"] if c in df.columns and c not in cols]
    if "gain_pct_net" in df.columns and "gain_pct_net" not in cols:
        cols = cols + ["gain_pct_net"]
    return cols + extra


def _parse_beyond_width_sweep(raw: str) -> List[Optional[float]]:
    """Always include None (off); then unique parsed floats."""
    out: List[Optional[float]] = [None]
    text = (raw or "").strip()
    if not text:
        return out
    seen = {None}
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        val = float(p)
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _beyond_width_ab(
    trades: pd.DataFrame,
    *,
    thresholds: Sequence[Optional[float]],
    require_in_channel: bool,
    max_channel_span_days: Optional[float],
    max_channel_age_days: Optional[float],
    max_entries_per_day: int,
    friction_pct: float,
    min_adv: Optional[float] = None,
    min_atr_pct: Optional[float] = None,
    geo_kwargs: Optional[dict] = None,
    spy_regime: bool = False,
) -> pd.DataFrame:
    """Filter-then-RS A/B for max_beyond_width on one already-enriched trade frame."""
    rows = []
    geo_kwargs = geo_kwargs or {}
    for thresh in thresholds:
        filtered = filter_trades(
            trades,
            min_adv=min_adv,
            min_atr_pct=min_atr_pct,
            require_in_channel=bool(require_in_channel),
            max_channel_span_days=max_channel_span_days,
            max_channel_age_days=max_channel_age_days,
            require_spy_above_sma=bool(spy_regime),
            max_beyond_width=thresh,
            **geo_kwargs,
        )
        if max_entries_per_day and max_entries_per_day > 0:
            filtered = select_same_day_rs(
                filtered, rs_col="rs_spy_126d", max_per_day=int(max_entries_per_day)
            )
        gain_col = "gain_pct"
        if friction_pct and friction_pct > 0:
            filtered = apply_friction(filtered, friction_pct)
            gain_col = "gain_pct_net"
        s = _summarize(filtered, gain_col=gain_col)
        label = "off" if thresh is None else str(thresh)
        rows.append({"max_beyond_width": label, **s})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel bottom-touch long backtest")
    ap.add_argument(
        "--preset",
        default="",
        choices=("", "15m"),
        help="15m = IB 15m scaled hunt (shorter pivots, vol-scaled %%, session RS). Daily defaults unchanged.",
    )
    ap.add_argument("--n-symbols", type=int, default=100)
    ap.add_argument(
        "--all-symbols",
        action="store_true",
        help="Scan full provider/timeframe universe (fast DISTINCT list; skips liquidity HAVING)",
    )
    ap.add_argument("--symbols", default="", help="Comma list override (e.g. GLD,SPY)")
    ap.add_argument(
        "--entry-touch",
        type=int,
        default=3,
        help="First bottom-touch number to buy (pivot: detector touches; "
        "l3_touch: 3=L3 after H2, 4=L4 after L3 leaves the rail)",
    )
    ap.add_argument("--stop-pct", type=float, default=0.03)
    ap.add_argument("--trail-pct", type=float, default=0.10)
    ap.add_argument(
        "--trail-pct-wide",
        type=float,
        default=0.18,
        help="Wider trail when TTM Squeeze momentum is strong (with --squeeze-adaptive)",
    )
    ap.add_argument(
        "--squeeze-adaptive",
        action="store_true",
        help="Widen trail when TTM Squeeze mom is +rising and strong vs recent pctile",
    )
    ap.add_argument("--squeeze-pctile", type=float, default=75.0, help="Mom strength percentile threshold")
    ap.add_argument("--squeeze-lookback", type=int, default=100, help="Bars for mom percentile window")
    ap.add_argument("--pivot-len", type=int, default=15)
    ap.add_argument("--error-pct", type=float, default=1.2)
    ap.add_argument(
        "--touch-error-pct",
        type=float,
        default=None,
        help="L3/tag tolerance %% of price (default = --error-pct). 0 requires the bar "
        "range to intersect support (no near-miss). Does not change detector pivot fitting.",
    )
    ap.add_argument("--min-rally-pct", type=float, default=4.0)
    ap.add_argument("--min-pullback-pct", type=float, default=3.0)
    ap.add_argument("--min-total-rise-pct", type=float, default=3.0)
    ap.add_argument("--flat-pct", type=float, default=0.04)
    ap.add_argument("--min-bars-apart", type=int, default=15)
    ap.add_argument("--max-low-pivots", type=int, default=16)
    ap.add_argument(
        "--window-bars",
        type=int,
        default=0,
        help="If >0, slide find_channels over overlapping windows. Daily long-history default "
        "is applied automatically (~504 bars) unless --no-window-scan.",
    )
    ap.add_argument("--window-step-bars", type=int, default=0)
    ap.add_argument(
        "--no-window-scan",
        action="store_true",
        help="Daily only: disable automatic sliding-window channel scan (legacy last-16-pivot pass)",
    )
    ap.add_argument("--adv-lookback", type=int, default=20)
    ap.add_argument(
        "--bars-per-session",
        type=int,
        default=1,
        help="Multiply RS 63/126 session lookbacks by this (26 on RTH 15m)",
    )
    ap.add_argument(
        "--include-time",
        action="store_true",
        help="Write buy_time/sell_time (HH:MM) for intraday bars",
    )
    ap.add_argument(
        "--rs-symbol",
        default="SPY",
        help="Benchmark for relative-strength ranking (default SPY)",
    )
    ap.add_argument("--provider", default="ALPACA")
    ap.add_argument(
        "--fallback-provider",
        default="",
        help="Secondary provider for daily prefix stitch (e.g. IB). Empty=disabled until IB 1d backfilled.",
    )
    ap.add_argument(
        "--merge-mode",
        choices=("", "prefix", "none"),
        default="prefix",
        help="How to combine fallback when set: prefix=IB bars before first Alpaca bar",
    )
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", default="2018-11-01")
    ap.add_argument("--end", default="2026-08-23")
    ap.add_argument("--min-bars", type=int, default=180)
    ap.add_argument("--workers", type=int, default=1, help="Process workers for scan/backtest")
    ap.add_argument("--load-workers", type=int, default=4, help="Thread workers for OHLCV load")
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--min-adv", type=float, default=None, help="Filter: min 20d ADV dollars")
    ap.add_argument("--min-atr-pct", type=float, default=None, help="Filter: min ATR%% of price")
    ap.add_argument(
        "--max-entries-per-day",
        type=int,
        default=0,
        help="If >0, keep top-N same-day entries by RS vs SPY (126d)",
    )
    ap.add_argument(
        "--friction-pct",
        type=float,
        default=0.0,
        help="Round-trip friction %% deducted from each trade gain (e.g. 0.25)",
    )
    ap.add_argument(
        "--edge-improve",
        action="store_true",
        help="Write liquidity/vol/RS/friction scenario comparison table",
    )
    ap.add_argument(
        "--edge-v2",
        action="store_true",
        help="Run long-history H0-H5 improvement matrix + year splits (one data load)",
    )
    ap.add_argument(
        "--geometry-filter",
        action="store_true",
        help="Keep entries in lower 40%% of channel with moderate width/slope",
    )
    ap.add_argument(
        "--require-in-channel",
        action="store_true",
        help="Reject entries outside the rails (channel_pos < 0 or > 1)",
    )
    ap.add_argument(
        "--max-beyond-width",
        type=float,
        default=None,
        help="Reject trades whose max (high-resist)/width from first support touch "
        "through entry exceeds this (0=no pierce; 1=one extra channel above)",
    )
    ap.add_argument(
        "--beyond-width-sweep",
        default="0,0.25,0.5,1.0",
        help="Comma list of max-beyond-width caps to A/B after in-channel/span (empty=skip). "
        "Always includes an off row.",
    )
    ap.add_argument(
        "--entry-features",
        dest="entry_features",
        action="store_true",
        default=True,
        help="Snapshot RSI/MA/vol/squeeze/calendar features at entry (default on)",
    )
    ap.add_argument(
        "--no-entry-features",
        dest="entry_features",
        action="store_false",
        help="Skip per-symbol entry feature snapshot",
    )
    ap.add_argument(
        "--max-channel-span-days",
        type=float,
        default=None,
        help="Reject channels whose start->end span exceeds N calendar days",
    )
    ap.add_argument(
        "--max-channel-age-days",
        type=float,
        default=None,
        help="Reject entries where buy_date - channel_start exceeds N calendar days",
    )
    ap.add_argument(
        "--spy-regime",
        action="store_true",
        help="Only take entries when SPY close > SMA50",
    )
    ap.add_argument(
        "--entry-mode",
        choices=("pivot", "reclaim", "l3_touch"),
        default="pivot",
        help="pivot=buy at touch+pivot_len close; l3_touch=arm at H2, buy Nth support tag via --entry-touch (+slip)",
    )
    ap.add_argument(
        "--entry-slip-pct",
        type=float,
        default=0.001,
        help="l3_touch: slippage added to support limit fill (0.001=0.1%%)",
    )
    ap.add_argument(
        "--max-l3-wait-bars",
        type=int,
        default=252,
        help="l3_touch: max bars after H2 print to wait for a support tag",
    )
    ap.add_argument(
        "--min-l3-wait-bars",
        type=int,
        default=1,
        help="l3_touch: abort if support is tagged before N bars after H2 (not a swing L3)",
    )
    ap.add_argument(
        "--shakeout-rebuy-bars",
        type=int,
        default=0,
        help="l3_touch: after L3, if price closes through support and reclaims "
        "from above within N bars, emit a second fill (rebuy, not hold-through). 0=off.",
    )
    ap.add_argument(
        "--h2-resist-break",
        action="store_true",
        help="l3_touch: fill a close above resistance after H2 (breakout) instead of cancelling",
    )
    ap.add_argument(
        "--h2-resist-break-only",
        action="store_true",
        help="l3_touch: drop L3 support-tag fills before occupancy (live H2 resist-break book)",
    )
    ap.add_argument(
        "--shakeout-breakout",
        action="store_true",
        help="l3_touch: after first H2 resist-break, fill a later close-above-resistance "
        "if price came back inside the channel and support held (rebuy, not hold-through)",
    )
    ap.add_argument(
        "--shakeout-breakout-min-inside",
        type=int,
        default=1,
        help="shakeout-breakout: require N closes back inside the channel before the rebreak (default 1)",
    )
    ap.add_argument(
        "--shakeout-breakout-hard-stop",
        action="store_true",
        help="shakeout-breakout: only take the extra fill if the previous trade was a hard stop",
    )
    ap.add_argument(
        "--realistic-fill",
        action="store_true",
        help="Use realistic purchase prices. 15m default is signal-bar close; "
        "pass --realistic-fill-mode next-mid for the old next-bar mid. "
        "1d default is the 15m close that printed X; next-mid blends X with the following 15m mid; "
        "open-cross fills at the close of the first 15m that opens above resist.",
    )
    ap.add_argument(
        "--realistic-fill-mode",
        choices=("signal-close", "next-mid", "open-cross"),
        default=DEFAULT_FILL_MODE,
        help="When --realistic-fill: signal-close (default) fills at the touch bar close; "
        "next-mid keeps the previous next-bar mid purchaser; "
        "open-cross (1d) waits for a 15m open above resist and buys that bar's close.",
    )
    ap.add_argument(
        "--max-low-to-mid-pct",
        type=float,
        default=DEFAULT_MAX_LOW_TO_MID_PCT,
        help="Realistic fill: cancel when (mid-low)/mid of the exec 15m bar exceeds this (default 0.005)",
    )
    ap.add_argument(
        "--max-chase-pct",
        type=float,
        default=None,
        help="Realistic fill: optional cap on (next_mid - signal)/signal. Off if omitted.",
    )
    ap.add_argument(
        "--no-causal-h2",
        action="store_true",
        help="l3_touch: allow highs after the first H2 to refit width (full-series look-ahead; old batch)",
    )
    ap.add_argument(
        "--max-rsi",
        type=float,
        default=None,
        help="Reject entries with rsi_14 above this (filter then RS)",
    )
    ap.add_argument(
        "--min-close-loc",
        type=float,
        default=None,
        help="Keep entries whose close_loc >= this (0.5=prior completed bar closed in upper half "
        "when --feature-asof prior-bar). Not the fill bar's close on a wick fill.",
    )
    ap.add_argument(
        "--intraday-fill",
        default="",
        choices=("", "15m"),
        help="Daily l3_touch: first IB 15m from-above tag that session. Skip symbols with no 15m. "
        "Nightly/pivot unchanged (default off).",
    )
    ap.add_argument(
        "--feature-asof",
        default="auto",
        choices=("auto", "prior-bar", "entry-bar"),
        help="Stock feature snapshot: prior-bar=last completed bar before fill (no fill-bar close); "
        "entry-bar=fill bar (leaky for wick fills); auto=prior-bar for l3_touch 15m/hybrid.",
    )
    ap.add_argument("--atr-stop-mult", type=float, default=None)
    ap.add_argument("--stop-pct-floor", type=float, default=0.015)
    ap.add_argument("--stop-pct-ceil", type=float, default=0.06)
    ap.add_argument("--resist-exit", action="store_true")
    ap.add_argument(
        "--resist-arm-trail",
        action="store_true",
        help="Alias for --peak-trail-mode fixed (from-entry peak trail; ATR/squeeze off).",
    )
    ap.add_argument(
        "--peak-trail-mode",
        default="off",
        choices=("off", "fixed", "time_decay", "gain_tighten"),
        help="From-entry peak trail: fixed | time_decay | gain_tighten (ATR/squeeze off).",
    )
    ap.add_argument("--trail-floor", type=float, default=0.01, help="Min trail width for dynamic peak trail")
    ap.add_argument(
        "--trail-decay-per-bar",
        type=float,
        default=0.0002,
        help="time_decay: subtract this width each bar (0.0002 = 0.02pp)",
    )
    ap.add_argument(
        "--trail-tighten-per-pct",
        type=float,
        default=0.0033,
        help="gain_tighten: subtract this width per full +1%% from entry (0.0033 = 0.33pp)",
    )
    ap.add_argument(
        "--peak-trail-sweep",
        action="store_true",
        help="One-load A/B: Keeper + F125/F200/F300/T4d/G4t peak-trail exits (15m L3 stack).",
    )
    ap.add_argument("--trail-pct-tight", type=float, default=None)
    ap.add_argument("--squeeze-fade-tighten", action="store_true")
    ap.add_argument("--max-hold-days", type=int, default=None)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
        help="Base report folder; a YYYY-MM-DD subfolder is created automatically",
    )
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)
    intraday_fill = (args.intraday_fill or "").strip().lower()
    feat_asof_mode = (args.feature_asof or "auto").strip().lower()
    use_prior_bar = feat_asof_mode == "prior-bar" or (
        feat_asof_mode == "auto"
        and str(args.entry_mode) == "l3_touch"
        and (
            (args.preset or "").strip() == "15m"
            or str(args.timeframe) == "15m"
            or intraday_fill == "15m"
        )
    )
    if feat_asof_mode == "entry-bar":
        use_prior_bar = False
    if (args.preset or "").strip() == "15m":
        overlay_preset(args, PRESET_15M, sys.argv[1:])
        logger.info(
            "Preset 15m: provider=%s timeframe=%s pivot_len=%d window=%d/%d "
            "trail=%.2f%%/%.2f%% stop_clamp=%.2f-%.2f%% rs_bars_per_session=%d span_days=%s",
            args.provider,
            args.timeframe,
            args.pivot_len,
            args.window_bars,
            args.window_step_bars,
            args.trail_pct * 100,
            args.trail_pct_wide * 100,
            args.stop_pct_floor * 100,
            args.stop_pct_ceil * 100,
            args.bars_per_session,
            args.max_channel_span_days,
        )
        if args.edge_v2:
            logger.warning("Ignoring --edge-v2 with --preset 15m (daily H0-H5 matrix)")
            args.edge_v2 = False

    if bool(getattr(args, "resist_arm_trail", False)) and str(
        getattr(args, "peak_trail_mode", "off")
    ).strip().lower() in ("", "off"):
        args.peak_trail_mode = "fixed"
    pt_mode = _normalize_peak_trail_mode(
        getattr(args, "peak_trail_mode", "off"),
        resist_arm_trail=bool(getattr(args, "resist_arm_trail", False)),
    )
    args.peak_trail_mode = pt_mode
    if pt_mode != "off":
        args.atr_stop_mult = None
        args.squeeze_adaptive = False
        args.squeeze_fade_tighten = False
        args.resist_exit = False
        logger.info(
            "peak_trail_mode=%s stop_pct=%.2f%% trail_pct=%.2f%% floor=%.2f%% "
            "decay/bar=%.4f tighten/%%=%.4f (ATR/squeeze exit overlays off)",
            pt_mode,
            float(args.stop_pct) * 100.0,
            float(args.trail_pct) * 100.0,
            float(args.trail_floor) * 100.0,
            float(args.trail_decay_per_bar),
            float(args.trail_tighten_per_pct),
        )

    apply_daily_long_history_defaults(args)
    if str(args.timeframe) == "1d" and int(args.window_bars) > 0:
        logger.info(
            "Daily windowed channel scan: window_bars=%d window_step_bars=%d",
            int(args.window_bars),
            int(args.window_step_bars or args.window_bars),
        )

    t0 = time.perf_counter()
    t_sym = time.perf_counter()
    pick_provider = args.provider
    pick_tf = args.timeframe
    if intraday_fill == "15m" and not args.symbols.strip() and not args.all_symbols:
        pick_provider = "IB"
        pick_tf = "15m"
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.all_symbols:
        symbols = list_symbols_fast(args.provider, args.timeframe)
        logger.info("Full universe list: %d symbols (%.1fs)", len(symbols), time.perf_counter() - t_sym)
    else:
        symbols = _pick_symbols(args.n_symbols, pick_provider, pick_tf, args.min_bars)
        if "GLD" not in symbols and str(args.timeframe) == "1d" and intraday_fill != "15m":
            symbols = ["GLD"] + symbols

    # Always include RS benchmark (SPY unless --rs-symbol)
    rs_symbol = (args.rs_symbol or "SPY").strip().upper()
    args.rs_symbol = rs_symbol
    if rs_symbol not in symbols:
        symbols = list(symbols) + [rs_symbol]

    logger.info(
        "Backtest %d symbols | entry touch>=%d | stop=%.1f%% trail=%.1f%% wide=%.1f%% adaptive=%s | workers=%d load_workers=%d",
        len(symbols),
        args.entry_touch,
        args.stop_pct * 100,
        args.trail_pct * 100,
        args.trail_pct_wide * 100,
        bool(args.squeeze_adaptive),
        args.workers,
        args.load_workers,
    )
    t_load = time.perf_counter()
    fb = (args.fallback_provider or "").strip()
    merge = (args.merge_mode or "").strip().lower()
    if merge in ("", "none"):
        fb = ""
        merge = None
    panels = load_ohlcv_many(
        symbols,
        timeframe=args.timeframe,
        provider=args.provider,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        use_cache=True,
        chunk_size=args.chunk_size,
        workers=max(1, int(args.load_workers)),
        fallback_provider=fb or None,
        merge_mode=merge,
    )
    logger.info(
        "Loaded %d/%d panels in %.1fs",
        len(panels),
        len(symbols),
        time.perf_counter() - t_load,
    )

    need_15m_purchase = intraday_fill == "15m" or (
        bool(args.realistic_fill) and str(args.timeframe) == "1d"
    )
    panels_15m: Dict[str, pd.DataFrame] = {}
    n_skip_15m = 0
    if need_15m_purchase:
        t_15 = time.perf_counter()
        names_15 = sorted({s.upper() for s in symbols})
        panels_15m = load_ohlcv_many(
            names_15,
            timeframe="15m",
            provider="IB",
            start=datetime.strptime(args.start, "%Y-%m-%d"),
            end=datetime.strptime(args.end, "%Y-%m-%d"),
            use_cache=True,
            chunk_size=args.chunk_size,
            workers=max(1, int(args.load_workers)),
        )
        have_15 = {s for s, df in panels_15m.items() if df is not None and not df.empty}
        n_skip_15m = sum(1 for s in symbols if s != rs_symbol and s not in have_15)
        logger.info(
            "IB 15m fill panels %d/%d in %.1fs (skip no-15m=%d; intersection only, no prior-day fallback)",
            len(have_15),
            len(names_15),
            time.perf_counter() - t_15,
            n_skip_15m,
        )

    spy_df = panels.get(rs_symbol)
    rs_panels = panels
    rs_bars_per_session = int(args.bars_per_session)
    rs_source = f"{args.provider} {args.timeframe}"
    if spy_df is None or spy_df.empty:
        logger.warning(
            "%s %s missing for RS; falling back to ALPACA 1d (ingest IB 15m %s for session-equivalent RS)",
            rs_symbol,
            args.timeframe,
            rs_symbol,
        )
        t_rs_load = time.perf_counter()
        rs_names = sorted({s.upper() for s in symbols} | {rs_symbol})
        rs_panels = load_ohlcv_many(
            rs_names,
            timeframe="1d",
            provider="ALPACA",
            start=datetime.strptime(args.start, "%Y-%m-%d"),
            end=datetime.strptime(args.end, "%Y-%m-%d"),
            use_cache=True,
            chunk_size=50,
            workers=max(1, int(args.load_workers)),
        )
        spy_df = rs_panels.get(rs_symbol)
        rs_bars_per_session = 1
        rs_source = "ALPACA 1d fallback"
        logger.info(
            "RS fallback loaded %d/%d daily panels in %.1fs",
            len(rs_panels),
            len(rs_names),
            time.perf_counter() - t_rs_load,
        )
        if spy_df is None or spy_df.empty:
            logger.error("%s panel missing; cannot compute relative strength", rs_symbol)
            return 1

    if (
        intraday_fill == "15m"
        and rs_symbol in panels_15m
        and panels_15m[rs_symbol] is not None
        and not panels_15m[rs_symbol].empty
    ):
        spy_df = panels_15m[rs_symbol]
        rs_panels = panels_15m
        rs_bars_per_session = 26
        rs_source = "IB 15m (completed bar before fill)"
        logger.info("Hybrid RS: IB 15m %s, bars_per_session=26, as-of last completed 15m", rs_symbol)

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.edge_v2:
        friction = args.friction_pct if args.friction_pct > 0 else 0.25
        max_day = args.max_entries_per_day if args.max_entries_per_day > 0 else 1
        summary_df, year_df, raw_by_name = run_edge_v2(
            panels,
            spy_df,
            symbols,
            workers=int(args.workers),
            friction_pct=friction,
            max_entries_per_day=max_day,
            squeeze_pctile=args.squeeze_pctile,
            squeeze_lookback=args.squeeze_lookback,
        )
        scenarios_csv = args.outdir / f"channel_touch_edge_v2_{stamp}.csv"
        year_csv = args.outdir / f"channel_touch_edge_v2_years_{stamp}.csv"
        summary_df.to_csv(scenarios_csv, index=False)
        year_df.to_csv(year_csv, index=False)
        h0 = raw_by_name.get("H0_baseline", pd.DataFrame())
        trades_csv = args.outdir / f"channel_touch_trades_{stamp}.csv"
        summary_txt = args.outdir / f"channel_touch_trades_summary_{stamp}.txt"
        if not h0.empty:
            filtered, gain_col = _finalize_report_trades(
                h0, max_entries_per_day=max_day, friction_pct=friction
            )
            export_cols = [c for c in REPORT_COLS if c in filtered.columns]
            if "gain_pct_net" in filtered.columns:
                export_cols = export_cols + ["gain_pct_net"]
            filtered = filtered.sort_values(
                [gain_col, "buy_date"], ascending=[False, True]
            ).reset_index(drop=True)
            filtered[export_cols].to_csv(trades_csv, index=False)
            s = _summarize(filtered, gain_col=gain_col)
            y = summarize_by_year(filtered, gain_col=gain_col)
            lines = [
                "Ascending channel bottom-touch long backtest (edge-v2 H0 baseline)",
                f"start={args.start} end={args.end}",
                f"friction_pct={friction} max_entries_per_day={max_day}",
                f"elapsed_sec={time.perf_counter() - t0:.1f}",
                "",
                *[f"{k}={v}" for k, v in s.items()],
                "",
                "Year splits:",
                y.to_string(index=False) if not y.empty else "(none)",
                "",
                "Edge-v2 scenarios:",
                summary_df.to_string(index=False),
            ]
            summary_txt.write_text("\n".join(lines), encoding="utf-8")
        else:
            pd.DataFrame().to_csv(trades_csv, index=False)
            summary_txt.write_text("No H0 trades\n", encoding="utf-8")
        logger.info("edge-v2 scenarios -> %s", scenarios_csv)
        print("\nEdge-v2 scenario summary:")
        print(summary_df.to_string(index=False))
        print("\nYear splits:")
        print(year_df.to_string(index=False))
        print(f"\nScenarios CSV: {scenarios_csv}")
        print(f"Years CSV: {year_csv}")
        print(f"H0 trades: {trades_csv}")
        print(f"Summary: {summary_txt}")
        return 0

    if bool(getattr(args, "peak_trail_sweep", False)):
        friction = float(args.friction_pct) if args.friction_pct > 0 else float(PRESET_15M["friction_pct"])
        max_day = int(args.max_entries_per_day) if args.max_entries_per_day > 0 else 1
        channel_kwargs = {
            "error_pct": float(args.error_pct),
            "flat_pct": float(args.flat_pct),
            "min_bars_apart": int(args.min_bars_apart),
            "min_intervening_rally_pct": float(args.min_rally_pct),
            "min_intervening_pullback_pct": float(args.min_pullback_pct),
            "min_total_rise_pct": float(args.min_total_rise_pct),
            "max_low_pivots": int(args.max_low_pivots),
            "causal_h2": not bool(args.no_causal_h2),
        }
        summary_df, year_df, raw_by_name = run_peak_trail_sweep(
            panels,
            spy_df,
            symbols,
            workers=int(args.workers),
            friction_pct=friction,
            max_entries_per_day=max_day,
            require_in_channel=bool(args.require_in_channel),
            max_channel_span_days=args.max_channel_span_days,
            bars_per_session=int(args.bars_per_session),
            channel_kwargs=channel_kwargs,
        )
        scenarios_csv = args.outdir / f"channel_touch_peak_trail_sweep_{stamp}.csv"
        year_csv = args.outdir / f"channel_touch_peak_trail_sweep_years_{stamp}.csv"
        summary_txt = args.outdir / f"channel_touch_peak_trail_sweep_summary_{stamp}.txt"
        summary_df.to_csv(scenarios_csv, index=False)
        year_df.to_csv(year_csv, index=False)
        lines = [
            "Peak-trail exit sweep (Keeper + F125/F200/F300/T4d/G4t)",
            f"start={args.start} end={args.end} friction_pct={friction} max_entries_per_day={max_day}",
            f"require_in_channel={args.require_in_channel} max_channel_span_days={args.max_channel_span_days}",
            f"elapsed_sec={time.perf_counter() - t0:.1f}",
            "",
            summary_df.to_string(index=False),
            "",
            "Year splits:",
            year_df.to_string(index=False) if not year_df.empty else "(none)",
        ]
        summary_txt.write_text("\n".join(lines), encoding="utf-8")
        logger.info("peak-trail sweep -> %s", scenarios_csv)
        print("\nPeak-trail sweep summary:")
        print(summary_df.to_string(index=False))
        print("\nYear splits:")
        print(year_df.to_string(index=False) if not year_df.empty else "(none)")
        print(f"\nScenarios CSV: {scenarios_csv}")
        print(f"Years CSV: {year_csv}")
        print(f"Summary: {summary_txt}")
        return 0

    t_scan = time.perf_counter()
    all_trades: List[dict] = []
    channel_kwargs = {
        "error_pct": float(args.error_pct),
        "flat_pct": float(args.flat_pct),
        "min_bars_apart": int(args.min_bars_apart),
        "min_intervening_rally_pct": float(args.min_rally_pct),
        "min_intervening_pullback_pct": float(args.min_pullback_pct),
        "min_total_rise_pct": float(args.min_total_rise_pct),
        "max_low_pivots": int(args.max_low_pivots),
        "causal_h2": not bool(args.no_causal_h2),
    }
    window_bars = int(args.window_bars) if int(args.window_bars) > 0 else None
    window_step = int(args.window_step_bars) if int(args.window_step_bars) > 0 else window_bars
    payloads = [
        {
            "symbol": sym,
            "df": panels[sym],
            "entry_touch": args.entry_touch,
            "stop_pct": args.stop_pct,
            "trail_pct": args.trail_pct,
            "trail_pct_wide": args.trail_pct_wide,
            "squeeze_adaptive": bool(args.squeeze_adaptive),
            "squeeze_pctile": args.squeeze_pctile,
            "squeeze_lookback": args.squeeze_lookback,
            "pivot_len": args.pivot_len,
            "entry_mode": str(args.entry_mode),
            "atr_stop_mult": args.atr_stop_mult,
            "stop_pct_floor": float(args.stop_pct_floor),
            "stop_pct_ceil": float(args.stop_pct_ceil),
            "resist_exit": bool(args.resist_exit),
            "resist_arm_trail": bool(args.resist_arm_trail) or str(args.peak_trail_mode) != "off",
            "peak_trail_mode": str(args.peak_trail_mode),
            "trail_floor": float(args.trail_floor),
            "trail_decay_per_bar": float(args.trail_decay_per_bar),
            "trail_tighten_per_pct": float(args.trail_tighten_per_pct),
            "trail_pct_tight": args.trail_pct_tight,
            "squeeze_fade_tighten": bool(args.squeeze_fade_tighten),
            "max_hold_days": args.max_hold_days,
            "adv_lookback": int(args.adv_lookback),
            "window_bars": window_bars,
            "window_step_bars": window_step,
            "include_time": bool(args.include_time),
            "entry_features": bool(args.entry_features),
            "entry_slip_pct": float(args.entry_slip_pct),
            "max_l3_wait_bars": int(args.max_l3_wait_bars),
            "min_l3_wait_bars": int(args.min_l3_wait_bars),
            "shakeout_rebuy_bars": int(args.shakeout_rebuy_bars),
            "h2_resist_break": bool(args.h2_resist_break),
            "h2_resist_break_only": bool(args.h2_resist_break_only),
            "shakeout_breakout": bool(args.shakeout_breakout),
            "shakeout_breakout_min_inside": int(args.shakeout_breakout_min_inside),
            "shakeout_breakout_hard_stop": bool(args.shakeout_breakout_hard_stop),
            "df_15m": panels_15m.get(sym) if need_15m_purchase else None,
            "intraday_fill": intraday_fill,
            "feature_asof_prior_bar": bool(use_prior_bar),
            "realistic_fill": bool(args.realistic_fill),
            "realistic_fill_mode": str(args.realistic_fill_mode),
            "max_low_to_mid_pct": args.max_low_to_mid_pct,
            "max_chase_pct": args.max_chase_pct,
            "touch_error_pct": args.touch_error_pct,
            "channel_kwargs": channel_kwargs,
        }
        for sym in symbols
        if sym in panels and sym != rs_symbol
        and (
            not need_15m_purchase
            or (sym in panels_15m and panels_15m[sym] is not None and not panels_15m[sym].empty)
        )
    ]
    if need_15m_purchase:
        logger.info(
            "15m purchase panels: %d symbols with daily+15m (skipped %d without 15m) realistic=%s hybrid=%s",
            len(payloads),
            n_skip_15m,
            bool(args.realistic_fill),
            intraday_fill == "15m",
        )
    if args.workers and args.workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = [pool.submit(_worker_symbol_trades, p) for p in payloads]
            for fut in as_completed(futs):
                all_trades.extend(fut.result())
    else:
        for p in payloads:
            all_trades.extend(_worker_symbol_trades(p))
    logger.info("Scan+trade done in %.1fs (%d trades)", time.perf_counter() - t_scan, len(all_trades))

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_prefix = (
        "channel_touch_hybrid"
        if intraday_fill == "15m"
        else ("channel_touch_15m" if (args.preset or "").strip() == "15m" else "channel_touch")
    )
    trades_csv = args.outdir / f"{file_prefix}_trades_{stamp}.csv"
    raw_csv = args.outdir / f"{file_prefix}_trades_raw_{stamp}.csv"
    summary_txt = args.outdir / f"{file_prefix}_trades_summary_{stamp}.txt"
    scenarios_csv = args.outdir / f"{file_prefix}_edge_scenarios_{stamp}.csv"
    beyond_csv = args.outdir / f"{file_prefix}_beyond_width_ab_{stamp}.csv"

    if not all_trades:
        logger.warning("No trades generated")
        pd.DataFrame().to_csv(trades_csv, index=False)
        summary_txt.write_text("No trades\n", encoding="utf-8")
        print("No trades")
        return 0

    trades = pd.DataFrame(all_trades)
    t_rs = time.perf_counter()
    rs_lookbacks = (21, 63, 126) if bool(args.entry_features) else (63, 126)
    trades = enrich_rs(
        trades,
        rs_panels,
        spy_df,
        lookbacks=rs_lookbacks,
        bars_per_session=int(rs_bars_per_session),
    )
    if bool(args.entry_features):
        trades = enrich_spy_entry_features(trades, spy_df)
    logger.info("RS enrichment done in %.1fs", time.perf_counter() - t_rs)

    raw_export_cols = _export_trade_columns(trades)
    trades[[c for c in raw_export_cols if c in trades.columns]].to_csv(raw_csv, index=False)
    logger.info("Raw trades (pre quality/RS-top1) -> %s (%d rows)", raw_csv, len(trades))

    # Optional live filters for the primary report
    geo_kwargs = {}
    if args.geometry_filter:
        geo_kwargs.update(
            dict(
                max_channel_pos=0.40,
                min_width_pct=3.0,
                max_width_pct=35.0,
                min_slope_pct=0.02,
                max_slope_pct=0.50,
            )
        )
    filtered = filter_trades(
        trades,
        min_adv=args.min_adv,
        min_atr_pct=args.min_atr_pct,
        require_in_channel=bool(args.require_in_channel),
        max_channel_span_days=args.max_channel_span_days,
        max_channel_age_days=args.max_channel_age_days,
        require_spy_above_sma=bool(args.spy_regime),
        max_beyond_width=args.max_beyond_width,
        max_rsi=args.max_rsi,
        min_close_loc=args.min_close_loc,
        **geo_kwargs,
    )
    if args.max_entries_per_day and args.max_entries_per_day > 0:
        filtered = select_same_day_rs(
            filtered, rs_col="rs_spy_126d", max_per_day=int(args.max_entries_per_day)
        )
    if args.friction_pct and args.friction_pct > 0:
        filtered = apply_friction(filtered, args.friction_pct)
        gain_col = "gain_pct_net"
    else:
        gain_col = "gain_pct"

    filtered = filtered.sort_values(
        [gain_col if gain_col in filtered.columns else "gain_pct", "buy_date"],
        ascending=[False, True],
    ).reset_index(drop=True)

    export_cols = _export_trade_columns(filtered)
    filtered[export_cols].to_csv(trades_csv, index=False)

    summary = _summarize(filtered, gain_col=gain_col if gain_col in filtered.columns else "gain_pct")
    elapsed = time.perf_counter() - t0
    lines = [
        "Ascending channel bottom-touch long backtest",
        f"entry_touch>={args.entry_touch}",
        f"stop_pct={args.stop_pct}",
        f"trail_pct={args.trail_pct}",
        f"trail_pct_wide={args.trail_pct_wide}",
        f"squeeze_adaptive={args.squeeze_adaptive}",
        f"squeeze_pctile={args.squeeze_pctile}",
        f"squeeze_lookback={args.squeeze_lookback}",
        f"pivot_len={args.pivot_len} entry_mode={args.entry_mode} entry_slip_pct={args.entry_slip_pct}",
        f"preset={args.preset or 'daily'} window_bars={args.window_bars} window_step_bars={args.window_step_bars}",
        f"error_pct={args.error_pct} touch_error_pct={args.touch_error_pct} "
        f"min_rally_pct={args.min_rally_pct} min_total_rise_pct={args.min_total_rise_pct}",
        f"provider={args.provider} timeframe={args.timeframe}",
        f"fallback_provider={fb or ''}",
        f"merge_mode={merge or ''}",
        f"start={args.start} end={args.end}",
        f"symbols={len(panels)}",
        f"min_adv={args.min_adv}",
        f"min_atr_pct={args.min_atr_pct}",
        f"max_entries_per_day={args.max_entries_per_day}",
        f"friction_pct={args.friction_pct}",
        f"atr_stop_mult={args.atr_stop_mult}",
        f"bars_per_session={rs_bars_per_session} rs_source={rs_source} rs_symbol={rs_symbol}",
        f"require_in_channel={args.require_in_channel} max_channel_span_days={args.max_channel_span_days}",
        f"max_beyond_width={args.max_beyond_width} max_rsi={args.max_rsi} min_l3_wait_bars={args.min_l3_wait_bars} shakeout_rebuy_bars={args.shakeout_rebuy_bars} h2_resist_break={bool(args.h2_resist_break)} h2_resist_break_only={bool(args.h2_resist_break_only)} shakeout_breakout={bool(args.shakeout_breakout)} shakeout_breakout_min_inside={args.shakeout_breakout_min_inside} shakeout_breakout_hard_stop={bool(args.shakeout_breakout_hard_stop)} entry_features={bool(args.entry_features)}",
        f"intraday_fill={intraday_fill or 'off'} feature_asof={'prior-bar' if use_prior_bar else 'entry-bar'}",
        "no_buy_below_channel=True (re-entry through support or resist-break)",
        f"realistic_fill={bool(args.realistic_fill)} realistic_fill_mode={args.realistic_fill_mode}",
        f"elapsed_sec={elapsed:.1f}",
        "",
        *[f"{k}={v}" for k, v in summary.items()],
        "",
        "Exit: hard stop = entry*(1-stop); trail = peak*(1-trail); fill at max(hard,trail) when low hits",
        "Features: adv_20 = 20d mean(close*volume); atr_pct = ATR14/close*100; rs_spy_Nd = stock_ret - SPY_ret",
    ]
    summary_txt.write_text("\n".join(lines), encoding="utf-8")

    scenario_df = None
    if args.edge_improve:
        scenario_df = edge_scenarios(trades, friction_pcts=(0.10, 0.25))
        scenario_df.to_csv(scenarios_csv, index=False)
        logger.info("Edge scenarios -> %s", scenarios_csv)

    beyond_df = None
    sweep_spec = (args.beyond_width_sweep or "").strip()
    if "max_beyond_width" in trades.columns:
        thresh_list = _parse_beyond_width_sweep(sweep_spec)
        if len(thresh_list) > 1 or (len(thresh_list) == 1 and args.max_beyond_width is not None):
            beyond_df = _beyond_width_ab(
                trades,
                thresholds=thresh_list,
                require_in_channel=bool(args.require_in_channel),
                max_channel_span_days=args.max_channel_span_days,
                max_channel_age_days=args.max_channel_age_days,
                max_entries_per_day=int(args.max_entries_per_day or 0),
                friction_pct=float(args.friction_pct or 0.0),
                min_adv=args.min_adv,
                min_atr_pct=args.min_atr_pct,
                geo_kwargs=geo_kwargs,
                spy_regime=bool(args.spy_regime),
            )
            beyond_df.to_csv(beyond_csv, index=False)
            logger.info("Beyond-width A/B -> %s", beyond_csv)

    logger.info("Wrote %d trades -> %s", len(filtered), trades_csv)
    logger.info("Summary -> %s", summary_txt)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if scenario_df is not None and not scenario_df.empty:
        print("\nEdge improvement scenarios:")
        print(scenario_df.to_string(index=False))
        print(f"\nScenarios CSV: {scenarios_csv}")
    print("\nTop 20 trades by gain %:")
    show_cols = [c for c in export_cols if c in filtered.columns]
    print(filtered[show_cols].head(20).to_string(index=False))
    gld = filtered[filtered["stock"] == "GLD"]
    if not gld.empty:
        print("\nGLD trades:")
        print(gld[show_cols].to_string(index=False))
    print(f"\nFull trades: {trades_csv}")
    print(f"Raw trades: {raw_csv}")
    print(f"Summary: {summary_txt}")
    if beyond_df is not None and not beyond_df.empty:
        print("\nBeyond-width A/B (filter then RS top-N):")
        print(beyond_df.to_string(index=False))
        print(f"Beyond-width CSV: {beyond_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
