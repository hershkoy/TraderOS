"""Causal Cup & Handle swing claim-check (Cashflow Academy / Noah Davidson).

Daily bars only. Arm on completed cup+handle (prior bars), buy-stop fill when
high clears the rim at max(open, rim). Claim exit: 1.5x ATR trail + 50/50
scale-out at 1/2 and 2/3 cup depth. Robustness exit: ATR k=2 clamp + 10% trail.

Classical geometry (Edwards/Magee/Bulkowski-inspired): U-cup not V, depth and
duration bounds, handle shorter/shallower in the upper half. Original code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.roc_rebound import YEAR_BUCKETS, book_from_trades, occupancy_fcfs
from utils.scanning.channel_touch_bought import (
    ATR_STOP_MULT,
    TRAIL_PCT,
    current_stop_price,
    hard_stop_price,
)

EXIT_CLAIM = "claim"
EXIT_KEEPER = "keeper"

SECTOR_TO_ETF: Dict[str, str] = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


@dataclass(frozen=True)
class CupParams:
    sma_fast: int = 20
    sma_mid: int = 50
    sma_slow: int = 200
    sma_slope_lookback: int = 20
    min_cup_bars: int = 20
    max_cup_bars: int = 65
    min_cup_depth_pct: float = 12.0
    max_cup_depth_pct: float = 35.0
    rim_tolerance_pct: float = 3.0
    min_handle_bars: int = 5
    max_handle_bars: int = 25
    max_handle_depth_frac: float = 0.50
    near_high_lookback: int = 252
    near_high_pct: float = 0.05
    vol_ma: int = 50
    atr_period: int = 14
    claim_atr_mult: float = 1.5
    t1_cup_frac: float = 0.50
    t2_cup_frac: float = 2.0 / 3.0
    max_arm_wait_bars: int = 15
    v_side_min_frac: float = 0.25
    require_recovery_vol: bool = True
    require_handle_vol_dryup: bool = True
    exit_mode: str = EXIT_CLAIM


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    return np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    ).astype(float)


def wilder_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    length: int = 14,
) -> np.ndarray:
    tr = _true_range(high, low, close)
    atr = np.full(len(close), np.nan, dtype=float)
    n = int(length)
    if len(close) < n:
        return atr
    atr[n - 1] = float(np.nanmean(tr[:n]))
    alpha = 1.0 / float(n)
    for i in range(n, len(close)):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _sma(arr: np.ndarray, length: int) -> np.ndarray:
    n = int(length)
    out = np.full(len(arr), np.nan, dtype=float)
    if n < 1 or len(arr) < n:
        return out
    csum = np.cumsum(np.nan_to_num(arr, nan=0.0))
    csum[n:] = csum[n:] - csum[:-n]
    out[n - 1 :] = csum[n - 1 :] / float(n)
    # invalidate where any NaN in window
    nan_c = np.cumsum(np.isnan(arr).astype(float))
    nan_c[n:] = nan_c[n:] - nan_c[:-n]
    out[n - 1 :] = np.where(nan_c[n - 1 :] > 0, np.nan, out[n - 1 :])
    return out


def trend_ok(
    close: np.ndarray,
    i: int,
    params: CupParams,
    sma20: Optional[np.ndarray] = None,
    sma50: Optional[np.ndarray] = None,
    sma200: Optional[np.ndarray] = None,
) -> bool:
    """Price above SMA20/50/200 with rising SMA50."""
    if i < int(params.sma_slow) - 1:
        return False
    px = float(close[i])
    if not np.isfinite(px) or px <= 0:
        return False
    s20 = sma20 if sma20 is not None else _sma(close, params.sma_fast)
    s50 = sma50 if sma50 is not None else _sma(close, params.sma_mid)
    s200 = sma200 if sma200 is not None else _sma(close, params.sma_slow)
    a20, a50, a200 = float(s20[i]), float(s50[i]), float(s200[i])
    if not all(np.isfinite(x) for x in (a20, a50, a200)):
        return False
    if not (px > a20 and px > a50 and px > a200):
        return False
    j = i - int(params.sma_slope_lookback)
    if j < 0 or not np.isfinite(float(s50[j])):
        return False
    return float(s50[i]) > float(s50[j])


def _is_rounded_cup(
    lows: np.ndarray,
    trough_i: int,
    cup_start: int,
    cup_end: int,
    *,
    side_min_frac: float,
) -> bool:
    """Reject V-bottoms: trough in the middle and both sides have length."""
    span = int(cup_end) - int(cup_start)
    if span < 4:
        return False
    left = int(trough_i) - int(cup_start)
    right = int(cup_end) - int(trough_i)
    if left < 1 or right < 1:
        return False
    min_side = max(2, int(round(span * float(side_min_frac))))
    if left < min_side or right < min_side:
        return False
    # Mid-cup lows should stay near the trough (not a single spike).
    mid_lo = max(cup_start, trough_i - max(1, left // 3))
    mid_hi = min(cup_end, trough_i + max(1, right // 3))
    window = lows[mid_lo : mid_hi + 1]
    if len(window) < 2:
        return False
    trough_px = float(lows[trough_i])
    if trough_px <= 0:
        return False
    avg_mid = float(np.nanmean(window))
    if not np.isfinite(avg_mid):
        return False
    # Average mid lows within 8% of trough (rounded), else V-spike.
    return avg_mid <= trough_px * 1.08


def find_cup_handle_at(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    end_i: int,
    params: CupParams,
    *,
    sma20: Optional[np.ndarray] = None,
    sma50: Optional[np.ndarray] = None,
    sma200: Optional[np.ndarray] = None,
    vol_ma: Optional[np.ndarray] = None,
) -> Optional[dict]:
    """Find a cup+handle completing at ``end_i`` (inclusive, completed bars only)."""
    i = int(end_i)
    n = len(close)
    if i < 0 or i >= n:
        return None
    if not trend_ok(close, i, params, sma20=sma20, sma50=sma50, sma200=sma200):
        return None

    lo_hi = max(0, i - int(params.near_high_lookback) + 1)
    prior_high = float(np.nanmax(high[lo_hi : i + 1]))
    if prior_high <= 0:
        return None
    if float(close[i]) < prior_high * (1.0 - float(params.near_high_pct) - 0.03):
        return None

    vma = vol_ma if vol_ma is not None else _sma(volume.astype(float), params.vol_ma)
    vol_f = volume.astype(float)

    # Longer cups first; return first valid (classical preference).
    for cup_len in range(int(params.max_cup_bars), int(params.min_cup_bars) - 1, -1):
        for handle_len in range(int(params.min_handle_bars), int(params.max_handle_bars) + 1):
            handle_start = i - handle_len + 1
            if handle_start < 1:
                continue
            cup_start = handle_start - cup_len
            if cup_start < 0:
                continue
            cup_end = handle_start - 1
            if cup_end <= cup_start:
                continue

            trough_rel = int(np.nanargmin(low[cup_start : cup_end + 1]))
            trough_i = cup_start + trough_rel
            if not _is_rounded_cup(
                low,
                trough_i,
                cup_start,
                cup_end,
                side_min_frac=params.v_side_min_frac,
            ):
                continue

            left_rim_i = cup_start + int(np.nanargmax(high[cup_start : trough_i + 1]))
            right_rim_i = trough_i + int(np.nanargmax(high[trough_i : cup_end + 1]))
            left_rim = float(high[left_rim_i])
            right_rim = float(high[right_rim_i])
            if left_rim <= 0 or right_rim <= 0:
                continue
            rim_ref = max(left_rim, right_rim)
            if abs(left_rim - right_rim) / rim_ref * 100.0 > float(params.rim_tolerance_pct):
                continue

            trough_px = float(low[trough_i])
            if trough_px <= 0 or rim_ref <= trough_px:
                continue
            cup_depth = rim_ref - trough_px
            cup_depth_pct = cup_depth / rim_ref * 100.0
            if cup_depth_pct < float(params.min_cup_depth_pct):
                continue
            if cup_depth_pct > float(params.max_cup_depth_pct):
                continue

            handle_high = float(np.nanmax(high[handle_start : i + 1]))
            handle_low = float(np.nanmin(low[handle_start : i + 1]))
            if not np.isfinite(handle_high) or not np.isfinite(handle_low):
                continue
            entry_rim = handle_high
            if entry_rim <= 0:
                continue
            if entry_rim < prior_high * (1.0 - float(params.near_high_pct)):
                continue
            if handle_low < trough_px + 0.5 * cup_depth:
                continue
            if entry_rim - handle_low > float(params.max_handle_depth_frac) * cup_depth:
                continue
            if float(close[i]) >= entry_rim:
                continue

            if params.require_recovery_vol and i >= int(params.vol_ma):
                rec_lo = max(trough_i, right_rim_i - 5)
                rec_hi = min(cup_end, right_rim_i + 5)
                rec_vols = vol_f[rec_lo : rec_hi + 1]
                ma_slice = vma[rec_lo : rec_hi + 1]
                if len(rec_vols) == 0 or not np.any(
                    (rec_vols > ma_slice) & np.isfinite(ma_slice)
                ):
                    continue

            if params.require_handle_vol_dryup:
                cup_rec = vol_f[trough_i : cup_end + 1]
                hand_v = vol_f[handle_start : i + 1]
                if len(cup_rec) >= 3 and len(hand_v) >= 3:
                    if float(np.nanmean(hand_v)) >= float(np.nanmean(cup_rec)):
                        continue

            return {
                "arm_i": i,
                "cup_start_i": int(cup_start),
                "cup_end_i": int(cup_end),
                "trough_i": int(trough_i),
                "left_rim_i": int(left_rim_i),
                "right_rim_i": int(right_rim_i),
                "handle_start_i": int(handle_start),
                "handle_end_i": i,
                "left_rim": left_rim,
                "right_rim": right_rim,
                "rim": float(entry_rim),
                "trough": trough_px,
                "handle_low": handle_low,
                "cup_depth": float(cup_depth),
                "cup_depth_pct": float(cup_depth_pct),
                "cup_bars": int(cup_len),
                "handle_bars": int(handle_len),
                "prior_high": prior_high,
            }
    return None


def _prepare_arrays(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={d.index.name or "index": "ts"})
        else:
            raise ValueError("df needs ts or DatetimeIndex")
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col not in d.columns:
            if col == "volume":
                d[col] = 0.0
            else:
                raise ValueError("missing column %s" % col)
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    close = d["close"].to_numpy(dtype=float)
    high = d["high"].to_numpy(dtype=float)
    low = d["low"].to_numpy(dtype=float)
    volume = d["volume"].to_numpy(dtype=float)
    open_ = d["open"].to_numpy(dtype=float)
    return d, {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "sma20": _sma(close, 20),
        "sma50": _sma(close, 50),
        "sma200": _sma(close, 200),
        "vol_ma": _sma(volume, 50),
        "atr": wilder_atr(high, low, close, length=14),
    }


def arm_sessions(df: pd.DataFrame, params: CupParams) -> List[dict]:
    """Walk completed bars; emit arm rows when a cup+handle completes."""
    if df is None or df.empty:
        return []
    try:
        d, arr = _prepare_arrays(df)
    except ValueError:
        return []
    n = len(d)
    min_i = max(
        int(params.sma_slow),
        int(params.min_cup_bars) + int(params.min_handle_bars),
        int(params.near_high_lookback),
    )
    rows: List[dict] = []
    last_arm = -10**9
    close = arr["close"]
    high = arr["high"]
    for i in range(min_i, n):
        if i - last_arm < int(params.min_handle_bars):
            continue
        if not trend_ok(
            close,
            i,
            params,
            sma20=arr["sma20"],
            sma50=arr["sma50"],
            sma200=arr["sma200"],
        ):
            continue
        lo_hi = max(0, i - int(params.near_high_lookback) + 1)
        prior_high = float(np.nanmax(high[lo_hi : i + 1]))
        if prior_high <= 0 or float(close[i]) < prior_high * (
            1.0 - float(params.near_high_pct) - 0.03
        ):
            continue
        setup = find_cup_handle_at(
            arr["high"],
            arr["low"],
            arr["close"],
            arr["volume"],
            i,
            params,
            sma20=arr["sma20"],
            sma50=arr["sma50"],
            sma200=arr["sma200"],
            vol_ma=arr["vol_ma"],
        )
        if setup is None:
            continue
        atr_i = float(arr["atr"][i]) if np.isfinite(arr["atr"][i]) else float("nan")
        rows.append(
            {
                **setup,
                "arm_date": str(pd.Timestamp(d["ts"].iloc[i]).date()),
                "arm_close": float(arr["close"][i]),
                "atr_at_arm": atr_i,
            }
        )
        last_arm = i
    return rows


def buy_stop_fill(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    start_i: int,
    rim: float,
    handle_low: float,
    max_wait: int,
) -> Optional[dict]:
    """First bar with high >= rim; fill max(open, rim). Cancel if close < handle_low."""
    n = len(close)
    rim_f = float(rim)
    hlo = float(handle_low)
    for j in range(int(start_i), min(n, int(start_i) + int(max_wait))):
        if float(close[j]) < hlo:
            return {"filled": False, "reason": "cancel_handle_break", "i": j}
        if float(high[j]) >= rim_f:
            fill = max(float(open_[j]), rim_f)
            return {
                "filled": True,
                "reason": "buy_stop",
                "i": j,
                "entry_px": float(fill),
                "gap_open": bool(float(open_[j]) > rim_f),
            }
    return {"filled": False, "reason": "timeout", "i": None}


def _scale_out_ret(
    entry_px: float,
    exit_half_px: float,
    exit_rest_px: float,
    *,
    friction_pct: float = 0.0,
) -> float:
    """Equal-share scale-out return in percent (one round-trip friction each leg)."""
    e = float(entry_px)
    if e <= 0:
        return 0.0
    fr = float(friction_pct)
    r1 = (float(exit_half_px) / e - 1.0) * 100.0 - fr
    r2 = (float(exit_rest_px) / e - 1.0) * 100.0 - fr
    return 0.5 * r1 + 0.5 * r2


def simulate_claim_exit(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    entry_i: int,
    entry_px: float,
    atr_at_entry: float,
    cup_depth: float,
    params: CupParams,
    friction_pct: float = 0.0,
) -> Optional[dict]:
    """1.5x ATR trailing stop + 50% at T1 (1/2 depth) + 50% at T2 (2/3 depth)."""
    n = len(close)
    if entry_i < 0 or entry_i >= n - 1:
        return None
    e = float(entry_px)
    atr = float(atr_at_entry)
    if e <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    mult = float(params.claim_atr_mult)
    t1 = e + float(params.t1_cup_frac) * float(cup_depth)
    t2 = e + float(params.t2_cup_frac) * float(cup_depth)
    peak = e
    stop = e - mult * atr
    half_done = False
    exit_half_px: Optional[float] = None
    exit_half_i: Optional[int] = None

    for j in range(entry_i + 1, n):
        # Use prior-bar stop (do not ratchet peak before stop/target checks).
        lo = float(low[j])
        hi = float(high[j])
        op = float(open_[j])
        stop_hit = lo <= stop
        t1_hit = (not half_done) and hi >= t1
        t2_hit = half_done and hi >= t2

        if stop_hit and not half_done:
            # Gap through stop.
            px = min(op, stop) if op < stop else stop
            ret = (px / e - 1.0) * 100.0 - float(friction_pct)
            return {
                "exit_i": j,
                "exit_px": float(px),
                "exit_reason": "atr_trail",
                "ret_pct": float(ret),
                "t1_hit": False,
                "t2_hit": False,
                "peak_px": float(peak),
            }

        if stop_hit and half_done:
            px = min(op, stop) if op < stop else stop
            assert exit_half_px is not None
            ret = _scale_out_ret(e, exit_half_px, px, friction_pct=friction_pct)
            return {
                "exit_i": j,
                "exit_px": float(px),
                "exit_reason": "atr_trail_after_t1",
                "ret_pct": float(ret),
                "t1_hit": True,
                "t2_hit": False,
                "peak_px": float(peak),
                "exit_half_px": float(exit_half_px),
                "exit_half_i": exit_half_i,
            }

        if t1_hit:
            half_done = True
            exit_half_px = t1
            exit_half_i = j
            # Same bar can also hit T2 after T1.
            if hi >= t2:
                peak = max(peak, hi)
                stop = max(stop, peak - mult * atr)
                ret = _scale_out_ret(e, t1, t2, friction_pct=friction_pct)
                return {
                    "exit_i": j,
                    "exit_px": float(t2),
                    "exit_reason": "t1_t2",
                    "ret_pct": float(ret),
                    "t1_hit": True,
                    "t2_hit": True,
                    "peak_px": float(peak),
                    "exit_half_px": float(t1),
                    "exit_half_i": j,
                }
            peak = max(peak, hi)
            stop = max(stop, peak - mult * atr)
            continue

        if t2_hit:
            assert exit_half_px is not None
            peak = max(peak, hi)
            ret = _scale_out_ret(e, exit_half_px, t2, friction_pct=friction_pct)
            return {
                "exit_i": j,
                "exit_px": float(t2),
                "exit_reason": "t2",
                "ret_pct": float(ret),
                "t1_hit": True,
                "t2_hit": True,
                "peak_px": float(peak),
                "exit_half_px": float(exit_half_px),
                "exit_half_i": exit_half_i,
            }

        peak = max(peak, hi)
        stop = max(stop, peak - mult * atr)

    # Open at end: mark last close (exclude from completed if desired by caller).
    return None


def simulate_keeper_exit(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    *,
    entry_i: int,
    entry_px: float,
    atr_at_entry: float,
    friction_pct: float = 0.0,
    atr_stop_mult: float = ATR_STOP_MULT,
    trail_pct: float = TRAIL_PCT,
) -> Optional[dict]:
    """ATR k=2 clamp 1.5%-6% hard + 10% peak trail (channel-touch keeper)."""
    n = len(close)
    if entry_i < 0 or entry_i >= n - 1:
        return None
    e = float(entry_px)
    if e <= 0:
        return None
    hard = hard_stop_price(e, atr_at_entry=atr_at_entry, atr_stop_mult=atr_stop_mult)
    peak = e
    for j in range(entry_i + 1, n):
        hi = float(high[j])
        lo = float(low[j])
        peak = max(peak, hi)
        stop = current_stop_price(hard, peak, trail_pct)
        if lo <= stop:
            px = stop
            ret = (px / e - 1.0) * 100.0 - float(friction_pct)
            return {
                "exit_i": j,
                "exit_px": float(px),
                "exit_reason": "keeper_stop",
                "ret_pct": float(ret),
                "t1_hit": False,
                "t2_hit": False,
                "peak_px": float(peak),
                "hard_stop": float(hard),
            }
    return None


def simulate_cup_handle_trade(
    df: pd.DataFrame,
    setup: Mapping[str, Any],
    params: CupParams,
    *,
    symbol: str = "",
    friction_pct: float = 0.0,
    exit_mode: Optional[str] = None,
) -> Optional[dict]:
    """From an armed setup, attempt buy-stop fill and exit per ``exit_mode``."""
    try:
        d, arr = _prepare_arrays(df)
    except ValueError:
        return None
    arm_i = int(setup["arm_i"])
    rim = float(setup["rim"])
    handle_low = float(setup["handle_low"])
    cup_depth = float(setup["cup_depth"])
    fill = buy_stop_fill(
        arr["open"],
        arr["high"],
        arr["low"],
        arr["close"],
        start_i=arm_i + 1,
        rim=rim,
        handle_low=handle_low,
        max_wait=int(params.max_arm_wait_bars),
    )
    if fill is None or not fill.get("filled"):
        return None
    entry_i = int(fill["i"])
    entry_px = float(fill["entry_px"])
    atr_e = float(arr["atr"][entry_i])
    if not np.isfinite(atr_e) or atr_e <= 0:
        atr_e = float(setup.get("atr_at_arm") or float("nan"))
    if not np.isfinite(atr_e) or atr_e <= 0:
        return None

    mode = str(exit_mode or params.exit_mode or EXIT_CLAIM).lower()
    if mode == EXIT_KEEPER:
        ex = simulate_keeper_exit(
            arr["high"],
            arr["low"],
            arr["close"],
            entry_i=entry_i,
            entry_px=entry_px,
            atr_at_entry=atr_e,
            friction_pct=friction_pct,
        )
    else:
        ex = simulate_claim_exit(
            arr["open"],
            arr["high"],
            arr["low"],
            arr["close"],
            entry_i=entry_i,
            entry_px=entry_px,
            atr_at_entry=atr_e,
            cup_depth=cup_depth,
            params=params,
            friction_pct=friction_pct,
        )
    if ex is None:
        return None

    exit_i = int(ex["exit_i"])
    hold = exit_i - entry_i
    return {
        "symbol": str(symbol).upper(),
        "entry_date": str(pd.Timestamp(d["ts"].iloc[entry_i]).date()),
        "exit_date": str(pd.Timestamp(d["ts"].iloc[exit_i]).date()),
        "entry_px": round(entry_px, 4),
        "exit_px": round(float(ex["exit_px"]), 4),
        "ret_pct": round(float(ex["ret_pct"]), 4),
        "hold_days": int(hold),
        "exit_reason": ex.get("exit_reason"),
        "rim": round(rim, 4),
        "cup_depth": round(cup_depth, 4),
        "cup_depth_pct": round(float(setup.get("cup_depth_pct", 0.0)), 3),
        "arm_date": setup.get("arm_date")
        or str(pd.Timestamp(d["ts"].iloc[arm_i]).date()),
        "atr_at_entry": round(atr_e, 4),
        "exit_mode": mode,
        "t1_hit": bool(ex.get("t1_hit")),
        "t2_hit": bool(ex.get("t2_hit")),
        "gap_open": bool(fill.get("gap_open")),
        "buy_ts": pd.Timestamp(d["ts"].iloc[entry_i]),
        "sell_ts": pd.Timestamp(d["ts"].iloc[exit_i]),
    }


def trades_for_symbol(
    df: pd.DataFrame,
    params: CupParams,
    *,
    symbol: str = "",
    friction_pct: float = 0.0,
    exit_mode: Optional[str] = None,
) -> List[dict]:
    """Arm -> fill -> exit walk; one open position per symbol at a time."""
    arms = arm_sessions(df, params)
    if not arms:
        return []
    try:
        d, _ = _prepare_arrays(df)
    except ValueError:
        return []
    # Map arm_date -> index for chronological skip while open.
    date_to_i = {
        str(pd.Timestamp(ts).date()): i for i, ts in enumerate(d["ts"])
    }
    trades: List[dict] = []
    busy_until_i = -1
    for setup in arms:
        arm_i = int(setup["arm_i"])
        if arm_i <= busy_until_i:
            continue
        tr = simulate_cup_handle_trade(
            df,
            setup,
            params,
            symbol=symbol,
            friction_pct=friction_pct,
            exit_mode=exit_mode,
        )
        if tr is None:
            continue
        exit_i = date_to_i.get(str(tr["exit_date"]))
        if exit_i is None:
            # Fallback: parse from sell_ts
            exit_i = arm_i + int(tr.get("hold_days") or 1)
        busy_until_i = int(exit_i)
        trades.append(tr)
    return trades


def trifecta_ok(
    *,
    arm_date: str,
    spy_close: pd.Series,
    sector_etf_close: Optional[pd.Series],
    sma_len: int = 50,
) -> bool:
    """SPY and sector ETF both above their SMA50 on the arm session (causal)."""
    day = pd.Timestamp(arm_date).normalize()
    for series in (spy_close, sector_etf_close):
        if series is None or series.empty:
            return False
        s = series.astype(float).copy()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).tz_localize(None).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        prior = s[s.index <= day]
        if len(prior) < int(sma_len):
            return False
        window = prior.iloc[-int(sma_len) :]
        ma = float(window.mean())
        px = float(prior.iloc[-1])
        if not np.isfinite(px) or not np.isfinite(ma) or px <= ma:
            return False
    return True


def sector_etf_for(sector: str) -> Optional[str]:
    if not sector:
        return None
    return SECTOR_TO_ETF.get(str(sector).strip())


def max_concurrent(trades: Sequence[Mapping[str, Any]]) -> int:
    if not trades:
        return 0
    events: List[Tuple[pd.Timestamp, int]] = []
    for tr in trades:
        buy = pd.Timestamp(tr["entry_date"]).normalize()
        sell = pd.Timestamp(tr["exit_date"]).normalize()
        events.append((buy, 1))
        events.append((sell, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return int(peak)


def summarize_book(trades: List[dict]) -> Dict[str, Any]:
    base = book_from_trades(trades)
    base["max_concurrent"] = max_concurrent(trades)
    return base


def params_dict(params: CupParams) -> Dict[str, Any]:
    return asdict(params)


__all__ = [
    "CupParams",
    "EXIT_CLAIM",
    "EXIT_KEEPER",
    "SECTOR_TO_ETF",
    "YEAR_BUCKETS",
    "arm_sessions",
    "book_from_trades",
    "buy_stop_fill",
    "find_cup_handle_at",
    "max_concurrent",
    "occupancy_fcfs",
    "params_dict",
    "sector_etf_for",
    "simulate_claim_exit",
    "simulate_cup_handle_trade",
    "simulate_keeper_exit",
    "summarize_book",
    "trades_for_symbol",
    "trend_ok",
    "trifecta_ok",
    "wilder_atr",
]
