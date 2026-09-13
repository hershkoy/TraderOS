"""Causal VCP AVWAP breakout (Playbook #1) with realistic 15m fills.

Daily VCP / 52-week / AVWAP / ATR from completed bars through the prior
session. Buy the first RTH 15m **close** above the pivot and AVWAP with
RVOL confirmation (signal-close). Sells: ATR hard stop or weekly close
below the 10-week SMA, fill the **next** RTH 15m mid (overnight OK).

This is a research translation of ``strategies/vcp_avwap_breakout.py``.
There is no stored 1h universe; 15m close-cross is the 1h trigger analog.
Not a promote.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.realistic_exits import (
    EXIT_EOD,
    EXIT_HARD_STOP,
    ExitResult,
    REASON_EOD,
    REASON_FILLED,
    REASON_NO_15M,
    REASON_NO_ENTRY_BAR,
    REASON_NO_NEXT_BAR,
)
from utils.research.realistic_purchaser import (
    FILL_MODE_SIGNAL_CLOSE,
    PurchaseResult,
    bar_mid,
    normalize_fill_mode,
    purchase_after_close_signal,
    purchase_at_signal_close,
)
from utils.research.weekly_deepvue import occupancy_take

EXIT_WEEKLY_MA = "weekly_ma"
EXIT_AVWAP_STOP = "avwap_stop"

DEFAULT_VCP_MIN_WEEKS = 6
DEFAULT_VCP_MAX_WEEKS = 12
DEFAULT_VCP_MAX_DEPTH_PCT = 25.0
DEFAULT_VCP_MIN_CONTRACTIONS = 2
DEFAULT_VCP_VOLUME_DRYUP_WINDOW = 20
DEFAULT_ZIGZAG_PCT = 3.0
DEFAULT_NEAR_52W = 0.92
DEFAULT_BREAKOUT_RVOL = 1.5
DEFAULT_RVOL_WINDOW_DAYS = 20
DEFAULT_BARS_PER_DAY_15M = 26
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_MULT_STOP = 1.0
DEFAULT_WEEKLY_MA_PERIOD = 10
DEFAULT_TAKE_PROFIT_EXT_MULT = 0.25


@dataclass(frozen=True)
class VcpParams:
    vcp_min_weeks: int = DEFAULT_VCP_MIN_WEEKS
    vcp_max_weeks: int = DEFAULT_VCP_MAX_WEEKS
    vcp_max_depth_pct: float = DEFAULT_VCP_MAX_DEPTH_PCT
    vcp_min_contractions: int = DEFAULT_VCP_MIN_CONTRACTIONS
    vcp_volume_dryup_window: int = DEFAULT_VCP_VOLUME_DRYUP_WINDOW
    zigzag_pct: float = DEFAULT_ZIGZAG_PCT
    near_52w_threshold: float = DEFAULT_NEAR_52W
    breakout_rvol_min: float = DEFAULT_BREAKOUT_RVOL
    rvol_window_days: int = DEFAULT_RVOL_WINDOW_DAYS
    bars_per_day: int = DEFAULT_BARS_PER_DAY_15M
    atr_period_days: int = DEFAULT_ATR_PERIOD
    atr_mult_stop: float = DEFAULT_ATR_MULT_STOP
    weekly_ma_period: int = DEFAULT_WEEKLY_MA_PERIOD
    take_profit_ext_mult: float = DEFAULT_TAKE_PROFIT_EXT_MULT
    fill_mode: str = FILL_MODE_SIGNAL_CLOSE


def zigzag_swings(closes: Sequence[float], pct: float = 0.03) -> List[dict]:
    """Percent-threshold zigzag matching ``VcpAvwapBreakoutStrategy._zigzag_swings``."""
    swings: List[dict] = []
    if not closes:
        return swings
    last_pivot_idx = 0
    last_pivot_price = float(closes[0])
    trend = None
    n = len(closes)
    for i in range(1, n):
        px = float(closes[i])
        if last_pivot_price == 0:
            continue
        change = (px - last_pivot_price) / last_pivot_price
        if trend in (None, "down"):
            if change >= pct:
                swings.append(
                    {"idx": last_pivot_idx, "price": last_pivot_price, "type": "L"}
                )
                trend = "up"
                last_pivot_idx = i
                last_pivot_price = px
            elif change <= -pct:
                last_pivot_idx = i
                last_pivot_price = px
        if trend in (None, "up"):
            if change <= -pct:
                swings.append(
                    {"idx": last_pivot_idx, "price": last_pivot_price, "type": "H"}
                )
                trend = "down"
                last_pivot_idx = i
                last_pivot_price = px
            elif change >= pct:
                last_pivot_idx = i
                last_pivot_price = px
    swings.append(
        {
            "idx": last_pivot_idx,
            "price": last_pivot_price,
            "type": "H" if trend == "up" else "L",
        }
    )
    return swings


def volume_dryup(vols: Sequence[float], window: int = 20) -> bool:
    if len(vols) < int(window) * 2:
        return True
    first = float(np.mean(vols[:window]))
    last = float(np.mean(vols[-window:]))
    return last < first


def pullback_depths(swings: Sequence[Mapping[str, Any]]) -> List[float]:
    out: List[float] = []
    for k in range(1, len(swings)):
        prev = swings[k - 1]
        curr = swings[k]
        if prev.get("type") == "H" and curr.get("type") == "L":
            hp = float(prev["price"])
            if hp <= 0:
                continue
            depth = (hp - float(curr["price"])) / hp * 100.0
            out.append(-abs(depth))
    return out


def contractions_ok(
    pullbacks: Sequence[float],
    min_contractions: int,
) -> bool:
    if len(pullbacks) < int(min_contractions):
        return False
    for i in range(len(pullbacks) - 1):
        if abs(pullbacks[i]) <= abs(pullbacks[i + 1]):
            return False
    return True


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


def anchored_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    start_i: int,
    end_i: int,
) -> float:
    if start_i < 0 or end_i < start_i or end_i >= len(close):
        return float("nan")
    typical = (high[start_i : end_i + 1] + low[start_i : end_i + 1] + close[start_i : end_i + 1]) / 3.0
    vol = volume[start_i : end_i + 1].astype(float)
    pv = typical * vol
    denom = float(np.nansum(vol))
    if denom <= 0:
        return float("nan")
    return float(np.nansum(pv) / denom)


def highest_n(high: np.ndarray, i: int, lookback: int = 252) -> float:
    lo = max(0, int(i) - int(lookback) + 1)
    if i < 0 or lo > i:
        return float("nan")
    return float(np.nanmax(high[lo : i + 1]))


def weekly_sma_by_date(close: pd.Series, period: int = 10) -> Dict[date, float]:
    """Completed W-FRI SMA, ffilled onto each daily session date."""
    c = close.astype(float).copy()
    if not isinstance(c.index, pd.DatetimeIndex):
        c.index = pd.to_datetime(c.index)
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    weekly = c.resample("W-FRI").last().dropna()
    sma = weekly.rolling(int(period), min_periods=int(period)).mean()
    daily = sma.reindex(c.index, method="ffill")
    out: Dict[date, float] = {}
    for ts, val in daily.items():
        if val != val:
            continue
        out[pd.Timestamp(ts).date()] = float(val)
    return out


def weekly_close_below_ma(
    close: pd.Series,
    sma_by_date: Mapping[date, float],
) -> Dict[date, bool]:
    out: Dict[date, bool] = {}
    for ts, px in close.astype(float).items():
        d = pd.Timestamp(ts).date()
        ma = sma_by_date.get(d)
        if px != px or ma is None or ma != ma:
            continue
        out[d] = bool(float(px) < float(ma))
    return out


def find_vcp_base_at(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    end_i: int,
    params: VcpParams,
) -> Optional[dict]:
    """Longest valid 6-12 week VCP ending at ``end_i`` (inclusive, completed)."""
    min_days = int(params.vcp_min_weeks) * 5
    max_days = int(params.vcp_max_weeks) * 5
    if end_i < max(252, max_days + 10):
        return None
    zigzag_frac = float(params.zigzag_pct) / 100.0
    for days in range(max_days, min_days - 1, -1):
        start_i = end_i - days + 1
        if start_i < 0:
            continue
        closes = [float(x) for x in close[start_i : end_i + 1]]
        if len(closes) < days:
            continue
        highs = high[start_i : end_i + 1]
        lows = low[start_i : end_i + 1]
        vols = volume[start_i : end_i + 1]
        swings = zigzag_swings(closes, pct=zigzag_frac)
        if len(swings) < 3:
            continue
        pbs = pullback_depths(swings)
        if not contractions_ok(pbs, params.vcp_min_contractions):
            continue
        peak = float(np.nanmax(highs))
        trough = float(np.nanmin(lows))
        if peak <= 0:
            continue
        depth_pct = (peak - trough) / peak * 100.0
        if depth_pct > float(params.vcp_max_depth_pct):
            continue
        if not volume_dryup(
            [float(x) for x in vols],
            window=int(params.vcp_volume_dryup_window),
        ):
            continue
        avwap = anchored_vwap(high, low, close, volume, start_i, end_i)
        return {
            "start_i": int(start_i),
            "end_i": int(end_i),
            "pivot_price": peak,
            "depth_pct": float(depth_pct),
            "pullbacks": pbs,
            "avwap": float(avwap),
        }
    return None


def arm_sessions(
    daily: pd.DataFrame,
    params: VcpParams,
) -> List[dict]:
    """Watch-days: VCP + 52w + AVWAP + ATR from the **prior** completed session."""
    if daily is None or daily.empty:
        return []
    df = daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    need = ("open", "high", "low", "close", "volume")
    for col in need:
        if col not in df.columns:
            return []
    close = df["close"].astype(float).to_numpy()
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    dates = [pd.Timestamp(ts).date() for ts in df.index]
    atr = wilder_atr(high, low, close, length=int(params.atr_period_days))
    n = len(close)
    rows: List[dict] = []
    for i in range(n - 1):
        hi_52 = highest_n(high, i, 252)
        px = float(close[i])
        if hi_52 != hi_52 or hi_52 <= 0 or px != px or px <= 0:
            continue
        if (px / hi_52) < float(params.near_52w_threshold):
            continue
        base = find_vcp_base_at(close, high, low, volume, i, params)
        if base is None:
            continue
        pivot = float(base["pivot_price"])
        if px > pivot:
            continue
        atr_i = float(atr[i]) if atr[i] == atr[i] else float("nan")
        rows.append(
            {
                "signal_date": dates[i],
                "watch_date": dates[i + 1],
                "pivot_price": float(base["pivot_price"]),
                "avwap": float(base["avwap"]),
                "depth_pct": float(base["depth_pct"]),
                "atr": atr_i,
                "close": px,
                "high_52w": float(hi_52),
                "near_52w": px / hi_52,
                "base_start": dates[int(base["start_i"])],
                "base_end": dates[int(base["end_i"])],
                "base_start_i": int(base["start_i"]),
                "base_end_i": int(base["end_i"]),
            }
        )
    return rows


def _bar_px(bar: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _trailing_rvol(
    vol_now: float,
    prior_vols: Sequence[float],
) -> float:
    if not prior_vols:
        return float("nan")
    avg = float(np.mean(prior_vols))
    if avg <= 0:
        return float("nan")
    return float(vol_now) / avg


def index_rth_15m_fast(df: pd.DataFrame, *, naive_tz: str = "UTC") -> Dict[date, List[dict]]:
    """Vectorized RTH 15m session index (avoids per-bar ZoneInfo in as_et)."""
    if df is None or df.empty:
        return {}
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize(str(naive_tz))
    et = idx.tz_convert("America/New_York")
    hour = et.hour
    minute = et.minute
    rth = (
        (et.weekday < 5)
        & (minute % 15 == 0)
        & (
            ((hour == 9) & (minute >= 30))
            | ((hour > 9) & (hour < 15))
            | ((hour == 15) & (minute <= 45))
        )
    )
    if not bool(np.any(rth)):
        return {}
    sub = df.loc[rth]
    et_sub = et[rth]
    opens = sub["open"].to_numpy(dtype=float, copy=False)
    highs = sub["high"].to_numpy(dtype=float, copy=False)
    lows = sub["low"].to_numpy(dtype=float, copy=False)
    closes = sub["close"].to_numpy(dtype=float, copy=False)
    vols = sub["volume"].to_numpy(dtype=float, copy=False) if "volume" in sub.columns else np.ones(len(sub))
    ts_list = et_sub.to_pydatetime()
    dates = et_sub.date
    out: Dict[date, List[dict]] = {}
    for i in range(len(sub)):
        sess = pd.Timestamp(dates[i]).date()
        bar = {
            "ts": ts_list[i],
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(vols[i]),
        }
        bucket = out.get(sess)
        if bucket is None:
            out[sess] = [bar]
        else:
            bucket.append(bar)
    return out


def flatten_session_volumes(
    by_day: Mapping[date, Sequence[dict]],
    before: date,
) -> List[float]:
    out: List[float] = []
    for day in sorted(by_day.keys()):
        if day >= before:
            break
        for bar in by_day.get(day) or []:
            v = _bar_px(bar, "volume")
            if v is not None:
                out.append(v)
    return out


def find_vcp_breakout_fill(
    bars: Sequence[dict],
    *,
    pivot: float,
    avwap: float,
    rvol_min: float,
    prior_volumes: Sequence[float],
    rvol_window_bars: int,
    fill_mode: str = FILL_MODE_SIGNAL_CLOSE,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    """First RTH 15m close above pivot and AVWAP with RVOL; fill by ``fill_mode``."""
    empty = PurchaseResult(filled=False, reason="no_close_cross", signal_px=pivot)
    try:
        pivot_f = float(pivot)
        avwap_f = float(avwap)
        rvol_need = float(rvol_min)
    except (TypeError, ValueError):
        return PurchaseResult(filled=False, reason="bad_signal")
    if pivot_f != pivot_f or pivot_f <= 0:
        return PurchaseResult(filled=False, reason="bad_signal")
    if avwap_f != avwap_f:
        return empty
    mode = normalize_fill_mode(fill_mode)
    win = max(2, int(rvol_window_bars))
    hist = [float(x) for x in prior_volumes[-win:]]
    if not bars:
        return empty
    for i, bar in enumerate(bars):
        close_px = _bar_px(bar, "close")
        vol = _bar_px(bar, "volume")
        opened = _bar_px(bar, "open")
        high = _bar_px(bar, "high")
        low = _bar_px(bar, "low")
        if close_px is None or vol is None or opened is None or high is None or low is None:
            if vol is not None:
                hist.append(vol)
                if len(hist) > win:
                    hist = hist[-win:]
            continue
        rvol = _trailing_rvol(vol, hist[-win:] if len(hist) >= 2 else hist)
        hist.append(vol)
        if len(hist) > win * 2:
            hist = hist[-win:]
        if close_px <= pivot_f:
            continue
        if close_px <= avwap_f:
            continue
        if rvol != rvol or rvol < rvol_need:
            continue
        next_bar = bars[i + 1] if i + 1 < len(bars) else None
        if mode == FILL_MODE_SIGNAL_CLOSE:
            inner = purchase_at_signal_close(bar, naive_tz=naive_tz)
        else:
            inner = purchase_after_close_signal(bar, next_bar, naive_tz=naive_tz)
        if not inner.filled:
            return inner
        return inner
    return empty


def _clock_tuple(ts: Any) -> Optional[tuple]:
    if ts is None:
        return None
    t = ts
    if hasattr(t, "to_pydatetime"):
        try:
            t = t.to_pydatetime()
        except (TypeError, ValueError):
            t = ts
    if not isinstance(t, datetime):
        try:
            t = pd.Timestamp(ts).to_pydatetime()
        except (TypeError, ValueError):
            return None
    return (int(t.year), int(t.month), int(t.day), int(t.hour), int(t.minute))


def _index_by_clock(bars: Sequence[dict], ts: Any) -> Optional[int]:
    want = _clock_tuple(ts)
    if want is None:
        return None
    for i, bar in enumerate(bars):
        got = _clock_tuple(bar.get("ts"))
        if got is not None and got == want:
            return i
    return None


def _is_last_rth_clock(ts: Any) -> bool:
    if ts is None:
        return False
    try:
        return int(ts.hour) == 15 and int(ts.minute) == 45
    except (AttributeError, TypeError, ValueError):
        clock = _clock_tuple(ts)
        return bool(clock is not None and clock[3] == 15 and clock[4] == 45)


def simulate_vcp_exit(
    by_day: Dict[date, Sequence[dict]],
    *,
    entry_ts: Any,
    entry_px: float,
    atr_at_entry: Optional[float],
    atr_mult: float = DEFAULT_ATR_MULT_STOP,
    weekly_below: Mapping[date, bool],
    avwap_by_prior_session: Mapping[date, float],
) -> ExitResult:
    """ATR*k stop or Friday weekly-MA loss; fill next 15m mid. No peak trail.

    AVWAP loss of control (15m close below AVWAP) raises the stop to
    ``avwap - 0.5 * ATR``; fill is still the next 15m mid after the tag.
    """
    bars: List[dict] = []
    sess_of: List[date] = []
    for sess_d in sorted(by_day.keys()):
        day_bars = list(by_day.get(sess_d) or [])
        day_bars.sort(key=lambda b: b.get("ts"))
        for bar in day_bars:
            bars.append(bar)
            sess_of.append(sess_d)
    px = float(entry_px)
    if not bars:
        return ExitResult(filled=False, reason=REASON_NO_15M)
    if px != px or px <= 0:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_i = _index_by_clock(bars, entry_ts)
    if entry_i is None:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    atr = float(atr_at_entry) if atr_at_entry is not None else float("nan")
    if atr == atr and atr > 0:
        hard = px - float(atr_mult) * atr
    else:
        hard = px * 0.97
    last_i = len(bars) - 1
    if entry_i >= last_i:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            hard_stop=hard,
            peak_px=px,
        )
    peak = px
    stop = hard
    sessions = sorted(by_day.keys())
    prior_of: Dict[date, Optional[date]] = {}
    prev: Optional[date] = None
    for sess_d in sessions:
        prior_of[sess_d] = prev
        prev = sess_d

    for i in range(entry_i + 1, len(bars)):
        hi = _bar_px(bars[i], "high")
        lo = _bar_px(bars[i], "low")
        cl = _bar_px(bars[i], "close")
        if hi is not None:
            peak = max(peak, hi)
        ts = bars[i].get("ts")
        sess = sess_of[i]
        if cl is not None:
            prior = prior_of.get(sess)
            avwap = avwap_by_prior_session.get(prior) if prior is not None else None
            if avwap is not None and avwap == avwap and cl < float(avwap):
                tighten = float(avwap)
                if atr == atr and atr > 0:
                    tighten = float(avwap) - 0.5 * atr
                if tighten > stop:
                    stop = tighten
        if lo is not None and lo <= stop + 1e-12:
            reason = EXIT_AVWAP_STOP if stop > hard + 1e-9 else EXIT_HARD_STOP
            return _next_mid(
                bars,
                decision_i=i,
                peak=peak,
                hard=hard,
                stop=stop,
                exit_reason=reason,
            )
        if _is_last_rth_clock(ts):
            if bool(weekly_below.get(sess, False)):
                return _next_mid(
                    bars,
                    decision_i=i,
                    peak=peak,
                    hard=hard,
                    stop=stop,
                    exit_reason=EXIT_WEEKLY_MA,
                )
    last = bars[-1]
    cl = _bar_px(last, "close")
    if cl is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    return ExitResult(
        filled=True,
        reason=REASON_EOD,
        sell_px=float(cl),
        decision_ts=last.get("ts"),
        exec_bar_ts=last.get("ts"),
        exit_reason=EXIT_EOD,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=stop,
    )


def avwap_series(
    daily: pd.DataFrame,
    start_i: int,
) -> Dict[date, float]:
    """AVWAP from ``start_i`` through each later session (inclusive)."""
    df = daily
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    volume = df["volume"].astype(float).to_numpy()
    dates = [pd.Timestamp(ts).date() for ts in df.index]
    out: Dict[date, float] = {}
    for i in range(max(0, int(start_i)), len(close)):
        val = anchored_vwap(high, low, close, volume, int(start_i), i)
        if val == val:
            out[dates[i]] = val
    return out


def _next_mid(
    bars: Sequence[dict],
    *,
    decision_i: int,
    peak: float,
    hard: float,
    stop: float,
    exit_reason: str,
) -> ExitResult:
    nxt_i = int(decision_i) + 1
    if nxt_i >= len(bars):
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            exit_reason=exit_reason,
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    nxt = bars[nxt_i]
    hi = _bar_px(nxt, "high")
    lo = _bar_px(nxt, "low")
    if hi is None or lo is None:
        return ExitResult(
            filled=False,
            reason=REASON_NO_NEXT_BAR,
            decision_ts=bars[decision_i].get("ts"),
            peak_px=peak,
            hard_stop=hard,
            stop_at_decision=stop,
        )
    return ExitResult(
        filled=True,
        reason=REASON_FILLED,
        sell_px=float(bar_mid(hi, lo)),
        decision_ts=bars[decision_i].get("ts"),
        exec_bar_ts=nxt.get("ts"),
        exit_reason=exit_reason,
        peak_px=peak,
        hard_stop=hard,
        stop_at_decision=stop,
    )


def rvol_window_bars(params: VcpParams) -> int:
    return int(params.rvol_window_days) * int(params.bars_per_day)


__all__ = [
    "EXIT_AVWAP_STOP",
    "EXIT_WEEKLY_MA",
    "VcpParams",
    "arm_sessions",
    "avwap_series",
    "contractions_ok",
    "find_vcp_base_at",
    "find_vcp_breakout_fill",
    "flatten_session_volumes",
    "index_rth_15m_fast",
    "occupancy_take",
    "pullback_depths",
    "rvol_window_bars",
    "simulate_vcp_exit",
    "volume_dryup",
    "weekly_close_below_ma",
    "weekly_sma_by_date",
    "wilder_atr",
    "zigzag_swings",
]
