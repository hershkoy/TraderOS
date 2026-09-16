"""SMB-style LOD reclaim: 9 EMA cross up through session VWAP.

Intraday long after the session low is in and momentum rebuilds toward VWAP.
Trigger is the first completed 15m where 9 EMA crosses above session VWAP.
Entry and exit use realistic purchasing (not same-bar clips):

- **Buy:** ``next-mid`` (default) or ``signal-close`` of the cross bar.
  Last RTH 15:45 is cancelled (no same-session next bar; no time to work
  the measured move). Video "enter at the cross, do not wait for the
  candle to close" is intra-bar; close-based 9 EMA is only known when the
  15m completes, so signal-close is the earliest causal analog.
- **Sell:** measured-move target / 1/3 stop. Decision on bar N (high/low
  tag). Fill the **next same-session** 15m mid. Overnight is not allowed
  (day trade). Last-bar tag or no tag flattens that bar's close (MOC).
  Same-bar stop+target is the stop (pessimistic).

Not a promote until the book clears E/PF + year-bucket geo + 2022-23.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.realistic_exits import (
    EXIT_EOD,
    ExitResult,
    REASON_EOD,
    REASON_FILLED,
    REASON_NO_15M,
    REASON_NO_ENTRY_BAR,
    REASON_NO_NEXT_BAR,
)
from utils.research.realistic_purchaser import (
    FILL_MODE_NEXT_MID,
    FILL_MODE_SIGNAL_CLOSE,
    PurchaseResult,
    as_et,
    bar_mid,
    normalize_fill_mode,
    purchase_after_close_signal,
    purchase_at_signal_close,
)
from utils.research.vcp_avwap import index_rth_15m_fast
from utils.research.weekly_deepvue import occupancy_take

EXIT_TARGET = "target"
EXIT_STOP = "stop"

EMA_SPAN = 9
STOP_FRACTION = 1.0 / 3.0
LAST_ENTRY_CLOCK = time(15, 30)
DEFAULT_FILL_MODE = FILL_MODE_NEXT_MID
TRADE_COLUMNS = (
    "stock",
    "buy_date",
    "buy_ts",
    "buy_time",
    "buy_price",
    "sell_date",
    "sell_ts",
    "sell_time",
    "sell_price",
    "gain_pct",
    "exit_reason",
    "lod",
    "cross_px",
    "unit",
    "stop_px",
    "target_px",
    "signal_ts",
    "fill_mode",
)


@dataclass(frozen=True)
class ReclaimParams:
    ema_span: int = EMA_SPAN
    stop_fraction: float = STOP_FRACTION
    fill_mode: str = DEFAULT_FILL_MODE
    min_lod_age_bars: int = 1
    last_entry_clock: time = LAST_ENTRY_CLOCK


def _bar_px(bar: Mapping[str, Any], key: str) -> Optional[float]:
    try:
        val = float(bar[key])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _clock_hm(ts: Any) -> Optional[Tuple[int, int]]:
    if ts is None:
        return None
    try:
        et = as_et(ts)
    except (TypeError, ValueError):
        return None
    return (int(et.hour), int(et.minute))


def _is_after_last_entry(ts: Any, last_clock: time) -> bool:
    hm = _clock_hm(ts)
    if hm is None:
        return True
    return (hm[0], hm[1]) > (int(last_clock.hour), int(last_clock.minute))


def session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    """Session VWAP from typical price (H+L+C)/3."""
    tp = (high + low + close) / 3.0
    vol = np.asarray(volume, dtype=float)
    vol = np.where(vol == vol, np.maximum(vol, 0.0), 0.0)
    pv = tp * vol
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(vol)
    out = np.full(len(close), np.nan, dtype=float)
    ok = cum_vol > 0
    out[ok] = cum_pv[ok] / cum_vol[ok]
    return out


def ema_next(prev: Optional[float], close: float, *, span: int = EMA_SPAN) -> float:
    px = float(close)
    if prev is None or prev != prev:
        return px
    n = max(1, int(span))
    alpha = 2.0 / (n + 1.0)
    return alpha * px + (1.0 - alpha) * float(prev)


def measured_move_levels(
    lod: float,
    cross_px: float,
    *,
    stop_fraction: float = STOP_FRACTION,
) -> Optional[Tuple[float, float, float]]:
    """Return (unit, stop, target) from LOD to the cross. None if unit <= 0."""
    try:
        lo = float(lod)
        cr = float(cross_px)
        frac = float(stop_fraction)
    except (TypeError, ValueError):
        return None
    if lo != lo or cr != cr or cr <= lo or frac <= 0:
        return None
    unit = cr - lo
    if unit <= 0:
        return None
    stop = cr - frac * unit
    target = cr + unit
    return (unit, stop, target)


def first_reclaim_index(
    ema: np.ndarray,
    vwap: np.ndarray,
    low: np.ndarray,
    *,
    min_lod_age_bars: int = 1,
) -> Optional[int]:
    """First completed bar where 9 EMA crosses up through VWAP after LOD is in.

    LOD must print on a prior bar (age >= min_lod_age_bars), stay below VWAP,
    and the signal bar must not make a new session low.
    """
    n = len(ema)
    if n < 2 or len(vwap) != n or len(low) != n:
        return None
    age = max(1, int(min_lod_age_bars))
    lod_run = np.minimum.accumulate(low)
    for i in range(1, n):
        e0 = float(ema[i - 1])
        e1 = float(ema[i])
        v0 = float(vwap[i - 1])
        v1 = float(vwap[i])
        lo = float(low[i])
        if e0 != e0 or e1 != e1 or v0 != v0 or v1 != v1 or lo != lo:
            continue
        if not (e0 <= v0 and e1 > v1):
            continue
        lod_before = float(lod_run[i - 1])
        if lod_before != lod_before:
            continue
        if lo < lod_before - 1e-12:
            continue
        lod_i = int(np.argmin(low[:i]))
        if (i - lod_i) < age:
            continue
        if lod_before >= v1 - 1e-12:
            continue
        return i
    return None


def _fill_from_signal(
    bars: Sequence[dict],
    signal_i: int,
    *,
    fill_mode: str,
    naive_tz: str = "UTC",
) -> PurchaseResult:
    mode = normalize_fill_mode(fill_mode)
    bar = bars[signal_i]
    nxt = bars[signal_i + 1] if signal_i + 1 < len(bars) else None
    if mode == FILL_MODE_SIGNAL_CLOSE:
        return purchase_at_signal_close(bar, naive_tz=naive_tz)
    return purchase_after_close_signal(bar, nxt, naive_tz=naive_tz)


def _index_by_clock(bars: Sequence[dict], ts: Any) -> Optional[int]:
    want = _clock_hm(ts)
    if want is None:
        return None
    want_et = as_et(ts)
    want_day = want_et.date()
    for i, bar in enumerate(bars):
        got = _clock_hm(bar.get("ts"))
        if got is None:
            continue
        try:
            day = as_et(bar.get("ts")).date()
        except (TypeError, ValueError):
            continue
        if day == want_day and got == want:
            return i
    return None


def simulate_measured_move_exit(
    session_bars: Sequence[dict],
    *,
    entry_ts: Any,
    stop_px: float,
    target_px: float,
) -> ExitResult:
    """Same-session N+1 mid after a stop/target tag. Else last close (EOD).

    Entry bar is skipped. Overnight next-mid is not used.
    """
    bars = list(session_bars or [])
    if not bars:
        return ExitResult(filled=False, reason=REASON_NO_15M)
    try:
        stop = float(stop_px)
        target = float(target_px)
    except (TypeError, ValueError):
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    if stop != stop or target != target or target <= stop:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)
    entry_i = _index_by_clock(bars, entry_ts)
    if entry_i is None:
        return ExitResult(filled=False, reason=REASON_NO_ENTRY_BAR)

    def _next_mid_or_close(decision_i: int, exit_reason: str) -> ExitResult:
        nxt_i = int(decision_i) + 1
        if nxt_i < len(bars):
            nxt = bars[nxt_i]
            hi = _bar_px(nxt, "high")
            lo = _bar_px(nxt, "low")
            if hi is not None and lo is not None:
                return ExitResult(
                    filled=True,
                    reason=REASON_FILLED,
                    sell_px=float(bar_mid(hi, lo)),
                    decision_ts=bars[decision_i].get("ts"),
                    exec_bar_ts=nxt.get("ts"),
                    exit_reason=exit_reason,
                    hard_stop=stop,
                    stop_at_decision=stop,
                )
        cl = _bar_px(bars[decision_i], "close")
        if cl is None:
            return ExitResult(
                filled=False,
                reason=REASON_NO_NEXT_BAR,
                decision_ts=bars[decision_i].get("ts"),
                exit_reason=exit_reason,
                hard_stop=stop,
                stop_at_decision=stop,
            )
        return ExitResult(
            filled=True,
            reason=REASON_EOD,
            sell_px=float(cl),
            decision_ts=bars[decision_i].get("ts"),
            exec_bar_ts=bars[decision_i].get("ts"),
            exit_reason=EXIT_EOD,
            hard_stop=stop,
            stop_at_decision=stop,
        )

    last_i = len(bars) - 1
    if entry_i >= last_i:
        cl = _bar_px(bars[entry_i], "close")
        if cl is None:
            return ExitResult(
                filled=False,
                reason=REASON_NO_NEXT_BAR,
                hard_stop=stop,
            )
        return ExitResult(
            filled=True,
            reason=REASON_EOD,
            sell_px=float(cl),
            decision_ts=bars[entry_i].get("ts"),
            exec_bar_ts=bars[entry_i].get("ts"),
            exit_reason=EXIT_EOD,
            hard_stop=stop,
            stop_at_decision=stop,
        )

    for i in range(entry_i + 1, len(bars)):
        hi = _bar_px(bars[i], "high")
        lo = _bar_px(bars[i], "low")
        if hi is None or lo is None:
            continue
        hit_stop = lo <= stop + 1e-12
        hit_tgt = hi >= target - 1e-12
        if hit_stop:
            return _next_mid_or_close(i, EXIT_STOP)
        if hit_tgt:
            return _next_mid_or_close(i, EXIT_TARGET)
    last = bars[-1]
    cl = _bar_px(last, "close")
    if cl is None:
        return ExitResult(
            filled=False,
            reason=REASON_EOD,
            hard_stop=stop,
        )
    return ExitResult(
        filled=True,
        reason=REASON_EOD,
        sell_px=float(cl),
        decision_ts=last.get("ts"),
        exec_bar_ts=last.get("ts"),
        exit_reason=EXIT_EOD,
        hard_stop=stop,
        stop_at_decision=stop,
    )


def scan_session(
    bars: Sequence[dict],
    *,
    ema_prev: Optional[float],
    params: ReclaimParams,
    naive_tz: str = "UTC",
) -> Tuple[Optional[dict], Optional[float]]:
    """Walk one RTH session. Returns (trade_or_none, ema after last bar)."""
    n = len(bars)
    span = max(1, int(params.ema_span))
    ema = np.full(n, np.nan, dtype=float)
    high = np.full(n, np.nan, dtype=float)
    low = np.full(n, np.nan, dtype=float)
    close = np.full(n, np.nan, dtype=float)
    vol = np.zeros(n, dtype=float)
    prev = ema_prev
    for i, bar in enumerate(bars):
        h = _bar_px(bar, "high")
        lo = _bar_px(bar, "low")
        c = _bar_px(bar, "close")
        v = _bar_px(bar, "volume")
        if h is not None:
            high[i] = h
        if lo is not None:
            low[i] = lo
        if c is not None:
            close[i] = c
            prev = ema_next(prev, c, span=span)
            ema[i] = prev
        if v is not None:
            vol[i] = v
    vwap = session_vwap(high, low, close, vol)
    sig_i = first_reclaim_index(
        ema,
        vwap,
        low,
        min_lod_age_bars=int(params.min_lod_age_bars),
    )
    if sig_i is None:
        return None, prev
    sig_bar = bars[sig_i]
    if _is_after_last_entry(sig_bar.get("ts"), params.last_entry_clock):
        return None, prev
    lod = float(np.min(low[:sig_i]))
    cross_px = float(vwap[sig_i])
    levels = measured_move_levels(
        lod,
        cross_px,
        stop_fraction=float(params.stop_fraction),
    )
    if levels is None:
        return None, prev
    unit, stop_px, target_px = levels
    bought = _fill_from_signal(
        bars,
        sig_i,
        fill_mode=params.fill_mode,
        naive_tz=naive_tz,
    )
    if not bought.filled or bought.fill_px is None or bought.exec_bar_ts is None:
        return None, prev
    fill_px = float(bought.fill_px)
    if fill_px <= stop_px + 1e-12 or fill_px >= target_px - 1e-12:
        return None, prev
    exited = simulate_measured_move_exit(
        bars,
        entry_ts=bought.exec_bar_ts,
        stop_px=stop_px,
        target_px=target_px,
    )
    if not exited.filled or exited.sell_px is None or exited.exec_bar_ts is None:
        return None, prev
    sell_px = float(exited.sell_px)
    gain = (sell_px / fill_px - 1.0) * 100.0
    trade = {
        "buy_ts": bought.exec_bar_ts,
        "buy_price": fill_px,
        "sell_ts": exited.exec_bar_ts,
        "sell_price": sell_px,
        "gain_pct": gain,
        "exit_reason": exited.exit_reason,
        "lod": lod,
        "cross_px": cross_px,
        "unit": unit,
        "stop_px": stop_px,
        "target_px": target_px,
        "signal_ts": sig_bar.get("ts"),
        "fill_mode": normalize_fill_mode(params.fill_mode),
        "ema_at_signal": float(ema[sig_i]),
        "vwap_at_signal": cross_px,
        "lod_age_bars": int(sig_i - int(np.argmin(low[:sig_i]))),
        "signal_bar_i": int(sig_i),
    }
    return trade, prev


def collect_symbol_trades(
    by_day: Mapping[date, Sequence[dict]],
    *,
    params: Optional[ReclaimParams] = None,
    naive_tz: str = "UTC",
) -> List[dict]:
    cfg = params or ReclaimParams()
    trades: List[dict] = []
    ema_prev: Optional[float] = None
    for sess in sorted(by_day.keys()):
        bars = list(by_day.get(sess) or [])
        bars.sort(key=lambda b: as_et(b["ts"]))
        if not bars:
            continue
        rec, ema_prev = scan_session(
            bars,
            ema_prev=ema_prev,
            params=cfg,
            naive_tz=naive_tz,
        )
        if rec is not None:
            rec["session_date"] = sess
            trades.append(rec)
    return trades


def max_concurrent(trades: Sequence[Mapping[str, Any]]) -> Tuple[int, float]:
    """Return (max, median) concurrent names from buy_ts/sell_ts."""
    events: List[Tuple[Any, int]] = []
    for rec in trades:
        buy = rec.get("buy_ts")
        sell = rec.get("sell_ts")
        if buy is None or sell is None:
            continue
        events.append((buy, 1))
        events.append((sell, -1))
    if not events:
        return 0, 0.0
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    peak = 0
    path: List[int] = []
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
        if cur >= 0:
            path.append(cur)
    med = float(np.median(path)) if path else 0.0
    return int(peak), med


def attach_unit_clock(df: pd.DataFrame) -> pd.DataFrame:
    """Add unit_pct, signal_et_hour, session_date from stored fill fields."""
    out = df.copy()
    unit = pd.to_numeric(out.get("unit"), errors="coerce")
    cross = pd.to_numeric(out.get("cross_px"), errors="coerce")
    out["unit_pct"] = unit / cross * 100.0
    raw = out["signal_ts"] if "signal_ts" in out.columns else out.get("buy_ts")
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    et = ts.dt.tz_convert("America/New_York")
    out["signal_et_hour"] = et.dt.hour + et.dt.minute / 60.0
    out["session_date"] = et.dt.strftime("%Y-%m-%d")
    return out


def mask_min_unit(df: pd.DataFrame, min_pct: float) -> pd.Series:
    return pd.to_numeric(df["unit_pct"], errors="coerce") >= float(min_pct)


def mask_morning(df: pd.DataFrame, before_hour: float = 11.0) -> pd.Series:
    """Signal clock strictly before ``before_hour`` ET (default 11:00)."""
    return pd.to_numeric(df["signal_et_hour"], errors="coerce") < float(before_hour)


def mask_min_lod_age(df: pd.DataFrame, min_bars: int) -> pd.Series:
    return pd.to_numeric(df["lod_age_bars"], errors="coerce") >= int(min_bars)


def take_daily_topn(
    df: pd.DataFrame,
    *,
    n: int = 1,
    rank_col: str = "unit_pct",
    day_col: str = "session_date",
) -> pd.DataFrame:
    """Keep the ``n`` largest ``rank_col`` names per session date (ties: stock)."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    cap = max(1, int(n))
    work = df.copy()
    work["_rank"] = pd.to_numeric(work[rank_col], errors="coerce")
    work["_sym"] = work["stock"].astype(str) if "stock" in work.columns else ""
    work = work.sort_values([day_col, "_rank", "_sym"], ascending=[True, False, True], kind="mergesort")
    kept = work.groupby(day_col, sort=False, as_index=False).head(cap)
    return kept.drop(columns=["_rank", "_sym"])


def drop_top_n_expectancy(gains: Sequence[float], n: int = 3) -> Optional[float]:
    vals = [float(x) for x in gains if x is not None and pd.notna(x)]
    if len(vals) <= int(n):
        return None
    ordered = sorted(vals, reverse=True)
    kept = ordered[int(n) :]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


__all__ = [
    "DEFAULT_FILL_MODE",
    "EMA_SPAN",
    "EXIT_STOP",
    "EXIT_TARGET",
    "LAST_ENTRY_CLOCK",
    "ReclaimParams",
    "STOP_FRACTION",
    "TRADE_COLUMNS",
    "attach_unit_clock",
    "collect_symbol_trades",
    "drop_top_n_expectancy",
    "ema_next",
    "mask_min_lod_age",
    "mask_min_unit",
    "mask_morning",
    "take_daily_topn",
    "first_reclaim_index",
    "index_rth_15m_fast",
    "max_concurrent",
    "measured_move_levels",
    "occupancy_take",
    "scan_session",
    "session_vwap",
    "simulate_measured_move_exit",
]
