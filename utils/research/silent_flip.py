"""Silent Flip (ProRealAlgos) claim-check on RTH 15m bars.

Pre-open levels from completed prior sessions:

- Range High / Range Low: prior RTH session high / low
- Flip High: high of the most recent earlier session whose high is above Range High
- Flip Low: low of the most recent earlier session whose low is below Range Low

Trade only when the opening 15m sequence tags one of those four lines,
holds it on the close, prints an opposite-color silent candle, then fills a
stop at that silent candle's extreme. Stop sits just beyond the tested line.
Target is the inner range's opposite rail (Range High for longs, Range Low
for shorts). Day trade only; flatten the 15:45 ET close. Same-bar stop and
target is the stop.

Not a promote until E/PF, year-bucket geo, 2022-23, and occupancy vs SPY
clear the gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from utils.research.vcp_avwap import index_rth_15m_fast
from utils.research.vwap_ema_reclaim import drop_top_n_expectancy, max_concurrent
from utils.research.weekly_deepvue import occupancy_take

EXIT_TARGET = "target"
EXIT_STOP = "stop"
EXIT_EOD = "eod"
EXIT_TRAIL = "trail"

SIDE_LONG = "long"
SIDE_SHORT = "short"

LEVEL_RANGE_HIGH = "range_high"
LEVEL_RANGE_LOW = "range_low"
LEVEL_FLIP_HIGH = "flip_high"
LEVEL_FLIP_LOW = "flip_low"

WINDOW_OPEN_ONLY = "open_only"
WINDOW_FIRST_60 = "first_60"

TRAIL_NONE = "none"
TRAIL_BREAKEVEN = "breakeven"

VIDEO_SYMBOLS: Tuple[str, ...] = ("NVDA", "BAC", "PFE", "UBER", "XOM")

CLAIMED: Dict[str, Any] = {
    "source": "ProRealAlgos Q2QjB5jBTro (2026-09-29)",
    "title": "Silent Flip Strategy",
    "timeframe": "15m",
    "window": "first 60 minutes RTH",
    "notes": (
        "Video live scan of NVDA/BAC/PFE/UBER/XOM; Uber Flip High short "
        "claimed ~1:9 and a Range Low target. Transcript was unavailable; "
        "rules follow the pasted summary plus the public description."
    ),
}

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
    "side",
    "level_name",
    "level_px",
    "range_high",
    "range_low",
    "flip_high",
    "flip_low",
    "stop_px",
    "target_px",
    "silent_px",
    "exit_reason",
    "rr_planned",
)


@dataclass(frozen=True)
class SilentFlipParams:
    window: str = WINDOW_FIRST_60
    lookback_sessions: int = 60
    min_prior_bars: int = 10
    touch_tol_pct: float = 0.10
    min_body_frac: float = 0.30
    stop_buffer_pct: float = 0.05
    min_stop_abs: float = 0.01
    trail: str = TRAIL_NONE
    one_per_day: bool = True


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
        et = pd.Timestamp(ts)
        if et.tzinfo is None:
            return (int(et.hour), int(et.minute))
        et = et.tz_convert("America/New_York")
        return (int(et.hour), int(et.minute))
    except (TypeError, ValueError):
        return None


def _is_green(bar: Mapping[str, Any]) -> bool:
    o = _bar_px(bar, "open")
    c = _bar_px(bar, "close")
    if o is None or c is None:
        return False
    return c > o + 1e-12


def _is_red(bar: Mapping[str, Any]) -> bool:
    o = _bar_px(bar, "open")
    c = _bar_px(bar, "close")
    if o is None or c is None:
        return False
    return c < o - 1e-12


def _body_frac(bar: Mapping[str, Any]) -> float:
    hi = _bar_px(bar, "high")
    lo = _bar_px(bar, "low")
    o = _bar_px(bar, "open")
    c = _bar_px(bar, "close")
    if hi is None or lo is None or o is None or c is None:
        return 0.0
    rng = hi - lo
    if rng <= 1e-12:
        return 0.0
    return abs(c - o) / rng


def session_hl(bars: Sequence[Mapping[str, Any]]) -> Optional[Tuple[float, float]]:
    highs: List[float] = []
    lows: List[float] = []
    for bar in bars:
        hi = _bar_px(bar, "high")
        lo = _bar_px(bar, "low")
        if hi is None or lo is None:
            continue
        highs.append(hi)
        lows.append(lo)
    if not highs:
        return None
    return (float(max(highs)), float(min(lows)))


def prior_flip_levels(
    session_hls: Sequence[Optional[Tuple[float, float]]],
    day_i: int,
    *,
    lookback: int = 60,
) -> Optional[Dict[str, Optional[float]]]:
    """Causal Range / Flip rails using completed sessions before ``day_i``."""
    if day_i < 1:
        return None
    prev = session_hls[day_i - 1]
    if prev is None:
        return None
    range_high, range_low = prev
    flip_high: Optional[float] = None
    flip_low: Optional[float] = None
    start = day_i - 2
    stop = max(-1, day_i - 2 - int(lookback))
    for j in range(start, stop, -1):
        hl = session_hls[j]
        if hl is None:
            continue
        high, low = hl
        if flip_high is None and high > range_high + 1e-12:
            flip_high = high
        if flip_low is None and low < range_low - 1e-12:
            flip_low = low
        if flip_high is not None and flip_low is not None:
            break
    return {
        "range_high": float(range_high),
        "range_low": float(range_low),
        "flip_high": None if flip_high is None else float(flip_high),
        "flip_low": None if flip_low is None else float(flip_low),
    }


def _candle1_ok_for_window(bar: Mapping[str, Any], window: str) -> bool:
    hm = _clock_hm(bar.get("ts"))
    if hm is None:
        return False
    if window == WINDOW_OPEN_ONLY:
        return hm == (9, 30)
    if window == WINDOW_FIRST_60:
        return hm in ((9, 30), (9, 45), (10, 0))
    return False


def _touches_support(bar: Mapping[str, Any], level: float, tol_pct: float) -> bool:
    lo = _bar_px(bar, "low")
    hi = _bar_px(bar, "high")
    o = _bar_px(bar, "open")
    c = _bar_px(bar, "close")
    if lo is None or hi is None or o is None or c is None:
        return False
    band = abs(level) * float(tol_pct) / 100.0
    if lo > level + band:
        return False
    if hi < level - band:
        return False
    if o < level - 1e-12:
        return False
    if c < level - 1e-12:
        return False
    return True


def _touches_resist(bar: Mapping[str, Any], level: float, tol_pct: float) -> bool:
    lo = _bar_px(bar, "low")
    hi = _bar_px(bar, "high")
    o = _bar_px(bar, "open")
    c = _bar_px(bar, "close")
    if lo is None or hi is None or o is None or c is None:
        return False
    band = abs(level) * float(tol_pct) / 100.0
    if hi < level - band:
        return False
    if lo > level + band:
        return False
    if o > level + 1e-12:
        return False
    if c > level + 1e-12:
        return False
    return True


def _stop_buffer(level: float, params: SilentFlipParams) -> float:
    return max(float(params.min_stop_abs), abs(level) * float(params.stop_buffer_pct) / 100.0)


def _fill_buy_stop(bar: Mapping[str, Any], stop_px: float) -> Optional[float]:
    hi = _bar_px(bar, "high")
    o = _bar_px(bar, "open")
    if hi is None or o is None:
        return None
    if hi + 1e-12 < stop_px:
        return None
    if o >= stop_px - 1e-12:
        return float(o)
    return float(stop_px)


def _fill_sell_stop(bar: Mapping[str, Any], stop_px: float) -> Optional[float]:
    lo = _bar_px(bar, "low")
    o = _bar_px(bar, "open")
    if lo is None or o is None:
        return None
    if lo - 1e-12 > stop_px:
        return None
    if o <= stop_px + 1e-12:
        return float(o)
    return float(stop_px)


def _exit_fill_long(
    bar: Mapping[str, Any],
    *,
    stop_px: float,
    target_px: float,
    is_entry_bar: bool,
    entry_px: float,
) -> Optional[Tuple[float, str]]:
    hi = _bar_px(bar, "high")
    lo = _bar_px(bar, "low")
    o = _bar_px(bar, "open")
    if hi is None or lo is None or o is None:
        return None
    if not is_entry_bar:
        if o <= stop_px + 1e-12:
            return (float(o), EXIT_STOP)
        if o >= target_px - 1e-12:
            return (float(o), EXIT_TARGET)
        if lo <= stop_px + 1e-12:
            return (float(stop_px), EXIT_STOP)
        if hi >= target_px - 1e-12:
            return (float(target_px), EXIT_TARGET)
        return None
    if lo <= stop_px + 1e-12 and stop_px < entry_px:
        return (float(stop_px), EXIT_STOP)
    if hi >= target_px - 1e-12:
        return (float(target_px), EXIT_TARGET)
    return None


def _exit_fill_short(
    bar: Mapping[str, Any],
    *,
    stop_px: float,
    target_px: float,
    is_entry_bar: bool,
    entry_px: float,
) -> Optional[Tuple[float, str]]:
    hi = _bar_px(bar, "high")
    lo = _bar_px(bar, "low")
    o = _bar_px(bar, "open")
    if hi is None or lo is None or o is None:
        return None
    if not is_entry_bar:
        if o >= stop_px - 1e-12:
            return (float(o), EXIT_STOP)
        if o <= target_px + 1e-12:
            return (float(o), EXIT_TARGET)
        if hi >= stop_px - 1e-12:
            return (float(stop_px), EXIT_STOP)
        if lo <= target_px + 1e-12:
            return (float(target_px), EXIT_TARGET)
        return None
    if hi >= stop_px - 1e-12 and stop_px > entry_px:
        return (float(stop_px), EXIT_STOP)
    if lo <= target_px + 1e-12:
        return (float(target_px), EXIT_TARGET)
    return None


def _candidate_levels(levels: Mapping[str, Optional[float]]) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    rh = levels.get("range_high")
    rl = levels.get("range_low")
    fh = levels.get("flip_high")
    fl = levels.get("flip_low")
    if fl is not None:
        out.append((LEVEL_FLIP_LOW, SIDE_LONG, float(fl)))
    if rl is not None:
        out.append((LEVEL_RANGE_LOW, SIDE_LONG, float(rl)))
    if rh is not None:
        out.append((LEVEL_RANGE_HIGH, SIDE_SHORT, float(rh)))
    if fh is not None:
        out.append((LEVEL_FLIP_HIGH, SIDE_SHORT, float(fh)))
    return out


def _pick_setup(
    candle1: Mapping[str, Any],
    levels: Mapping[str, Optional[float]],
    params: SilentFlipParams,
) -> Optional[Tuple[str, str, float]]:
    hits: List[Tuple[str, str, float, float]] = []
    for name, side, level in _candidate_levels(levels):
        if side == SIDE_LONG:
            if not _touches_support(candle1, level, params.touch_tol_pct):
                continue
            if not _is_red(candle1):
                continue
            dist = abs((_bar_px(candle1, "low") or level) - level)
        else:
            if not _touches_resist(candle1, level, params.touch_tol_pct):
                continue
            if not _is_green(candle1):
                continue
            dist = abs((_bar_px(candle1, "high") or level) - level)
        hits.append((name, side, level, dist))
    if not hits:
        return None
    hits.sort(key=lambda row: row[3])
    name, side, level, _ = hits[0]
    return (name, side, level)


def simulate_exit(
    bars: Sequence[Mapping[str, Any]],
    *,
    entry_i: int,
    entry_px: float,
    side: str,
    stop_px: float,
    target_px: float,
    trail: str = TRAIL_NONE,
) -> Optional[dict]:
    if entry_i < 0 or entry_i >= len(bars):
        return None
    cur_stop = float(stop_px)
    for i in range(entry_i, len(bars)):
        bar = bars[i]
        is_entry = i == entry_i
        if side == SIDE_LONG:
            hit = _exit_fill_long(
                bar,
                stop_px=cur_stop,
                target_px=target_px,
                is_entry_bar=is_entry,
                entry_px=entry_px,
            )
        else:
            hit = _exit_fill_short(
                bar,
                stop_px=cur_stop,
                target_px=target_px,
                is_entry_bar=is_entry,
                entry_px=entry_px,
            )
        if hit is not None:
            px, reason = hit
            return {
                "sell_px": float(px),
                "sell_ts": bar.get("ts"),
                "exit_reason": reason if trail == TRAIL_NONE or reason != EXIT_STOP else (
                    EXIT_TRAIL if abs(cur_stop - stop_px) > 1e-9 else EXIT_STOP
                ),
            }
        if trail == TRAIL_BREAKEVEN and not is_entry:
            c = _bar_px(bar, "close")
            if c is None:
                continue
            if side == SIDE_LONG and c > entry_px + 1e-12:
                cur_stop = max(cur_stop, float(entry_px))
            elif side == SIDE_SHORT and c < entry_px - 1e-12:
                cur_stop = min(cur_stop, float(entry_px))
    last = bars[-1]
    last_c = _bar_px(last, "close")
    if last_c is None:
        return None
    return {
        "sell_px": float(last_c),
        "sell_ts": last.get("ts"),
        "exit_reason": EXIT_EOD,
    }


def scan_session(
    bars: Sequence[Mapping[str, Any]],
    levels: Mapping[str, Optional[float]],
    *,
    params: Optional[SilentFlipParams] = None,
) -> Optional[dict]:
    cfg = params or SilentFlipParams()
    if len(bars) < 3:
        return None
    rh = levels.get("range_high")
    rl = levels.get("range_low")
    if rh is None or rl is None or rh <= rl:
        return None
    last_i = len(bars) - 2
    for i1 in range(0, last_i):
        c1 = bars[i1]
        if not _candle1_ok_for_window(c1, cfg.window):
            continue
        if _body_frac(c1) + 1e-12 < float(cfg.min_body_frac):
            continue
        setup = _pick_setup(c1, levels, cfg)
        if setup is None:
            continue
        level_name, side, level_px = setup
        c2 = bars[i1 + 1]
        c3 = bars[i1 + 2]
        if side == SIDE_LONG:
            if not _is_green(c2):
                continue
            silent_px = _bar_px(c2, "high")
            entry = _fill_buy_stop(c3, float(silent_px)) if silent_px is not None else None
            stop_px = min(level_px, _bar_px(c1, "low") or level_px) - _stop_buffer(level_px, cfg)
            target_px = float(rh)
        else:
            if not _is_red(c2):
                continue
            silent_px = _bar_px(c2, "low")
            entry = _fill_sell_stop(c3, float(silent_px)) if silent_px is not None else None
            stop_px = max(level_px, _bar_px(c1, "high") or level_px) + _stop_buffer(level_px, cfg)
            target_px = float(rl)
        if entry is None or silent_px is None:
            continue
        if side == SIDE_LONG:
            if stop_px >= entry - 1e-12 or target_px <= entry + 1e-12:
                continue
            risk = entry - stop_px
            reward = target_px - entry
            gain_sign = 1.0
        else:
            if stop_px <= entry + 1e-12 or target_px >= entry - 1e-12:
                continue
            risk = stop_px - entry
            reward = entry - target_px
            gain_sign = -1.0
        if risk <= 1e-12:
            continue
        exit_rec = simulate_exit(
            bars,
            entry_i=i1 + 2,
            entry_px=float(entry),
            side=side,
            stop_px=float(stop_px),
            target_px=float(target_px),
            trail=cfg.trail,
        )
        if exit_rec is None:
            continue
        sell_px = float(exit_rec["sell_px"])
        gain = gain_sign * (sell_px - float(entry)) / float(entry) * 100.0
        return {
            "buy_ts": c3.get("ts"),
            "buy_price": float(entry),
            "sell_ts": exit_rec["sell_ts"],
            "sell_price": sell_px,
            "gain_pct": float(gain),
            "side": side,
            "level_name": level_name,
            "level_px": float(level_px),
            "range_high": float(rh),
            "range_low": float(rl),
            "flip_high": levels.get("flip_high"),
            "flip_low": levels.get("flip_low"),
            "stop_px": float(stop_px),
            "target_px": float(target_px),
            "silent_px": float(silent_px),
            "exit_reason": exit_rec["exit_reason"],
            "rr_planned": float(reward / risk),
            "signal_ts": c1.get("ts"),
        }
    return None


def collect_symbol_trades(
    by_day: Mapping[date, Sequence[dict]],
    *,
    params: Optional[SilentFlipParams] = None,
) -> List[dict]:
    cfg = params or SilentFlipParams()
    days = sorted(by_day.keys())
    hls: List[Optional[Tuple[float, float]]] = []
    for sess in days:
        bars = list(by_day.get(sess) or [])
        if len(bars) < int(cfg.min_prior_bars):
            hls.append(None)
        else:
            hls.append(session_hl(bars))
    trades: List[dict] = []
    for i, sess in enumerate(days):
        levels = prior_flip_levels(hls, i, lookback=cfg.lookback_sessions)
        if levels is None:
            continue
        bars = list(by_day.get(sess) or [])
        if not bars:
            continue
        rec = scan_session(bars, levels, params=cfg)
        if rec is None:
            continue
        rec["session_date"] = sess
        trades.append(rec)
        if cfg.one_per_day:
            continue
    return trades


def rows_from_raw(raw: Sequence[dict], stock: str) -> List[dict]:
    out: List[dict] = []
    for rec in raw:
        buy_ts = rec["buy_ts"]
        sell_ts = rec["sell_ts"]
        out.append(
            {
                "stock": stock,
                "buy_date": _utc_date(buy_ts),
                "buy_ts": buy_ts,
                "buy_time": _et_clock(buy_ts),
                "buy_price": round(float(rec["buy_price"]), 4),
                "sell_date": _utc_date(sell_ts),
                "sell_ts": sell_ts,
                "sell_time": _et_clock(sell_ts),
                "sell_price": round(float(rec["sell_price"]), 4),
                "gain_pct": round(float(rec["gain_pct"]), 4),
                "side": rec.get("side"),
                "level_name": rec.get("level_name"),
                "level_px": rec.get("level_px"),
                "range_high": rec.get("range_high"),
                "range_low": rec.get("range_low"),
                "flip_high": rec.get("flip_high"),
                "flip_low": rec.get("flip_low"),
                "stop_px": rec.get("stop_px"),
                "target_px": rec.get("target_px"),
                "silent_px": rec.get("silent_px"),
                "exit_reason": rec.get("exit_reason"),
                "rr_planned": None if rec.get("rr_planned") is None else round(float(rec["rr_planned"]), 3),
            }
        )
    return out


def _et_clock(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    t = t.tz_convert("America/New_York")
    return t.strftime("%Y-%m-%d %H:%M")


def _utc_date(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.strftime("%Y-%m-%d")


__all__ = [
    "CLAIMED",
    "EXIT_EOD",
    "EXIT_STOP",
    "EXIT_TARGET",
    "EXIT_TRAIL",
    "SilentFlipParams",
    "TRADE_COLUMNS",
    "TRAIL_BREAKEVEN",
    "TRAIL_NONE",
    "VIDEO_SYMBOLS",
    "WINDOW_FIRST_60",
    "WINDOW_OPEN_ONLY",
    "collect_symbol_trades",
    "drop_top_n_expectancy",
    "index_rth_15m_fast",
    "max_concurrent",
    "occupancy_take",
    "prior_flip_levels",
    "rows_from_raw",
    "scan_session",
    "session_hl",
    "simulate_exit",
]
