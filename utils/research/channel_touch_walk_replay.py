"""Causal walk-replay of channel-touch: detector sees bars 0..t only.

Default ``causal_h2`` freezes rails at the first completing H2. Live
``h2_resist_break_only`` drops L3 support-tag fills *before* occupancy so
they cannot block a later resist-break (post-filter after occupancy was a
silent miss). ``walk_replay_trades`` is the as-of-t check; detector v1
``find_channels`` is unchanged.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from find_ascending_channels import (  # noqa: E402
    _remap_setup_indices,
    find_h2_l3_setups,
    find_h2_l3_setups_windowed,
)
from backtest_channel_touch_trades import (  # noqa: E402
    _adv_20,
    _atr,
    _h2_rail_tag_fills,
    _line_at,
    _normalize_ohlcv_frame,
    _simulate_trade,
    _summarize,
    apply_friction,
    filter_trades,
    trades_for_symbol,
)
from utils.research.channel_touch_entry_features import max_beyond_width  # noqa: E402

logger = logging.getLogger(__name__)

SetupFinder = Callable[..., List[dict]]

DETECTOR_KEYS = {
    "error_pct",
    "flat_pct",
    "min_bars_apart",
    "min_intervening_rally_pct",
    "min_intervening_pullback_pct",
    "min_total_rise_pct",
    "max_support_violation_frac",
    "max_low_pivots",
    "min_top_touches",
    "min_rally_pct",
    "min_pullback_pct",
    "causal_h2",
}


def _channel_kwargs(raw: dict) -> dict:
    kw = {k: v for k, v in raw.items() if k in DETECTOR_KEYS}
    if "min_rally_pct" in kw:
        kw.setdefault("min_intervening_rally_pct", kw.pop("min_rally_pct"))
    if "min_pullback_pct" in kw:
        kw.setdefault("min_intervening_pullback_pct", kw.pop("min_pullback_pct"))
    return kw


def _setup_key(ch: dict) -> Tuple[int, int, int, float, float]:
    idxs = ch.get("touch_indices") or []
    i0 = int(idxs[0]) if idxs else int(ch.get("l1_idx") or ch.get("support_x0") or -1)
    i1 = int(idxs[-1]) if idxs else int(ch.get("l2_idx") or -1)
    return (
        i0,
        i1,
        int(ch.get("h2_idx", -1)),
        round(float(ch.get("support_slope", 0.0)), 8),
        round(float(ch.get("channel_width", 0.0)), 6),
    )


def _buy_key(row: dict, include_time: bool = False) -> Tuple[str, str]:
    if include_time and row.get("buy_time"):
        ts = str(row["buy_time"])
    else:
        ts = str(row.get("buy_date") or "")
    px = row.get("buy_price")
    try:
        px_s = "%.4f" % float(px)
    except (TypeError, ValueError):
        px_s = ""
    return ts, px_s


class WindowedSetupCache:
    """Reuse completed sliding windows as the prefix grows by one bar."""

    def __init__(
        self,
        finder: SetupFinder,
        *,
        window_bars: int,
        step_bars: int,
        pivot_len: int,
        channel_kwargs: dict,
    ) -> None:
        self.finder = finder
        self.window_bars = int(window_bars or 0)
        self.step_bars = int(step_bars or window_bars or 0)
        self.pivot_len = int(pivot_len)
        self.channel_kwargs = dict(channel_kwargs)
        self._complete: Dict[int, List[dict]] = {}

    def setups(self, df: pd.DataFrame) -> List[dict]:
        n = len(df)
        win = self.window_bars
        step = self.step_bars if self.step_bars > 0 else win
        kwargs = dict(self.channel_kwargs)
        kwargs["pivot_len"] = self.pivot_len
        if win <= 0 or win >= n:
            return self.finder(df, **kwargs)
        if step <= 0:
            step = win
        seen = set()
        out: List[dict] = []
        start = 0
        while start < n:
            end = min(n, start + win)
            complete = end - start == win
            if complete and start in self._complete:
                chunk = self._complete[start]
            else:
                raw = self.finder(df.iloc[start:end], **kwargs)
                chunk = [_remap_setup_indices(ch, start) for ch in raw]
                if complete:
                    self._complete[start] = chunk
            for mapped in chunk:
                idxs = mapped.get("touch_indices") or []
                if not idxs:
                    continue
                key = _setup_key(mapped)
                if key in seen:
                    continue
                seen.add(key)
                out.append(mapped)
            if end >= n:
                break
            start += step
        out.sort(key=lambda c: (int(c.get("support_x0", 0)), int(c.get("bars_span", 0))))
        return out


def last_bar_fills(
    df: Optional[pd.DataFrame],
    setups: Sequence[dict],
    *,
    error_pct: float,
    slip: float,
    wait: int,
    min_wait: int,
    entry_touch: int,
    h2_resist_break: bool,
    h2_resist_break_only: bool,
    high: Optional[np.ndarray] = None,
    low: Optional[np.ndarray] = None,
    close: Optional[np.ndarray] = None,
    n: Optional[int] = None,
    resolved_keys: Optional[set] = None,
) -> List[Tuple[dict, int, float, int, bool]]:
    """Fills whose entry index is the last bar of the prefix (no future bars).

    Pass precomputed ``high``/``low``/``close`` and prefix length ``n`` to avoid
    copying the growing frame on every walk step. Setups whose H2 wait window
    cannot contain the last bar are skipped (same as ``_h2_rail_tag_fills``).
    """
    if not setups:
        return []
    if high is None or low is None or close is None:
        if df is None or df.empty:
            return []
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)
        close = df["close"].to_numpy(dtype=float)
        n = len(df)
    n = int(n if n is not None else len(high))
    if n < 2:
        return []
    last_i = n - 1
    wait_n = max(1, int(wait))
    min_w = max(1, int(min_wait))
    out: List[Tuple[dict, int, float, int, bool]] = []
    for ch in setups:
        h2 = int(ch.get("h2_idx", -1))
        if h2 < 0 or h2 >= last_i:
            continue
        if last_i < h2 + min_w or last_i > h2 + wait_n:
            continue
        tags = _h2_rail_tag_fills(
            high,
            low,
            close,
            support_x0=int(ch["support_x0"]),
            support_y0=float(ch["support_y0"]),
            support_slope=float(ch["support_slope"]),
            width=float(ch.get("channel_width") or 0.0),
            h2=h2,
            n=n,
            error_pct=float(error_pct),
            slip=float(slip),
            wait=wait_n,
            min_wait=min_w,
            entry_touch=int(entry_touch),
            h2_resist_break=bool(h2_resist_break),
        )
        if tags and int(tags[0][0]) != last_i:
            if resolved_keys is not None:
                resolved_keys.add(_setup_key(ch))
            continue
        for tag in tags:
            i, fill, tnum = int(tag[0]), float(tag[1]), int(tag[2])
            is_brk = bool(tag[4]) if len(tag) > 4 else False
            if bool(h2_resist_break_only) and not is_brk:
                continue
            if i != last_i:
                continue
            out.append((ch, i, fill, tnum, is_brk))
    return out


def _trade_row(
    *,
    symbol: str,
    df: pd.DataFrame,
    ch: dict,
    entry_i: int,
    fill_px: float,
    touch_num: int,
    is_brk: bool,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    dates: pd.DatetimeIndex,
    stop_pct: float,
    trail_pct: float,
    trail_pct_wide: Optional[float],
    squeeze_mom: Optional[np.ndarray],
    squeeze_pctile: float,
    squeeze_lookback: int,
    atr_stop_mult: Optional[float],
    stop_pct_floor: float,
    stop_pct_ceil: float,
    resist_exit: bool,
    resist_arm_trail: bool,
    peak_trail_mode: str = "off",
    trail_floor: float = 0.01,
    trail_decay_per_bar: float = 0.0002,
    trail_tighten_per_pct: float = 0.0033,
    trail_pct_tight: Optional[float],
    squeeze_fade_tighten: bool,
    max_hold_days: Optional[int],
    include_time: bool,
    adv_lookback: int,
    source: str,
) -> Optional[dict]:
    n = len(close)
    if entry_i < 0 or entry_i >= n:
        return None
    sx0 = int(ch["support_x0"])
    sy0 = float(ch["support_y0"])
    sslope = float(ch["support_slope"])
    width = float(ch.get("channel_width") or 0.0)
    atr_i = float(atr[entry_i]) if entry_i < len(atr) else float("nan")
    sim = _simulate_trade(
        high,
        low,
        close,
        dates,
        entry_i,
        stop_pct=stop_pct,
        trail_pct=trail_pct,
        trail_pct_wide=trail_pct_wide,
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
        resist_exit=resist_exit,
        resist_arm_trail=resist_arm_trail,
        peak_trail_mode=peak_trail_mode,
        trail_floor=trail_floor,
        trail_decay_per_bar=trail_decay_per_bar,
        trail_tighten_per_pct=trail_tighten_per_pct,
        trail_pct_tight=trail_pct_tight,
        squeeze_fade_tighten=squeeze_fade_tighten,
        max_hold_days=max_hold_days,
        include_time=include_time,
        entry_px=fill_px,
    )
    if sim is None:
        return None
    entry_px = float(sim["buy_price"])
    atr_pct = (atr_i / entry_px * 100.0) if entry_px > 0 and np.isfinite(atr_i) else float("nan")
    adv = _adv_20(close, volume, entry_i, lookback=adv_lookback)
    support_at = _line_at(sy0, sx0, sslope, entry_i)
    resist_at = support_at + width
    channel_pos = (
        (entry_px - support_at) / width
        if width > 0 and np.isfinite(support_at)
        else float("nan")
    )
    h2_idx = int(ch.get("h2_idx", -1))
    buy_ts = pd.Timestamp(dates[entry_i])
    try:
        ch_start_ts = pd.Timestamp(ch["start_date"])
        ch_end_ts = pd.Timestamp(ch.get("h2_date") or ch["end_date"])
        span_days = int((ch_end_ts - ch_start_ts).days)
        age_days = int((buy_ts - ch_start_ts).days)
    except Exception:
        span_days = None
        age_days = None
    beyond = max_beyond_width(high, sy0, sx0, sslope, width, sx0, entry_i)
    row = {
        "stock": symbol.upper(),
        "channel_start": ch.get("start_date"),
        "channel_end": ch.get("h2_date") or ch.get("end_date"),
        "h2_date": ch.get("h2_date"),
        "touch_num": int(touch_num),
        "resist_break": bool(is_brk),
        "wait_bars": int(entry_i - h2_idx) if h2_idx >= 0 else None,
        "l1_idx": int(ch.get("l1_idx", sx0)),
        "l2_idx": int(ch.get("l2_idx", -1)),
        "h2_idx": h2_idx,
        "support_x0": sx0,
        "support_slope": sslope,
        "channel_width": width,
        "slope_pct_per_bar": ch.get("slope_pct_per_bar"),
        "channel_width_pct": ch.get("channel_width_pct"),
        "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
        "max_beyond_width": round(float(beyond), 4) if np.isfinite(beyond) else None,
        "channel_span_days": span_days,
        "channel_age_at_buy_days": age_days,
        "adv_20": round(adv, 2) if np.isfinite(adv) else None,
        "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
        "source": source,
        **{k: v for k, v in sim.items() if k not in ("entry_i", "exit_i")},
        "entry_i": sim["entry_i"],
        "exit_i": sim["exit_i"],
    }
    return row


def pending_to_trades(
    symbol: str,
    df: pd.DataFrame,
    pending: Sequence[Tuple[dict, int, float, int, bool]],
    *,
    stop_pct: float = 0.03,
    trail_pct: float = 0.10,
    trail_pct_wide: Optional[float] = None,
    squeeze_adaptive: bool = False,
    squeeze_pctile: float = 75.0,
    squeeze_lookback: int = 100,
    atr_stop_mult: Optional[float] = 2.0,
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
    include_time: bool = False,
    adv_lookback: int = 20,
    atr_len: int = 14,
    source: str = "walk",
) -> List[dict]:
    """One-position occupancy + exits on the full frame (exits look only forward)."""
    if df is None or df.empty or not pending:
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
    squeeze_mom = None
    wide = None
    if squeeze_adaptive or squeeze_fade_tighten:
        from indicators.ttm_squeeze import calculate_squeeze_momentum

        squeeze_mom = calculate_squeeze_momentum(out, lengthKC=20, use_logging=False).to_numpy(
            dtype=float
        )
        if squeeze_adaptive and trail_pct_wide is not None:
            wide = float(trail_pct_wide)
    items = sorted(pending, key=lambda t: (int(t[1]), int(t[0].get("h2_idx", 0))))
    trades: List[dict] = []
    busy_until = -1
    for ch, entry_i, fill_px, tnum, is_brk in items:
        if entry_i <= busy_until:
            continue
        row = _trade_row(
            symbol=symbol,
            df=out,
            ch=ch,
            entry_i=int(entry_i),
            fill_px=float(fill_px),
            touch_num=int(tnum),
            is_brk=bool(is_brk),
            high=high,
            low=low,
            close=close,
            volume=volume,
            atr=atr,
            dates=dates,
            stop_pct=stop_pct,
            trail_pct=trail_pct,
            trail_pct_wide=wide,
            squeeze_mom=squeeze_mom,
            squeeze_pctile=squeeze_pctile,
            squeeze_lookback=squeeze_lookback,
            atr_stop_mult=atr_stop_mult,
            stop_pct_floor=stop_pct_floor,
            stop_pct_ceil=stop_pct_ceil,
            resist_exit=resist_exit,
            resist_arm_trail=resist_arm_trail,
            peak_trail_mode=peak_trail_mode,
            trail_floor=trail_floor,
            trail_decay_per_bar=trail_decay_per_bar,
            trail_tighten_per_pct=trail_tighten_per_pct,
            trail_pct_tight=trail_pct_tight,
            squeeze_fade_tighten=squeeze_fade_tighten,
            max_hold_days=max_hold_days,
            include_time=include_time,
            adv_lookback=adv_lookback,
            source=source,
        )
        if row is None:
            continue
        trades.append(row)
        busy_until = int(row["exit_i"])
    return trades


def walk_replay_trades(
    symbol: str,
    df: pd.DataFrame,
    *,
    pivot_len: int = 15,
    window_bars: Optional[int] = None,
    window_step_bars: Optional[int] = None,
    entry_touch: int = 3,
    entry_slip_pct: float = 0.001,
    max_l3_wait_bars: int = 252,
    min_l3_wait_bars: int = 6,
    h2_resist_break: bool = True,
    h2_resist_break_only: bool = True,
    freeze_h2: bool = False,
    progress_every: int = 0,
    min_prefix_bars: Optional[int] = None,
    **kwargs: Any,
) -> List[dict]:
    """Walk bar-by-bar; detector is called on ``df.iloc[:t+1]`` only.

    ``freeze_h2``: keep the first geometry seen for each L1-L2-H2 index triple
    (live watchlist style). Off = re-scan the prefix every bar (Pine-like).
    Extra kwargs are split into ``trades_for_symbol`` occupancy args and
    detector channel kwargs.
    """
    if df is None or df.empty:
        return []
    out = _normalize_ohlcv_frame(df)
    n = len(out)
    occ_keys = {
        "stop_pct",
        "trail_pct",
        "trail_pct_wide",
        "squeeze_adaptive",
        "squeeze_pctile",
        "squeeze_lookback",
        "atr_stop_mult",
        "stop_pct_floor",
        "stop_pct_ceil",
        "resist_exit",
        "resist_arm_trail",
        "peak_trail_mode",
        "trail_floor",
        "trail_decay_per_bar",
        "trail_tighten_per_pct",
        "trail_pct_tight",
        "squeeze_fade_tighten",
        "max_hold_days",
        "include_time",
        "adv_lookback",
        "atr_len",
    }
    occ = {k: kwargs[k] for k in occ_keys if k in kwargs}
    channel_kwargs = _channel_kwargs({k: v for k, v in kwargs.items() if k not in occ_keys})
    error_pct = float(channel_kwargs.get("error_pct", 1.2))
    win = int(window_bars or 0)
    step = int(window_step_bars or window_bars or 0)
    cache = WindowedSetupCache(
        find_h2_l3_setups,
        window_bars=win,
        step_bars=step,
        pivot_len=int(pivot_len),
        channel_kwargs=channel_kwargs,
    )
    min_len = int(min_prefix_bars) if min_prefix_bars else max(int(pivot_len) * 4 + 40, 80)
    min_len = min(max(min_len, 2), n)
    frozen: Dict[Tuple[int, int, int], dict] = {}
    resolved: set = set()
    pending: List[Tuple[dict, int, float, int, bool]] = []
    wait = max(1, int(max_l3_wait_bars))
    min_wait = max(1, int(min_l3_wait_bars))
    slip = float(entry_slip_pct)
    want = max(3, int(entry_touch))
    log_every = max(0, int(progress_every))
    high_a = out["high"].to_numpy(dtype=float)
    low_a = out["low"].to_numpy(dtype=float)
    close_a = out["close"].to_numpy(dtype=float)

    for t in range(min_len - 1, n):
        prefix = out.iloc[: t + 1]
        setups = cache.setups(prefix)
        if freeze_h2:
            for ch in setups:
                key = (
                    int(ch.get("l1_idx", ch.get("support_x0", -1))),
                    int(ch.get("l2_idx", -1)),
                    int(ch.get("h2_idx", -1)),
                )
                if key not in frozen:
                    frozen[key] = ch
            dead = [
                k
                for k, ch in frozen.items()
                if int(ch.get("h2_idx", -1)) >= 0 and t > int(ch["h2_idx"]) + wait
            ]
            for k in dead:
                frozen.pop(k, None)
            setups = list(frozen.values())
        active: List[dict] = []
        for ch in setups:
            skey = _setup_key(ch)
            if skey in resolved:
                continue
            h2 = int(ch.get("h2_idx", -1))
            if h2 < 0 or t < h2 + min_wait:
                continue
            if t > h2 + wait:
                resolved.add(skey)
                continue
            active.append(ch)
        fills = last_bar_fills(
            None,
            active,
            error_pct=error_pct,
            slip=slip,
            wait=wait,
            min_wait=min_wait,
            entry_touch=want,
            h2_resist_break=bool(h2_resist_break),
            h2_resist_break_only=bool(h2_resist_break_only),
            high=high_a,
            low=low_a,
            close=close_a,
            n=t + 1,
            resolved_keys=resolved,
        )
        for item in fills:
            resolved.add(_setup_key(item[0]))
            pending.append(item)
        if log_every and (t + 1) % log_every == 0:
            logger.info(
                "walk %s bar %d/%d setups=%d active=%d pending_fills=%d",
                symbol,
                t + 1,
                n,
                len(setups),
                len(active),
                len(pending),
            )

    return pending_to_trades(symbol, out, pending, source="walk", **occ)


def batch_trades_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    h2_resist_break_only: bool = True,
    **kwargs: Any,
) -> List[dict]:
    """Original full-series `trades_for_symbol`, with optional resist-break-only."""
    occ_keys = {
        "stop_pct",
        "trail_pct",
        "trail_pct_wide",
        "squeeze_adaptive",
        "squeeze_pctile",
        "squeeze_lookback",
        "atr_stop_mult",
        "stop_pct_floor",
        "stop_pct_ceil",
        "resist_exit",
        "resist_arm_trail",
        "peak_trail_mode",
        "trail_floor",
        "trail_decay_per_bar",
        "trail_tighten_per_pct",
        "trail_pct_tight",
        "squeeze_fade_tighten",
        "max_hold_days",
        "include_time",
        "adv_lookback",
        "entry_features",
        "entry_slip_pct",
        "max_l3_wait_bars",
        "min_l3_wait_bars",
        "shakeout_rebuy_bars",
        "h2_resist_break",
        "h2_resist_break_only",
        "shakeout_breakout",
        "shakeout_breakout_min_inside",
        "shakeout_breakout_hard_stop",
        "window_bars",
        "window_step_bars",
        "pivot_len",
        "entry_mode",
        "entry_touch",
        "feature_asof_prior_bar",
    }
    scan = {k: kwargs[k] for k in occ_keys if k in kwargs}
    scan.setdefault("entry_mode", "l3_touch")
    scan.setdefault("entry_features", False)
    scan["h2_resist_break_only"] = bool(h2_resist_break_only)
    channel_kwargs = _channel_kwargs({k: v for k, v in kwargs.items() if k not in occ_keys})
    rows = trades_for_symbol(symbol, df, **scan, **channel_kwargs)
    if h2_resist_break_only:
        rows = [r for r in rows if bool(r.get("resist_break"))]
    for r in rows:
        r["source"] = "batch"
    return rows


def compare_trade_lists(
    batch: Sequence[dict],
    walk: Sequence[dict],
    *,
    include_time: bool = False,
) -> dict:
    """Match on buy timestamp (+ price). Report only-batch / only-walk / both."""
    b_map = {_buy_key(r, include_time=include_time): r for r in batch}
    w_map = {_buy_key(r, include_time=include_time): r for r in walk}
    b_keys = set(b_map)
    w_keys = set(w_map)
    matched_keys = sorted(b_keys & w_keys)
    only_batch = [b_map[k] for k in sorted(b_keys - w_keys)]
    only_walk = [w_map[k] for k in sorted(w_keys - b_keys)]
    matched = [{"batch": b_map[k], "walk": w_map[k]} for k in matched_keys]
    date_only_b = {str(r.get("buy_date") or "") for r in batch}
    date_only_w = {str(r.get("buy_date") or "") for r in walk}
    return {
        "n_batch": len(batch),
        "n_walk": len(walk),
        "n_matched": len(matched),
        "n_only_batch": len(only_batch),
        "n_only_walk": len(only_walk),
        "n_matched_buy_date": len(date_only_b & date_only_w),
        "matched": matched,
        "only_batch": only_batch,
        "only_walk": only_walk,
    }


def summarize_side(rows: Sequence[dict], friction_pct: float = 0.0) -> dict:
    if not rows:
        return _summarize(pd.DataFrame())
    df = pd.DataFrame(list(rows))
    if friction_pct:
        df = apply_friction(df, friction_pct)
        col = "gain_pct_net"
    else:
        col = "gain_pct"
    return _summarize(df, gain_col=col)


def apply_quality_filters(rows: Sequence[dict], **filter_kwargs: Any) -> List[dict]:
    if not rows:
        return []
    if not any(v is not None and v is not False for v in filter_kwargs.values()):
        return list(rows)
    out = filter_trades(pd.DataFrame(list(rows)), **filter_kwargs)
    return out.to_dict("records")


def comparison_frame(cmp: dict) -> pd.DataFrame:
    rows: List[dict] = []
    for pair in cmp.get("matched") or []:
        b, w = pair["batch"], pair["walk"]
        rows.append(
            {
                "bucket": "matched",
                "buy_date": b.get("buy_date"),
                "batch_price": b.get("buy_price"),
                "walk_price": w.get("buy_price"),
                "batch_h2": b.get("h2_date") or b.get("channel_end"),
                "walk_h2": w.get("h2_date") or w.get("channel_end"),
                "batch_gain_pct": b.get("gain_pct"),
                "walk_gain_pct": w.get("gain_pct"),
                "price_diff": (
                    None
                    if b.get("buy_price") is None or w.get("buy_price") is None
                    else round(float(w["buy_price"]) - float(b["buy_price"]), 4)
                ),
            }
        )
    for r in cmp.get("only_batch") or []:
        rows.append(
            {
                "bucket": "only_batch",
                "buy_date": r.get("buy_date"),
                "batch_price": r.get("buy_price"),
                "walk_price": None,
                "batch_h2": r.get("h2_date") or r.get("channel_end"),
                "walk_h2": None,
                "batch_gain_pct": r.get("gain_pct"),
                "walk_gain_pct": None,
                "price_diff": None,
            }
        )
    for r in cmp.get("only_walk") or []:
        rows.append(
            {
                "bucket": "only_walk",
                "buy_date": r.get("buy_date"),
                "batch_price": None,
                "walk_price": r.get("buy_price"),
                "batch_h2": None,
                "walk_h2": r.get("h2_date") or r.get("channel_end"),
                "batch_gain_pct": None,
                "walk_gain_pct": r.get("gain_pct"),
                "price_diff": None,
            }
        )
    return pd.DataFrame(rows)


def setups_match_windowed_finder(
    df: pd.DataFrame,
    *,
    window_bars: int,
    step_bars: int,
    pivot_len: int = 15,
    **channel_kwargs: Any,
) -> bool:
    """True when the cache on the full frame equals `find_h2_l3_setups_windowed`."""
    kw = _channel_kwargs(channel_kwargs)
    cache = WindowedSetupCache(
        find_h2_l3_setups,
        window_bars=int(window_bars),
        step_bars=int(step_bars),
        pivot_len=int(pivot_len),
        channel_kwargs=kw,
    )
    cached = [_setup_key(c) for c in cache.setups(df)]
    direct = [
        _setup_key(c)
        for c in find_h2_l3_setups_windowed(
            df,
            window_bars=int(window_bars),
            step_bars=int(step_bars),
            pivot_len=int(pivot_len),
            **kw,
        )
    ]
    return cached == direct
