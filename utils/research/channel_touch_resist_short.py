"""Short ascending-channel resistance tags (H3, H4, ...).

After a causal H2 arm, fade subsequent from-below resistance tags. Opposite of
the long L3 / H2-resist-break books. Detector rails stay frozen at H2.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from backtest_channel_touch_trades import (  # noqa: E402
    _adv_20,
    _atr,
    _hard_stop_price,
    _line_at,
    _support_tagged,
    channel_rail_fields,
)
from find_ascending_channels import (  # noqa: E402
    find_h2_l3_setups,
    find_h2_l3_setups_windowed,
)
from utils.research.channel_touch_entry_features import max_beyond_width  # noqa: E402


def resist_rail_touch(
    bar_high: float,
    bar_low: float,
    bar_close: float,
    resist: float,
    error_pct: float,
) -> bool:
    """Resistance tag from below: range intersects rail, high tags, close not broken out."""
    if not _support_tagged(bar_low, bar_high, resist, error_pct):
        return False
    if not np.isfinite(bar_close):
        return False
    tol = float(error_pct) / 100.0
    if bar_low > resist:
        return False
    if bar_close > resist * (1.0 + tol):
        return False
    return True


def limit_fill_at_resist(
    resist: float,
    bar_low: float,
    bar_high: float,
    slip_pct: float,
) -> Optional[float]:
    """Limit short at resist minus slippage, clipped to the bar. Never above the rail."""
    if not np.isfinite(resist) or resist <= 0:
        return None
    if not np.isfinite(bar_low) or not np.isfinite(bar_high) or bar_high < bar_low:
        return None
    if float(bar_low) - 1e-12 > float(resist):
        return None
    if float(bar_high) + 1e-12 < float(resist) * (1.0 - max(0.0, float(slip_pct))):
        # Never traded near the rail
        if float(bar_high) + 1e-12 < float(resist):
            return None
    raw = resist * (1.0 - max(0.0, float(slip_pct)))
    fill = min(float(bar_high), max(float(bar_low), raw))
    if not np.isfinite(fill) or fill <= 0:
        return None
    if fill - 1e-12 > float(resist):
        return None
    return float(fill)


def find_h3_plus_resist_tags(
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
) -> List[Tuple[int, float, int]]:
    """Post-H2 resistance tags. First is H3 (touch_num=3).

    ``entry_touch`` 3 fills H3+; 4 skips H3 and fills H4 after leave-rail.
    Close through support or close above resistance cancels. Early tag before
    ``min_wait`` aborts the current arm (no retarget). After a filled tag,
    require leave-rail before the next count.
    """
    want = max(3, int(entry_touch))
    wait_n = max(1, int(wait))
    min_w = max(1, int(min_wait))
    leave_frac = max(0.0, float(leave_width_frac))
    out: List[Tuple[int, float, int]] = []
    touch_count = 2
    arm_i = int(h2)
    need_leave = True  # leave H2 peak before first H3
    for i in range(int(h2) + 1, min(n, int(h2) + 1 + wait_n)):
        sup = _line_at(support_y0, support_x0, support_slope, i)
        resist = float(sup) + float(width or 0.0)
        if not np.isfinite(sup) or not np.isfinite(resist) or resist <= 0:
            continue
        tol = float(error_pct) / 100.0
        if float(close[i]) < float(sup) * (1.0 - tol):
            break
        if float(close[i]) > float(resist) * (1.0 + tol):
            break
        touched = resist_rail_touch(
            float(high[i]), float(low[i]), float(close[i]), resist, error_pct
        )
        if need_leave:
            leave_lvl = float(resist) - leave_frac * max(float(width or 0.0), 0.0)
            if np.isfinite(float(close[i])) and float(close[i]) < leave_lvl:
                need_leave = False
            continue
        if i < arm_i + min_w:
            if touched:
                break
            continue
        if touched:
            fill = limit_fill_at_resist(resist, float(low[i]), float(high[i]), slip)
            touch_count += 1
            if fill is not None and touch_count >= want:
                out.append((i, float(fill), int(touch_count)))
            arm_i = int(i)
            need_leave = True
    return out


def simulate_short_trade(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    dates: pd.DatetimeIndex,
    entry_i: int,
    *,
    stop_pct: float,
    trail_pct: float,
    atr_at_entry: Optional[float] = None,
    atr_stop_mult: Optional[float] = None,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    support_x0: Optional[int] = None,
    support_y0: Optional[float] = None,
    support_slope: Optional[float] = None,
    channel_width: Optional[float] = None,
    support_exit: bool = True,
    max_hold_days: Optional[int] = None,
    entry_px: Optional[float] = None,
    include_time: bool = False,
    skip_entry_bar_stop: bool = True,
) -> Optional[dict]:
    """Short from entry_i. Hard stop above entry; trail from trough; optional support cover."""
    n = len(close)
    if entry_i < 0 or entry_i >= n - 1:
        return None
    fill = float(close[entry_i]) if entry_px is None else float(entry_px)
    if not np.isfinite(fill) or fill <= 0:
        return None
    entry_px = fill

    # Long helper returns entry - dist; short stop is entry + dist.
    long_stop = _hard_stop_price(
        entry_px,
        stop_pct=stop_pct,
        atr_at_entry=atr_at_entry,
        atr_stop_mult=atr_stop_mult,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )
    dist = entry_px - float(long_stop)
    if not np.isfinite(dist) or dist <= 0:
        dist = entry_px * float(stop_pct)
    hard_stop = entry_px + dist

    trough = entry_px
    stop_level = hard_stop
    exit_i = n - 1
    exit_px = float(close[exit_i])
    exit_reason = "eod"
    have_line = (
        support_exit
        and support_x0 is not None
        and support_y0 is not None
        and support_slope is not None
        and channel_width is not None
        and np.isfinite(support_y0)
        and np.isfinite(support_slope)
    )

    hi0 = float(high[entry_i]) if entry_i < n else float("nan")
    if (not skip_entry_bar_stop) and np.isfinite(hi0) and hi0 >= hard_stop > entry_px:
        gain_pct = (entry_px - hard_stop) / entry_px * 100.0
        ts_buy = dates[entry_i]
        out = {
            "buy_date": ts_buy.strftime("%Y-%m-%d"),
            "sell_date": ts_buy.strftime("%Y-%m-%d"),
            "buy_price": round(entry_px, 4),
            "sell_price": round(float(hard_stop), 4),
            "gain_pct": round(gain_pct, 2),
            "hold_days": 0,
            "exit_reason": "hard_stop",
            "peak_price": round(float(trough), 4),
            "hard_stop_price": round(float(hard_stop), 4),
            "entry_i": int(entry_i),
            "exit_i": int(entry_i),
            "side": "short",
        }
        if include_time:
            out["buy_time"] = ts_buy.strftime("%Y-%m-%d %H:%M")
            out["sell_time"] = ts_buy.strftime("%Y-%m-%d %H:%M")
            out["hold_bars"] = 0
        return out

    for i in range(entry_i + 1, n):
        hi = float(high[i])
        lo = float(low[i])
        cl = float(close[i])
        if np.isfinite(lo):
            trough = min(trough, lo)

        if max_hold_days is not None and (i - entry_i) >= int(max_hold_days):
            exit_i = i
            exit_px = cl if np.isfinite(cl) else exit_px
            exit_reason = "time_stop"
            break

        trail_stop = trough * (1.0 + float(trail_pct)) if trough > 0 else hard_stop
        stop_level = min(hard_stop, trail_stop)

        if have_line:
            sup = _line_at(float(support_y0), int(support_x0), float(support_slope), i)
            if np.isfinite(sup) and np.isfinite(lo) and lo <= float(sup):
                # Cover at support (clip to bar); prefer support if tagged.
                cover = min(hi, max(lo, float(sup))) if np.isfinite(hi) else float(sup)
                exit_i = i
                exit_px = float(cover)
                exit_reason = "support_exit"
                break

        if np.isfinite(hi) and hi >= stop_level:
            exit_i = i
            exit_px = float(stop_level)
            exit_reason = "hard_stop" if abs(float(stop_level) - float(hard_stop)) <= 1e-9 else "trail_stop"
            break

        if i == n - 1:
            exit_i = i
            exit_px = cl if np.isfinite(cl) else exit_px
            exit_reason = "eod"

    gain_pct = (entry_px - float(exit_px)) / entry_px * 100.0
    ts_buy = dates[entry_i]
    ts_sell = dates[exit_i]
    hold = int(exit_i - entry_i)
    out = {
        "buy_date": ts_buy.strftime("%Y-%m-%d"),
        "sell_date": ts_sell.strftime("%Y-%m-%d"),
        "buy_price": round(entry_px, 4),
        "sell_price": round(float(exit_px), 4),
        "gain_pct": round(gain_pct, 2),
        "hold_days": hold,
        "exit_reason": exit_reason,
        "peak_price": round(float(trough), 4),
        "hard_stop_price": round(float(hard_stop), 4),
        "entry_i": int(entry_i),
        "exit_i": int(exit_i),
        "side": "short",
    }
    if include_time:
        out["buy_time"] = ts_buy.strftime("%Y-%m-%d %H:%M")
        out["sell_time"] = ts_sell.strftime("%Y-%m-%d %H:%M")
        out["hold_bars"] = hold
    return out


def trades_for_symbol_resist_short(
    df: pd.DataFrame,
    symbol: str,
    *,
    stop_pct: float = 0.03,
    trail_pct: float = 0.10,
    atr_stop_mult: Optional[float] = 2.0,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    entry_slip_pct: float = 0.0,
    max_l3_wait_bars: int = 120,
    min_h3_wait_bars: int = 6,
    entry_touch: int = 3,
    leave_width_frac: float = 0.20,
    touch_error_pct: Optional[float] = None,
    support_exit: bool = True,
    max_hold_days: Optional[int] = None,
    window_bars: int = 0,
    window_step_bars: int = 0,
    include_time: bool = False,
    channel_kwargs: Optional[dict] = None,
) -> List[dict]:
    """Detect causal H2 channels and short H3+ resistance tags (one position at a time)."""
    if df is None or df.empty:
        return []
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    volume = out["volume"].to_numpy(dtype=float) if "volume" in out.columns else np.zeros(len(out))
    dates = out.index
    n = len(out)
    atr = _atr(high, low, close, 14)
    ck = dict(channel_kwargs or {})
    error_pct = float(ck.get("error_pct", 1.2))
    tag_err = float(touch_error_pct) if touch_error_pct is not None else error_pct
    if int(window_bars) > 0:
        setups = find_h2_l3_setups_windowed(
            out,
            window_bars=int(window_bars),
            step_bars=int(window_step_bars or window_bars),
            **ck,
        )
    else:
        setups = find_h2_l3_setups(out, **ck)

    pending: List[Tuple[dict, int, float, int]] = []
    for ch in setups:
        h2 = int(ch.get("h2_idx", -1))
        if h2 < 0 or h2 >= n - 1:
            continue
        tags = find_h3_plus_resist_tags(
            high,
            low,
            close,
            support_x0=int(ch["support_x0"]),
            support_y0=float(ch["support_y0"]),
            support_slope=float(ch["support_slope"]),
            width=float(ch["channel_width"]),
            h2=h2,
            n=n,
            error_pct=tag_err,
            slip=float(entry_slip_pct),
            wait=int(max_l3_wait_bars),
            min_wait=int(min_h3_wait_bars),
            entry_touch=int(entry_touch),
            leave_width_frac=float(leave_width_frac),
        )
        for entry_i, fill_px, touch_num in tags:
            pending.append((ch, int(entry_i), float(fill_px), int(touch_num)))

    pending.sort(key=lambda t: t[1])
    trades: List[dict] = []
    busy_until = -1
    for ch, entry_i, fill_px, touch_num in pending:
        if entry_i <= busy_until or entry_i >= n:
            continue
        sx0 = int(ch["support_x0"])
        sy0 = float(ch["support_y0"])
        sslope = float(ch["support_slope"])
        width = float(ch["channel_width"])
        atr_i = float(atr[max(0, entry_i - 1)]) if entry_i > 0 else float(atr[entry_i])
        sim = simulate_short_trade(
            high,
            low,
            close,
            dates,
            entry_i,
            stop_pct=stop_pct,
            trail_pct=trail_pct,
            atr_at_entry=atr_i if np.isfinite(atr_i) else None,
            atr_stop_mult=atr_stop_mult,
            stop_pct_floor=stop_pct_floor,
            stop_pct_ceil=stop_pct_ceil,
            support_x0=sx0,
            support_y0=sy0,
            support_slope=sslope,
            channel_width=width,
            support_exit=bool(support_exit),
            max_hold_days=max_hold_days,
            entry_px=fill_px,
            include_time=include_time,
            skip_entry_bar_stop=True,
        )
        if sim is None:
            continue
        entry_px = float(sim["buy_price"])
        atr_pct = (atr_i / entry_px * 100.0) if entry_px > 0 and np.isfinite(atr_i) else float("nan")
        adv = _adv_20(close, volume, max(0, entry_i - 1))
        support_at = _line_at(sy0, sx0, sslope, entry_i)
        resist_at = support_at + width
        channel_pos = (
            (entry_px - support_at) / width
            if width > 0 and np.isfinite(support_at)
            else float("nan")
        )
        beyond = max_beyond_width(high, sy0, sx0, sslope, width, sx0, entry_i)
        h2_idx = int(ch.get("h2_idx", entry_i))
        formation_beyond = max_beyond_width(high, sy0, sx0, sslope, width, sx0, max(sx0, h2_idx))
        buy_ts = pd.Timestamp(dates[entry_i])
        try:
            ch_start_ts = pd.Timestamp(ch["start_date"])
            ch_end_ts = pd.Timestamp(ch.get("h2_date") or ch["end_date"])
            span_days = int((ch_end_ts - ch_start_ts).days)
            age_days = int((buy_ts - ch_start_ts).days)
        except Exception:
            span_days = None
            age_days = None
        trades.append(
            {
                "stock": symbol.upper(),
                "channel_start": ch["start_date"],
                "channel_end": ch.get("h2_date") or ch["end_date"],
                "touch_num": touch_num,
                "touch_date": buy_ts.strftime("%Y-%m-%d"),
                **{k: v for k, v in sim.items() if k not in ("entry_i", "exit_i")},
                "entry_i": sim["entry_i"],
                "exit_i": sim["exit_i"],
                "adv_20": round(adv, 2) if np.isfinite(adv) else None,
                "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
                **channel_rail_fields(ch, dates, low, include_time=include_time),
                "slope_pct_per_bar": ch.get("slope_pct_per_bar"),
                "channel_width_pct": ch.get("channel_width_pct"),
                "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
                "bars_span": ch.get("bars_span"),
                "entry_mode": "resist_short",
                "wait_bars": int(entry_i - h2_idx),
                "max_beyond_width": round(float(beyond), 4) if np.isfinite(beyond) else None,
                "formation_beyond_width": (
                    round(float(formation_beyond), 4) if np.isfinite(formation_beyond) else None
                ),
                "channel_span_days": span_days,
                "channel_age_at_buy_days": age_days,
                "resist_at_entry": round(float(resist_at), 4) if np.isfinite(resist_at) else None,
                "support_at_entry": round(float(support_at), 4) if np.isfinite(support_at) else None,
            }
        )
        busy_until = int(sim["exit_i"])
    return trades


def _worker(payload: dict) -> List[dict]:
    return trades_for_symbol_resist_short(
        payload["df"],
        payload["symbol"],
        stop_pct=float(payload.get("stop_pct", 0.03)),
        trail_pct=float(payload.get("trail_pct", 0.10)),
        atr_stop_mult=payload.get("atr_stop_mult", 2.0),
        stop_pct_floor=float(payload.get("stop_pct_floor", 0.015)),
        stop_pct_ceil=float(payload.get("stop_pct_ceil", 0.06)),
        entry_slip_pct=float(payload.get("entry_slip_pct", 0.0)),
        max_l3_wait_bars=int(payload.get("max_l3_wait_bars", 120)),
        min_h3_wait_bars=int(payload.get("min_h3_wait_bars", 6)),
        entry_touch=int(payload.get("entry_touch", 3)),
        leave_width_frac=float(payload.get("leave_width_frac", 0.20)),
        touch_error_pct=payload.get("touch_error_pct"),
        support_exit=bool(payload.get("support_exit", True)),
        max_hold_days=payload.get("max_hold_days"),
        window_bars=int(payload.get("window_bars", 0) or 0),
        window_step_bars=int(payload.get("window_step_bars", 0) or 0),
        include_time=bool(payload.get("include_time", False)),
        channel_kwargs=payload.get("channel_kwargs") or {},
    )


def scan_resist_short(
    panels: Dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    workers: int,
    base: dict,
) -> pd.DataFrame:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    payloads = []
    for sym in symbols:
        if sym not in panels or str(sym).upper() == "SPY":
            continue
        payloads.append({"symbol": sym, "df": panels[sym], **base})
    all_trades: List[dict] = []
    if workers and workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = [pool.submit(_worker, p) for p in payloads]
            for fut in as_completed(futs):
                all_trades.extend(fut.result())
    else:
        for p in payloads:
            all_trades.extend(_worker(p))
    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def geo_year_pf(year_df: pd.DataFrame, *, min_n: int = 30) -> Optional[float]:
    """Equal-weight geometric mean of YEAR_BUCKETS PFs; skip buckets with n < min_n."""
    if year_df is None or year_df.empty or "profit_factor" not in year_df.columns:
        return None
    pfs: List[float] = []
    for _, row in year_df.iterrows():
        if str(row.get("bucket", "")) == "ALL":
            continue
        n = int(row.get("n_trades") or 0)
        pf = row.get("profit_factor")
        if n < int(min_n) or pf is None or not np.isfinite(float(pf)) or float(pf) <= 0:
            continue
        pfs.append(float(pf))
    if not pfs:
        return None
    return float(np.exp(np.mean(np.log(pfs))))
