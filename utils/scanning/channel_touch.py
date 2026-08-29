"""
Live ascending-channel bottom-touch triggers (EOD).

Detects l3_touch fills on the as-of bar (last bar by default): arm at H2,
fill the first from-above support tag after min-wait, then quality-filter
and RS-rank. Detector v1 find_channels is unchanged; live setups use
find_h2_l3_setups.

Production defaults match the 2026-08-28 l3_touch opt keeper:
  entry_mode=l3_touch, min_l3_wait_bars=6, max_rsi=50,
  require_in_channel, max_channel_span_days=365, max_beyond_width=0.25,
  RS top1/day, ATR hard-stop k=2.0 clamped, windowed 504/252.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from find_ascending_channels import (  # noqa: E402
    find_channels,
    find_channels_windowed,
    find_h2_l3_setups,
    find_h2_l3_setups_windowed,
)
from backtest_channel_touch_trades import (  # noqa: E402
    _adv_20,
    _atr,
    _h2_rail_tag_fills,
    _hard_stop_price,
    _line_at,
    _resolve_entry_i,
    enrich_rs,
    filter_trades,
    select_same_day_rs,
)
from utils.research.channel_touch_entry_features import (  # noqa: E402
    max_beyond_width,
    stock_entry_feature_series,
)
from utils.research.channel_touch_scale import (  # noqa: E402
    DAILY_WINDOW_BARS,
    DAILY_WINDOW_STEP_BARS,
)

logger = logging.getLogger(__name__)

# Daily l3_touch keeper (report 20260829_105855 / trades 20260828_194314).
LIVE_DEFAULTS: Dict[str, Any] = {
    "entry_mode": "l3_touch",
    "entry_touch": 3,
    "pivot_len": 15,
    "min_l3_wait_bars": 6,
    "max_l3_wait_bars": 252,
    "entry_slip_pct": 0.001,
    "max_rsi": 50.0,
    "require_in_channel": True,
    "max_channel_span_days": 365.0,
    "max_beyond_width": 0.25,
    "atr_stop_mult": 2.0,
    "stop_pct": 0.03,
    "stop_pct_floor": 0.015,
    "stop_pct_ceil": 0.06,
    "max_entries_per_day": 1,
    "window_bars": DAILY_WINDOW_BARS,
    "window_step_bars": DAILY_WINDOW_STEP_BARS,
    "error_pct": 1.2,
    "flat_pct": 0.04,
    "min_bars_apart": 15,
    "min_rally_pct": 4.0,
    "min_pullback_pct": 3.0,
    "min_total_rise_pct": 3.0,
    "max_low_pivots": 16,
}


def _remap_channel_kwargs(channel_kwargs: dict) -> dict:
    kw = dict(channel_kwargs)
    if "min_rally_pct" in kw:
        kw.setdefault("min_intervening_rally_pct", kw.pop("min_rally_pct"))
    if "min_pullback_pct" in kw:
        kw.setdefault("min_intervening_pullback_pct", kw.pop("min_pullback_pct"))
    return kw


def _asof_index(dates: pd.DatetimeIndex, as_of: Optional[pd.Timestamp], n: int) -> Optional[int]:
    if n < 1:
        return None
    if as_of is None:
        return n - 1
    asof_ts = pd.Timestamp(as_of)
    if asof_ts.tzinfo is not None:
        asof_ts = asof_ts.tz_convert(None)
    hist = dates[dates <= asof_ts]
    if len(hist) == 0:
        return None
    target_i = dates.get_loc(hist[-1])
    if isinstance(target_i, slice):
        return None
    return int(target_i)


def _span_age_days(ch: dict, buy_ts: pd.Timestamp) -> tuple:
    try:
        ch_start_ts = pd.Timestamp(ch["start_date"])
        ch_end_ts = pd.Timestamp(ch.get("h2_date") or ch["end_date"])
        span_days = int((ch_end_ts - ch_start_ts).days)
        age_days = int((buy_ts - ch_start_ts).days)
        return span_days, age_days
    except Exception:
        return None, None


def _build_trigger(
    *,
    symbol: str,
    dates: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    atr: np.ndarray,
    rsi: np.ndarray,
    ch: dict,
    entry_i: int,
    entry_px: float,
    touch_num: int,
    t_idx: int,
    entry_mode: str,
    atr_stop_mult: Optional[float],
    stop_pct: float,
    stop_pct_floor: float,
    stop_pct_ceil: float,
    adv_lookback: int,
) -> Optional[dict]:
    if not np.isfinite(entry_px) or entry_px <= 0:
        return None
    sx0 = int(ch["support_x0"])
    sy0 = float(ch["support_y0"])
    sslope = float(ch["support_slope"])
    width = float(ch["channel_width"])
    atr_i = float(atr[entry_i]) if entry_i < len(atr) else float("nan")
    hard_stop = _hard_stop_price(
        entry_px,
        stop_pct=stop_pct,
        atr_at_entry=atr_i if np.isfinite(atr_i) else None,
        atr_stop_mult=atr_stop_mult,
        stop_pct_floor=stop_pct_floor,
        stop_pct_ceil=stop_pct_ceil,
    )
    atr_pct = (atr_i / entry_px * 100.0) if entry_px > 0 and np.isfinite(atr_i) else float("nan")
    adv = _adv_20(close, volume, entry_i, lookback=adv_lookback)
    support_at = _line_at(sy0, sx0, sslope, entry_i)
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
    beyond = max_beyond_width(high, sy0, sx0, sslope, width, sx0, entry_i)
    buy_ts = pd.Timestamp(dates[entry_i])
    span_days, age_days = _span_age_days(ch, buy_ts)
    h2_idx = int(ch.get("h2_idx", t_idx))
    wait_bars = int(entry_i - h2_idx) if h2_idx >= 0 else None
    rsi_i = float(rsi[entry_i]) if 0 <= entry_i < len(rsi) else float("nan")
    touch_px = float(low[int(t_idx)]) if 0 <= int(t_idx) < len(low) else float("nan")
    return {
        "stock": symbol.upper(),
        "buy_date": buy_ts.strftime("%Y-%m-%d"),
        "buy_price": round(entry_px, 4),
        "hard_stop": round(float(hard_stop), 4),
        "stop_pct_used": round((1.0 - float(hard_stop) / entry_px) * 100.0, 3),
        "atr_stop_mult": atr_stop_mult,
        "channel_start": ch["start_date"],
        "channel_end": ch.get("h2_date") or ch["end_date"],
        "h2_date": ch.get("h2_date"),
        "touch_num": touch_num,
        "touch_date": dates[int(t_idx)].strftime("%Y-%m-%d") if 0 <= int(t_idx) < len(dates) else None,
        "touch_price": round(touch_px, 4) if np.isfinite(touch_px) else None,
        "adv_20": round(adv, 2) if np.isfinite(adv) else None,
        "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
        "slope_pct_per_bar": ch.get("slope_pct_per_bar"),
        "channel_width_pct": ch.get("channel_width_pct"),
        "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
        "room_to_resist_pct": (
            round(float(room_to_resist_pct), 3) if np.isfinite(room_to_resist_pct) else None
        ),
        "entry_mode": entry_mode,
        "pivot_len": int(ch.get("pivot_len", LIVE_DEFAULTS["pivot_len"])),
        "wait_bars": wait_bars,
        "max_beyond_width": round(float(beyond), 4) if np.isfinite(beyond) else None,
        "channel_span_days": span_days,
        "channel_age_at_buy_days": age_days,
        "rsi_14": round(rsi_i, 4) if np.isfinite(rsi_i) else None,
    }


def live_entries_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    as_of: Optional[pd.Timestamp] = None,
    entry_touch: int = 3,
    pivot_len: int = 15,
    entry_mode: str = "l3_touch",
    stop_pct: float = 0.03,
    atr_stop_mult: Optional[float] = 2.0,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    adv_lookback: int = 20,
    atr_len: int = 14,
    min_l3_wait_bars: int = 6,
    max_l3_wait_bars: int = 252,
    entry_slip_pct: float = 0.001,
    window_bars: Optional[int] = None,
    window_step_bars: Optional[int] = None,
    **channel_kwargs,
) -> List[dict]:
    """
    Return channel-touch long entries whose fill bar equals as_of
    (or the last bar if as_of is None). Does not require a future exit bar.
    """
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
    volume = (
        out["volume"].to_numpy(dtype=float)
        if "volume" in out.columns
        else np.full(len(out), np.nan, dtype=float)
    )
    atr = _atr(high, low, close, length=atr_len)
    dates = out.index
    n = len(out)
    target_i = _asof_index(dates, as_of, n)
    if target_i is None:
        return []

    mode = (entry_mode or "l3_touch").lower().strip()
    ck = _remap_channel_kwargs(channel_kwargs)
    error_pct = float(ck.get("error_pct", LIVE_DEFAULTS["error_pct"]))
    rsi = stock_entry_feature_series(out)["rsi_14"]
    triggers: List[dict] = []
    use_window = window_bars is not None and int(window_bars) > 0
    step = int(window_step_bars or window_bars or 0)

    if mode == "l3_touch":
        setups = (
            find_h2_l3_setups_windowed(
                out,
                window_bars=int(window_bars),
                step_bars=step,
                pivot_len=pivot_len,
                **ck,
            )
            if use_window
            else find_h2_l3_setups(out, pivot_len=pivot_len, **ck)
        )
        wait = max(1, int(max_l3_wait_bars))
        min_wait = max(1, int(min_l3_wait_bars))
        slip = float(entry_slip_pct)
        want_touch = max(3, int(entry_touch))
        for ch in setups:
            h2 = int(ch.get("h2_idx", -1))
            if h2 < 0:
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
                error_pct=error_pct,
                slip=slip,
                wait=wait,
                min_wait=min_wait,
                entry_touch=want_touch,
            )
            for i, fill, tnum in tags:
                if int(i) != int(target_i):
                    continue
                row = _build_trigger(
                    symbol=symbol,
                    dates=dates,
                    high=high,
                    low=low,
                    close=close,
                    volume=volume,
                    atr=atr,
                    rsi=rsi,
                    ch=ch,
                    entry_i=int(i),
                    entry_px=float(fill),
                    touch_num=int(tnum),
                    t_idx=int(i),
                    entry_mode=mode,
                    atr_stop_mult=atr_stop_mult,
                    stop_pct=stop_pct,
                    stop_pct_floor=stop_pct_floor,
                    stop_pct_ceil=stop_pct_ceil,
                    adv_lookback=adv_lookback,
                )
                if row is not None:
                    triggers.append(row)
        return triggers

    channels = (
        find_channels_windowed(
            out,
            window_bars=int(window_bars),
            step_bars=step,
            pivot_len=pivot_len,
            **ck,
        )
        if use_window
        else find_channels(out, pivot_len=pivot_len, **ck)
    )
    for ch in channels:
        touch_idxs: List[int] = list(ch.get("touch_indices") or [])
        if len(touch_idxs) < entry_touch:
            continue
        sx0 = int(ch["support_x0"])
        sy0 = float(ch["support_y0"])
        sslope = float(ch["support_slope"])
        pl = int(ch.get("pivot_len", pivot_len))
        for touch_num, t_idx in enumerate(touch_idxs, start=1):
            if touch_num < entry_touch:
                continue
            entry_i = _resolve_entry_i(
                t_idx=int(t_idx),
                pivot_len=pl,
                entry_mode=mode,
                close=close,
                n=n,
                support_x0=sx0,
                support_y0=sy0,
                support_slope=sslope,
            )
            if entry_i is None or int(entry_i) != int(target_i):
                continue
            row = _build_trigger(
                symbol=symbol,
                dates=dates,
                high=high,
                low=low,
                close=close,
                volume=volume,
                atr=atr,
                rsi=rsi,
                ch=ch,
                entry_i=int(entry_i),
                entry_px=float(close[entry_i]),
                touch_num=int(touch_num),
                t_idx=int(t_idx),
                entry_mode=mode,
                atr_stop_mult=atr_stop_mult,
                stop_pct=stop_pct,
                stop_pct_floor=stop_pct_floor,
                stop_pct_ceil=stop_pct_ceil,
                adv_lookback=adv_lookback,
            )
            if row is not None:
                triggers.append(row)
    return triggers


def _worker_live_entries(payload: dict) -> List[dict]:
    return live_entries_for_symbol(
        payload["symbol"],
        payload["df"],
        as_of=payload.get("as_of"),
        entry_touch=int(payload.get("entry_touch", LIVE_DEFAULTS["entry_touch"])),
        pivot_len=int(payload.get("pivot_len", LIVE_DEFAULTS["pivot_len"])),
        entry_mode=str(payload.get("entry_mode", LIVE_DEFAULTS["entry_mode"])),
        stop_pct=float(payload.get("stop_pct", LIVE_DEFAULTS["stop_pct"])),
        atr_stop_mult=payload.get("atr_stop_mult", LIVE_DEFAULTS["atr_stop_mult"]),
        stop_pct_floor=float(payload.get("stop_pct_floor", LIVE_DEFAULTS["stop_pct_floor"])),
        stop_pct_ceil=float(payload.get("stop_pct_ceil", LIVE_DEFAULTS["stop_pct_ceil"])),
        min_l3_wait_bars=int(payload.get("min_l3_wait_bars", LIVE_DEFAULTS["min_l3_wait_bars"])),
        max_l3_wait_bars=int(payload.get("max_l3_wait_bars", LIVE_DEFAULTS["max_l3_wait_bars"])),
        entry_slip_pct=float(payload.get("entry_slip_pct", LIVE_DEFAULTS["entry_slip_pct"])),
        window_bars=payload.get("window_bars"),
        window_step_bars=payload.get("window_step_bars"),
        **(payload.get("channel_kwargs") or {}),
    )


def scan_live_triggers(
    panels: Dict[str, pd.DataFrame],
    *,
    symbols: Sequence[str],
    spy_df: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    workers: int = 4,
    max_entries_per_day: int = 1,
    entry_touch: int = 3,
    pivot_len: int = 15,
    atr_stop_mult: float = 2.0,
    stop_pct: float = 0.03,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    entry_mode: str = "l3_touch",
    min_l3_wait_bars: int = 6,
    max_l3_wait_bars: int = 252,
    entry_slip_pct: float = 0.001,
    window_bars: Optional[int] = DAILY_WINDOW_BARS,
    window_step_bars: Optional[int] = DAILY_WINDOW_STEP_BARS,
    require_in_channel: bool = True,
    max_channel_span_days: Optional[float] = 365.0,
    max_beyond_width: Optional[float] = 0.25,
    max_rsi: Optional[float] = 50.0,
    stats: Optional[dict] = None,
    **channel_kwargs,
) -> pd.DataFrame:
    """
    Scan universe for as-of entries, apply quality filters, enrich RS vs SPY, keep top-N per day.
    """
    ck = {
        "error_pct": float(channel_kwargs.get("error_pct", LIVE_DEFAULTS["error_pct"])),
        "flat_pct": float(channel_kwargs.get("flat_pct", LIVE_DEFAULTS["flat_pct"])),
        "min_bars_apart": int(channel_kwargs.get("min_bars_apart", LIVE_DEFAULTS["min_bars_apart"])),
        "min_rally_pct": float(channel_kwargs.get("min_rally_pct", LIVE_DEFAULTS["min_rally_pct"])),
        "min_pullback_pct": float(
            channel_kwargs.get("min_pullback_pct", LIVE_DEFAULTS["min_pullback_pct"])
        ),
        "min_total_rise_pct": float(
            channel_kwargs.get("min_total_rise_pct", LIVE_DEFAULTS["min_total_rise_pct"])
        ),
        "max_low_pivots": int(channel_kwargs.get("max_low_pivots", LIVE_DEFAULTS["max_low_pivots"])),
    }
    base = {
        "as_of": as_of,
        "entry_touch": entry_touch,
        "pivot_len": pivot_len,
        "entry_mode": entry_mode,
        "stop_pct": stop_pct,
        "atr_stop_mult": atr_stop_mult,
        "stop_pct_floor": stop_pct_floor,
        "stop_pct_ceil": stop_pct_ceil,
        "min_l3_wait_bars": min_l3_wait_bars,
        "max_l3_wait_bars": max_l3_wait_bars,
        "entry_slip_pct": entry_slip_pct,
        "window_bars": window_bars,
        "window_step_bars": window_step_bars,
        "channel_kwargs": ck,
    }
    payloads = [
        {"symbol": sym, "df": panels[sym], **base}
        for sym in symbols
        if sym in panels and str(sym).upper() != "SPY"
    ]
    rows: List[dict] = []
    if workers and workers > 1 and len(payloads) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as pool:
            futs = [pool.submit(_worker_live_entries, p) for p in payloads]
            for fut in as_completed(futs):
                rows.extend(fut.result())
    else:
        for p in payloads:
            rows.extend(_worker_live_entries(p))

    n_raw = len(rows)
    if not rows:
        if stats is not None:
            stats["n_raw"] = 0
            stats["n_quality"] = 0
        return pd.DataFrame()

    trades = pd.DataFrame(rows)
    trades = filter_trades(
        trades,
        require_in_channel=bool(require_in_channel),
        max_channel_span_days=max_channel_span_days,
        max_beyond_width=max_beyond_width,
        max_rsi=max_rsi,
    )
    n_quality = 0 if trades.empty else len(trades)
    if stats is not None:
        stats["n_raw"] = n_raw
        stats["n_quality"] = n_quality
    if trades.empty:
        return trades.reset_index(drop=True)

    trades = enrich_rs(trades, panels, spy_df)
    if max_entries_per_day and max_entries_per_day > 0:
        trades = select_same_day_rs(
            trades, rs_col="rs_spy_126d", max_per_day=int(max_entries_per_day)
        )
    return trades.reset_index(drop=True)


def format_triggers_message(
    triggers: pd.DataFrame,
    *,
    as_of: str,
    n_candidates: int,
    update_stats: Optional[dict] = None,
    n_raw: Optional[int] = None,
    entry_mode: str = "l3_touch",
    min_l3_wait_bars: int = 6,
    max_rsi: float = 50.0,
    max_beyond_width: float = 0.25,
    max_channel_span_days: float = 365.0,
) -> str:
    """Plain-text Telegram / log summary."""
    lines = [
        "Channel-touch nightly",
        f"as_of={as_of}",
        (
            f"mode={entry_mode} min_wait={min_l3_wait_bars} max_rsi={max_rsi} "
            f"in_channel span<={max_channel_span_days} beyond<={max_beyond_width}"
        ),
    ]
    if n_raw is not None:
        lines.append(
            f"raw={n_raw} | quality={n_candidates} | triggers={len(triggers)} (RS top filtered)"
        )
    else:
        lines.append(f"candidates={n_candidates} | triggers={len(triggers)} (RS top filtered)")
    if update_stats:
        lines.append(
            "data_update: saved={saved} failed={failed} skipped={skipped}".format(
                saved=update_stats.get("successful", update_stats.get("saved", "?")),
                failed=update_stats.get("failed", "?"),
                skipped=update_stats.get("skipped", "?"),
            )
        )
    if triggers is None or triggers.empty:
        lines.append("No new triggers.")
        return "\n".join(lines)

    lines.append("")
    for _, row in triggers.iterrows():
        rs = row.get("rs_spy_126d")
        rs_s = f"{rs:+.2f}%" if rs is not None and np.isfinite(float(rs)) else "n/a"
        rsi = row.get("rsi_14")
        rsi_s = f"{float(rsi):.1f}" if rsi is not None and np.isfinite(float(rsi)) else "n/a"
        wait = row.get("wait_bars")
        wait_s = str(int(wait)) if wait is not None and np.isfinite(float(wait)) else "n/a"
        lines.append(
            "{stock} @ {px} | stop={stop} (-{spct:.2f}%) | RS126={rs} | touch#{tn}".format(
                stock=row["stock"],
                px=row["buy_price"],
                stop=row["hard_stop"],
                spct=float(row.get("stop_pct_used") or 0.0),
                rs=rs_s,
                tn=int(row.get("touch_num") or 0),
            )
        )
        lines.append(
            "  channel {start}..{end} | pos={pos} | atr%={atr} | rsi={rsi} | wait={wait}".format(
                start=row.get("channel_start"),
                end=row.get("channel_end"),
                pos=row.get("channel_pos"),
                atr=row.get("atr_pct"),
                rsi=rsi_s,
                wait=wait_s,
            )
        )
    return "\n".join(lines)


def resolve_as_of_from_panels(panels: Dict[str, pd.DataFrame], spy_df: pd.DataFrame) -> pd.Timestamp:
    """Prefer SPY last bar; else max last timestamp across panels."""
    if spy_df is not None and not spy_df.empty:
        idx = spy_df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        ts = idx.max()
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None)
        return pd.Timestamp(ts)
    lasts = []
    for df in panels.values():
        if df is None or df.empty:
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.DatetimeIndex(idx)
        lasts.append(idx.max())
    if not lasts:
        return pd.Timestamp(datetime.utcnow().date())
    ts = max(lasts)
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return pd.Timestamp(ts)
