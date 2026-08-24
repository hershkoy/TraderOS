"""
Live ascending-channel bottom-touch triggers (EOD).

Detects pivot-confirmation entries on the as-of bar (last bar by default).
Production defaults follow edge-v2 keepers:
  RS top1/day, squeeze-adaptive trail research params, ATR hard-stop k=2.0 clamped.
"""
from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "scripts" / "research"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from find_ascending_channels import find_channels  # noqa: E402
from backtest_channel_touch_trades import (  # noqa: E402
    _adv_20,
    _atr,
    _hard_stop_price,
    _line_at,
    _resolve_entry_i,
    enrich_rs,
    select_same_day_rs,
)

logger = logging.getLogger(__name__)


def live_entries_for_symbol(
    symbol: str,
    df: pd.DataFrame,
    *,
    as_of: Optional[pd.Timestamp] = None,
    entry_touch: int = 3,
    pivot_len: int = 15,
    entry_mode: str = "pivot",
    stop_pct: float = 0.03,
    atr_stop_mult: Optional[float] = 2.0,
    stop_pct_floor: float = 0.015,
    stop_pct_ceil: float = 0.06,
    adv_lookback: int = 20,
    atr_len: int = 14,
    **channel_kwargs,
) -> List[dict]:
    """
    Return channel-touch long entries whose pivot confirmation bar equals as_of
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
    if n < 1:
        return []

    if as_of is None:
        target_i = n - 1
    else:
        asof_ts = pd.Timestamp(as_of)
        if asof_ts.tzinfo is not None:
            asof_ts = asof_ts.tz_convert(None)
        hist = dates[dates <= asof_ts]
        if len(hist) == 0:
            return []
        target_i = int(dates.get_loc(hist[-1]))
        if isinstance(target_i, slice):
            return []

    channels = find_channels(out, pivot_len=pivot_len, **channel_kwargs)
    triggers: List[dict] = []

    for ch in channels:
        touch_idxs: List[int] = list(ch.get("touch_indices") or [])
        if len(touch_idxs) < entry_touch:
            continue
        sx0 = int(ch["support_x0"])
        sy0 = float(ch["support_y0"])
        sslope = float(ch["support_slope"])
        width = float(ch["channel_width"])
        pl = int(ch.get("pivot_len", pivot_len))

        for touch_num, t_idx in enumerate(touch_idxs, start=1):
            if touch_num < entry_touch:
                continue
            entry_i = _resolve_entry_i(
                t_idx=int(t_idx),
                pivot_len=pl,
                entry_mode=entry_mode,
                close=close,
                n=n,
                support_x0=sx0,
                support_y0=sy0,
                support_slope=sslope,
            )
            if entry_i is None or int(entry_i) != int(target_i):
                continue

            entry_px = float(close[entry_i])
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
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
            triggers.append(
                {
                    "stock": symbol.upper(),
                    "buy_date": dates[entry_i].strftime("%Y-%m-%d"),
                    "buy_price": round(entry_px, 4),
                    "hard_stop": round(float(hard_stop), 4),
                    "stop_pct_used": round((1.0 - float(hard_stop) / entry_px) * 100.0, 3),
                    "atr_stop_mult": atr_stop_mult,
                    "channel_start": ch["start_date"],
                    "channel_end": ch["end_date"],
                    "touch_num": touch_num,
                    "touch_date": dates[int(t_idx)].strftime("%Y-%m-%d"),
                    "touch_price": round(float(low[int(t_idx)]), 4),
                    "adv_20": round(adv, 2) if np.isfinite(adv) else None,
                    "atr_pct": round(atr_pct, 3) if np.isfinite(atr_pct) else None,
                    "slope_pct_per_bar": ch.get("slope_pct_per_bar"),
                    "channel_width_pct": ch.get("channel_width_pct"),
                    "channel_pos": round(float(channel_pos), 3) if np.isfinite(channel_pos) else None,
                    "room_to_resist_pct": (
                        round(float(room_to_resist_pct), 3) if np.isfinite(room_to_resist_pct) else None
                    ),
                    "entry_mode": entry_mode,
                    "pivot_len": pl,
                }
            )
    return triggers


def _worker_live_entries(payload: dict) -> List[dict]:
    return live_entries_for_symbol(
        payload["symbol"],
        payload["df"],
        as_of=payload.get("as_of"),
        entry_touch=int(payload.get("entry_touch", 3)),
        pivot_len=int(payload.get("pivot_len", 15)),
        entry_mode=str(payload.get("entry_mode", "pivot")),
        stop_pct=float(payload.get("stop_pct", 0.03)),
        atr_stop_mult=payload.get("atr_stop_mult", 2.0),
        stop_pct_floor=float(payload.get("stop_pct_floor", 0.015)),
        stop_pct_ceil=float(payload.get("stop_pct_ceil", 0.06)),
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
) -> pd.DataFrame:
    """
    Scan universe for as-of entries, enrich RS vs SPY, keep top-N per day.
    """
    base = {
        "as_of": as_of,
        "entry_touch": entry_touch,
        "pivot_len": pivot_len,
        "entry_mode": "pivot",
        "stop_pct": stop_pct,
        "atr_stop_mult": atr_stop_mult,
        "stop_pct_floor": stop_pct_floor,
        "stop_pct_ceil": stop_pct_ceil,
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

    if not rows:
        return pd.DataFrame()

    trades = pd.DataFrame(rows)
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
) -> str:
    """Plain-text Telegram / log summary."""
    lines = [
        "Channel-touch nightly",
        f"as_of={as_of}",
        f"candidates={n_candidates} | triggers={len(triggers)} (RS top filtered)",
    ]
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
            "  channel {start}..{end} | pos={pos} | atr%={atr}".format(
                start=row.get("channel_start"),
                end=row.get("channel_end"),
                pos=row.get("channel_pos"),
                atr=row.get("atr_pct"),
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
