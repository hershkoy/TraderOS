"""Fill-session RTH extension: 09:30 open to 15:45 open, in ATR units.

Last-15m fills decide at 15:45 ET. The 15:45 open is known; the 15:45 close
is not. 15m ATR uses completed bars before that open. Daily ATR uses the
prior session (today's daily bar is not closed).
"""
from __future__ import annotations

from datetime import date, time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from utils.research.channel_touch_entry_features import completed_asof
from utils.research.last_15m_volume_geometry import fill_day_et
from utils.research.realistic_purchaser import as_et
from indicators.trend import ATR

RTH_OPEN = time(9, 30)
LAST_15M = time(15, 45)
ATR_15M_LEN = 14
MIN_ATR_BARS = 5


def _bar_open(bar: Mapping[str, Any]) -> Optional[float]:
    try:
        px = float(bar.get("open"))
    except (TypeError, ValueError):
        return None
    if px != px or px <= 0:
        return None
    return px


def _session_bars(bars: Sequence[Mapping[str, Any]], fill_day: date) -> list:
    out = []
    for bar in bars:
        ts = bar.get("ts")
        if ts is None:
            continue
        try:
            et = as_et(ts)
        except (TypeError, ValueError):
            continue
        if et.date() != fill_day:
            continue
        out.append(bar)
    out.sort(key=lambda b: as_et(b["ts"]))
    return out


def session_rth_extension(
    bars: Sequence[Mapping[str, Any]],
    fill_day: date,
    *,
    atr_len: int = ATR_15M_LEN,
) -> dict:
    """Return rth_open, last15_open, run, atr_15m (pre-15:45), run_atr."""
    empty = {
        "rth_open": None,
        "last15_open": None,
        "run": None,
        "atr_15m": None,
        "run_atr_15m": None,
    }
    sess = _session_bars(bars, fill_day)
    if not sess:
        return empty
    open_bar = None
    last_bar = None
    for bar in sess:
        clock = as_et(bar["ts"]).replace(microsecond=0, tzinfo=None).time()
        if clock == RTH_OPEN:
            open_bar = bar
        if open_bar is None and clock >= RTH_OPEN:
            open_bar = bar
        if clock == LAST_15M:
            last_bar = bar
    if open_bar is None:
        open_bar = sess[0]
    if last_bar is None:
        return empty
    rth_open = _bar_open(open_bar)
    last15_open = _bar_open(last_bar)
    if rth_open is None or last15_open is None:
        return empty
    run = float(last15_open - rth_open)
    last_ts = as_et(last_bar["ts"])
    prior = [b for b in sess if as_et(b["ts"]) < last_ts]
    atr_15m = None
    if len(prior) >= MIN_ATR_BARS:
        highs = pd.Series([float(b["high"]) for b in prior], dtype=float)
        lows = pd.Series([float(b["low"]) for b in prior], dtype=float)
        closes = pd.Series([float(b["close"]) for b in prior], dtype=float)
        n = min(int(atr_len), len(prior))
        atr_s = ATR(highs, lows, closes, period=n)
        if atr_s is not None and len(atr_s.dropna()):
            atr_15m = float(atr_s.dropna().iloc[-1])
            if not np.isfinite(atr_15m) or atr_15m <= 0:
                atr_15m = None
    run_atr = None if atr_15m is None else float(run / atr_15m)
    return {
        "rth_open": rth_open,
        "last15_open": last15_open,
        "run": run,
        "atr_15m": atr_15m,
        "run_atr_15m": run_atr,
    }


def prior_session_daily_atr(
    daily: pd.DataFrame,
    asof: pd.Timestamp,
    *,
    period: int = 14,
) -> Optional[float]:
    if daily is None or daily.empty or asof is None:
        return None
    df = daily.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    ts = pd.Timestamp(asof)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    hist = df.loc[: ts.normalize()]
    need = ("high", "low", "close")
    if hist.empty or not set(need).issubset(hist.columns) or len(hist) < max(5, period // 2):
        return None
    atr = ATR(
        hist["high"].astype(float),
        hist["low"].astype(float),
        hist["close"].astype(float),
        period=int(period),
    )
    val = atr.dropna()
    if val.empty:
        return None
    out = float(val.iloc[-1])
    if not np.isfinite(out) or out <= 0:
        return None
    return out


def attach_exhaustion(
    trades: pd.DataFrame,
    indexed_15m: Mapping[str, Mapping[date, Sequence[Mapping[str, Any]]]],
    daily_panels: Optional[Mapping[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    out = trades.copy()
    rows = []
    for _, row in out.iterrows():
        rec = {
            "rth_open": None,
            "last15_open": None,
            "run": None,
            "atr_15m": None,
            "run_atr_15m": None,
            "atr_1d_prior": None,
            "run_atr_1d": None,
        }
        fill_day = fill_day_et(row)
        sym = str(row.get("stock") or "").upper()
        by_day = indexed_15m.get(sym) if indexed_15m is not None else None
        bars = None
        if fill_day is not None and by_day:
            bars = by_day.get(fill_day) or []
        if bars:
            rec.update(session_rth_extension(bars, fill_day))
        asof = completed_asof(row, series_is_daily=True)
        dpanel = None if daily_panels is None else daily_panels.get(sym)
        atr_d = prior_session_daily_atr(dpanel, asof) if asof is not None else None
        rec["atr_1d_prior"] = atr_d
        if atr_d is not None and rec.get("run") is not None:
            rec["run_atr_1d"] = float(rec["run"]) / float(atr_d)
        rows.append(rec)
    extra = pd.DataFrame(rows, index=out.index)
    for col in extra.columns:
        out[col] = extra[col]
    return out


def keep_unless_run_atr(
    trades: pd.DataFrame,
    *,
    col: str,
    max_run_atr: float,
) -> pd.Series:
    """Keep when metric missing or run/ATR < cap. Skip already-extended names."""
    x = pd.to_numeric(trades[col], errors="coerce")
    skip = x >= float(max_run_atr)
    return ~skip.fillna(False)
