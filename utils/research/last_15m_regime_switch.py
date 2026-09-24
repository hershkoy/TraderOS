"""Causal H2 health / TLT / UUP regime for last-15m vs 15m L3 switch.

Rolling stats use only H2 trades whose sell_time is strictly before the
candidate fill (paper book, including skipped H2 names). Macro joins use
prior-session daily close. Occupancy is not re-walked.

This is not a 2022 calendar splice.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.causal_regime_hmm import lookup_asof
from utils.research.channel_touch_entry_features import completed_asof

DEFAULT_WR_WINDOW = 20
DEFAULT_WR_OFF = 0.28
DEFAULT_WR_ON = 0.32
DEFAULT_PF_WINDOW = 20
DEFAULT_PF_OFF = 1.0
DEFAULT_PF_ON = 1.2
DEFAULT_SMA_TLT = 100
DEFAULT_SMA_UUP = 50
DEFAULT_ROC = 63


def _ts_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_datetime(df[col], errors="coerce")


def rolling_h2_regime(
    h2: pd.DataFrame,
    *,
    window: int = DEFAULT_WR_WINDOW,
    off_wr: float = DEFAULT_WR_OFF,
    on_wr: float = DEFAULT_WR_ON,
    mode: str = "wr",
    pf_off: float = DEFAULT_PF_OFF,
    pf_on: float = DEFAULT_PF_ON,
) -> pd.DataFrame:
    """After each H2 close, paper rolling WR or PF and the live ON/OFF flag.

    Warmup (fewer than ``window`` closed trades) stays ON.
    """
    if h2.empty:
        return pd.DataFrame(
            columns=["sell_time", "h2_on", "roll_wr", "roll_pf", "n_closed"]
        )
    closed = h2.copy()
    closed["_sell"] = _ts_series(closed, "sell_time")
    closed["_gain"] = pd.to_numeric(closed["gain_pct"], errors="coerce")
    closed = closed.dropna(subset=["_sell"]).sort_values("_sell", kind="mergesort")
    wins: List[float] = []
    gains: List[float] = []
    on = True
    rows: List[dict] = []
    win_n = max(1, int(window))
    for _, rec in closed.iterrows():
        g = rec["_gain"]
        gains.append(float(g) if g == g else 0.0)
        wins.append(1.0 if g == g and g > 0 else 0.0)
        n = len(gains)
        wr = float(np.mean(wins[-win_n:])) if n >= win_n else float("nan")
        pf = _pf(gains[-win_n:]) if n >= win_n else float("nan")
        if n >= win_n:
            if mode == "pf":
                if on and pf == pf and pf < float(pf_off):
                    on = False
                elif (not on) and pf == pf and pf >= float(pf_on):
                    on = True
            else:
                if on and wr == wr and wr < float(off_wr):
                    on = False
                elif (not on) and wr == wr and wr >= float(on_wr):
                    on = True
        rows.append(
            {
                "sell_time": rec["_sell"],
                "h2_on": bool(on),
                "roll_wr": wr,
                "roll_pf": pf,
                "n_closed": n,
            }
        )
    return pd.DataFrame(rows)


def _pf(gains: Sequence[float]) -> float:
    w = float(sum(g for g in gains if g > 0))
    loss = float(sum(-g for g in gains if g < 0))
    if loss <= 0:
        return float("inf") if w > 0 else float("nan")
    return w / loss


def h2_on_at(
    timestamps: Sequence[Any],
    regime: pd.DataFrame,
) -> pd.Series:
    """ON/OFF as of last H2 close strictly before each timestamp. Default ON."""
    if regime is None or regime.empty:
        return pd.Series([True] * len(timestamps), dtype=bool)
    flags = regime["h2_on"].astype(bool).to_numpy()
    sell_ns = pd.to_datetime(regime["sell_time"], errors="coerce").astype("int64").to_numpy()
    out: List[bool] = []
    for ts in timestamps:
        t = pd.to_datetime(ts, errors="coerce")
        if t is None or pd.isna(t):
            out.append(True)
            continue
        t_ns = int(pd.Timestamp(t).value)
        i = int(np.searchsorted(sell_ns, t_ns, side="left")) - 1
        if i < 0:
            out.append(True)
        else:
            out.append(bool(flags[i]))
    return pd.Series(out, dtype=bool)


def attach_h2_regime(
    trades: pd.DataFrame,
    regime: pd.DataFrame,
    *,
    time_col: str = "buy_time",
) -> pd.Series:
    ts = trades[time_col] if time_col in trades.columns else trades["buy_date"]
    out = h2_on_at(list(ts), regime)
    out.index = trades.index
    return out


def combine_h2_l3(
    h2: pd.DataFrame,
    l3: pd.DataFrame,
    h2_on: pd.Series,
    l3_on: pd.Series,
) -> pd.DataFrame:
    """Take last-15m when H2 is ON, 15m L3 when H2 is OFF. Occupancy not re-walked."""
    return combine_h2_alt(
        h2,
        l3,
        h2_on,
        l3_on,
        alt_sleeve="l3_wait12",
    )


def combine_h2_alt(
    h2: pd.DataFrame,
    alt: pd.DataFrame,
    h2_on: pd.Series,
    alt_on: pd.Series,
    *,
    alt_sleeve: str = "alt",
) -> pd.DataFrame:
    """Take last-15m when H2 is ON, alternate sleeve when H2 is OFF.

    Occupancy is not re-walked. ``alt`` may be L3 long or resist-short.
    """
    keep_h2 = h2_on.reindex(h2.index).fillna(True).astype(bool)
    keep_alt = alt_on.reindex(alt.index).fillna(False).astype(bool)
    a = h2.loc[keep_h2].copy()
    a["sleeve"] = "last15m_h2"
    b = alt.loc[keep_alt].copy()
    b["sleeve"] = str(alt_sleeve)
    if a.empty and b.empty:
        return pd.DataFrame()
    return pd.concat([a, b], ignore_index=True, sort=False)


def sma(close: pd.Series, span: int) -> pd.Series:
    c = close.astype(float)
    return c.rolling(int(span), min_periods=int(span)).mean()


def roc(close: pd.Series, span: int) -> pd.Series:
    c = close.astype(float)
    return c.pct_change(int(span), fill_method=None)


def attach_macro_asof(
    trades: pd.DataFrame,
    close: pd.Series,
    *,
    prefix: str,
    sma_span: int,
    roc_span: int = DEFAULT_ROC,
) -> pd.DataFrame:
    """Prior-session close, SMA, ROC on each fill."""
    out = trades.copy()
    c = close.astype(float).copy()
    if not isinstance(c.index, pd.DatetimeIndex):
        c.index = pd.to_datetime(c.index)
    c.index = pd.DatetimeIndex(c.index).tz_localize(None).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    ma = sma(c, sma_span)
    rr = roc(c, roc_span)
    px_col = "%s_px" % prefix
    ma_col = "%s_sma" % prefix
    roc_col = "%s_roc" % prefix
    pxs: List[Optional[float]] = []
    mas: List[Optional[float]] = []
    rocs: List[Optional[float]] = []
    for _, row in out.iterrows():
        asof = completed_asof(row, series_is_daily=True)
        pxs.append(lookup_asof(c, asof) if asof is not None else None)
        mas.append(lookup_asof(ma, asof) if asof is not None else None)
        rocs.append(lookup_asof(rr, asof) if asof is not None else None)
    out[px_col] = pxs
    out[ma_col] = mas
    out[roc_col] = rocs
    return out


def keep_unless_tlt_below_sma(trades: pd.DataFrame) -> pd.Series:
    """Keep when TLT missing or TLT close >= SMA. Skip crash/downtrend."""
    px = pd.to_numeric(trades["tlt_px"], errors="coerce")
    ma = pd.to_numeric(trades["tlt_sma"], errors="coerce")
    skip = px.notna() & ma.notna() & (px < ma)
    return ~skip.fillna(False)


def keep_unless_uup_above_sma(trades: pd.DataFrame) -> pd.Series:
    """Keep when UUP missing or UUP close <= SMA. Skip dollar uptrend."""
    px = pd.to_numeric(trades["uup_px"], errors="coerce")
    ma = pd.to_numeric(trades["uup_sma"], errors="coerce")
    skip = px.notna() & ma.notna() & (px > ma)
    return ~skip.fillna(False)


def keep_unless_tlt_roc_neg(trades: pd.DataFrame) -> pd.Series:
    roc_s = pd.to_numeric(trades["tlt_roc"], errors="coerce")
    skip = roc_s.notna() & (roc_s < 0)
    return ~skip.fillna(False)


def keep_unless_uup_roc_pos(trades: pd.DataFrame) -> pd.Series:
    roc_s = pd.to_numeric(trades["uup_roc"], errors="coerce")
    skip = roc_s.notna() & (roc_s > 0)
    return ~skip.fillna(False)


def keep_unless_spy_below_sma(trades: pd.DataFrame) -> pd.Series:
    """Keep when SPY missing or SPY close >= SMA. Skip bear market."""
    px = pd.to_numeric(trades["spy_px"], errors="coerce")
    ma = pd.to_numeric(trades["spy_sma"], errors="coerce")
    skip = px.notna() & ma.notna() & (px < ma)
    return ~skip.fillna(False)


def keep_when_spy_below_sma(trades: pd.DataFrame) -> pd.Series:
    """Alternate sleeve ON only when SPY is below SMA (missing stays OFF)."""
    px = pd.to_numeric(trades["spy_px"], errors="coerce")
    ma = pd.to_numeric(trades["spy_sma"], errors="coerce")
    return (px.notna() & ma.notna() & (px < ma)).fillna(False)


def keep_when_tlt_below_sma(trades: pd.DataFrame) -> pd.Series:
    """Alternate sleeve ON when TLT is below SMA (missing stays OFF)."""
    px = pd.to_numeric(trades["tlt_px"], errors="coerce")
    ma = pd.to_numeric(trades["tlt_sma"], errors="coerce")
    return (px.notna() & ma.notna() & (px < ma)).fillna(False)


def keep_when_uup_above_sma(trades: pd.DataFrame) -> pd.Series:
    """Alternate sleeve ON when UUP is above SMA (missing stays OFF)."""
    px = pd.to_numeric(trades["uup_px"], errors="coerce")
    ma = pd.to_numeric(trades["uup_sma"], errors="coerce")
    return (px.notna() & ma.notna() & (px > ma)).fillna(False)
