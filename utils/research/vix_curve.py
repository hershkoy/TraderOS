"""VIX term-structure (spot vs 3m) regime overlay on SPY."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from utils.research.costs import bps_to_frac
from utils.research.metrics import equity_from_returns
from utils.research.vol_target import vol_target_weights


def align_series(*series: pd.Series) -> Tuple[pd.Series, ...]:
    """Normalize timestamps and inner-join on shared dates."""
    cleaned = []
    for s in series:
        x = s.astype(float).copy()
        x.index = pd.DatetimeIndex(x.index).tz_localize(None).normalize()
        x = x[~x.index.duplicated(keep="last")].sort_index()
        cleaned.append(x)
    idx = cleaned[0].index
    for x in cleaned[1:]:
        idx = idx.intersection(x.index)
    return tuple(x.reindex(idx) for x in cleaned)


def curve_ratio(vix: pd.Series, vix3m: pd.Series) -> pd.Series:
    """VIX / VIX3M. Contango < 1; backwardation > 1."""
    v, m = align_series(vix, vix3m)
    return (v / m.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def risk_on_weights(
    ratio: pd.Series,
    *,
    mode: str = "binary",
    back_thresh: float = 1.0,
    soft_floor: float = 0.0,
) -> pd.Series:
    """
    Map term-structure ratio to SPY weight in [0, 1].

    binary: 1 if ratio < back_thresh else 0
    soft:   clip(1 / ratio, soft_floor, 1)  (full weight in deep contango)
    """
    r = ratio.astype(float)
    if mode == "soft":
        w = (1.0 / r.replace(0.0, np.nan)).clip(lower=float(soft_floor), upper=1.0)
        return w.fillna(0.0)
    # binary
    return (r < float(back_thresh)).astype(float)


def simulate_vix_curve_overlay(
    spy_close: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    cash_close: pd.Series | None = None,
    *,
    eval_start: str,
    eval_end: str,
    mode: str = "binary",
    back_thresh: float = 1.0,
    soft_floor: float = 0.0,
    min_vix: float | None = None,
    vol_target: float | None = None,
    vol_lookback: int = 20,
    vol_cap: float = 1.0,
    cost_bps_rt: float = 5.0,
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    Scale SPY by VIX term-structure regime; residual earns cash (BIL) or 0%.

    Optional min_vix: only force risk-off when VIX >= min_vix (binary mode).
    Optional vol_target: multiply curve weight by Moreira-Muir vol-target weight.

    Signal on close t; apply at t+1.
    Returns (eq_gross, eq_net, invested_spy_weight, notes).
    """
    spy, vx, vx3 = align_series(spy_close, vix, vix3m)
    spy_ret = spy.pct_change(fill_method=None)

    if cash_close is not None and not cash_close.empty:
        cash = cash_close.astype(float).copy()
        cash.index = pd.DatetimeIndex(cash.index).tz_localize(None).normalize()
        cash = cash[~cash.index.duplicated(keep="last")].sort_index()
        cash = cash.reindex(spy.index).ffill()
        cash_ret = cash.pct_change(fill_method=None).fillna(0.0)
        cash_note = "BIL"
    else:
        cash_ret = pd.Series(0.0, index=spy.index)
        cash_note = "cash0"

    ratio = (vx / vx3.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    w_curve = risk_on_weights(ratio, mode=mode, back_thresh=back_thresh, soft_floor=soft_floor)

    if min_vix is not None and mode == "binary":
        # Stay invested unless both backwardation AND elevated spot VIX
        risk_off = (ratio >= float(back_thresh)) & (vx >= float(min_vix))
        w_curve = (~risk_off).astype(float)

    if vol_target is not None:
        w_vol = vol_target_weights(
            spy_ret, target_vol=float(vol_target), lookback=vol_lookback, leverage_cap=vol_cap
        )
        w_sig = (w_curve * w_vol).clip(upper=float(vol_cap))
        vt_note = f" vt={vol_target:.0%}/{vol_lookback}d/cap{vol_cap:g}"
    else:
        w_sig = w_curve
        vt_note = ""

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    mask = (spy.index >= s) & (spy.index <= e)
    w_sig = w_sig.loc[mask]
    spy_ret_e = spy_ret.reindex(w_sig.index).fillna(0.0)
    cash_ret_e = cash_ret.reindex(w_sig.index).fillna(0.0)
    ratio_e = ratio.reindex(w_sig.index)

    w = w_sig.shift(1).fillna(0.0)
    w_cash = 1.0 - w
    port_gross = w * spy_ret_e + w_cash * cash_ret_e

    w_change = w.diff().abs().fillna(0.0)
    cost = (w_change / 2.0) * bps_to_frac(cost_bps_rt)
    port_net = port_gross - cost

    eq_gross = equity_from_returns(port_gross)
    eq_net = equity_from_returns(port_net)
    invested = w.clip(lower=0.0)

    frac_back = float((ratio_e >= float(back_thresh)).mean()) if len(ratio_e) else 0.0
    notes = (
        f"vix_curve mode={mode} thresh={back_thresh:g}"
        f"{f' minVIX={min_vix:g}' if min_vix is not None else ''}"
        f"{vt_note} cash={cash_note} cost={cost_bps_rt:.0f}bps RT; "
        f"avg_w={float(w.mean()):.2f} frac_back={frac_back:.2f}"
    )
    return eq_gross, eq_net, invested, notes
