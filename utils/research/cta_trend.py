"""Multi-asset trend (CTA-lite) sleeve from liquid ETFs."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.costs import apply_turnover_costs
from utils.research.metrics import equity_from_returns
from utils.research.panel import month_ends


def _sma_signal(close: pd.DataFrame, sma_period: int = 200) -> pd.DataFrame:
    sma = close.rolling(int(sma_period), min_periods=int(sma_period)).mean()
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig = sig.mask(close > sma, 1.0)
    sig = sig.mask(close < sma, -1.0)
    return sig.fillna(0.0)


def _mom_signal(close: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    mom = close.pct_change(int(lookback_days), fill_method=None)
    sig = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    sig = sig.mask(mom > 0, 1.0)
    sig = sig.mask(mom < 0, -1.0)
    return sig.fillna(0.0)


def simulate_cta_sleeve(
    closes: pd.DataFrame,
    *,
    eval_start: str,
    eval_end: str,
    method: str = "sma200",
    sma_period: int = 200,
    mom_days: int = 252,
    cost_bps_rt: float = 10.0,
    rebalance: str = "monthly",
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    Equal-weight long/short trend across columns of `closes`.

    method: 'sma200' or 'mom12'.
    rebalance: 'daily' or 'monthly' (month-end signal).
    Returns (eq_gross, eq_net, gross_exposure, notes).
    """
    close = closes.astype(float).sort_index()
    close = close.dropna(how="all")
    if method == "mom12":
        raw = _mom_signal(close, lookback_days=mom_days)
        method_note = f"mom{mom_days}d"
    else:
        raw = _sma_signal(close, sma_period=sma_period)
        method_note = f"sma{sma_period}"

    n = float(close.shape[1])
    w_full = raw / n

    if rebalance == "monthly":
        me = month_ends(close.index)
        w_me = w_full.reindex(me).dropna(how="all")
        w_sig = w_me.reindex(close.index, method="ffill").fillna(0.0)
    else:
        w_sig = w_full

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    w_sig = w_sig.loc[(w_sig.index >= s) & (w_sig.index <= e)]
    rets = close.pct_change(fill_method=None).reindex(w_sig.index).fillna(0.0)

    port_gross, port_net, _inv = apply_turnover_costs(
        w_sig, rets, cost_bps_rt=cost_bps_rt, lag_weights=True
    )
    # Gross exposure = sum abs weights (invested measure for long/short)
    w_use = w_sig.shift(1).fillna(0.0)
    gross_exp = w_use.abs().sum(axis=1)

    eq_g = equity_from_returns(port_gross)
    eq_n = equity_from_returns(port_net)
    notes = (
        f"CTA {method_note} equal_w n={int(n)} assets={list(close.columns)} "
        f"rebal={rebalance} cost={cost_bps_rt:.0f}bps RT; avg_gross_exp={float(gross_exp.mean()):.2f}"
    )
    return eq_g, eq_n, gross_exp, notes


def blend_spy_cta(
    spy_eq: pd.Series,
    cta_eq: pd.Series,
    spy_weight: float = 0.70,
) -> pd.Series:
    """Fixed capital mix of two equity curves (rebased to overlapping window)."""
    idx = spy_eq.index.intersection(cta_eq.index)
    if idx.empty:
        return spy_eq.copy() * 0.0 + 1.0
    s = spy_eq.reindex(idx).astype(float)
    c = cta_eq.reindex(idx).astype(float)
    s = s / float(s.iloc[0])
    c = c / float(c.iloc[0])
    w = float(spy_weight)
    return w * s + (1.0 - w) * c


def load_macro_closes(
    symbols: Sequence[str],
    start,
    end,
    preferred_provider: str = "IB",
    fallback_provider: str = "ALPACA",
) -> Dict[str, pd.Series]:
    """Load daily closes per symbol; try preferred then fallback provider."""
    from datetime import datetime

    from utils.data.ohlcv_loader import load_ohlcv_many
    from utils.research.panel import to_naive_index

    if isinstance(start, str):
        start = datetime.strptime(start, "%Y-%m-%d")
    if isinstance(end, str):
        end = datetime.strptime(end, "%Y-%m-%d")

    out: Dict[str, pd.Series] = {}
    missing = [s.upper() for s in symbols]
    for provider in (preferred_provider, fallback_provider):
        if not missing:
            break
        frames = load_ohlcv_many(
            missing,
            timeframe="1d",
            provider=provider,
            start=start,
            end=end,
            use_cache=True,
            workers=2,
            chunk_size=20,
        )
        still = []
        for sym in missing:
            df = frames.get(sym)
            if df is None or df.empty or "close" not in df.columns:
                still.append(sym)
                continue
            out[sym] = to_naive_index(df["close"].astype(float))
        missing = still
    return out
