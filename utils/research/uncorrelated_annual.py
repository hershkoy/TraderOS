"""
Yearly low-correlation sleeve.

Causal design
-------------
At the end of calendar year T, form daily returns over year T among PIT-liquid
names, build a Pearson correlation matrix, greedily pick ``x`` symbols with
the lowest average pairwise correlation, then hold them equal-weight through
calendar year T+1 (signal on year-T close, implement from first trading day
of T+1 via lagged weights).

Rebalance each year-end into the next year's picks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.costs import apply_turnover_costs
from utils.research.metrics import equity_from_returns
from utils.research.panel import adv_dollar


@dataclass
class YearPick:
    lookback_year: int
    hold_year: int
    symbols: List[str]
    mean_pairwise_corr: float
    n_eligible: int
    notes: str = ""


@dataclass
class UncorrAnnualResult:
    equity_gross: pd.Series
    equity_net: pd.Series
    invested: pd.Series
    picks: List[YearPick] = field(default_factory=list)
    weights: Optional[pd.DataFrame] = None
    notes: str = ""


def _year_slice(index: pd.DatetimeIndex, year: int) -> pd.DatetimeIndex:
    return index[(index.year == int(year))]


def eligible_symbols_for_year(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    year: int,
    *,
    liquid_n: int = 500,
    min_adv: float = 1_000_000.0,
    min_price: float = 5.0,
    min_obs: int = 180,
    adv_window: int = 30,
) -> List[str]:
    """
    Names with enough closes in ``year``, then PIT top-``liquid_n`` by ADV.

    History filter first: ALPACA panels often list names mid-year, so taking
    top ADV then requiring ``min_obs`` can leave almost nobody early on.
    """
    close = close.astype(float).sort_index()
    volume = volume.astype(float).reindex_like(close)
    days = _year_slice(close.index, year)
    if days.empty:
        return []
    asof = days[-1]
    counts = close.loc[days].notna().sum(axis=0)
    hist_ok = counts[counts >= int(min_obs)].index.tolist()
    if not hist_ok:
        return []
    px = close.loc[asof, hist_ok]
    priced = [s for s in hist_ok if pd.notna(px.get(s)) and float(px.get(s)) >= float(min_price)]
    if not priced:
        return []
    adv = adv_dollar(close, volume, window=adv_window)
    if asof not in adv.index:
        return priced[: int(liquid_n)]
    adv_row = adv.loc[asof, priced].astype(float)
    adv_row = adv_row[adv_row >= float(min_adv)].dropna()
    if adv_row.empty:
        return []
    ranked = adv_row.sort_values(ascending=False)
    return [str(s) for s in ranked.index[: int(liquid_n)].tolist()]

def correlation_matrix(
    close: pd.DataFrame,
    symbols: Sequence[str],
    year: int,
    *,
    min_periods: int = 60,
) -> pd.DataFrame:
    """Pearson corr of daily returns over calendar ``year`` for ``symbols``."""
    days = _year_slice(close.index, year)
    if days.empty or not symbols:
        return pd.DataFrame()
    px = close.loc[days, list(symbols)].astype(float)
    rets = px.pct_change(fill_method=None).iloc[1:]
    if rets.empty:
        return pd.DataFrame()
    # Drop columns that are all-NaN after pct_change
    rets = rets.dropna(axis=1, how="all")
    if rets.shape[1] < 2:
        return pd.DataFrame()
    corr = rets.corr(min_periods=int(min_periods))
    # Symmetrize / clean diagonal
    corr = corr.astype(float)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def greedy_uncorrelated(
    corr: pd.DataFrame,
    x: int,
    *,
    seed: Optional[str] = None,
) -> Tuple[List[str], float]:
    """
    Greedy pick of ``x`` symbols minimizing mean pairwise correlation.

    Seed = symbol with lowest mean correlation to all others (or ``seed``).
    Each next pick minimizes average corr to the already-selected set.
    """
    if corr is None or corr.empty or corr.shape[0] < 2:
        return [], float("nan")
    names = [str(c) for c in corr.columns]
    mat = corr.reindex(index=names, columns=names).astype(float)
    # Treat missing pairwise as high corr so they are avoided
    filled = mat.fillna(1.0).values.copy()
    n = len(names)
    x = int(max(1, min(int(x), n)))

    mean_to_all = filled.mean(axis=1)
    if seed is not None and seed in names:
        first = names.index(seed)
    else:
        first = int(np.argmin(mean_to_all))

    selected = [first]
    remaining = set(range(n)) - {first}

    while len(selected) < x and remaining:
        best_i = None
        best_score = float("inf")
        for i in remaining:
            score = float(np.mean([filled[i, j] for j in selected]))
            if score < best_score:
                best_score = score
                best_i = i
        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    picks = [names[i] for i in selected]
    if len(picks) < 2:
        return picks, float("nan")
    # Mean of upper triangle among picks
    sub = filled[np.ix_(selected, selected)]
    iu = np.triu_indices(len(selected), k=1)
    mean_pair = float(sub[iu].mean()) if len(iu[0]) else float("nan")
    return picks, mean_pair


def pick_for_lookback_year(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback_year: int,
    *,
    x: int = 10,
    liquid_n: int = 500,
    min_adv: float = 1_000_000.0,
    min_price: float = 5.0,
    min_obs: int = 180,
    min_corr_periods: int = 60,
    adv_window: int = 30,
) -> YearPick:
    hold_year = int(lookback_year) + 1
    elig = eligible_symbols_for_year(
        close,
        volume,
        lookback_year,
        liquid_n=liquid_n,
        min_adv=min_adv,
        min_price=min_price,
        min_obs=min_obs,
        adv_window=adv_window,
    )
    if len(elig) < 2:
        return YearPick(
            lookback_year=lookback_year,
            hold_year=hold_year,
            symbols=[],
            mean_pairwise_corr=float("nan"),
            n_eligible=len(elig),
            notes="insufficient eligible names",
        )
    corr = correlation_matrix(
        close, elig, lookback_year, min_periods=min_corr_periods
    )
    # Drop rows/cols that are all-NaN (too few overlapping pairs)
    if not corr.empty:
        keep = corr.notna().sum(axis=1) >= max(2, int(0.5 * len(corr)))
        corr = corr.loc[keep, keep]
    picks, mean_pair = greedy_uncorrelated(corr, x=x)
    return YearPick(
        lookback_year=lookback_year,
        hold_year=hold_year,
        symbols=picks,
        mean_pairwise_corr=mean_pair,
        n_eligible=len(elig),
        notes=f"corr_n={corr.shape[0] if corr is not None else 0}",
    )


def simulate_uncorrelated_annual(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    *,
    eval_start: str,
    eval_end: str,
    x: int = 10,
    liquid_n: int = 500,
    min_adv: float = 1_000_000.0,
    min_price: float = 5.0,
    min_obs: int = 180,
    min_corr_periods: int = 60,
    adv_window: int = 30,
    cost_bps_rt: float = 10.0,
) -> UncorrAnnualResult:
    """
    Build year-end picks from prior-year corr and hold equal-weight next year.
    """
    close = close.astype(float).sort_index()
    volume = volume.astype(float).reindex_like(close)

    s = pd.Timestamp(eval_start)
    e = pd.Timestamp(eval_end)
    # Need lookback year before first hold year
    first_hold = int(s.year)
    last_hold = int(e.year)
    lookback_years = list(range(first_hold - 1, last_hold))

    picks: List[YearPick] = []
    for y in lookback_years:
        picks.append(
            pick_for_lookback_year(
                close,
                volume,
                y,
                x=x,
                liquid_n=liquid_n,
                min_adv=min_adv,
                min_price=min_price,
                min_obs=min_obs,
                min_corr_periods=min_corr_periods,
                adv_window=adv_window,
            )
        )

    all_days = close.index[(close.index >= s) & (close.index <= e)]
    weight = pd.DataFrame(0.0, index=all_days, columns=close.columns)

    for p in picks:
        if not p.symbols:
            continue
        hold_days = _year_slice(all_days, p.hold_year)
        if hold_days.empty:
            continue
        w = 1.0 / float(len(p.symbols))
        for sym in p.symbols:
            if sym in weight.columns:
                weight.loc[hold_days, sym] = w

    daily_ret = close.pct_change(fill_method=None)
    port_gross, port_net, invested = apply_turnover_costs(
        weight, daily_ret, cost_bps_rt=cost_bps_rt, lag_weights=True
    )
    eq_g = equity_from_returns(port_gross.reindex(all_days).fillna(0.0))
    eq_n = equity_from_returns(port_net.reindex(all_days).fillna(0.0))
    inv = invested.reindex(all_days).fillna(0.0)

    n_ok = sum(1 for p in picks if p.symbols)
    mean_c = float(
        np.nanmean([p.mean_pairwise_corr for p in picks if p.symbols])
    ) if n_ok else float("nan")
    notes = (
        f"x={x} liquid_n={liquid_n} cost_bps_rt={cost_bps_rt}; "
        f"years_with_picks={n_ok}/{len(picks)}; avg_mean_pair_corr={mean_c:.3f}"
    )
    return UncorrAnnualResult(
        equity_gross=eq_g,
        equity_net=eq_n,
        invested=inv,
        picks=picks,
        weights=weight,
        notes=notes,
    )


def picks_to_frame(picks: Sequence[YearPick]) -> pd.DataFrame:
    rows: List[Dict] = []
    for p in picks:
        rows.append(
            {
                "lookback_year": p.lookback_year,
                "hold_year": p.hold_year,
                "n_eligible": p.n_eligible,
                "n_picks": len(p.symbols),
                "mean_pairwise_corr": p.mean_pairwise_corr,
                "symbols": ",".join(p.symbols),
                "notes": p.notes,
            }
        )
    return pd.DataFrame(rows)
