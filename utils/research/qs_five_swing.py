"""Quantified Strategies five SPY swing strategies (claim-check helpers).

Shared exit (QS): sell at close when close > prior day's high.

Entries (buy at close):
  S1 Band+IBS: close < (10d high - 2.5 * 25d avg range) and IBS < 0.3
  S2 Turnaround Tuesday: Monday close down two days in a row
  S3 5-day low: close < min(low of prior 5 sessions)
  S4 Vol+ADX: today's range < min(range of prior 6) and ADX(5) > 40
  S5 10d high reject: high > max(high of prior 10) and IBS < 0.15

Combined: single slot, priority S1->S5 when flat; ignore buys while in position.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.deep_pullback import YEAR_BUCKETS, geom_mean_year_pf, year_bucket_stats

STRATEGY_IDS: Sequence[str] = ("S1", "S2", "S3", "S4", "S5")

STRATEGY_NAMES: Dict[str, str] = {
    "S1": "Band+IBS",
    "S2": "Turnaround Tuesday",
    "S3": "5-Day Low",
    "S4": "Vol Contraction + ADX",
    "S5": "10-Day High Rejection",
}


def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Internal Bar Strength: (close - low) / (high - low); 0 when range is 0."""
    rng = (high - low).astype(float)
    out = (close.astype(float) - low.astype(float)) / rng
    out = out.where(rng > 0, 0.0)
    return out


def _wilder_rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder RMA: first value = SMA(period), then ((n-1)*prev + x) / n."""
    x = series.astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    n = int(period)
    if len(x) < n or n < 1:
        return out
    # Find first index where the preceding n values are all finite.
    start = None
    for i in range(n - 1, len(x)):
        window = x.iloc[i - n + 1 : i + 1]
        if bool(window.notna().all()):
            start = i
            break
    if start is None:
        return out
    prev = float(x.iloc[start - n + 1 : start + 1].mean())
    out.iloc[start] = prev
    for i in range(start + 1, len(x)):
        v = float(x.iloc[i])
        if not np.isfinite(v) or not np.isfinite(prev):
            out.iloc[i] = np.nan
            prev = np.nan
            continue
        prev = ((n - 1) * prev + v) / float(n)
        out.iloc[i] = prev
    return out


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 5,
) -> pd.Series:
    """Wilder Average Directional Index."""
    h = high.astype(float)
    l = low.astype(float)
    c = close.astype(float)
    prev_h = h.shift(1)
    prev_l = l.shift(1)
    prev_c = c.shift(1)

    tr = pd.concat(
        [
            (h - l).abs(),
            (h - prev_c).abs(),
            (l - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up = h - prev_h
    down = prev_l - l
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=h.index, dtype=float
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=h.index, dtype=float
    )
    # First bar has no prior; force DM/TR NaN so RMA starts cleanly.
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    tr.iloc[0] = np.nan

    atr = _wilder_rma(tr, period)
    plus_sm = _wilder_rma(plus_dm, period)
    minus_sm = _wilder_rma(minus_dm, period)

    plus_di = 100.0 * plus_sm / atr
    minus_di = 100.0 * minus_sm / atr
    di_sum = plus_di + minus_di
    dx = (100.0 * (plus_di - minus_di).abs() / di_sum).where(di_sum > 0)
    return _wilder_rma(dx, period)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV frame and attach shared feature columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "ts"})
        else:
            return pd.DataFrame()

    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return pd.DataFrame()
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "close", "high", "low"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    if d.empty:
        return d

    d["range"] = d["high"] - d["low"]
    d["ibs"] = ibs(d["high"], d["low"], d["close"])
    d["prior_high"] = d["high"].shift(1)
    d["avg_range_25"] = d["range"].rolling(25, min_periods=25).mean()
    d["high_10"] = d["high"].rolling(10, min_periods=10).max()
    d["band_s1"] = d["high_10"] - 2.5 * d["avg_range_25"]
    d["adx5"] = adx(d["high"], d["low"], d["close"], period=5)
    d["min_low_prior_5"] = d["low"].shift(1).rolling(5, min_periods=5).min()
    d["min_range_prior_6"] = d["range"].shift(1).rolling(6, min_periods=6).min()
    d["max_high_prior_10"] = d["high"].shift(1).rolling(10, min_periods=10).max()
    # Yahoo daily bars are US equity sessions; weekday in UTC date == ET calendar date.
    d["weekday"] = d["ts"].dt.weekday  # Mon=0
    return d


def entry_mask_s1(d: pd.DataFrame) -> pd.Series:
    return (
        d["band_s1"].notna()
        & (d["close"] < d["band_s1"])
        & (d["ibs"] < 0.3)
    )


def entry_mask_s2(d: pd.DataFrame) -> pd.Series:
    c = d["close"]
    return (
        (d["weekday"] == 0)
        & (c < c.shift(1))
        & (c.shift(1) < c.shift(2))
    )


def entry_mask_s3(d: pd.DataFrame) -> pd.Series:
    return d["min_low_prior_5"].notna() & (d["close"] < d["min_low_prior_5"])


def entry_mask_s4(d: pd.DataFrame) -> pd.Series:
    return (
        d["min_range_prior_6"].notna()
        & d["adx5"].notna()
        & (d["range"] < d["min_range_prior_6"])
        & (d["adx5"] > 40.0)
    )


def entry_mask_s5(d: pd.DataFrame) -> pd.Series:
    return (
        d["max_high_prior_10"].notna()
        & (d["high"] > d["max_high_prior_10"])
        & (d["ibs"] < 0.15)
    )


ENTRY_MASKS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "S1": entry_mask_s1,
    "S2": entry_mask_s2,
    "S3": entry_mask_s3,
    "S4": entry_mask_s4,
    "S5": entry_mask_s5,
}


def _metrics_from_trades(
    trades: List[Dict[str, Any]],
    d: pd.DataFrame,
    *,
    friction_pct: float,
    n_entry_signals: int,
    strategy_id: str,
    symbol: str,
    start_i: int = 30,
) -> Dict[str, Any]:
    tdf = pd.DataFrame(trades)
    n = len(tdf)
    base = {
        "symbol": symbol,
        "strategy_id": strategy_id,
        "strategy_name": STRATEGY_NAMES.get(strategy_id, strategy_id),
        "n": n,
        "n_entry_signals": n_entry_signals,
        "friction_pct": friction_pct,
        "bars": len(d),
        "first_bar": str(pd.Timestamp(d["ts"].iloc[0]).date()) if len(d) else None,
        "last_bar": str(pd.Timestamp(d["ts"].iloc[-1]).date()) if len(d) else None,
        "by_year": [],
        "year_buckets": [],
        "geo_mean_year_pf": None,
        "trades": trades,
    }
    if n == 0:
        return base

    gains = tdf["ret_pct"].to_numpy(dtype=float)
    wins = gains[gains > 0]
    losses = gains[gains <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")

    equity = [1.0]
    for r in gains:
        equity.append(equity[-1] * (1.0 + r / 100.0))
    eq = np.array(equity)
    peak = np.maximum.accumulate(eq)
    mdd_trade = float((eq / peak - 1.0).min() * 100.0)

    hold_sum = int(tdf["hold_days"].sum())
    calendar_days = max(len(d) - start_i, 1)
    exposure = hold_sum / calendar_days

    full_start = pd.Timestamp(d.iloc[min(start_i, len(d) - 1)]["ts"])
    full_end = pd.Timestamp(d.iloc[-1]["ts"])
    full_years = max((full_end - full_start).days / 365.25, 1e-9)
    cagr_full = (eq[-1] ** (1.0 / full_years) - 1.0) * 100.0
    risk_adj = cagr_full / exposure if exposure > 0 else None

    # MTM equity / MDD (cash when flat)
    mdd_mtm, equity_monthly, bh_monthly = _mtm_paths(
        d, trades, friction_pct=friction_pct, start_i=start_i
    )

    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    by_year: List[Dict[str, Any]] = []
    for y, g in tdf.groupby("year"):
        gy = g["ret_pct"].to_numpy(dtype=float)
        w = float(gy[gy > 0].sum())
        l = float(-gy[gy <= 0].sum())
        by_year.append(
            {
                "year": int(y),
                "n": int(len(g)),
                "E": round(float(gy.mean()), 3),
                "WR": round(float((gy > 0).mean() * 100), 1),
                "PF": None if l <= 0 else round(w / l, 2),
                "sum_ret": round(float(gy.sum()), 2),
            }
        )

    buckets = year_bucket_stats(trades)
    geo = geom_mean_year_pf(buckets, min_n=30)

    losing_years = [
        {"year": int(r["year"]), "sum_ret": r["sum_ret"]}
        for r in by_year
        if r["sum_ret"] < 0
    ]

    bh0 = float(d.iloc[min(start_i, len(d) - 1)]["close"])
    bh1 = float(d.iloc[-1]["close"])
    bh_cagr = ((bh1 / bh0) ** (1.0 / full_years) - 1.0) * 100.0

    base.update(
        {
            "win_rate_pct": round(float((gains > 0).mean() * 100), 2),
            "profit_factor": round(float(pf), 3) if np.isfinite(pf) else None,
            "avg_gain_pct": round(float(gains.mean()), 3),
            "median_gain_pct": round(float(np.median(gains)), 3),
            "mdd_trade_eq_pct": round(mdd_trade, 2),
            "mdd_mtm_pct": round(float(mdd_mtm), 2),
            "cagr_full_window_pct": round(float(cagr_full), 2),
            "spy_bh_cagr_pct": round(float(bh_cagr), 2),
            "exposure_pct": round(exposure * 100, 2),
            "risk_adj_return_pct": None if risk_adj is None else round(float(risk_adj), 1),
            "avg_hold_days": round(float(tdf["hold_days"].mean()), 2),
            "final_equity": round(float(eq[-1]), 4),
            "final_equity_100k": round(float(eq[-1]) * 100_000.0, 0),
            "first_trade": tdf["entry_date"].iloc[0],
            "last_trade": tdf["exit_date"].iloc[-1],
            "by_year": by_year,
            "losing_years": losing_years,
            "year_buckets": buckets,
            "geo_mean_year_pf": None if geo is None else round(float(geo), 3),
            "equity_monthly": equity_monthly,
            "bh_monthly": bh_monthly,
        }
    )
    return base


def _mtm_paths(
    d: pd.DataFrame,
    trades: List[Dict[str, Any]],
    *,
    friction_pct: float,
    start_i: int,
) -> Tuple[float, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuild MTM from trade list (entry/exit dates) for chart + MDD."""
    if not trades:
        return 0.0, [], []

    entry_map = {t["entry_date"]: t for t in trades}
    # Multiple strategies could theoretically share a date in solo books — one trade/day.
    exit_set = {t["exit_date"]: t for t in trades}

    pos = False
    e_px: Optional[float] = None
    eq_mtm = 1.0
    peak_mtm = 1.0
    mdd_mtm = 0.0
    equity_monthly: List[Dict[str, Any]] = []
    last_month = None

    for i in range(len(d)):
        row = d.iloc[i]
        day = str(pd.Timestamp(row["ts"]).date())
        if not pos and day in entry_map:
            pos = True
            e_px = float(entry_map[day]["entry_px"])
        elif pos and e_px is not None and day in exit_set:
            r = (float(row["close"]) / e_px - 1.0) - friction_pct / 100.0
            eq_mtm *= 1.0 + r
            pos = False
            e_px = None
        mtm = eq_mtm * (float(row["close"]) / e_px) if (pos and e_px) else eq_mtm
        peak_mtm = max(peak_mtm, mtm)
        mdd_mtm = min(mdd_mtm, mtm / peak_mtm - 1.0)
        month = (int(row["ts"].year), int(row["ts"].month))
        if i >= start_i and (month != last_month or i == len(d) - 1):
            equity_monthly.append(
                {"date": day, "equity": round(float(mtm), 6)}
            )
            last_month = month

    bh0 = float(d.iloc[min(start_i, len(d) - 1)]["close"])
    bh_monthly: List[Dict[str, Any]] = []
    last_month = None
    for i in range(start_i, len(d)):
        row = d.iloc[i]
        month = (int(row["ts"].year), int(row["ts"].month))
        if month != last_month or i == len(d) - 1:
            bh_monthly.append(
                {
                    "date": str(pd.Timestamp(row["ts"]).date()),
                    "equity": round(float(row["close"]) / bh0, 6),
                }
            )
            last_month = month

    return float(mdd_mtm * 100.0), equity_monthly, bh_monthly


def backtest_strategy(
    df: pd.DataFrame,
    entry_mask: pd.Series,
    *,
    symbol: str = "SPY",
    strategy_id: str = "S?",
    friction_pct: float = 0.0,
) -> Dict[str, Any]:
    """Run one strategy: entry at close on mask, QS prior-high exit (not same bar)."""
    # Always re-prepare so feature columns stay consistent with masks.
    if "band_s1" in getattr(df, "columns", []) and "prior_high" in df.columns:
        d = df.copy()
    else:
        d = prepare_frame(df)
    if d.empty:
        return {
            "symbol": symbol,
            "strategy_id": strategy_id,
            "n": 0,
            "trades": [],
            "reason": "empty",
        }

    mask = entry_mask.reindex(d.index).fillna(False).astype(bool)
    d["entry_signal"] = mask

    trades: List[Dict[str, Any]] = []
    in_pos = False
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None

    for i in range(len(d)):
        row = d.iloc[i]
        if not in_pos:
            if bool(row["entry_signal"]):
                in_pos = True
                entry_i = i
                entry_px = float(row["close"])
        else:
            assert entry_i is not None and entry_px is not None
            if pd.notna(row["prior_high"]) and float(row["close"]) > float(row["prior_high"]):
                exit_px = float(row["close"])
                ret = (exit_px / entry_px - 1.0) * 100.0 - friction_pct
                trades.append(
                    {
                        "symbol": symbol,
                        "strategy_id": strategy_id,
                        "entry_date": str(pd.Timestamp(d.iloc[entry_i]["ts"]).date()),
                        "exit_date": str(pd.Timestamp(row["ts"]).date()),
                        "entry_px": round(entry_px, 4),
                        "exit_px": round(exit_px, 4),
                        "ret_pct": round(ret, 4),
                        "hold_days": int(i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_px = None

    return _metrics_from_trades(
        trades,
        d,
        friction_pct=friction_pct,
        n_entry_signals=int(mask.sum()),
        strategy_id=strategy_id,
        symbol=symbol,
    )


def backtest_named(
    df: pd.DataFrame,
    strategy_id: str,
    *,
    symbol: str = "SPY",
    friction_pct: float = 0.0,
) -> Dict[str, Any]:
    """Backtest one of S1..S5 by id."""
    sid = strategy_id.upper()
    if sid not in ENTRY_MASKS:
        raise KeyError("unknown strategy_id: %s" % strategy_id)
    d = prepare_frame(df)
    if d.empty:
        return {
            "symbol": symbol,
            "strategy_id": sid,
            "n": 0,
            "trades": [],
            "reason": "empty",
        }
    return backtest_strategy(
        d,
        ENTRY_MASKS[sid](d),
        symbol=symbol,
        strategy_id=sid,
        friction_pct=friction_pct,
    )


def backtest_combined(
    df: pd.DataFrame,
    *,
    symbol: str = "SPY",
    friction_pct: float = 0.0,
    priority: Sequence[str] = STRATEGY_IDS,
) -> Dict[str, Any]:
    """Single-slot book: when flat take first firing strategy in priority order."""
    d = prepare_frame(df)
    if d.empty:
        return {
            "symbol": symbol,
            "strategy_id": "COMBINED",
            "n": 0,
            "trades": [],
            "reason": "empty",
        }

    masks: Dict[str, pd.Series] = {}
    for sid in priority:
        sid_u = sid.upper()
        if sid_u not in ENTRY_MASKS:
            raise KeyError("unknown strategy_id: %s" % sid)
        masks[sid_u] = ENTRY_MASKS[sid_u](d).fillna(False).astype(bool)

    any_signal = False
    for m in masks.values():
        any_signal = any_signal | m
    n_entry_signals = int(any_signal.sum())

    trades: List[Dict[str, Any]] = []
    in_pos = False
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None
    entry_sid: Optional[str] = None

    for i in range(len(d)):
        row = d.iloc[i]
        if not in_pos:
            chosen: Optional[str] = None
            for sid in priority:
                sid_u = sid.upper()
                if bool(masks[sid_u].iloc[i]):
                    chosen = sid_u
                    break
            if chosen is not None:
                in_pos = True
                entry_i = i
                entry_px = float(row["close"])
                entry_sid = chosen
        else:
            assert entry_i is not None and entry_px is not None and entry_sid is not None
            if pd.notna(row["prior_high"]) and float(row["close"]) > float(row["prior_high"]):
                exit_px = float(row["close"])
                ret = (exit_px / entry_px - 1.0) * 100.0 - friction_pct
                trades.append(
                    {
                        "symbol": symbol,
                        "strategy_id": entry_sid,
                        "entry_date": str(pd.Timestamp(d.iloc[entry_i]["ts"]).date()),
                        "exit_date": str(pd.Timestamp(row["ts"]).date()),
                        "entry_px": round(entry_px, 4),
                        "exit_px": round(exit_px, 4),
                        "ret_pct": round(ret, 4),
                        "hold_days": int(i - entry_i),
                    }
                )
                in_pos = False
                entry_i = None
                entry_px = None
                entry_sid = None

    out = _metrics_from_trades(
        trades,
        d,
        friction_pct=friction_pct,
        n_entry_signals=n_entry_signals,
        strategy_id="COMBINED",
        symbol=symbol,
    )
    out["strategy_name"] = "Combined (S1-S5 priority)"
    out["priority"] = list(priority)
    # Attribution: which sleeve filled
    if trades:
        tdf = pd.DataFrame(trades)
        counts = tdf["strategy_id"].value_counts().to_dict()
        out["fills_by_strategy"] = {k: int(v) for k, v in counts.items()}
    else:
        out["fills_by_strategy"] = {}
    return out


def run_all(
    df: pd.DataFrame,
    *,
    symbol: str = "SPY",
    friction_pct: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Run S1..S5 + combined at one friction level."""
    d = prepare_frame(df)
    results: Dict[str, Dict[str, Any]] = {}
    for sid in STRATEGY_IDS:
        results[sid] = backtest_strategy(
            d,
            ENTRY_MASKS[sid](d),
            symbol=symbol,
            strategy_id=sid,
            friction_pct=friction_pct,
        )
    results["COMBINED"] = backtest_combined(
        d, symbol=symbol, friction_pct=friction_pct
    )
    return results
