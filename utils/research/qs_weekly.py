"""Quantified Strategies weekly claim-check helpers.

1) SPY down-week (free rules):
   Signal Friday: close < prior Friday close -> long for the following week.
   Claim fill: Friday close -> next Friday close.
   Honest fill: next Monday open -> that week's Friday close.

2) Strategy #69 reconstruction (rules paywalled):
   Universe SPY / EFA / GLD / TLT.
   Every Friday rank by N-week ROC; buy weakest or strongest; hold one week.
   Also monthly end-of-month variant.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.metrics import equity_from_returns, passes_phase5_gates, perf_stats

ETFS_69: Tuple[str, ...] = ("SPY", "EFA", "GLD", "TLT")

CLAIMED_DOWN_WEEK: Dict[str, Any] = {
    "avg_gain_pct": 0.44,
    "cagr_pct": 9.3,
    "mdd_pct": 36.0,
    "exposure_pct": 42.0,
    "spy_bh_cagr_pct": 10.4,
    "spy_bh_mdd_pct": 55.0,
    "source": "https://www.quantifiedstrategies.com/sp500-down-week-trading-strategy/",
}

CLAIMED_69_WEEKLY: Dict[str, Any] = {
    "n": 728,
    "avg_gain_pct": 0.4,
    "cagr_pct": 15.1,
    "mdd_pct": 43.0,
    "keep_same_pct": 20.0,
    "source": "https://www.quantifiedstrategies.com/weekly-trading-strategy/",
}

CLAIMED_69_MONTHLY: Dict[str, Any] = {
    "n": 179,
    "avg_gain_pct": 1.2,
    "win_rate_pct": 58.0,
    "profit_factor": 1.9,
    "cagr_pct": 10.1,
    "mdd_pct": 21.0,
    "source": "https://www.quantifiedstrategies.com/product/strategy69/",
}

WEEKLY_LOOKBACKS: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12)
MONTHLY_LOOKBACKS: Tuple[int, ...] = (1, 2, 3, 6, 9, 12)


def _ensure_ts(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("df needs ts column or DatetimeIndex")
    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    return d


def to_weekly_friday(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to week-ending bars; keep the last bar's real timestamp."""
    d = _ensure_ts(df)
    d = d.set_index("ts").sort_index()
    d.index = d.index.tz_convert("UTC")
    rows = []
    for _, chunk in d.groupby(pd.Grouper(freq="W-FRI")):
        if chunk.empty:
            continue
        sub = chunk.dropna(subset=["close"])
        if sub.empty:
            continue
        last = sub.iloc[-1]
        row = {
            "ts": sub.index[-1],
            "open": float(sub["open"].iloc[0]) if "open" in sub.columns else float(last["close"]),
            "high": float(sub["high"].max()) if "high" in sub.columns else float(last["close"]),
            "low": float(sub["low"].min()) if "low" in sub.columns else float(last["close"]),
            "close": float(last["close"]),
        }
        if "volume" in sub.columns:
            row["volume"] = float(sub["volume"].sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    return pd.DataFrame(rows)



def _trade_stats(rets: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in rets if x is not None and pd.notna(x)]
    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "avg_gain_pct": 0.0,
        }
    wins = [v for v in vals if v > 0]
    losses = [v for v in vals if v < 0]
    gw = float(sum(wins)) if wins else 0.0
    gl = float(-sum(losses)) if losses else 0.0
    pf = (gw / gl) if gl > 1e-12 else (99.0 if gw > 0 else 0.0)
    return {
        "n": n,
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "profit_factor": round(pf, 3),
        "expectancy_pct": round(100.0 * float(sum(vals) / n), 3),
        "avg_gain_pct": round(100.0 * float(sum(vals) / n), 3),
    }


def _compound_equity(trade_rets: Sequence[float], start: float = 1.0) -> Tuple[float, float]:
    eq = float(start)
    peak = eq
    mdd = 0.0
    for r in trade_rets:
        eq *= 1.0 + float(r)
        if eq > peak:
            peak = eq
        dd = eq / peak - 1.0
        if dd < mdd:
            mdd = dd
    return eq, abs(mdd)


def backtest_spy_down_week(
    daily: pd.DataFrame,
    *,
    fill_mode: str = "claim_close",
    friction_pct: float = 0.0,
) -> Dict[str, Any]:
    """SPY down-week mean-reversion claim-check.

    fill_mode:
      claim_close — buy Friday close after down week, sell next Friday close
      next_open   — buy Monday open of following week, sell that Friday close
    """
    d = _ensure_ts(daily)
    if "open" not in d.columns:
        return {"n": 0, "reason": "missing_open", "trades": []}

    weekly = to_weekly_friday(d)
    if len(weekly) < 3:
        return {"n": 0, "reason": "short_weekly", "trades": []}

    # Map each Friday to the next session open (first daily bar strictly after Friday).
    daily_ts = d["ts"].to_numpy()
    daily_open = d["open"].to_numpy(dtype=float)
    daily_close = d["close"].to_numpy(dtype=float)

    trades: List[Dict[str, Any]] = []
    fric = float(friction_pct) / 100.0

    for i in range(1, len(weekly) - 1):
        c0 = float(weekly.iloc[i - 1]["close"])
        c1 = float(weekly.iloc[i]["close"])
        if not (c1 < c0):
            continue
        sig_ts = pd.Timestamp(weekly.iloc[i]["ts"])
        exit_ts = pd.Timestamp(weekly.iloc[i + 1]["ts"])
        exit_px = float(weekly.iloc[i + 1]["close"])

        if fill_mode == "claim_close":
            entry_ts = sig_ts
            entry_px = c1
        elif fill_mode == "next_open":
            # First daily bar after signal Friday (compare as UTC ns ints).
            sig_ns = int(pd.Timestamp(sig_ts).value)
            daily_ns = np.array([int(pd.Timestamp(t).value) for t in daily_ts], dtype=np.int64)
            j = int(np.searchsorted(daily_ns, sig_ns, side="right"))
            if j >= len(daily_ts):
                continue
            entry_ts = pd.Timestamp(daily_ts[j])
            entry_px = float(daily_open[j])
            exit_px = float(weekly.iloc[i + 1]["close"])
            exit_ts = pd.Timestamp(weekly.iloc[i + 1]["ts"])
        else:
            raise ValueError("unknown fill_mode: %s" % fill_mode)

        if entry_px <= 0 or exit_px <= 0:
            continue
        ret = exit_px / entry_px - 1.0 - 2.0 * fric
        trades.append(
            {
                "signal_date": str(sig_ts.date()),
                "entry_date": str(pd.Timestamp(entry_ts).date()),
                "exit_date": str(exit_ts.date()),
                "entry_px": round(entry_px, 4),
                "exit_px": round(exit_px, 4),
                "ret_pct": round(100.0 * ret, 4),
                "down_week_pct": round(100.0 * (c1 / c0 - 1.0), 4),
            }
        )

    rets = [t["ret_pct"] / 100.0 for t in trades]
    stats = _trade_stats(rets)
    final_eq, mdd_trade = _compound_equity(rets)

    # Daily equity for Sharpe gate: cash when flat, SPY when in a trade.
    eq_daily, invested = _down_week_daily_equity(d, trades, fill_mode=fill_mode, friction_pct=friction_pct)
    spy_eq = (1.0 + d.set_index("ts")["close"].pct_change().fillna(0.0)).cumprod()
    spy_eq.index = pd.DatetimeIndex(spy_eq.index).tz_localize(None)
    eq_daily.index = pd.DatetimeIndex(eq_daily.index).tz_localize(None)
    invested.index = pd.DatetimeIndex(invested.index).tz_localize(None)
    spy_eq = spy_eq.reindex(eq_daily.index).ffill().bfill()
    spy_stats = perf_stats("SPY_buy_hold", spy_eq)
    strat_stats = perf_stats(
        "SPY_down_week_%s" % fill_mode,
        eq_daily,
        invested=invested,
        spy_cagr=spy_stats.cagr,
        spy_sharpe=spy_stats.sharpe,
        trade_rets=rets,
    )
    _ok, gate = passes_phase5_gates(strat_stats, spy_stats)

    return {
        "name": "SPY_down_week_%s" % fill_mode,
        "fill_mode": fill_mode,
        "friction_pct": friction_pct,
        "trades": trades,
        **stats,
        "final_equity": round(final_eq, 4),
        "mdd_trade_eq_pct": round(100.0 * mdd_trade, 2),
        "exposure_pct": round(100.0 * float(invested.mean()), 2),
        "cagr_pct": round(100.0 * strat_stats.cagr, 2),
        "sharpe": round(strat_stats.sharpe, 3),
        "mdd_daily_pct": round(100.0 * abs(strat_stats.max_drawdown), 2),
        "spy_cagr_pct": round(100.0 * spy_stats.cagr, 2),
        "spy_sharpe": round(spy_stats.sharpe, 3),
        "spy_mdd_pct": round(100.0 * abs(spy_stats.max_drawdown), 2),
        "gate": gate,
        "start": strat_stats.start,
        "end": strat_stats.end,
        "claimed": CLAIMED_DOWN_WEEK,
    }


def _down_week_daily_equity(
    daily: pd.DataFrame,
    trades: List[Dict[str, Any]],
    *,
    fill_mode: str,
    friction_pct: float,
) -> Tuple[pd.Series, pd.Series]:
    d = _ensure_ts(daily).set_index("ts").sort_index()
    idx = d.index
    close = d["close"].astype(float)
    open_ = d["open"].astype(float) if "open" in d.columns else close
    rets = close.pct_change().fillna(0.0)
    invested = pd.Series(0.0, index=idx)
    strat_rets = pd.Series(0.0, index=idx)

    for t in trades:
        a_date = pd.Timestamp(t["entry_date"]).date()
        b_date = pd.Timestamp(t["exit_date"]).date()
        dates = pd.Series([x.date() for x in idx], index=idx)
        if fill_mode == "claim_close":
            mask = (dates > a_date) & (dates <= b_date)
            invested.loc[mask] = 1.0
            strat_rets.loc[mask] = rets.loc[mask]
        else:
            mask = (dates >= a_date) & (dates <= b_date)
            invested.loc[mask] = 1.0
            # Entry day: open -> close; later days: close-to-close.
            entry_rows = idx[dates == a_date]
            rest = (dates > a_date) & (dates <= b_date)
            if len(entry_rows):
                e = entry_rows[0]
                if float(open_.loc[e]) > 0:
                    strat_rets.loc[e] = float(close.loc[e] / open_.loc[e] - 1.0)
            strat_rets.loc[rest] = rets.loc[rest]

    fric = float(friction_pct) / 100.0
    if fric > 0 and trades:
        for t in trades:
            a_date = pd.Timestamp(t["entry_date"]).date()
            b_date = pd.Timestamp(t["exit_date"]).date()
            dates = pd.Series([x.date() for x in idx], index=idx)
            entry_rows = idx[dates == a_date]
            exit_rows = idx[dates == b_date]
            if fill_mode == "claim_close":
                # Friction on first invested day after entry and on exit day.
                after = idx[dates > a_date]
                if len(after):
                    strat_rets.loc[after[0]] -= fric
            elif len(entry_rows):
                strat_rets.loc[entry_rows[0]] -= fric
            if len(exit_rows):
                strat_rets.loc[exit_rows[0]] -= fric
    eq = equity_from_returns(strat_rets)
    return eq, invested



def _weekly_panel(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide Friday close panel for ETFs."""
    closes = {}
    for sym, df in frames.items():
        w = to_weekly_friday(df)
        closes[sym] = w.set_index("ts")["close"]
    panel = pd.DataFrame(closes).sort_index().dropna(how="any")
    return panel


def _monthly_panel(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {}
    for sym, df in frames.items():
        d = _ensure_ts(df).set_index("ts").sort_index()
        m = d["close"].resample("ME").last().dropna()
        closes[sym] = m
    panel = pd.DataFrame(closes).sort_index().dropna(how="any")
    return panel


def backtest_rotation69(
    frames: Dict[str, pd.DataFrame],
    *,
    lookback: int = 4,
    pick: str = "lowest",
    cadence: str = "weekly",
    friction_pct: float = 0.0,
) -> Dict[str, Any]:
    """Reconstruct Strategy #69-style rotation.

    pick: 'lowest' (Substack teaser) or 'highest' (classic momentum).
    cadence: 'weekly' (W-FRI) or 'monthly' (month-end).
    Ranking variable: lookback-period simple ROC on close.
    """
    missing = [s for s in ETFS_69 if s not in frames or frames[s] is None or frames[s].empty]
    if missing:
        return {"n": 0, "reason": "missing_%s" % ",".join(missing), "trades": []}

    if cadence == "weekly":
        panel = _weekly_panel({s: frames[s] for s in ETFS_69})
    elif cadence == "monthly":
        panel = _monthly_panel({s: frames[s] for s in ETFS_69})
    else:
        raise ValueError("cadence must be weekly or monthly")

    lb = int(lookback)
    if lb < 1 or len(panel) < lb + 2:
        return {"n": 0, "reason": "short_panel", "trades": []}

    roc = panel / panel.shift(lb) - 1.0
    fric = float(friction_pct) / 100.0
    trades: List[Dict[str, Any]] = []
    prev_sym: Optional[str] = None
    keep_same = 0

    for i in range(lb, len(panel) - 1):
        row = roc.iloc[i]
        if row.isna().any():
            continue
        if pick == "lowest":
            sym = str(row.idxmin())
        elif pick == "highest":
            sym = str(row.idxmax())
        else:
            raise ValueError("pick must be lowest or highest")
        if prev_sym is not None and sym == prev_sym:
            keep_same += 1
        entry_px = float(panel.iloc[i][sym])
        exit_px = float(panel.iloc[i + 1][sym])
        if entry_px <= 0 or exit_px <= 0:
            continue
        ret = exit_px / entry_px - 1.0 - 2.0 * fric
        trades.append(
            {
                "signal_date": str(pd.Timestamp(panel.index[i]).date()),
                "exit_date": str(pd.Timestamp(panel.index[i + 1]).date()),
                "symbol": sym,
                "rank_roc_pct": round(100.0 * float(row[sym]), 3),
                "entry_px": round(entry_px, 4),
                "exit_px": round(exit_px, 4),
                "ret_pct": round(100.0 * ret, 4),
            }
        )
        prev_sym = sym

    rets = [t["ret_pct"] / 100.0 for t in trades]
    stats = _trade_stats(rets)
    final_eq, mdd_trade = _compound_equity(rets)
    keep_pct = (100.0 * keep_same / len(trades)) if trades else 0.0

    if not trades:
        return {
            "name": "rot69_%s_lb%s_%s" % (cadence, lb, pick),
            "n": 0,
            "trades": [],
            "reason": "no_trades",
        }

    eq_daily, invested, spy_eq = _rotation_daily_equity(frames, trades, friction_pct=friction_pct)
    start = pd.Timestamp(trades[0]["signal_date"], tz="UTC")
    eq_daily = eq_daily.loc[eq_daily.index >= start]
    invested = invested.reindex(eq_daily.index).fillna(0.0)
    spy_eq = spy_eq.reindex(eq_daily.index).ffill().bfill()
    eq_daily.index = pd.DatetimeIndex(eq_daily.index).tz_localize(None)
    invested.index = pd.DatetimeIndex(invested.index).tz_localize(None)
    spy_eq.index = pd.DatetimeIndex(spy_eq.index).tz_localize(None)

    spy_stats = perf_stats("SPY_buy_hold", spy_eq)
    name = "rot69_%s_lb%s_%s" % (cadence, lb, pick)
    strat_stats = perf_stats(
        name,
        eq_daily,
        invested=invested,
        spy_cagr=spy_stats.cagr,
        spy_sharpe=spy_stats.sharpe,
        trade_rets=rets,
    )
    _ok, gate = passes_phase5_gates(strat_stats, spy_stats)

    claimed = CLAIMED_69_WEEKLY if cadence == "weekly" else CLAIMED_69_MONTHLY
    return {
        "name": name,
        "cadence": cadence,
        "lookback": lb,
        "pick": pick,
        "friction_pct": friction_pct,
        "trades": trades,
        **stats,
        "final_equity": round(final_eq, 4),
        "mdd_trade_eq_pct": round(100.0 * mdd_trade, 2),
        "keep_same_pct": round(keep_pct, 1),
        "cagr_pct": round(100.0 * strat_stats.cagr, 2),
        "sharpe": round(strat_stats.sharpe, 3),
        "mdd_daily_pct": round(100.0 * abs(strat_stats.max_drawdown), 2),
        "spy_cagr_pct": round(100.0 * spy_stats.cagr, 2),
        "spy_sharpe": round(spy_stats.sharpe, 3),
        "spy_mdd_pct": round(100.0 * abs(spy_stats.max_drawdown), 2),
        "gate": gate,
        "start": strat_stats.start,
        "end": strat_stats.end,
        "panel_start": str(pd.Timestamp(panel.index[0]).date()),
        "panel_end": str(pd.Timestamp(panel.index[-1]).date()),
        "claimed": claimed,
    }


def _rotation_daily_equity(
    frames: Dict[str, pd.DataFrame],
    trades: List[Dict[str, Any]],
    *,
    friction_pct: float = 0.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Daily MTM equity while holding the selected ETF between signal and exit."""
    closes = {}
    for sym in ETFS_69:
        d = _ensure_ts(frames[sym]).set_index("ts").sort_index()
        closes[sym] = d["close"].astype(float)
    close_df = pd.DataFrame(closes).sort_index().ffill()
    idx = close_df.index
    date_s = pd.Series([x.date() for x in idx], index=idx)
    held = pd.Series(index=idx, dtype=object)
    for t in trades:
        a_date = pd.Timestamp(t["signal_date"]).date()
        b_date = pd.Timestamp(t["exit_date"]).date()
        mask = (date_s > a_date) & (date_s <= b_date)
        held.loc[mask] = t["symbol"]

    rets = close_df.pct_change().fillna(0.0)
    strat_rets = pd.Series(0.0, index=idx)
    invested = pd.Series(0.0, index=idx)
    for sym in ETFS_69:
        m = held == sym
        strat_rets.loc[m] = rets.loc[m, sym]
        invested.loc[m] = 1.0

    fric = float(friction_pct) / 100.0
    if fric > 0:
        prev = None
        for t in trades:
            sym = t["symbol"]
            if prev is None or prev != sym:
                a_date = pd.Timestamp(t["signal_date"]).date()
                after = idx[date_s > a_date]
                if len(after):
                    strat_rets.loc[after[0]] -= fric
            prev = sym
        last = trades[-1]
        b_date = pd.Timestamp(last["exit_date"]).date()
        exit_rows = idx[date_s == b_date]
        if len(exit_rows):
            strat_rets.loc[exit_rows[0]] -= fric

    eq = equity_from_returns(strat_rets)
    spy_eq = equity_from_returns(rets["SPY"].fillna(0.0))
    return eq, invested, spy_eq


def sweep_rotation69(
    frames: Dict[str, pd.DataFrame],
    *,
    friction_pct: float = 0.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cadence, lookbacks in (("weekly", WEEKLY_LOOKBACKS), ("monthly", MONTHLY_LOOKBACKS)):
        for pick in ("lowest", "highest"):
            for lb in lookbacks:
                r = backtest_rotation69(
                    frames,
                    lookback=lb,
                    pick=pick,
                    cadence=cadence,
                    friction_pct=friction_pct,
                )
                rows.append({k: v for k, v in r.items() if k != "trades"})
    return rows


def best_rotation_rows(rows: Sequence[Dict[str, Any]], *, top_n: int = 8) -> List[Dict[str, Any]]:
    scored = [r for r in rows if r.get("n", 0) >= 30]
    scored.sort(key=lambda r: (r.get("sharpe", 0.0), r.get("cagr_pct", 0.0)), reverse=True)
    return scored[:top_n]
