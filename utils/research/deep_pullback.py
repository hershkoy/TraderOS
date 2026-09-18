"""Deep Pullback (Quantified Strategies) vectorized research helpers.

Entry at close when all true:
  1) close is lowest close of last 15 sessions
  2) 1-day % return is lowest of last 10 sessions
  3) close > SMA200
Exit (QS): sell at close when close > prior day's high.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

YEAR_BUCKETS: Sequence[tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)

# Broad equity index ETFs. Most are absent from TimescaleDB stock panels;
# the universe runner loads DB when present and Yahoo otherwise.
INDEX_SYMBOLS: Sequence[str] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "VTI",
    "MDY",
)


def backtest_deep_pullback(
    df: pd.DataFrame,
    *,
    symbol: str,
    friction_pct: float = 0.0,
) -> Dict[str, Any]:
    """Run QS deep-pullback rules on one OHLCV frame.

    ``df`` needs columns ts/open/high/low/close (volume optional).
    Returns summary dict with ``trades`` list and book metrics.
    """
    if df is None or df.empty:
        return {"symbol": symbol, "n": 0, "trades": [], "by_year": [], "reason": "empty"}

    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "ts"})
        else:
            return {"symbol": symbol, "n": 0, "trades": [], "by_year": [], "reason": "no_ts"}

    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return {
                "symbol": symbol,
                "n": 0,
                "trades": [],
                "by_year": [],
                "reason": "missing_%s" % col,
            }
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "close", "high"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    if len(d) < 220:
        return {
            "symbol": symbol,
            "n": 0,
            "trades": [],
            "by_year": [],
            "reason": "short_history",
            "bars": len(d),
        }

    d["ret1"] = d["close"].pct_change()
    d["sma200"] = d["close"].rolling(200, min_periods=200).mean()
    d["min_close_15"] = d["close"].rolling(15, min_periods=15).min()
    d["min_ret_10"] = d["ret1"].rolling(10, min_periods=10).min()
    d["prior_high"] = d["high"].shift(1)

    cond1 = d["close"] <= d["min_close_15"]
    cond2 = d["ret1"] <= d["min_ret_10"]
    cond3 = d["close"] > d["sma200"]
    d["entry_signal"] = cond1 & cond2 & cond3 & d["sma200"].notna()

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

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    if n == 0:
        return {
            "symbol": symbol,
            "n": 0,
            "n_entry_signals": int(d["entry_signal"].sum()),
            "trades": [],
            "by_year": [],
            "friction_pct": friction_pct,
            "bars": len(d),
            "first_bar": str(pd.Timestamp(d["ts"].iloc[0]).date()),
            "last_bar": str(pd.Timestamp(d["ts"].iloc[-1]).date()),
        }

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
    mdd = float((eq / peak - 1.0).min() * 100.0)

    hold_sum = int(tdf["hold_days"].sum())
    start_i = 199
    calendar_days = max(len(d) - start_i, 1)
    exposure = hold_sum / calendar_days

    full_start = pd.Timestamp(d.iloc[start_i]["ts"])
    full_end = pd.Timestamp(d.iloc[-1]["ts"])
    full_years = max((full_end - full_start).days / 365.25, 1e-9)
    cagr_full = (eq[-1] ** (1.0 / full_years) - 1.0) * 100.0
    risk_adj = cagr_full / exposure if exposure > 0 else None

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

    return {
        "symbol": symbol,
        "n": n,
        "n_entry_signals": int(d["entry_signal"].sum()),
        "win_rate_pct": round(float((gains > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else None,
        "avg_gain_pct": round(float(gains.mean()), 3),
        "median_gain_pct": round(float(np.median(gains)), 3),
        "mdd_trade_eq_pct": round(mdd, 2),
        "cagr_full_window_pct": round(float(cagr_full), 2),
        "exposure_pct": round(exposure * 100, 2),
        "risk_adj_return_pct": None if risk_adj is None else round(float(risk_adj), 1),
        "avg_hold_days": round(float(tdf["hold_days"].mean()), 2),
        "final_equity": round(float(eq[-1]), 4),
        "bars": len(d),
        "first_bar": str(pd.Timestamp(d["ts"].iloc[0]).date()),
        "last_bar": str(pd.Timestamp(d["ts"].iloc[-1]).date()),
        "first_trade": tdf["entry_date"].iloc[0],
        "last_trade": tdf["exit_date"].iloc[-1],
        "friction_pct": friction_pct,
        "by_year": by_year,
        "trades": trades,
    }


def book_from_trades(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pooled trade-list stats (equal-weight each fill, not portfolio)."""
    if not trades:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "median_pct": None,
            "avg_hold_days": None,
        }
    tdf = pd.DataFrame(trades)
    g = pd.to_numeric(tdf["ret_pct"], errors="coerce").dropna()
    n = int(len(g))
    if n == 0:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "profit_factor": None,
            "median_pct": None,
            "avg_hold_days": None,
        }
    wins = g[g > 0]
    losses = g[g <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = (gw / gl) if gl > 0 else float("inf")
    hold = pd.to_numeric(tdf.get("hold_days"), errors="coerce")
    return {
        "n": n,
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "expectancy_pct": round(float(g.mean()), 3),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
        "median_pct": round(float(g.median()), 3),
        "avg_hold_days": None if hold.isna().all() else round(float(hold.mean()), 2),
    }


def year_bucket_stats(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """YEAR_BUCKETS rows for geo-PF ranking (entry_date based)."""
    if not trades:
        return []
    tdf = pd.DataFrame(trades)
    tdf["entry_ts"] = pd.to_datetime(tdf["entry_date"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for label, start, end in YEAR_BUCKETS:
        mask = (tdf["entry_ts"] >= pd.Timestamp(start)) & (
            tdf["entry_ts"] <= pd.Timestamp(end)
        )
        g = pd.to_numeric(tdf.loc[mask, "ret_pct"], errors="coerce").dropna()
        n = int(len(g))
        if n == 0:
            rows.append(
                {
                    "bucket": label,
                    "n_trades": 0,
                    "expectancy_pct": None,
                    "profit_factor": None,
                    "win_rate_pct": None,
                }
            )
            continue
        wins = g[g > 0]
        losses = g[g <= 0]
        gw = float(wins.sum()) if len(wins) else 0.0
        gl = float(-losses.sum()) if len(losses) else 0.0
        pf = (gw / gl) if gl > 0 else float("inf")
        rows.append(
            {
                "bucket": label,
                "n_trades": n,
                "expectancy_pct": round(float(g.mean()), 3),
                "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
                "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
            }
        )
    return rows


def geom_mean_year_pf(bucket_rows: List[Dict[str, Any]], *, min_n: int = 30) -> Optional[float]:
    import math

    pfs: List[float] = []
    for row in bucket_rows:
        n = int(row.get("n_trades") or 0)
        pf = row.get("profit_factor")
        if n < int(min_n) or pf is None:
            continue
        try:
            pf_f = float(pf)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(pf_f) or pf_f <= 0:
            continue
        pfs.append(pf_f)
    if not pfs:
        return None
    return float(math.exp(sum(math.log(p) for p in pfs) / len(pfs)))
