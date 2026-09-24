"""7-day ROC rebound (Quantified Strategies) vectorized research helpers.

Rules (SPY daily):
  Entry: if flat and 7-day ROC closes below -entry_thresh_pct, buy next open.
  Exit:  if long and 7-day ROC closes above +exit_thresh_pct, sell next open.
  No stop-loss, no profit target. One long at a time.

ROC(n) = (close / close.shift(n) - 1) * 100
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Video-era splits (entry date).
ERA_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("1993-2004", "1993-01-01", "2004-12-31"),
    ("2005-2015", "2005-01-01", "2015-12-31"),
    ("2016-2026", "2016-01-01", "2026-12-31"),
)

YEAR_BUCKETS: Sequence[Tuple[str, str, str]] = (
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)

CLAIMED: Dict[str, Any] = {
    "n": 146,
    "win_rate_pct": 71.2,
    "profit_factor": 2.40,
    "final_equity": 9.15,
    "mdd_pct": 29.0,
    "eras": {
        "1993-2004": {"n": 54, "win_rate_pct": 72.2, "profit_factor": 2.71},
        "2005-2015": {"n": 51, "win_rate_pct": 64.7, "profit_factor": 2.07},
        "2016-2026": {"n": 41, "win_rate_pct": 78.0, "profit_factor": 2.45},
    },
    "source": "Quantified Strategies video SF1OtXR5crQ (2026-09-22)",
}


def roc_series(close: pd.Series, length: int) -> pd.Series:
    """Percent ROC over ``length`` trading days."""
    n = int(length)
    if n < 1:
        raise ValueError("roc length must be >= 1")
    return (close / close.shift(n) - 1.0) * 100.0


def _prior_roc(d: pd.DataFrame, entry_i: int) -> Optional[float]:
    """ROC on the signal bar (session before the next-open fill)."""
    if entry_i <= 0:
        return None
    v = d.iloc[entry_i - 1]["roc"]
    if not pd.notna(v):
        return None
    return round(float(v), 3)


def backtest_roc_rebound(
    df: pd.DataFrame,
    *,
    symbol: str = "SPY",
    roc_length: int = 7,
    entry_thresh_pct: float = 3.0,
    exit_thresh_pct: float = 3.0,
    friction_pct: float = 0.0,
    require_sma200: bool = False,
    vol_scale: bool = False,
    vol_lookback: int = 20,
    vol_k: float = 1.5,
    time_stop_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Run ROC rebound rules on one OHLCV frame.

    ``df`` needs columns ts/open/high/low/close (volume optional).
    Signals fire on the close; fills are next-session open.
    Incomplete open positions at series end are omitted from completed trades.

    Overlays:
      require_sma200: enter only if signal-bar close > SMA200
      vol_scale: entry when ROC < -vol_k * daily_vol% * sqrt(roc_length)
                 (exit stays on fixed exit_thresh_pct unless also below)
      time_stop_bars: force sell next open after N bars in the trade
    """
    if df is None or df.empty:
        return {"symbol": symbol, "n": 0, "trades": [], "by_era": [], "reason": "empty"}

    d = df.copy()
    if "ts" not in d.columns:
        if isinstance(d.index, pd.DatetimeIndex):
            d = d.reset_index().rename(columns={"index": "ts"})
        else:
            return {"symbol": symbol, "n": 0, "trades": [], "by_era": [], "reason": "no_ts"}

    d["ts"] = pd.to_datetime(d["ts"], utc=True, errors="coerce")
    for col in ("open", "high", "low", "close"):
        if col not in d.columns:
            return {
                "symbol": symbol,
                "n": 0,
                "trades": [],
                "by_era": [],
                "reason": "missing_%s" % col,
            }
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = (
        d.dropna(subset=["ts", "open", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )
    min_bars = int(roc_length) + 5
    if require_sma200:
        min_bars = max(min_bars, 210)
    if vol_scale:
        min_bars = max(min_bars, int(vol_lookback) + int(roc_length) + 2)
    if len(d) < min_bars:
        return {
            "symbol": symbol,
            "n": 0,
            "trades": [],
            "by_era": [],
            "reason": "short_history",
            "bars": len(d),
        }

    d["roc"] = roc_series(d["close"], roc_length)
    d["sma200"] = d["close"].rolling(200, min_periods=200).mean()
    # Daily realized vol in percent; scale to ROC horizon via sqrt(time).
    d["daily_vol_pct"] = d["close"].pct_change().rolling(
        int(vol_lookback), min_periods=int(vol_lookback)
    ).std() * 100.0
    d["entry_th_dyn"] = (
        -abs(float(vol_k)) * d["daily_vol_pct"] * float(np.sqrt(int(roc_length)))
    )

    fixed_entry_th = -abs(float(entry_thresh_pct))
    exit_th = abs(float(exit_thresh_pct))
    tstop = None if time_stop_bars is None else int(time_stop_bars)

    trades: List[Dict[str, Any]] = []
    in_pos = False
    entry_i: Optional[int] = None
    entry_px: Optional[float] = None
    pending_entry = False
    pending_exit = False
    pending_exit_reason: Optional[str] = None

    for i in range(len(d)):
        row = d.iloc[i]
        o = float(row["open"])
        roc = row["roc"]
        roc_ok = pd.notna(roc)

        if pending_entry and not in_pos:
            if np.isfinite(o) and o > 0:
                in_pos = True
                entry_i = i
                entry_px = o
            pending_entry = False

        if pending_exit and in_pos:
            assert entry_i is not None and entry_px is not None
            if np.isfinite(o) and o > 0:
                exit_px = o
                ret = (exit_px / entry_px - 1.0) * 100.0 - float(friction_pct)
                trades.append(
                    {
                        "symbol": symbol,
                        "entry_date": str(pd.Timestamp(d.iloc[entry_i]["ts"]).date()),
                        "exit_date": str(pd.Timestamp(row["ts"]).date()),
                        "entry_px": round(float(entry_px), 4),
                        "exit_px": round(exit_px, 4),
                        "ret_pct": round(float(ret), 4),
                        "hold_days": int(i - entry_i),
                        "entry_roc": _prior_roc(d, entry_i),
                        "exit_roc": None if not roc_ok else round(float(roc), 3),
                        "exit_reason": pending_exit_reason or "roc",
                    }
                )
                in_pos = False
                entry_i = None
                entry_px = None
            pending_exit = False
            pending_exit_reason = None

        if in_pos and not pending_exit and entry_i is not None:
            # Time-stop: signal so next-open exit lands at hold_days == tstop.
            if tstop is not None:
                held = i - entry_i
                if held >= max(int(tstop) - 1, 0) and (
                    held > 0 or int(tstop) <= 1
                ):
                    pending_exit = True
                    pending_exit_reason = "time_stop"
                    continue

        if not roc_ok:
            continue

        if in_pos and not pending_exit:
            if float(roc) > exit_th:
                pending_exit = True
                pending_exit_reason = "roc"
        elif (not in_pos) and (not pending_entry):
            if vol_scale:
                th = row["entry_th_dyn"]
                if not pd.notna(th):
                    continue
                entry_ok = float(roc) < float(th)
            else:
                entry_ok = float(roc) < fixed_entry_th
            if not entry_ok:
                continue
            if require_sma200:
                sma = row["sma200"]
                if not pd.notna(sma) or float(row["close"]) <= float(sma):
                    continue
            pending_entry = True

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    params = {
        "roc_length": int(roc_length),
        "entry_thresh_pct": float(entry_thresh_pct),
        "exit_thresh_pct": float(exit_thresh_pct),
        "friction_pct": float(friction_pct),
        "require_sma200": bool(require_sma200),
        "vol_scale": bool(vol_scale),
        "vol_lookback": int(vol_lookback),
        "vol_k": float(vol_k),
        "time_stop_bars": tstop,
    }
    base = {
        "symbol": symbol,
        "params": params,
        "bars": len(d),
        "first_bar": str(pd.Timestamp(d["ts"].iloc[0]).date()),
        "last_bar": str(pd.Timestamp(d["ts"].iloc[-1]).date()),
        "open_at_end": bool(in_pos or pending_entry or pending_exit),
    }
    if n == 0:
        return {
            **base,
            "n": 0,
            "trades": [],
            "by_era": [],
            "by_year_bucket": [],
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
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min() * 100.0)

    hold_sum = int(tdf["hold_days"].sum())
    first_trade_ts = pd.Timestamp(tdf["entry_date"].iloc[0])
    last_bar_ts = pd.Timestamp(d["ts"].iloc[-1])
    if first_trade_ts.tzinfo is not None:
        first_trade_ts = first_trade_ts.tz_convert("UTC").tz_localize(None)
    if last_bar_ts.tzinfo is not None:
        last_bar_ts = last_bar_ts.tz_convert("UTC").tz_localize(None)
    calendar_days = max((last_bar_ts - first_trade_ts).days, 1)
    start_i = int(roc_length)
    trading_days = max(len(d) - start_i, 1)
    exposure = hold_sum / trading_days

    full_years = max((last_bar_ts - first_trade_ts).days / 365.25, 1e-9)
    cagr_full = (eq[-1] ** (1.0 / full_years) - 1.0) * 100.0

    return {
        **base,
        "n": n,
        "win_rate_pct": round(float((gains > 0).mean() * 100), 2),
        "profit_factor": round(float(pf), 3) if np.isfinite(pf) else None,
        "expectancy_pct": round(float(gains.mean()), 3),
        "median_pct": round(float(np.median(gains)), 3),
        "mdd_trade_eq_pct": round(mdd, 2),
        "cagr_full_window_pct": round(float(cagr_full), 2),
        "exposure_pct": round(exposure * 100, 2),
        "avg_hold_days": round(float(tdf["hold_days"].mean()), 2),
        "final_equity": round(float(eq[-1]), 4),
        "first_trade": tdf["entry_date"].iloc[0],
        "last_trade": tdf["exit_date"].iloc[-1],
        "by_era": _bucket_stats(tdf, ERA_BUCKETS),
        "by_year_bucket": _bucket_stats(tdf, YEAR_BUCKETS),
        "trades": trades,
        "calendar_span_days": int(calendar_days),
        "sum_win_pct": round(gross_win, 3),
        "sum_loss_pct": round(gross_loss, 3),
    }


def _bucket_stats(
    tdf: pd.DataFrame, buckets: Sequence[Tuple[str, str, str]]
) -> List[Dict[str, Any]]:
    if tdf is None or tdf.empty:
        return []
    work = tdf.copy()
    work["entry_ts"] = pd.to_datetime(work["entry_date"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for label, start, end in buckets:
        mask = (work["entry_ts"] >= pd.Timestamp(start)) & (
            work["entry_ts"] <= pd.Timestamp(end)
        )
        g = pd.to_numeric(work.loc[mask, "ret_pct"], errors="coerce").dropna()
        n = int(len(g))
        if n == 0:
            rows.append(
                {
                    "bucket": label,
                    "n": 0,
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
                "n": n,
                "expectancy_pct": round(float(g.mean()), 3),
                "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
                "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
            }
        )
    return rows


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
            "final_equity": None,
            "mdd_trade_eq_pct": None,
            "sum_win_pct": None,
            "sum_loss_pct": None,
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
            "final_equity": None,
            "mdd_trade_eq_pct": None,
            "sum_win_pct": None,
            "sum_loss_pct": None,
        }
    wins = g[g > 0]
    losses = g[g <= 0]
    gw = float(wins.sum()) if len(wins) else 0.0
    gl = float(-losses.sum()) if len(losses) else 0.0
    pf = (gw / gl) if gl > 0 else float("inf")
    hold = pd.to_numeric(tdf.get("hold_days"), errors="coerce")
    equity = [1.0]
    for r in g.to_numpy(dtype=float):
        equity.append(equity[-1] * (1.0 + float(r) / 100.0))
    eq = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min() * 100.0)
    return {
        "n": n,
        "win_rate_pct": round(float((g > 0).mean() * 100.0), 2),
        "expectancy_pct": round(float(g.mean()), 3),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
        "median_pct": round(float(g.median()), 3),
        "avg_hold_days": None if hold.isna().all() else round(float(hold.mean()), 2),
        "final_equity": round(float(eq[-1]), 4),
        "mdd_trade_eq_pct": round(mdd, 2),
        "sum_win_pct": round(gw, 3),
        "sum_loss_pct": round(gl, 3),
    }


def drop_top_n_book(trades: List[Dict[str, Any]], n_drop: int = 3) -> Dict[str, Any]:
    """Winner-$ stress: drop the n largest winning trades, recompute book."""
    if not trades or n_drop <= 0:
        return book_from_trades(trades)
    tdf = pd.DataFrame(trades)
    tdf["ret_pct"] = pd.to_numeric(tdf["ret_pct"], errors="coerce")
    tdf = tdf.dropna(subset=["ret_pct"]).sort_values("ret_pct", ascending=False)
    kept = tdf.iloc[int(n_drop) :].to_dict(orient="records")
    out = book_from_trades(kept)
    out["n_dropped"] = int(n_drop)
    return out


def year_bucket_stats(
    trades: List[Dict[str, Any]],
    buckets: Sequence[Tuple[str, str, str]] = YEAR_BUCKETS,
) -> List[Dict[str, Any]]:
    return _bucket_stats(pd.DataFrame(trades) if trades else pd.DataFrame(), buckets)


def geom_mean_year_pf(bucket_rows: List[Dict[str, Any]], *, min_n: int = 30) -> Optional[float]:
    import math

    pfs: List[float] = []
    for row in bucket_rows:
        n = int(row.get("n") or row.get("n_trades") or 0)
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


def sweep_roc_lengths(
    df: pd.DataFrame,
    *,
    lengths: Sequence[int] = range(4, 12),
    entry_thresh_pct: float = 3.0,
    exit_thresh_pct: float = 3.0,
    friction_pct: float = 0.0,
    symbol: str = "SPY",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for n in lengths:
        r = backtest_roc_rebound(
            df,
            symbol=symbol,
            roc_length=int(n),
            entry_thresh_pct=entry_thresh_pct,
            exit_thresh_pct=exit_thresh_pct,
            friction_pct=friction_pct,
        )
        rows.append(
            {
                "roc_length": int(n),
                "n": r.get("n", 0),
                "win_rate_pct": r.get("win_rate_pct"),
                "profit_factor": r.get("profit_factor"),
                "expectancy_pct": r.get("expectancy_pct"),
                "final_equity": r.get("final_equity"),
                "mdd_trade_eq_pct": r.get("mdd_trade_eq_pct"),
            }
        )
    return rows


def sweep_exit_thresholds(
    df: pd.DataFrame,
    *,
    exits: Sequence[float] = (1.0, 2.0, 3.0, 4.0, 5.0),
    roc_length: int = 7,
    entry_thresh_pct: float = 3.0,
    friction_pct: float = 0.0,
    symbol: str = "SPY",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for x in exits:
        r = backtest_roc_rebound(
            df,
            symbol=symbol,
            roc_length=roc_length,
            entry_thresh_pct=entry_thresh_pct,
            exit_thresh_pct=float(x),
            friction_pct=friction_pct,
        )
        rows.append(
            {
                "exit_thresh_pct": float(x),
                "n": r.get("n", 0),
                "win_rate_pct": r.get("win_rate_pct"),
                "profit_factor": r.get("profit_factor"),
                "expectancy_pct": r.get("expectancy_pct"),
                "final_equity": r.get("final_equity"),
                "mdd_trade_eq_pct": r.get("mdd_trade_eq_pct"),
            }
        )
    return rows


def occupancy_fcfs(
    trades: List[Dict[str, Any]],
    *,
    max_open: int = 1,
    unique_symbol: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Greedy FCFS occupancy on completed trades.

    Same-day candidates are ordered by more-negative ``entry_roc`` first
    (strongest dip), then symbol. A slot frees when ``exit_date <`` next
    ``entry_date`` (no same-day flip; both fills are opens).
    """
    if not trades:
        return [], {"n_kept": 0, "n_skipped": 0}
    rows = []
    for i, tr in enumerate(trades):
        buy = pd.Timestamp(tr["entry_date"])
        sell = pd.Timestamp(tr["exit_date"])
        if buy.tzinfo is not None:
            buy = buy.tz_convert("UTC").tz_localize(None)
        if sell.tzinfo is not None:
            sell = sell.tz_convert("UTC").tz_localize(None)
        roc = tr.get("entry_roc")
        try:
            roc_f = float(roc) if roc is not None and pd.notna(roc) else 0.0
        except (TypeError, ValueError):
            roc_f = 0.0
        rows.append(
            {
                **tr,
                "_i": i,
                "_buy": buy.normalize(),
                "_sell": sell.normalize(),
                "_sym": str(tr.get("symbol") or "").upper(),
                "_roc": roc_f,
            }
        )
    rows.sort(key=lambda r: (r["_buy"], r["_roc"], r["_sym"], r["_i"]))

    open_pos: List[Tuple[pd.Timestamp, str]] = []
    kept: List[Dict[str, Any]] = []
    n_skip = 0
    for row in rows:
        buy = row["_buy"]
        open_pos = [(s, sym) for s, sym in open_pos if s > buy]
        if unique_symbol and any(sym == row["_sym"] for _, sym in open_pos):
            n_skip += 1
            continue
        if int(max_open) > 0 and len(open_pos) >= int(max_open):
            n_skip += 1
            continue
        kept.append({k: v for k, v in row.items() if not str(k).startswith("_")})
        open_pos.append((row["_sell"], row["_sym"]))
    return kept, {"n_kept": len(kept), "n_skipped": n_skip, "max_open": int(max_open)}


def _naive_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if getattr(idx, "tz", None) is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def closes_panel_from_frames(
    frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Wide daily close panel (naive dates) from per-symbol OHLCV frames."""
    series: Dict[str, pd.Series] = {}
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        d = df.copy()
        if "ts" not in d.columns:
            if isinstance(d.index, pd.DatetimeIndex):
                d = d.reset_index().rename(columns={d.index.name or "index": "ts"})
            else:
                continue
        ts = pd.to_datetime(d["ts"], utc=True, errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        close = pd.to_numeric(d["close"], errors="coerce")
        s = pd.Series(close.values, index=pd.DatetimeIndex(ts).normalize())
        s = s[~s.index.isna()].groupby(level=0).last().sort_index()
        series[str(sym).upper()] = s
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    return panel


def single_slot_equity(
    trades: List[Dict[str, Any]],
    close_panel: pd.DataFrame,
    *,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Fully invest one slot when in a trade; cash=1 when flat. Daily MTM.

    Returns (equity, invested_flag) on the close panel calendar.
    """
    if close_panel is None or close_panel.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    panel = close_panel.copy()
    panel.index = _naive_index(pd.DatetimeIndex(panel.index)).normalize()
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()

    s = pd.Timestamp(eval_start) if eval_start else panel.index[0]
    e = pd.Timestamp(eval_end) if eval_end else panel.index[-1]
    if s.tzinfo is not None:
        s = s.tz_convert("UTC").tz_localize(None)
    if e.tzinfo is not None:
        e = e.tz_convert("UTC").tz_localize(None)
    calendar = panel.index[(panel.index >= s.normalize()) & (panel.index <= e.normalize())]
    if calendar.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()
    if tdf.empty:
        eq = pd.Series(1.0, index=calendar)
        inv = pd.Series(0.0, index=calendar)
        return eq, inv

    tdf = tdf.copy()
    buy = pd.to_datetime(tdf["entry_date"], errors="coerce")
    sell = pd.to_datetime(tdf["exit_date"], errors="coerce")
    if getattr(buy.dt, "tz", None) is not None:
        buy = buy.dt.tz_convert("UTC").dt.tz_localize(None)
    if getattr(sell.dt, "tz", None) is not None:
        sell = sell.dt.tz_convert("UTC").dt.tz_localize(None)
    tdf["_buy"] = buy.dt.normalize()
    tdf["_sell"] = sell.dt.normalize()
    tdf["_sym"] = tdf["symbol"].astype(str).str.upper()
    tdf["entry_px"] = pd.to_numeric(tdf["entry_px"], errors="coerce")
    tdf["exit_px"] = pd.to_numeric(tdf["exit_px"], errors="coerce")
    tdf = tdf.dropna(subset=["_buy", "_sell", "entry_px", "exit_px"])
    tdf = tdf.sort_values(["_buy", "_sym"]).reset_index(drop=True)

    by_buy: Dict[pd.Timestamp, List[int]] = {}
    for idx, row in tdf.iterrows():
        by_buy.setdefault(pd.Timestamp(row["_buy"]), []).append(int(idx))

    cash = 1.0
    pos: Optional[Dict[str, Any]] = None
    eq_pts: List[float] = []
    inv_pts: List[float] = []

    def _mark(dt: pd.Timestamp) -> float:
        if pos is None:
            return cash
        sym = pos["sym"]
        px = pos["last_px"]
        if dt in panel.index and sym in panel.columns and pd.notna(panel.loc[dt, sym]):
            px = float(panel.loc[dt, sym])
            pos["last_px"] = px
        return pos["shares"] * px

    for dt in calendar:
        # Exit before entry on the same calendar day.
        if pos is not None and pd.Timestamp(pos["sell_date"]) <= dt:
            cash = pos["shares"] * float(pos["exit_px"])
            pos = None

        if pos is None and dt in by_buy:
            for idx in by_buy[dt]:
                row = tdf.loc[idx]
                entry_px = float(row["entry_px"])
                if entry_px <= 0 or cash <= 0:
                    continue
                shares = cash / entry_px
                last_px = entry_px
                sym = str(row["_sym"])
                if dt in panel.index and sym in panel.columns and pd.notna(panel.loc[dt, sym]):
                    last_px = float(panel.loc[dt, sym])
                pos = {
                    "sym": sym,
                    "shares": shares,
                    "last_px": last_px,
                    "sell_date": pd.Timestamp(row["_sell"]),
                    "exit_px": float(row["exit_px"]),
                }
                cash = 0.0
                break  # single slot

        eq_pts.append(_mark(dt) if pos is not None else cash)
        inv_pts.append(1.0 if pos is not None else 0.0)

    if pos is not None:
        cash = pos["shares"] * float(pos["exit_px"])
        eq_pts[-1] = cash
        pos = None

    eq = pd.Series(eq_pts, index=calendar)
    inv = pd.Series(inv_pts, index=calendar)
    if float(eq.iloc[0]) > 0:
        eq = eq / float(eq.iloc[0])
    return eq, inv


def spy_buy_hold_equity(
    spy_close: pd.Series,
    *,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    s = spy_close.dropna().astype(float).sort_index()
    s.index = _naive_index(pd.DatetimeIndex(s.index)).normalize()
    s = s[~s.index.duplicated(keep="last")]
    start = pd.Timestamp(eval_start) if eval_start else s.index[0]
    end = pd.Timestamp(eval_end) if eval_end else s.index[-1]
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)
    w = s.loc[(s.index >= start.normalize()) & (s.index <= end.normalize())]
    if w.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    eq = w / float(w.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv
