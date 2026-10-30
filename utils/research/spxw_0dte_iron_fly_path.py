"""Intraday path MTM + Phase-5 equity for the SPXW 0DTE short iron fly.

Pulls 1m NBBO for all four legs from 13:45 ET to the close, marks the short
package each minute, and reports MAE/MFE plus an optional 50%-of-credit exit.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.metrics import (
    equity_from_returns,
    passes_phase5_gates,
    perf_stats,
)
from utils.research.spxw_0dte_iron_fly import (
    DEFAULT_MULT,
    ENTRY_CLOCK,
    FILL_MID,
    FILL_NATURAL,
    OPT_SYMBOL,
    _read_cached,
    _theta_csv,
    _write_cached,
    summarize_pnls,
)

LOG = logging.getLogger(__name__)

TP_FRAC = 0.50  # take 50% of credit
PATH_START = "13:45:00"
PATH_END = "16:00:00"
UNDERWATER_FRAC = 0.50  # "deeper than half credit" adverse


def fetch_leg_path(
    client: Any,
    *,
    day: date,
    strike: float,
    right: str,
    cache_dir: Path,
    start_time: str = PATH_START,
    end_time: str = PATH_END,
) -> Optional[pd.DataFrame]:
    right_s = "put" if str(right).lower().startswith("p") else "call"
    name = f"{day.isoformat()}_{right_s}_{float(strike):.3f}.csv.gz"
    path = cache_dir / "paths" / name
    cached = _read_cached(path)
    if cached is not None:
        return cached
    params = {
        "symbol": OPT_SYMBOL,
        "expiration": day.isoformat(),
        "strike": f"{float(strike):.3f}",
        "right": right_s,
        "date": day.isoformat(),
        "interval": "1m",
        "start_time": start_time,
        "end_time": end_time,
    }
    df = _theta_csv(client, "/v3/option/history/quote", params)
    if df is None:
        return None
    if not df.empty:
        df.columns = [str(c).strip().lower() for c in df.columns]
        keep = [c for c in ("timestamp", "bid", "ask") if c in df.columns]
        df = df[keep].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
        for col in ("bid", "ask"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _write_cached(path, df)
    return df


def _normalize_leg(df: Optional[pd.DataFrame], label: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", f"{label}_bid", f"{label}_ask", f"{label}_mid"])
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"])
    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize("America/New_York")
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert("America/New_York")
    # floor to minute for join
    out["timestamp"] = out["timestamp"].dt.floor("min")
    out[f"{label}_bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out[f"{label}_ask"] = pd.to_numeric(out["ask"], errors="coerce")
    out[f"{label}_mid"] = 0.5 * (out[f"{label}_bid"] + out[f"{label}_ask"])
    return out[["timestamp", f"{label}_bid", f"{label}_ask", f"{label}_mid"]].drop_duplicates(
        "timestamp", keep="last"
    )


def merge_package_paths(
    short_put: pd.DataFrame,
    short_call: pd.DataFrame,
    long_put: pd.DataFrame,
    long_call: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join four legs on minute timestamp; compute mid/natural close costs."""
    sp = _normalize_leg(short_put, "sp")
    sc = _normalize_leg(short_call, "sc")
    lp = _normalize_leg(long_put, "lp")
    lc = _normalize_leg(long_call, "lc")
    if sp.empty or sc.empty or lp.empty or lc.empty:
        return pd.DataFrame()
    m = sp.merge(sc, on="timestamp", how="inner")
    m = m.merge(lp, on="timestamp", how="inner")
    m = m.merge(lc, on="timestamp", how="inner")
    if m.empty:
        return m
    # Valid two-sided quotes
    for prefix in ("sp", "sc", "lp", "lc"):
        m = m.loc[
            (m[f"{prefix}_bid"] > 0)
            & (m[f"{prefix}_ask"] > 0)
            & (m[f"{prefix}_ask"] >= m[f"{prefix}_bid"])
        ].copy()
    if m.empty:
        return m
    # Short package close cost (buy shorts, sell longs)
    m["close_mid"] = m["sp_mid"] + m["sc_mid"] - m["lp_mid"] - m["lc_mid"]
    m["close_natural"] = m["sp_ask"] + m["sc_ask"] - m["lp_bid"] - m["lc_bid"]
    return m.sort_values("timestamp").reset_index(drop=True)


def mtm_series(
    package: pd.DataFrame,
    *,
    credit: float,
    mark: str = FILL_MID,
    mult: float = DEFAULT_MULT,
) -> pd.DataFrame:
    """MTM dollar P&L vs entry credit for each minute."""
    if package is None or package.empty:
        return pd.DataFrame()
    col = "close_natural" if mark == FILL_NATURAL else "close_mid"
    out = package[["timestamp", "close_mid", "close_natural"]].copy()
    out["close_cost"] = out[col]
    out["pnl"] = (float(credit) - out["close_cost"]) * float(mult)
    return out


def path_stats(
    mtm: pd.DataFrame,
    *,
    credit: float,
    settle_pnl: float,
    tp_frac: float = TP_FRAC,
    mult: float = DEFAULT_MULT,
) -> Dict[str, Any]:
    """MAE/MFE, 50% TP overlay, underwater flags from an MTM path."""
    empty = {
        "n_bars": 0,
        "mae": None,
        "mfe": None,
        "mae_ts": None,
        "mfe_ts": None,
        "settle_pnl": float(settle_pnl),
        "tp_pnl": float(settle_pnl),
        "tp_hit": False,
        "tp_ts": None,
        "exit_mode": "settle",
        "underwater_half": False,
        "winner_was_underwater_half": False,
    }
    if mtm is None or mtm.empty or "pnl" not in mtm.columns:
        return empty
    pnls = mtm["pnl"].astype(float)
    mae_i = int(pnls.idxmin())
    mfe_i = int(pnls.idxmax())
    mae = float(pnls.min())
    mfe = float(pnls.max())
    mae_ts = mtm.loc[mae_i, "timestamp"]
    mfe_ts = mtm.loc[mfe_i, "timestamp"]
    half_credit_dollars = float(tp_frac) * float(credit) * float(mult)
    # Adverse beyond half credit (more negative than -half)
    underwater = bool((pnls < -half_credit_dollars).any())
    # 50% TP: first time MTM >= +half credit
    tp_hit = False
    tp_ts = None
    tp_pnl = float(settle_pnl)
    exit_mode = "settle"
    hits = mtm.loc[pnls >= half_credit_dollars]
    if not hits.empty:
        row = hits.iloc[0]
        tp_hit = True
        tp_ts = row["timestamp"]
        tp_pnl = float(half_credit_dollars)  # fill at the target level
        exit_mode = "tp50"
    won_settle = float(settle_pnl) > 0
    return {
        "n_bars": int(len(mtm)),
        "mae": mae,
        "mfe": mfe,
        "mae_ts": mae_ts,
        "mfe_ts": mfe_ts,
        "settle_pnl": float(settle_pnl),
        "tp_pnl": tp_pnl,
        "tp_hit": tp_hit,
        "tp_ts": tp_ts,
        "exit_mode": exit_mode,
        "underwater_half": underwater,
        "winner_was_underwater_half": bool(won_settle and underwater),
        "half_credit_dollars": half_credit_dollars,
    }


def analyze_trade_path(
    client: Any,
    trade: Dict[str, Any],
    *,
    cache_dir: Path,
    mark: str = FILL_MID,
) -> Dict[str, Any]:
    """Fetch four legs and compute path stats for one trade row."""
    day = trade["date"]
    if isinstance(day, str):
        day = date.fromisoformat(day[:10])
    legs = {}
    for key, right in (
        ("short_put", "put"),
        ("short_call", "call"),
        ("long_put", "put"),
        ("long_call", "call"),
    ):
        df = fetch_leg_path(
            client,
            day=day,
            strike=float(trade[key]),
            right=right,
            cache_dir=cache_dir,
        )
        if df is None:
            return {"date": day, "skip": "theta_error", "leg": key}
        if df.empty:
            return {"date": day, "skip": "no_path", "leg": key}
        legs[key] = df
    package = merge_package_paths(
        legs["short_put"],
        legs["short_call"],
        legs["long_put"],
        legs["long_call"],
    )
    if package.empty:
        return {"date": day, "skip": "no_join"}
    credit = float(trade["credit"])
    settle_pnl = float(trade["pnl"])
    mtm = mtm_series(package, credit=credit, mark=mark)
    stats = path_stats(mtm, credit=credit, settle_pnl=settle_pnl)
    # Exit natural vs mid at TP bar or last bar
    exit_row = package.iloc[-1]
    if stats["tp_hit"] and stats["tp_ts"] is not None:
        hit = package.loc[package["timestamp"] == stats["tp_ts"]]
        if not hit.empty:
            exit_row = hit.iloc[0]
    exit_mid_pnl = (credit - float(exit_row["close_mid"])) * DEFAULT_MULT
    exit_nat_pnl = (credit - float(exit_row["close_natural"])) * DEFAULT_MULT
    out = {
        "date": day,
        "fill_entry": trade.get("fill"),
        "mark": mark,
        "credit": credit,
        "credit_mid": float(trade.get("credit_mid", credit)),
        "credit_natural": float(trade.get("credit_natural", credit)),
        "entry_slip_pts": float(trade.get("credit_mid", credit))
        - float(trade.get("credit_natural", credit)),
        "settle_pnl": settle_pnl,
        "exit_mid_pnl": float(exit_mid_pnl),
        "exit_nat_pnl": float(exit_nat_pnl),
        **{k: stats[k] for k in stats},
    }
    return out


def daily_returns_on_sessions(
    trades: Sequence[Dict[str, Any]],
    sessions: Iterable[Any],
    *,
    start_equity: float,
    pnl_key: str = "pnl",
) -> pd.Series:
    """Map trade P&L onto session dates; missing sessions are flat."""
    pnl_by_day: Dict[date, float] = {}
    for t in trades:
        d = t.get("date")
        if isinstance(d, str):
            d = date.fromisoformat(d[:10])
        if isinstance(d, datetime):
            d = d.date()
        if d is None:
            continue
        pnl_by_day[d] = pnl_by_day.get(d, 0.0) + float(t[pnl_key])
    equity = float(start_equity)
    out: Dict[pd.Timestamp, float] = {}
    for raw in sessions:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            d = raw
        else:
            d = pd.Timestamp(raw).date()
        pnl = float(pnl_by_day.get(d, 0.0))
        ret = pnl / equity if equity else 0.0
        equity = equity + pnl
        out[pd.Timestamp(d)] = ret
    return pd.Series(out, dtype=float).sort_index()


def spy_buyhold_equity(
    closes: pd.Series,
    *,
    start_equity: float = 25_000.0,
) -> pd.Series:
    s = closes.dropna().astype(float).sort_index()
    if s.empty:
        return pd.Series(dtype=float)
    idx = pd.DatetimeIndex(pd.to_datetime(s.index).normalize())
    s.index = idx
    return start_equity * s / float(s.iloc[0])


def phase5_report(
    name: str,
    trade_pnls: Sequence[Dict[str, Any]],
    sessions: Sequence[date],
    spy_closes: pd.Series,
    *,
    start_equity: float = 25_000.0,
    pnl_key: str = "pnl",
) -> Dict[str, Any]:
    rets = daily_returns_on_sessions(
        trade_pnls, sessions, start_equity=start_equity, pnl_key=pnl_key
    )
    eq = equity_from_returns(rets, start_equity=start_equity)
    # Align SPY to same sessions
    spy_s = spy_closes.copy()
    spy_s.index = pd.DatetimeIndex(pd.to_datetime(spy_s.index).normalize())
    spy_s = spy_s.reindex(eq.index).ffill().bfill()
    spy_eq = spy_buyhold_equity(spy_s, start_equity=start_equity)
    spy_eq = spy_eq.reindex(eq.index).ffill()
    trade_rets = [float(t[pnl_key]) / float(start_equity) for t in trade_pnls]
    strat = perf_stats(
        name,
        eq,
        invested=(rets != 0).astype(float),
        trade_rets=trade_rets,
    )
    spy = perf_stats("SPY_buy_hold", spy_eq)
    strat.excess_cagr_vs_spy = float(strat.cagr - spy.cagr)
    strat.excess_sharpe_vs_spy = float(strat.sharpe - spy.sharpe)
    ok, gate = passes_phase5_gates(strat, spy)
    book = summarize_pnls([float(t[pnl_key]) for t in trade_pnls])
    return {
        "name": name,
        "gate_pass": bool(ok),
        "gate": gate,
        "strat": strat.as_dict(),
        "spy": spy.as_dict(),
        "book": book,
        "n_trades": len(trade_pnls),
    }


def summarize_path_book(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if "skip" not in r and r.get("mae") is not None]
    if not usable:
        return {"n": 0}
    mae = np.array([float(r["mae"]) for r in usable], dtype=float)
    mfe = np.array([float(r["mfe"]) for r in usable], dtype=float)
    settle = summarize_pnls([float(r["settle_pnl"]) for r in usable])
    tp = summarize_pnls([float(r["tp_pnl"]) for r in usable])
    winners = [r for r in usable if float(r["settle_pnl"]) > 0]
    uw_winners = [r for r in winners if r.get("winner_was_underwater_half")]
    slips = [float(r.get("entry_slip_pts") or 0.0) for r in usable]
    exit_haircut = [
        float(r["exit_mid_pnl"]) - float(r["exit_nat_pnl"]) for r in usable
    ]
    return {
        "n": len(usable),
        "mae_mean": float(mae.mean()),
        "mae_median": float(np.median(mae)),
        "mae_p10": float(np.percentile(mae, 10)),
        "mfe_mean": float(mfe.mean()),
        "mfe_median": float(np.median(mfe)),
        "underwater_half_frac": float(
            np.mean([1.0 if r.get("underwater_half") else 0.0 for r in usable])
        ),
        "winner_underwater_half_frac": (
            float(len(uw_winners) / len(winners)) if winners else None
        ),
        "tp_hit_frac": float(np.mean([1.0 if r.get("tp_hit") else 0.0 for r in usable])),
        "settle": settle,
        "tp50": tp,
        "entry_slip_pts_median": float(np.median(slips)),
        "entry_slip_pts_p90": float(np.percentile(slips, 90)),
        "entry_slip_dollars_median": float(np.median(slips) * DEFAULT_MULT),
        "entry_slip_dollars_p90": float(np.percentile(slips, 90) * DEFAULT_MULT),
        "exit_mid_minus_nat_median": float(np.median(exit_haircut)),
        "exit_mid_minus_nat_p90": float(np.percentile(exit_haircut, 90)),
    }
