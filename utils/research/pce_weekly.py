"""Gozlan Precision Candle (PCE) weekly claim-check helpers.

Video (NQ vs ES): a Precision Candle is a bar that closes green on one index
and red on the correlated index (SMT close-direction divergence). Weekly PCE
is treated as HTF POI; the video then drops to 4H for FVG/OB entry.

This module tests the weekly signal edge only (ETF proxy QQQ vs SPY), because
TimescaleDB has no NQ/ES futures and QQQ has no 15m/5m for a 4H entry book.

Books (single-name, one week hold, non-overlapping):
  long_green_next_open  — long the green-leg ETF Mon open -> next Fri close
  long_strong_next_open — same, but only if green leg failed to sweep prior low
  long_green_claim_close — Fri close fill (optimistic upper bound)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.metrics import equity_from_returns, passes_phase5_gates, perf_stats
from utils.research.qs_weekly import to_weekly_friday

SOURCE_URL = "https://www.youtube.com/watch?v=8ru-XJOs7bU"
PAIR_NOTE = "QQQ/SPY daily ETF proxy for NQ/ES; weekly resample W-FRI; no 4H entry layer"


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
    return (
        d.dropna(subset=["ts", "open", "high", "low", "close"])
        .sort_values("ts")
        .drop_duplicates("ts", keep="last")
        .reset_index(drop=True)
    )


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


def align_weekly_pair(
    a_daily: pd.DataFrame,
    b_daily: pd.DataFrame,
    *,
    a_name: str = "QQQ",
    b_name: str = "SPY",
) -> pd.DataFrame:
    """Resample both legs to Friday weeks and inner-join on calendar date."""
    wa = to_weekly_friday(_ensure_ts(a_daily)).rename(
        columns={
            "open": "a_open",
            "high": "a_high",
            "low": "a_low",
            "close": "a_close",
            "ts": "a_ts",
        }
    )
    wb = to_weekly_friday(_ensure_ts(b_daily)).rename(
        columns={
            "open": "b_open",
            "high": "b_high",
            "low": "b_low",
            "close": "b_close",
            "ts": "b_ts",
        }
    )
    wa["week_date"] = pd.to_datetime(wa["a_ts"], utc=True).dt.normalize()
    wb["week_date"] = pd.to_datetime(wb["b_ts"], utc=True).dt.normalize()
    m = wa.merge(wb, on="week_date", how="inner").sort_values("week_date").reset_index(drop=True)
    m["ts"] = m["a_ts"]
    m["a_green"] = m["a_close"] > m["a_open"]
    m["b_green"] = m["b_close"] > m["b_open"]
    m["a_red"] = m["a_close"] < m["a_open"]
    m["b_red"] = m["b_close"] < m["b_open"]
    m["pce"] = (m["a_green"] & m["b_red"]) | (m["a_red"] & m["b_green"])
    m["green_leg"] = np.where(m["a_green"] & m["b_red"], a_name, np.where(m["a_red"] & m["b_green"], b_name, None))
    m["red_leg"] = np.where(m["a_green"] & m["b_red"], b_name, np.where(m["a_red"] & m["b_green"], a_name, None))
    # Failed prior-week low sweep on the green leg (relative strength proxy from the video).
    a_prior_low = m["a_low"].shift(1)
    b_prior_low = m["b_low"].shift(1)
    a_no_sweep = m["a_low"] > a_prior_low
    b_no_sweep = m["b_low"] > b_prior_low
    m["green_failed_sweep"] = np.where(
        m["green_leg"] == a_name,
        a_no_sweep,
        np.where(m["green_leg"] == b_name, b_no_sweep, False),
    )
    m["a_name"] = a_name
    m["b_name"] = b_name
    return m


def detect_pce_events(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return one row per completed weekly PCE (needs a following week for exit)."""
    if weekly is None or weekly.empty or "pce" not in weekly.columns:
        return pd.DataFrame()
    rows = []
    for i in range(1, len(weekly) - 1):
        row = weekly.iloc[i]
        if not bool(row["pce"]):
            continue
        green = row["green_leg"]
        if green is None or (isinstance(green, float) and pd.isna(green)):
            continue
        rows.append(
            {
                "i": i,
                "signal_ts": pd.Timestamp(row["ts"]),
                "signal_date": str(pd.Timestamp(row["ts"]).date()),
                "exit_ts": pd.Timestamp(weekly.iloc[i + 1]["ts"]),
                "exit_date": str(pd.Timestamp(weekly.iloc[i + 1]["ts"]).date()),
                "green_leg": str(green),
                "red_leg": str(row["red_leg"]),
                "green_failed_sweep": bool(row["green_failed_sweep"]),
                "a_close": float(row["a_close"]),
                "b_close": float(row["b_close"]),
                "next_a_close": float(weekly.iloc[i + 1]["a_close"]),
                "next_b_close": float(weekly.iloc[i + 1]["b_close"]),
            }
        )
    return pd.DataFrame(rows)


def _next_session_open(
    daily: pd.DataFrame,
    after_ts: pd.Timestamp,
) -> Optional[Tuple[pd.Timestamp, float]]:
    d = _ensure_ts(daily)
    sig_ns = int(pd.Timestamp(after_ts).value)
    daily_ns = np.array([int(pd.Timestamp(t).value) for t in d["ts"].to_numpy()], dtype=np.int64)
    j = int(np.searchsorted(daily_ns, sig_ns, side="right"))
    if j >= len(d):
        return None
    return pd.Timestamp(d["ts"].iloc[j]), float(d["open"].iloc[j])


def _leg_frames(
    a_daily: pd.DataFrame,
    b_daily: pd.DataFrame,
    *,
    a_name: str,
    b_name: str,
) -> Dict[str, pd.DataFrame]:
    return {a_name: _ensure_ts(a_daily), b_name: _ensure_ts(b_daily)}


def backtest_pce_weekly(
    a_daily: pd.DataFrame,
    b_daily: pd.DataFrame,
    *,
    a_name: str = "QQQ",
    b_name: str = "SPY",
    book: str = "long_green_next_open",
    friction_pct: float = 0.0,
    require_failed_sweep: bool = False,
) -> Dict[str, Any]:
    """Backtest weekly PCE long-the-green-leg books.

    book:
      long_green_next_open   — Mon open after signal Friday
      long_green_claim_close — signal Friday close (optimistic)
    """
    weekly = align_weekly_pair(a_daily, b_daily, a_name=a_name, b_name=b_name)
    events = detect_pce_events(weekly)
    frames = _leg_frames(a_daily, b_daily, a_name=a_name, b_name=b_name)
    fric = float(friction_pct) / 100.0

    trades: List[Dict[str, Any]] = []
    busy_until: Optional[pd.Timestamp] = None

    for _, ev in events.iterrows():
        if require_failed_sweep and not bool(ev["green_failed_sweep"]):
            continue
        sig_ts = pd.Timestamp(ev["signal_ts"])
        if busy_until is not None and sig_ts <= busy_until:
            continue
        leg = str(ev["green_leg"])
        leg_df = frames[leg]
        exit_ts = pd.Timestamp(ev["exit_ts"])
        if leg == a_name:
            exit_px = float(ev["next_a_close"])
            claim_px = float(ev["a_close"])
        else:
            exit_px = float(ev["next_b_close"])
            claim_px = float(ev["b_close"])

        if book == "long_green_claim_close":
            entry_ts = sig_ts
            entry_px = claim_px
        elif book == "long_green_next_open":
            nxt = _next_session_open(leg_df, sig_ts)
            if nxt is None:
                continue
            entry_ts, entry_px = nxt
        else:
            raise ValueError("unknown book: %s" % book)

        if entry_px <= 0 or exit_px <= 0:
            continue
        ret = exit_px / entry_px - 1.0 - 2.0 * fric
        trades.append(
            {
                "signal_date": ev["signal_date"],
                "entry_date": str(pd.Timestamp(entry_ts).date()),
                "exit_date": ev["exit_date"],
                "symbol": leg,
                "red_leg": str(ev["red_leg"]),
                "entry_px": round(entry_px, 4),
                "exit_px": round(exit_px, 4),
                "ret_pct": round(100.0 * ret, 4),
                "green_failed_sweep": bool(ev["green_failed_sweep"]),
            }
        )
        busy_until = exit_ts

    rets = [t["ret_pct"] / 100.0 for t in trades]
    stats = _trade_stats(rets)
    name = book + ("_sweep" if require_failed_sweep else "")
    if friction_pct:
        name = "%s_fric%.2f" % (name, friction_pct)

    eq_daily, invested, spy_eq = _multi_leg_daily_equity(
        frames,
        trades,
        spy_daily=frames[b_name],
        fill_mode="claim_close" if book == "long_green_claim_close" else "next_open",
        friction_pct=friction_pct,
    )
    spy_eq = spy_eq.reindex(eq_daily.index).ffill().bfill()
    spy_stats = perf_stats("SPY_buy_hold", spy_eq)
    strat_stats = perf_stats(
        name,
        eq_daily,
        invested=invested,
        spy_cagr=spy_stats.cagr,
        spy_sharpe=spy_stats.sharpe,
        trade_rets=rets,
    )
    _ok, gate = passes_phase5_gates(strat_stats, spy_stats)

    n_pce = int(weekly["pce"].sum()) if len(weekly) else 0
    return {
        "name": name,
        "book": book,
        "require_failed_sweep": require_failed_sweep,
        "friction_pct": friction_pct,
        "pair": "%s/%s" % (a_name, b_name),
        "n_pce_weeks": n_pce,
        "n_weeks": int(len(weekly)),
        "trades": trades,
        **stats,
        "exposure_pct": round(100.0 * float(invested.mean()), 2) if len(invested) else 0.0,
        "cagr_pct": round(100.0 * strat_stats.cagr, 2),
        "sharpe": round(strat_stats.sharpe, 3),
        "mdd_daily_pct": round(100.0 * abs(strat_stats.max_drawdown), 2),
        "spy_cagr_pct": round(100.0 * spy_stats.cagr, 2),
        "spy_sharpe": round(spy_stats.sharpe, 3),
        "spy_mdd_pct": round(100.0 * abs(spy_stats.max_drawdown), 2),
        "gate": gate,
        "start": strat_stats.start,
        "end": strat_stats.end,
        "source": SOURCE_URL,
        "notes": PAIR_NOTE,
    }


def _multi_leg_daily_equity(
    frames: Dict[str, pd.DataFrame],
    trades: List[Dict[str, Any]],
    *,
    spy_daily: pd.DataFrame,
    fill_mode: str,
    friction_pct: float,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Cash when flat; long the traded ETF while a trade is open. SPY path for benchmark."""
    spy = _ensure_ts(spy_daily).set_index("ts").sort_index()
    idx = spy.index
    spy_close = spy["close"].astype(float)
    spy_rets = spy_close.pct_change().fillna(0.0)
    spy_eq = equity_from_returns(spy_rets)

    # Union index across legs so QQQ-only stretches still mark-to-market.
    all_idx = idx
    for df in frames.values():
        di = _ensure_ts(df).set_index("ts").sort_index().index
        all_idx = all_idx.union(di)
    all_idx = all_idx.sort_values()

    invested = pd.Series(0.0, index=all_idx)
    strat_rets = pd.Series(0.0, index=all_idx)
    fric = float(friction_pct) / 100.0

    for t in trades:
        sym = str(t["symbol"])
        d = _ensure_ts(frames[sym]).set_index("ts").sort_index()
        close = d["close"].astype(float)
        open_ = d["open"].astype(float) if "open" in d.columns else close
        rets = close.pct_change().fillna(0.0)
        a_date = pd.Timestamp(t["entry_date"]).date()
        b_date = pd.Timestamp(t["exit_date"]).date()
        dates = pd.Series([x.date() for x in d.index], index=d.index)

        if fill_mode == "claim_close":
            mask = (dates > a_date) & (dates <= b_date)
            invested.loc[d.index[mask]] = 1.0
            strat_rets.loc[d.index[mask]] = rets.loc[mask]
        else:
            mask = (dates >= a_date) & (dates <= b_date)
            invested.loc[d.index[mask]] = 1.0
            entry_rows = d.index[dates == a_date]
            rest = (dates > a_date) & (dates <= b_date)
            if len(entry_rows):
                e = entry_rows[0]
                if float(open_.loc[e]) > 0:
                    strat_rets.loc[e] = float(close.loc[e] / open_.loc[e] - 1.0)
            strat_rets.loc[d.index[rest]] = rets.loc[rest]

        if fric > 0:
            entry_rows = d.index[dates == a_date]
            exit_rows = d.index[dates == b_date]
            if len(entry_rows):
                strat_rets.loc[entry_rows[0]] = float(strat_rets.loc[entry_rows[0]]) - fric
            if len(exit_rows):
                strat_rets.loc[exit_rows[0]] = float(strat_rets.loc[exit_rows[0]]) - fric

    # Align to SPY calendar for comparable Sharpe/MDD.
    strat_rets = strat_rets.reindex(idx).fillna(0.0)
    invested = invested.reindex(idx).fillna(0.0)
    eq = equity_from_returns(strat_rets)
    eq.index = pd.DatetimeIndex(eq.index).tz_localize(None)
    invested.index = pd.DatetimeIndex(invested.index).tz_localize(None)
    spy_eq.index = pd.DatetimeIndex(spy_eq.index).tz_localize(None)
    return eq, invested, spy_eq


def run_pce_weekly_suite(
    a_daily: pd.DataFrame,
    b_daily: pd.DataFrame,
    *,
    a_name: str = "QQQ",
    b_name: str = "SPY",
    friction_pct: float = 0.10,
) -> List[Dict[str, Any]]:
    specs = [
        ("long_green_claim_close", False, 0.0),
        ("long_green_next_open", False, 0.0),
        ("long_green_next_open", True, 0.0),
        ("long_green_next_open", False, friction_pct),
        ("long_green_next_open", True, friction_pct),
    ]
    out = []
    for book, sweep, fric in specs:
        out.append(
            backtest_pce_weekly(
                a_daily,
                b_daily,
                a_name=a_name,
                b_name=b_name,
                book=book,
                friction_pct=fric,
                require_failed_sweep=sweep,
            )
        )
    return out
