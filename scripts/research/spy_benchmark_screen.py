#!/usr/bin/env python3
"""
Edge hunt: compare price-only strategies vs SPY buy-and-hold.

Candidates:
  1) SPY buy-and-hold (benchmark)
  2) Absolute momentum: long SPY iff close > SMA200 (month-end signal, next open)
  3) Cross-sectional 12-1 momentum: equal-weight top N among SPX ∩ ALPACA daily
  4) Weekly BigVol portfolio: setups CSV + daily prices, stop + 10w MA exit

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\spy_benchmark_screen.py
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\spy_benchmark_screen.py --candidates bh abs xs
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data.ohlcv_loader import load_ohlcv_many
from utils.data.ticker_universe import get_cached_combined_universe, get_sp500_tickers
from utils.db.timescaledb_client import get_timescaledb_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("spy_benchmark_screen")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

# Match plan window; SPY ALPACA IEX currently starts ~2018-11-01
DEFAULT_START = "2018-11-01"
DEFAULT_END = "2025-11-26"
TRADING_DAYS = 252
DEFAULT_BIGVOL_SETUPS = (
    ROOT / "reports" / "examples" / "universe_full"
    / "weekly_bigvol_full_setups_20260822_104401.csv"
)


@dataclass
class PerfStats:
    name: str
    start: str
    end: str
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    pct_time_invested: float
    excess_cagr_vs_spy: float
    excess_sharpe_vs_spy: float
    n_obs: int
    notes: str = ""


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{seconds:.1f}s"


def _to_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "ts" in out.columns:
            out = out.set_index("ts")
        elif "datetime" in out.columns:
            out = out.set_index("datetime")
        else:
            out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    return out.sort_index()


def _clip_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return df.loc[(df.index >= s) & (df.index <= e)]


def equity_from_returns(returns: pd.Series, start_equity: float = 1.0) -> pd.Series:
    r = returns.fillna(0.0)
    return (1.0 + r).cumprod() * start_equity


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def perf_stats(
    name: str,
    equity: pd.Series,
    invested: Optional[pd.Series] = None,
    spy_cagr: Optional[float] = None,
    spy_sharpe: Optional[float] = None,
    notes: str = "",
) -> PerfStats:
    eq = equity.dropna()
    if len(eq) < 2:
        return PerfStats(
            name=name,
            start="",
            end="",
            total_return=0.0,
            cagr=0.0,
            ann_vol=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            calmar=0.0,
            pct_time_invested=0.0,
            excess_cagr_vs_spy=0.0,
            excess_sharpe_vs_spy=0.0,
            n_obs=0,
            notes=notes or "insufficient data",
        )

    rets = eq.pct_change().dropna()
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if eq.iloc[0] > 0 else 0.0
    ann_vol = float(rets.std(ddof=0) * math.sqrt(TRADING_DAYS)) if len(rets) else 0.0
    sharpe = float(cagr / ann_vol) if ann_vol > 1e-12 else 0.0
    mdd = max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if abs(mdd) > 1e-12 else 0.0
    if invested is None:
        pct_inv = 1.0
    else:
        pct_inv = float(invested.reindex(eq.index).fillna(0.0).mean())

    return PerfStats(
        name=name,
        start=str(eq.index[0].date()),
        end=str(eq.index[-1].date()),
        total_return=total_ret,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        max_drawdown=mdd,
        calmar=calmar,
        pct_time_invested=pct_inv,
        excess_cagr_vs_spy=(cagr - spy_cagr) if spy_cagr is not None else 0.0,
        excess_sharpe_vs_spy=(sharpe - spy_sharpe) if spy_sharpe is not None else 0.0,
        n_obs=int(len(eq)),
        notes=notes,
    )


def load_spy(start: str, end: str, provider: str = "ALPACA") -> pd.DataFrame:
    data = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider=provider,
        start=datetime.strptime(start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        use_cache=True,
        workers=1,
    )
    if "SPY" not in data or data["SPY"].empty:
        raise RuntimeError("SPY daily not found in TimescaleDB. Fetch with utils/data/fetch_data.py")
    return _clip_dates(_to_naive_index(data["SPY"]), start, end)


def strategy_buy_hold(spy: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    close = spy["close"].astype(float)
    equity = close / close.iloc[0]
    invested = pd.Series(1.0, index=equity.index)
    return equity, invested


def strategy_abs_momentum_sma200(
    spy: pd.DataFrame,
    sma_period: int = 200,
) -> Tuple[pd.Series, pd.Series]:
    """
    Month-end signal: long iff close > SMA200; execute next trading day on open return path.
    Cash earns 0%.
    """
    df = spy[["open", "close"]].astype(float).copy()
    df["sma"] = df["close"].rolling(sma_period).mean()
    # Month-end bars
    month_end = df.groupby(df.index.to_period("M")).tail(1)
    signal = (month_end["close"] > month_end["sma"]).astype(float)
    # Map signal to all days: hold previous month-end decision until next
    daily_signal = signal.reindex(df.index, method="ffill").fillna(0.0)
    # Avoid look-ahead: use prior month-end signal for today's return
    position = daily_signal.shift(1).fillna(0.0)

    open_ret = df["open"].pct_change().fillna(0.0)
    # Approximate: invested days earn close-to-close; simpler and standard for timing tests
    close_ret = df["close"].pct_change().fillna(0.0)
    strat_ret = position * close_ret
    equity = equity_from_returns(strat_ret)
    return equity, position


def _month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(1, index=index)
    return s.groupby(s.index.to_period("M")).tail(1).index


def _fetch_sp500_symbols_fallback() -> List[str]:
    """Wikipedia often 403s; try public CSV mirrors then local ticker cache."""
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            col = "Symbol" if "Symbol" in df.columns else ("symbol" if "symbol" in df.columns else None)
            if col is None:
                continue
            syms = [str(s).upper().replace(".", "-") for s in df[col].tolist()]
            if len(syms) >= 100:
                logger.info("Loaded %d SPX tickers from %s", len(syms), url)
                return syms
        except Exception as exc:
            logger.warning("SPX CSV fetch failed (%s): %s", url, exc)
    cached = get_cached_combined_universe() or []
    if cached:
        logger.warning("Using cached combined universe (%d) as SPX proxy", len(cached))
        return [s.upper() for s in cached]
    return []


def _available_symbols(provider: str, timeframe: str, retries: int = 5) -> List[str]:
    client = get_timescaledb_client()
    for i in range(retries):
        if not client.ensure_connection():
            time.sleep(1.0 + i)
            continue
        syms = client.get_available_symbols(provider=provider, timeframe=timeframe) or []
        if syms:
            return [s.upper() for s in syms]
        time.sleep(1.0 + i)
    return []


def strategy_cross_sectional_momentum(
    start: str,
    end: str,
    top_n: int = 20,
    lookback_months: int = 12,
    skip_months: int = 1,
    cost_bps_rt: float = 10.0,
    provider: str = "ALPACA",
    workers: int = 4,
) -> Tuple[pd.Series, pd.Series, pd.Series, str]:
    """
    12-1 momentum, monthly rebalance, equal-weight top_n.
    Returns gross equity, net equity, invested fraction series, notes.
    """
    available = set(_available_symbols(provider, "1d"))
    spx = [s.upper() for s in get_sp500_tickers()]
    if len(spx) < 50:
        spx = _fetch_sp500_symbols_fallback()
    universe = sorted(set(spx) & available)
    # If SPX list still empty but DB is up, fall back to all available names (broader, noisier)
    if len(universe) < top_n + 5 and available:
        universe = sorted(available)
        univ_note = f"ALL_{provider}_1d"
    else:
        univ_note = f"SPX_intersect_{provider}_1d"
    # Warmup for 12m lookback
    load_start = (pd.Timestamp(start) - pd.DateOffset(months=lookback_months + skip_months + 2)).strftime(
        "%Y-%m-%d"
    )
    notes = (
        f"universe={len(universe)} ({univ_note}); "
        f"survivorship bias (current SPX list); top_n={top_n}; cost={cost_bps_rt:.0f}bps RT"
    )
    if len(universe) < top_n + 5:
        raise RuntimeError(f"Universe too small for XS momentum: {len(universe)}")

    logger.info("Loading %d symbols for XS momentum (workers=%d)...", len(universe), workers)
    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        universe,
        timeframe="1d",
        provider=provider,
        start=datetime.strptime(load_start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        use_cache=True,
        workers=workers,
        chunk_size=50,
    )
    logger.info("Loaded %d/%d symbols in %s", len(panels), len(universe), _format_elapsed(time.perf_counter() - t0))

    closes = {}
    for sym, df in panels.items():
        d = _to_naive_index(df)
        if "close" not in d.columns or d.empty:
            continue
        closes[sym] = d["close"].astype(float)
    price = pd.DataFrame(closes).sort_index()
    price = price.loc[(price.index >= pd.Timestamp(load_start)) & (price.index <= pd.Timestamp(end))]
    daily_ret = price.pct_change(fill_method=None)

    me_idx = _month_ends(price.index)
    # Formation: return from t-(12) to t-(1) month ends
    # Use trading-day approx: 21 days/month
    skip_days = skip_months * 21
    form_days = lookback_months * 21

    holdings: Dict[pd.Timestamp, List[str]] = {}
    for dt in me_idx:
        loc = price.index.get_loc(dt)
        if isinstance(loc, slice):
            continue
        if loc < form_days + skip_days:
            continue
        end_px = price.iloc[loc - skip_days]
        start_px = price.iloc[loc - form_days - skip_days]
        mom = (end_px / start_px) - 1.0
        mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
        if len(mom) < top_n:
            continue
        picks = mom.nlargest(top_n).index.tolist()
        holdings[dt] = picks

    # Build daily weights: after month-end signal, hold until next month-end (shift 1 day)
    all_days = price.loc[start:end].index
    weight = pd.DataFrame(0.0, index=all_days, columns=price.columns)
    me_sorted = sorted(holdings.keys())
    for i, dt in enumerate(me_sorted):
        start_hold = all_days[all_days > dt]
        if start_hold.empty:
            continue
        start_d = start_hold[0]
        if i + 1 < len(me_sorted):
            end_hold = all_days[(all_days >= start_d) & (all_days <= me_sorted[i + 1])]
        else:
            end_hold = all_days[all_days >= start_d]
        picks = holdings[dt]
        w = 1.0 / len(picks)
        for sym in picks:
            if sym in weight.columns:
                weight.loc[end_hold, sym] = w

    # Lag weights by 1 day already started after signal date; align returns
    w_use = weight.shift(1).fillna(0.0)
    r = daily_ret.reindex(w_use.index).fillna(0.0)
    port_ret_gross = (w_use * r).sum(axis=1)

    # Turnover cost: 0.5 * L1 weight change * one-way; RT bps applied on turnover
    w_change = w_use.diff().abs().sum(axis=1).fillna(0.0)
    # Full turnover (sum abs) / 2 = fraction of book traded one-way; RT cost on that
    cost = (w_change / 2.0) * (cost_bps_rt / 10000.0)
    port_ret_net = port_ret_gross - cost

    invested = (w_use.sum(axis=1) > 0).astype(float)
    eq_gross = equity_from_returns(port_ret_gross)
    eq_net = equity_from_returns(port_ret_net)
    notes += f"; rebalances={len(holdings)}"
    return eq_gross, eq_net, invested, notes


def _weekly_ma(close_daily: pd.Series, period: int = 10) -> pd.Series:
    weekly = close_daily.resample("W-FRI").last().dropna()
    ma = weekly.rolling(period).mean()
    return ma.reindex(close_daily.index, method="ffill")


def strategy_bigvol_portfolio(
    setups_csv: Path,
    start: str,
    end: str,
    alloc_frac: float = 0.08,
    max_positions: int = 12,
    stop_loss_pct: float = 0.15,
    split_filter: float = 0.45,
    provider: str = "ALPACA",
    workers: int = 4,
) -> Tuple[pd.Series, pd.Series, str]:
    """
    Portfolio simulation from Weekly BigVol confirms.
    Entry: next daily open after confirm_week.
    Exit: stop -15% from entry OR close < weekly MA10 (checked daily on close).
    Position size: alloc_frac of equity at entry, capped at max_positions.
    """
    raw = pd.read_csv(setups_csv)
    if raw.empty:
        raise RuntimeError(f"Empty setups CSV: {setups_csv}")

    dedup = raw.drop_duplicates(subset=["symbol", "confirm_week"], keep="first").copy()
    dedup["confirm_week"] = pd.to_datetime(dedup["confirm_week"])
    if "fwd_4w" in dedup.columns:
        dedup = dedup[dedup["fwd_4w"].abs() < split_filter]
    dedup = dedup[
        (dedup["confirm_week"] >= pd.Timestamp(start) - pd.Timedelta(days=14))
        & (dedup["confirm_week"] <= pd.Timestamp(end))
    ]
    symbols = sorted(dedup["symbol"].astype(str).str.upper().unique().tolist())
    notes = (
        f"setups={len(dedup)} symbols={len(symbols)} alloc={alloc_frac:.0%} "
        f"max_pos={max_positions} stop={stop_loss_pct:.0%} + MA10 exit; "
        f"source={setups_csv.name}"
    )
    if not symbols:
        raise RuntimeError("No BigVol setups in window")

    load_start = (pd.Timestamp(start) - pd.DateOffset(months=10)).strftime("%Y-%m-%d")
    logger.info("Loading %d BigVol symbols...", len(symbols))
    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider=provider,
        start=datetime.strptime(load_start, "%Y-%m-%d"),
        end=datetime.strptime(end, "%Y-%m-%d"),
        use_cache=True,
        workers=workers,
        chunk_size=50,
    )
    logger.info("Loaded %d/%d in %s", len(panels), len(symbols), _format_elapsed(time.perf_counter() - t0))

    ohlc: Dict[str, pd.DataFrame] = {}
    for sym, df in panels.items():
        d = _clip_dates(_to_naive_index(df), load_start, end)
        if len(d) < 50:
            continue
        ohlc[sym] = d

    # Calendar from union of closes
    all_idx = sorted(set().union(*[set(df.index) for df in ohlc.values()]))
    calendar = pd.DatetimeIndex(all_idx)
    calendar = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]

    # Precompute MA10 weekly for each symbol
    ma10 = {sym: _weekly_ma(df["close"].astype(float), 10) for sym, df in ohlc.items()}

    # Entry schedule: confirm_week -> next session
    entries: List[Tuple[pd.Timestamp, str]] = []
    for _, row in dedup.iterrows():
        sym = str(row["symbol"]).upper()
        if sym not in ohlc:
            continue
        cw = pd.Timestamp(row["confirm_week"])
        idx = ohlc[sym].index
        future = idx[idx > cw]
        if future.empty:
            continue
        entries.append((future[0], sym))
    entries.sort(key=lambda x: x[0])

    cash = 1.0
    # positions: sym -> dict entry_px, shares, entry_date
    positions: Dict[str, dict] = {}
    equity_points = []
    invested_points = []
    entry_i = 0
    n_trades = 0
    n_stops = 0
    n_ma_exits = 0

    for dt in calendar:
        # Exits on close
        to_close = []
        for sym, pos in positions.items():
            df = ohlc[sym]
            if dt not in df.index:
                continue
            px = float(df.loc[dt, "close"])
            stop_px = pos["entry_px"] * (1.0 - stop_loss_pct)
            ma = ma10[sym]
            ma_v = float(ma.loc[dt]) if dt in ma.index and pd.notna(ma.loc[dt]) else None
            if px <= stop_px:
                to_close.append((sym, px, "stop"))
            elif ma_v is not None and px < ma_v and dt > pos["entry_date"]:
                to_close.append((sym, px, "ma10"))
        for sym, px, reason in to_close:
            pos = positions.pop(sym)
            cash += pos["shares"] * px
            n_trades += 1
            if reason == "stop":
                n_stops += 1
            else:
                n_ma_exits += 1

        # Mark-to-market equity before new entries
        mtm = cash
        for sym, pos in positions.items():
            df = ohlc[sym]
            if dt in df.index:
                mtm += pos["shares"] * float(df.loc[dt, "close"])
            else:
                mtm += pos["shares"] * pos["last_px"]
            if dt in df.index:
                pos["last_px"] = float(df.loc[dt, "close"])

        # Entries at open (use open if available else close)
        while entry_i < len(entries) and entries[entry_i][0] == dt:
            _, sym = entries[entry_i]
            entry_i += 1
            if sym in positions:
                continue
            if len(positions) >= max_positions:
                continue
            df = ohlc[sym]
            if dt not in df.index:
                continue
            entry_px = float(df.loc[dt, "open"] if pd.notna(df.loc[dt, "open"]) else df.loc[dt, "close"])
            if entry_px <= 0:
                continue
            # Recompute equity for sizing
            eq_now = cash
            for s2, p2 in positions.items():
                d2 = ohlc[s2]
                px2 = float(d2.loc[dt, "close"]) if dt in d2.index else p2["last_px"]
                eq_now += p2["shares"] * px2
            alloc = eq_now * alloc_frac
            if alloc <= 0 or cash < alloc * 0.5:
                continue
            spend = min(alloc, cash)
            shares = spend / entry_px
            cash -= spend
            positions[sym] = {
                "entry_px": entry_px,
                "shares": shares,
                "entry_date": dt,
                "last_px": entry_px,
            }

        # End-of-day equity
        eq = cash
        for sym, pos in positions.items():
            df = ohlc[sym]
            if dt in df.index:
                eq += pos["shares"] * float(df.loc[dt, "close"])
            else:
                eq += pos["shares"] * pos["last_px"]
        equity_points.append((dt, eq))
        invested_points.append((dt, 1.0 if positions else 0.0))

    # Liquidate remaining at last close
    if positions and equity_points:
        last_dt = equity_points[-1][0]
        for sym, pos in list(positions.items()):
            df = ohlc[sym]
            px = float(df.loc[last_dt, "close"]) if last_dt in df.index else pos["last_px"]
            cash += pos["shares"] * px
            n_trades += 1
        positions.clear()
        equity_points[-1] = (last_dt, cash)

    eq_s = pd.Series({d: v for d, v in equity_points}).sort_index()
    inv_s = pd.Series({d: v for d, v in invested_points}).sort_index()
    # Normalize to 1.0 start
    if not eq_s.empty and eq_s.iloc[0] != 0:
        eq_s = eq_s / eq_s.iloc[0]
    notes += f"; closed_trades~={n_trades} stops={n_stops} ma_exits={n_ma_exits}"
    return eq_s, inv_s, notes


def passes_gates(stats: PerfStats, spy: PerfStats, cagr_tol: float = 0.02) -> Tuple[bool, str]:
    """Plan gates: Sharpe >= SPY OR (MDD better and CAGR within ~2pp), preferably both."""
    sharpe_ok = stats.sharpe >= spy.sharpe - 1e-9
    mdd_ok = abs(stats.max_drawdown) + 1e-9 < abs(spy.max_drawdown)
    cagr_ok = stats.cagr + 1e-9 >= spy.cagr - cagr_tol
    cagr_beat = stats.cagr + 1e-9 >= spy.cagr
    both = sharpe_ok and mdd_ok and cagr_ok
    either = (sharpe_ok and cagr_ok) or (mdd_ok and cagr_ok) or (cagr_beat and sharpe_ok)
    if both:
        return True, "PASS (Sharpe+MDD+CAGR)"
    if either:
        return True, "PASS (partial gate)"
    return False, "FAIL"


def format_table(rows: Sequence[PerfStats], spy: PerfStats) -> str:
    headers = [
        "name", "CAGR", "Sharpe", "MaxDD", "Calmar", "Vol", "TotRet",
        "Invested%", "xsCAGR", "xsSharpe", "gate",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        ok, gate = passes_gates(r, spy) if r.name != spy.name else (True, "BENCHMARK")
        lines.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    f"{r.cagr:.2%}",
                    f"{r.sharpe:.2f}",
                    f"{r.max_drawdown:.2%}",
                    f"{r.calmar:.2f}",
                    f"{r.ann_vol:.2%}",
                    f"{r.total_return:.2%}",
                    f"{r.pct_time_invested:.0%}",
                    f"{r.excess_cagr_vs_spy:+.2%}",
                    f"{r.excess_sharpe_vs_spy:+.2f}",
                    gate,
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_status_log(
    out_dir: Path,
    spy_stats: PerfStats,
    all_stats: List[PerfStats],
    timings: Dict[str, float],
    spy_range_note: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d")
    path = out_dir / f"{ts}_edge_hunt_phase1_scorecard.md"
    winners = []
    for s in all_stats:
        if s.name == spy_stats.name:
            continue
        ok, gate = passes_gates(s, spy_stats)
        if ok:
            winners.append((s, gate))

    body = []
    body.append("# Edge hunt Phase 1 - scorecard vs SPY buy-and-hold")
    body.append("")
    body.append(f"Date: {ts}")
    body.append("")
    body.append("## Window / data")
    body.append("")
    body.append(f"- Evaluation window: `{spy_stats.start}` -> `{spy_stats.end}`")
    body.append(f"- {spy_range_note}")
    body.append("- Edge gate: Sharpe >= SPY and/or better MDD with CAGR within ~2pp of SPY (or higher CAGR).")
    body.append("- Cross-sectional results use current SPX membership (survivorship bias) - optimistic.")
    body.append("")
    body.append("## Scorecard")
    body.append("")
    body.append(format_table(all_stats, spy_stats))
    body.append("")
    body.append("## Notes per candidate")
    body.append("")
    for s in all_stats:
        body.append(f"- **{s.name}**: {s.notes or 'n/a'}")
    body.append("")
    body.append("## Timings")
    body.append("")
    for k, v in timings.items():
        body.append(f"- {k}: {_format_elapsed(v)}")
    body.append("")
    body.append("## Promotion")
    body.append("")
    if winners:
        winners_sorted = sorted(winners, key=lambda x: (x[0].sharpe, x[0].cagr), reverse=True)
        best, gate = winners_sorted[0]
        body.append(f"**Promoted (Phase 1):** `{best.name}` - {gate}")
        body.append("")
        body.append(
            f"Best Sharpe among passers: CAGR={best.cagr:.2%}, Sharpe={best.sharpe:.2f}, "
            f"MDD={best.max_drawdown:.2%} vs SPY CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} "
            f"MDD={spy_stats.max_drawdown:.2%}."
        )
        if len(winners) > 1:
            body.append("")
            body.append("Other passers: " + ", ".join(f"`{s.name}` ({g})" for s, g in winners_sorted[1:]))
    else:
        body.append("**No Phase-1 candidate cleared gates.**")
        body.append("")
        # Highlight near-miss risk-adjusted names
        near = [
            s for s in all_stats
            if s.name != spy_stats.name and abs(s.max_drawdown) < abs(spy_stats.max_drawdown) - 0.05
        ]
        if near:
            body.append("Near-miss (much better MDD, but CAGR too far below SPY):")
            for s in near:
                body.append(
                    f"- `{s.name}`: CAGR={s.cagr:.2%} Sharpe={s.sharpe:.2f} MDD={s.max_drawdown:.2%} "
                    f"(invested {s.pct_time_invested:.0%})"
                )
            body.append("")
        body.append(
            "Next data to add for Phase 2 dual momentum: daily **SHY** (or BIL) and **EFA**/**VXUS**, "
            "plus ideally longer SPY history pre-2018 if available."
        )
    body.append("")
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Score strategies vs SPY buy-and-hold")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=["bh", "abs", "xs", "bigvol"],
        choices=["bh", "abs", "xs", "bigvol"],
    )
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--xs-cost-bps", type=float, default=10.0)
    ap.add_argument("--bigvol-setups", type=Path, default=DEFAULT_BIGVOL_SETUPS)
    ap.add_argument("--bigvol-alloc", type=float, default=0.08)
    ap.add_argument("--bigvol-max-pos", type=int, default=12)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "docs" / "status_log" / "edge_hunt",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports" / "edge_hunt",
    )
    args = ap.parse_args()

    t_all = time.perf_counter()
    timings: Dict[str, float] = {}
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    spy = load_spy(args.start, args.end)
    timings["load_spy"] = time.perf_counter() - t0
    spy_range_note = (
        f"SPY ALPACA 1d bars used: {spy.index[0].date()} -> {spy.index[-1].date()} "
        f"(n={len(spy)}). Earlier than ~2018-11 may be unavailable on current Alpaca feed."
    )
    logger.info(spy_range_note)

    all_stats: List[PerfStats] = []
    curves: Dict[str, pd.Series] = {}

    # Buy and hold
    t0 = time.perf_counter()
    eq_bh, inv_bh = strategy_buy_hold(spy)
    spy_stats = perf_stats("SPY_buy_hold", eq_bh, inv_bh, notes="100% SPY")
    timings["bh"] = time.perf_counter() - t0
    all_stats.append(spy_stats)
    curves["SPY_buy_hold"] = eq_bh

    # Absolute momentum
    if "abs" in args.candidates:
        t0 = time.perf_counter()
        eq_abs, inv_abs = strategy_abs_momentum_sma200(spy)
        st = perf_stats(
            "SPY_SMA200_abs_mom",
            eq_abs,
            inv_abs,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes="Month-end close>SMA200; cash=0%; next-day position",
        )
        timings["abs"] = time.perf_counter() - t0
        all_stats.append(st)
        curves["SPY_SMA200_abs_mom"] = eq_abs

    # Cross-sectional
    if "xs" in args.candidates:
        t0 = time.perf_counter()
        eq_g, eq_n, inv_xs, notes = strategy_cross_sectional_momentum(
            args.start,
            args.end,
            top_n=args.top_n,
            cost_bps_rt=args.xs_cost_bps,
            workers=args.workers,
        )
        # Align to SPY calendar overlap
        eq_g = eq_g.reindex(eq_bh.index).ffill().bfill()
        eq_n = eq_n.reindex(eq_bh.index).ffill().bfill()
        inv_xs = inv_xs.reindex(eq_bh.index).fillna(0.0)
        st_g = perf_stats(
            "XS_mom_12_1_gross",
            eq_g,
            inv_xs,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=notes + " [gross]",
        )
        st_n = perf_stats(
            f"XS_mom_12_1_net_{int(args.xs_cost_bps)}bps",
            eq_n,
            inv_xs,
            spy_cagr=spy_stats.cagr,
            spy_sharpe=spy_stats.sharpe,
            notes=notes + " [net]",
        )
        timings["xs"] = time.perf_counter() - t0
        all_stats.extend([st_g, st_n])
        curves["XS_mom_12_1_gross"] = eq_g
        curves[f"XS_mom_12_1_net_{int(args.xs_cost_bps)}bps"] = eq_n

    # BigVol portfolio
    if "bigvol" in args.candidates:
        t0 = time.perf_counter()
        if not args.bigvol_setups.exists():
            logger.error("BigVol setups CSV missing: %s", args.bigvol_setups)
        else:
            eq_bv, inv_bv, notes = strategy_bigvol_portfolio(
                args.bigvol_setups,
                args.start,
                args.end,
                alloc_frac=args.bigvol_alloc,
                max_positions=args.bigvol_max_pos,
                workers=args.workers,
            )
            eq_bv = eq_bv.reindex(eq_bh.index)
            first = eq_bv.first_valid_index()
            if first is not None:
                eq_bv.loc[:first] = eq_bv.loc[first]
            eq_bv = eq_bv.ffill().fillna(1.0)
            inv_bv = inv_bv.reindex(eq_bh.index).fillna(0.0)
            st = perf_stats(
                "WeeklyBigVol_portfolio",
                eq_bv,
                inv_bv,
                spy_cagr=spy_stats.cagr,
                spy_sharpe=spy_stats.sharpe,
                notes=notes,
            )
            all_stats.append(st)
            curves["WeeklyBigVol_portfolio"] = eq_bv
        timings["bigvol"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_all

    # Persist curves + stats JSON first (before console prints that may fail on Windows)
    curves_df = pd.DataFrame(curves)
    curves_path = args.reports_dir / "equity_curves.csv"
    curves_df.to_csv(curves_path)
    stats_path = args.reports_dir / "scorecard.json"
    stats_path.write_text(json.dumps([asdict(s) for s in all_stats], indent=2), encoding="utf-8")
    log_path = write_status_log(args.outdir, spy_stats, all_stats, timings, spy_range_note)

    print("\n=== SPY benchmark ===")
    print(
        f"SPY B&H {spy_stats.start} -> {spy_stats.end}: "
        f"CAGR={spy_stats.cagr:.2%} Sharpe={spy_stats.sharpe:.2f} "
        f"MDD={spy_stats.max_drawdown:.2%} Tot={spy_stats.total_return:.2%}"
    )
    print("\n=== Scorecard ===")
    print(format_table(all_stats, spy_stats))
    for s in all_stats:
        note = (s.notes or "").encode("ascii", "replace").decode("ascii")
        print(f"  note[{s.name}]: {note}")

    print(f"\nWrote status log: {log_path}")
    print(f"Wrote curves: {curves_path}")
    print(f"Wrote stats: {stats_path}")
    print(f"Total elapsed: {_format_elapsed(timings['total'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
