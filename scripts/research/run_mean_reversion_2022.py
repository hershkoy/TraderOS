#!/usr/bin/env python3
"""2022 test of daily Bollinger mean reversion vs SPY and Phase 5 dump3.

Faithful to ``strategies/mean_reversion_strategy.py``: buy close <= lower BB
(20, 2), exit close >= SMA20. Signal on completed close; fill next open.
No stop in the strategy. Commission 0.1% per side (20 bps RT).

Also re-sims Phase 5 ``dump3_stock`` on the same 2022 window (3-day dump,
SMA20 exit / 8d / 10% stop, 10 bps RT) so the two MR books can be compared.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_mean_reversion_2022.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import summarize_by_year  # noqa: E402
from utils.research.last_15m_volume_geometry import book_stats  # noqa: E402
from utils.research.mean_reversion_bb import (  # noqa: E402
    DEFAULT_COST_BPS_RT,
    DEFAULT_DEVFACTOR,
    DEFAULT_PERIOD,
    collect_bb_trades,
    simulate_bb_occupancy,
)
from utils.research.metrics import annual_returns, perf_stats  # noqa: E402
from utils.research.panel import list_daily_symbols, load_spy_close, load_wide_panels  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.swing_mr import VARIANT_DUMP_STOCK, simulate_swing_mr  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("mean_reversion_2022")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

EVAL_START = "2022-01-01"
EVAL_END = "2022-12-31"
PANEL_START = "2021-08-02"
SPY_START = "2021-01-04"


def _spy_buy_hold(spy_close: pd.Series, start: str, end: str):
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv


def _align_to(eq: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    out = eq.reindex(idx)
    first = out.first_valid_index()
    if first is not None:
        out.loc[:first] = out.loc[first]
    return out.ffill().fillna(1.0)


def _print_book(label: str, trades: pd.DataFrame) -> dict:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    s = book_stats(g)
    LOG.info(
        "%s n=%s E=%s PF=%s WR=%s median=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_pct"),
    )
    return s


def _monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["month"] = pd.to_datetime(t["buy_date"]).dt.to_period("M").astype(str)
    rows = []
    for month, part in t.groupby("month", sort=True):
        s = book_stats(pd.to_numeric(part["gain_pct"], errors="coerce"))
        s["month"] = month
        rows.append(s)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="2022 Bollinger mean reversion vs SPY")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--panel-start", default=PANEL_START)
    ap.add_argument("--spy-start", default=SPY_START)
    ap.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    ap.add_argument("--devfactor", type=float, default=DEFAULT_DEVFACTOR)
    ap.add_argument("--max-pos", type=int, default=15)
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS_RT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "mean_reversion",
    )
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    symbols = list_daily_symbols("ALPACA", "1d")
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("ALPACA 1d symbols: %d", len(symbols))
    if len(symbols) < 50:
        LOG.error("Universe too small (%d)", len(symbols))
        return 1

    t0 = time.perf_counter()
    panels = load_wide_panels(
        symbols,
        start=datetime.strptime(args.panel_start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        provider="ALPACA",
        timeframe="1d",
        workers=args.workers,
    )
    timings["load_panel"] = time.perf_counter() - t0
    open_px, close, volume = panels["open"], panels["close"], panels["volume"]
    LOG.info("Panel close shape=%s", close.shape)

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.spy_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    timings["load_spy"] = time.perf_counter() - t0
    spy_eq, spy_inv = _spy_buy_hold(spy, args.start, args.end)
    spy_st = perf_stats("SPY_buy_hold_2022", spy_eq, spy_inv, notes="IB/ALPACA SPY 2022")

    common_liq = dict(
        liquid_n=args.liquid_n,
        min_price=args.min_price,
        min_adv=args.min_adv,
    )

    t0 = time.perf_counter()
    trades = collect_bb_trades(
        open_px,
        close,
        volume,
        period=args.period,
        devfactor=args.devfactor,
        eval_start=args.start,
        eval_end=args.end,
        cost_bps_rt=args.cost_bps,
        stop_loss_pct=0.0,
        **common_liq,
    )
    timings["collect_bb"] = time.perf_counter() - t0
    bb_book = _print_book("BB unconstrained", trades)
    if not trades.empty:
        LOG.info("BB exits: %s", trades["exit_reason"].value_counts().to_dict())
        LOG.info(
            "BB hold_bars median=%.1f mean=%.1f",
            float(trades["hold_bars"].median()),
            float(trades["hold_bars"].mean()),
        )

    t0 = time.perf_counter()
    eq, inv, occ_rets, exits, notes = simulate_bb_occupancy(
        open_px,
        close,
        volume,
        period=args.period,
        devfactor=args.devfactor,
        eval_start=args.start,
        eval_end=args.end,
        max_positions=args.max_pos,
        cost_bps_rt=args.cost_bps,
        stop_loss_pct=0.0,
        **common_liq,
    )
    timings["sim_bb_occ"] = time.perf_counter() - t0
    eq_a = _align_to(eq, spy_eq.index)
    inv_a = inv.reindex(spy_eq.index).fillna(0.0)
    bb_occ = perf_stats(
        "BB_occ15_2022",
        eq_a,
        inv_a,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes=notes,
        trade_rets=occ_rets,
    )
    LOG.info(
        "BB occ n=%d CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% E=%.4f PF=%.2f WR=%.0f%% (%s)",
        bb_occ.n_trades,
        bb_occ.cagr * 100,
        bb_occ.sharpe,
        bb_occ.max_drawdown * 100,
        bb_occ.expectancy,
        bb_occ.profit_factor,
        bb_occ.win_rate * 100,
        notes,
    )

    t0 = time.perf_counter()
    d_eq, d_inv, d_rets, d_exits, d_notes = simulate_swing_mr(
        open_px,
        close,
        panels["low"] if "low" in panels else close,
        volume,
        spy,
        variant=VARIANT_DUMP_STOCK,
        eval_start=args.start,
        eval_end=args.end,
        max_positions=args.max_pos,
        hold_days=8,
        stop_loss_pct=0.10,
        cost_bps_rt=10.0,
        **common_liq,
    )
    timings["sim_dump3"] = time.perf_counter() - t0
    d_eq_a = _align_to(d_eq, spy_eq.index)
    d_inv_a = d_inv.reindex(spy_eq.index).fillna(0.0)
    dump_st = perf_stats(
        "MR_dump3_stock_2022",
        d_eq_a,
        d_inv_a,
        spy_cagr=spy_st.cagr,
        spy_sharpe=spy_st.sharpe,
        notes=d_notes,
        trade_rets=d_rets,
    )
    LOG.info(
        "dump3 n=%d CAGR=%.2f%% Sharpe=%.2f MDD=%.2f%% E=%.4f PF=%.2f WR=%.0f%%",
        dump_st.n_trades,
        dump_st.cagr * 100,
        dump_st.sharpe,
        dump_st.max_drawdown * 100,
        dump_st.expectancy,
        dump_st.profit_factor,
        dump_st.win_rate * 100,
    )

    monthly = _monthly(trades)
    if not monthly.empty:
        LOG.info("BB monthly:\n%s", monthly.to_string(index=False))

    years = summarize_by_year(trades, gain_col="gain_pct") if not trades.empty else pd.DataFrame()
    spy_years = annual_returns(spy_eq)
    bb_years = annual_returns(eq_a)
    dump_years = annual_returns(d_eq_a)

    outdir = dated_outdir(args.outdir)
    trades_path = outdir / "mean_reversion_bb_2022_trades.csv"
    if not trades.empty:
        trades.to_csv(trades_path, index=False)
    if not monthly.empty:
        monthly.to_csv(outdir / "mean_reversion_bb_2022_monthly.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "name": spy_st.name,
                "n": spy_st.n_trades,
                "cagr": spy_st.cagr,
                "sharpe": spy_st.sharpe,
                "max_drawdown": spy_st.max_drawdown,
                "total_return": spy_st.total_return,
                "expectancy": spy_st.expectancy,
                "profit_factor": spy_st.profit_factor,
                "win_rate": spy_st.win_rate,
                "notes": spy_st.notes,
            },
            {
                "name": "BB_unconstrained_2022",
                "n": bb_book.get("n"),
                "cagr": None,
                "sharpe": None,
                "max_drawdown": None,
                "total_return": None,
                "expectancy": None if bb_book.get("expectancy_pct") is None else bb_book["expectancy_pct"] / 100.0,
                "profit_factor": bb_book.get("profit_factor"),
                "win_rate": None if bb_book.get("win_rate_pct") is None else bb_book["win_rate_pct"] / 100.0,
                "notes": "per-symbol sequential; gain_pct mean is E",
            },
            {
                "name": bb_occ.name,
                "n": bb_occ.n_trades,
                "cagr": bb_occ.cagr,
                "sharpe": bb_occ.sharpe,
                "max_drawdown": bb_occ.max_drawdown,
                "total_return": bb_occ.total_return,
                "expectancy": bb_occ.expectancy,
                "profit_factor": bb_occ.profit_factor,
                "win_rate": bb_occ.win_rate,
                "notes": bb_occ.notes,
            },
            {
                "name": dump_st.name,
                "n": dump_st.n_trades,
                "cagr": dump_st.cagr,
                "sharpe": dump_st.sharpe,
                "max_drawdown": dump_st.max_drawdown,
                "total_return": dump_st.total_return,
                "expectancy": dump_st.expectancy,
                "profit_factor": dump_st.profit_factor,
                "win_rate": dump_st.win_rate,
                "notes": dump_st.notes,
            },
        ]
    )
    summary.to_csv(outdir / "mean_reversion_2022_summary.csv", index=False)

    timings["total"] = time.perf_counter() - t_all
    LOG.info("Wrote %s (total %.1fs)", outdir, timings["total"])
    LOG.info("timings=%s", {k: round(v, 1) for k, v in timings.items()})
    LOG.info("SPY 2022 tot=%.2f%% Sharpe=%.2f MDD=%.2f%%", spy_st.total_return * 100, spy_st.sharpe, spy_st.max_drawdown * 100)
    LOG.info("SPY annual %s", {str(k): round(float(v) * 100, 2) for k, v in spy_years.items()})
    LOG.info("BB occ annual %s", {str(k): round(float(v) * 100, 2) for k, v in bb_years.items()})
    LOG.info("dump3 annual %s", {str(k): round(float(v) * 100, 2) for k, v in dump_years.items()})
    if not years.empty:
        LOG.info("BB year buckets:\n%s", years.to_string(index=False))
    LOG.info("dump3 exits=%s", d_exits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
