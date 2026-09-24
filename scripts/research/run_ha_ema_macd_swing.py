#!/usr/bin/env python3
"""Claim-check Animesh/Dhan HA + 55 EMA channel + MACD swing.

Daily long-only translation. Signal on completed HA close; fill next open.
Transcript for https://www.youtube.com/watch?v=2YBmiyVmNNw was unavailable;
rules follow the published summary (not a parameter sweep).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_ha_ema_macd_swing.py --examples-only
  python scripts\\research\\run_ha_ema_macd_swing.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    apply_friction,
    summarize_by_year,
)
from utils.research.choppy_regime import occupancy_ranked  # noqa: E402
from utils.research.ha_ema_macd_swing import (  # noqa: E402
    DEFAULT_EMA_SPAN,
    DEFAULT_RR,
    collect_ha_ema_macd_trades,
    max_concurrent,
    occupancy_equity,
    resample_weekly_ohlc,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
)
from utils.research.metrics import annual_returns, perf_stats  # noqa: E402
from utils.research.panel import list_daily_symbols, load_spy_close, load_wide_panels  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("ha_ema_macd_swing")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

EVAL_START = "2018-01-01"
EVAL_END = "2026-09-18"
PANEL_START = "2017-01-03"
FRICTION_PCTS = (0.10, 0.25)
FIELDS = ("open", "high", "low", "close", "volume")
EXAMPLE_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")


def _spy_buy_hold(spy_close: pd.Series, start: str, end: str):
    s = spy_close.copy()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s = s[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))].dropna()
    eq = s / float(s.iloc[0])
    inv = pd.Series(1.0, index=eq.index)
    return eq, inv


def _print_book(label: str, trades: pd.DataFrame, years: Optional[pd.DataFrame] = None) -> dict:
    g = pd.to_numeric(trades["gain_pct"], errors="coerce") if not trades.empty else pd.Series(dtype=float)
    s = book_stats(g)
    geo = geom_mean_year_pf(years) if years is not None else None
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s.get("win_rate_pct"),
        s.get("median_pct"),
    )
    return s


def _drop_top_n(trades: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if trades.empty or len(trades) <= n:
        return trades.iloc[0:0].copy()
    g = pd.to_numeric(trades["gain_pct"], errors="coerce")
    drop = set(g.nlargest(int(n)).index)
    return trades.loc[~trades.index.isin(drop)].copy()


def _calendar_slice(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades.loc[pd.to_datetime(trades["buy_date"]).dt.year == int(year)]


def _book_block(label: str, trades: pd.DataFrame, outdir: Path, stem: str) -> dict:
    years = summarize_by_year(trades, gain_col="gain_pct", buckets=YEAR_BUCKETS) if not trades.empty else pd.DataFrame()
    stats = _print_book(label, trades, years)
    if not trades.empty:
        trades.to_csv(outdir / ("%s_trades.csv" % stem), index=False)
        LOG.info("%s exits: %s", label, trades["exit_reason"].value_counts().to_dict())
        LOG.info(
            "%s hold_bars median=%.1f mean=%.1f max_concurrent=%s",
            label,
            float(trades["hold_bars"].median()),
            float(trades["hold_bars"].mean()),
            max_concurrent(trades),
        )
        LOG.info(
            "%s examples:\n%s",
            label,
            trades.head(12)[
                [
                    "symbol",
                    "signal_date",
                    "buy_date",
                    "sell_date",
                    "buy_price",
                    "sell_price",
                    "gain_pct",
                    "exit_reason",
                ]
            ].to_string(index=False),
        )
    if not years.empty:
        years.to_csv(outdir / ("%s_years.csv" % stem), index=False)
        LOG.info("%s year buckets:\n%s", label, years.to_string(index=False))
    if not trades.empty and len(trades) >= 4:
        dropped = _drop_top_n(trades, 3)
        d_years = summarize_by_year(dropped, gain_col="gain_pct", buckets=YEAR_BUCKETS)
        _print_book(label + " drop-top-3", dropped, d_years)
    if not trades.empty:
        for fr in FRICTION_PCTS:
            net = apply_friction(trades, fr)
            n_years = summarize_by_year(net, gain_col="gain_pct_net", buckets=YEAR_BUCKETS)
            g = pd.to_numeric(net["gain_pct_net"], errors="coerce")
            ns = book_stats(g)
            geo = geom_mean_year_pf(n_years)
            LOG.info(
                "%s fric%.2f n=%s E=%s PF=%s geo=%s",
                label,
                fr,
                ns.get("n"),
                ns.get("expectancy_pct"),
                ns.get("profit_factor"),
                None if geo is None else round(geo, 3),
            )
    for year in (2022, 2023):
        part = _calendar_slice(trades, year)
        if part is None or part.empty:
            LOG.info("%s calendar %s n=0", label, year)
            continue
        ydf = summarize_by_year(
            part,
            gain_col="gain_pct",
            buckets=((str(year), "%s-01-01" % year, "%s-12-31" % year),),
        )
        _print_book("%s calendar %s" % (label, year), part, ydf)
    stats["years"] = years
    return stats


def _collect(args, panels, eval_start, eval_end, **overrides):
    kw = dict(
        eval_start=eval_start,
        eval_end=eval_end,
        ema_span=args.ema_span,
        rr=args.rr,
        friday_exit=not args.no_friday_exit,
        require_zero_cross=args.require_zero_cross,
        require_color_flip=not args.no_color_flip,
        liquid_n=args.liquid_n,
        min_price=args.min_price,
        min_adv=args.min_adv,
        cost_bps_rt=0.0,
    )
    kw.update(overrides)
    return collect_ha_ema_macd_trades(
        panels["open"],
        panels["high"],
        panels["low"],
        panels["close"],
        panels["volume"],
        **kw,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="HA + 55 EMA + MACD swing claim-check")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--panel-start", default=PANEL_START)
    ap.add_argument("--n-symbols", type=int, default=0, help="0 = full ALPACA 1d universe")
    ap.add_argument("--symbols", default="", help="Comma list; overrides universe if set")
    ap.add_argument("--examples-only", action="store_true", help="SPY/QQQ/AAPL/MSFT/NVDA only")
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--max-pos", type=int, default=15)
    ap.add_argument("--ema-span", type=int, default=DEFAULT_EMA_SPAN)
    ap.add_argument("--rr", type=float, default=DEFAULT_RR)
    ap.add_argument("--no-friday-exit", action="store_true")
    ap.add_argument("--require-zero-cross", action="store_true")
    ap.add_argument("--no-color-flip", action="store_true")
    ap.add_argument("--weekly", action="store_true", help="Also run Friday-weekly resample")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "research" / "ha_ema_macd_swing")
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.examples_only:
        symbols = list(EXAMPLE_SYMBOLS)
        args.liquid_n = max(args.liquid_n, 10)
        args.min_price = min(args.min_price, 1.0)
        args.min_adv = min(args.min_adv, 1.0)
    else:
        symbols = list_daily_symbols("ALPACA", "1d")
        if args.n_symbols and args.n_symbols > 0:
            symbols = symbols[: int(args.n_symbols)]
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("symbols: %d %s", len(symbols), symbols[:8])
    if len(symbols) < 1:
        LOG.error("No symbols")
        return 1

    t0 = time.perf_counter()
    panels = load_wide_panels(
        symbols,
        start=datetime.strptime(args.panel_start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        provider="ALPACA",
        timeframe="1d",
        fields=FIELDS,
        workers=args.workers,
    )
    timings["load_panel"] = time.perf_counter() - t0
    close = panels["close"]
    LOG.info("Panel close shape=%s", close.shape)

    t0 = time.perf_counter()
    spy = load_spy_close(
        datetime.strptime(args.panel_start, "%Y-%m-%d"),
        datetime.strptime(args.end, "%Y-%m-%d"),
    )
    timings["load_spy"] = time.perf_counter() - t0
    spy_eq, spy_inv = _spy_buy_hold(spy, args.start, args.end)
    spy_st = perf_stats("SPY_buy_hold", spy_eq, spy_inv, notes="IB/ALPACA SPY")
    LOG.info(
        "SPY tot=%.2f%% Sharpe=%.2f MDD=%.2f%%",
        spy_st.total_return * 100,
        spy_st.sharpe,
        spy_st.max_drawdown * 100,
    )

    outdir = dated_outdir(args.outdir)

    t0 = time.perf_counter()
    trades = _collect(args, panels, args.start, args.end)
    timings["collect_daily"] = time.perf_counter() - t0
    _book_block("HA-EMA-MACD daily unconstrained", trades, outdir, "daily")

    occ = occupancy_ranked(trades, max_open=args.max_pos, score_ascending=False)
    _book_block("HA-EMA-MACD daily occ%s" % args.max_pos, occ, outdir, "daily_occ")
    if not occ.empty:
        eq, inv, notes = occupancy_equity(
            occ,
            close,
            max_open=args.max_pos,
            eval_start=args.start,
            eval_end=args.end,
        )
        if not eq.empty:
            st = perf_stats("HA_EMA_MACD_occ", eq, inv, notes=notes)
            LOG.info(
                "occ%s vs SPY: tot=%.2f%% Sharpe=%.2f MDD=%.2f%% invested=%.1f%% (SPY Sharpe=%.2f MDD=%.2f%%)",
                args.max_pos,
                st.total_return * 100,
                st.sharpe,
                st.max_drawdown * 100,
                st.pct_time_invested * 100,
                spy_st.sharpe,
                spy_st.max_drawdown * 100,
            )
            eq.to_csv(outdir / "daily_occ_equity.csv", header=["equity"])
            LOG.info("SPY annual %s", {str(k): round(float(v) * 100, 2) for k, v in annual_returns(spy_eq).items()})
            LOG.info(
                "occ annual %s",
                {str(k): round(float(v) * 100, 2) for k, v in annual_returns(eq).items()},
            )

    # Video-named variants only (not a grid search).
    if not args.examples_only:
        t0 = time.perf_counter()
        t4 = _collect(args, panels, args.start, args.end, rr=4.0)
        timings["collect_rr4"] = time.perf_counter() - t0
        _book_block("HA-EMA-MACD daily rr4 unconstrained", t4, outdir, "daily_rr4")

        t0 = time.perf_counter()
        no_fri = _collect(args, panels, args.start, args.end, friday_exit=False)
        timings["collect_no_friday"] = time.perf_counter() - t0
        _book_block("HA-EMA-MACD daily no-friday unconstrained", no_fri, outdir, "daily_no_friday")

        t0 = time.perf_counter()
        zc = _collect(args, panels, args.start, args.end, require_zero_cross=True)
        timings["collect_zero_cross"] = time.perf_counter() - t0
        _book_block("HA-EMA-MACD daily MACD-zero-cross unconstrained", zc, outdir, "daily_zero_cross")

    if args.weekly:
        t0 = time.perf_counter()
        weekly = resample_weekly_ohlc(
            panels["open"], panels["high"], panels["low"], close, panels["volume"]
        )
        w_trades = _collect(args, weekly, args.start, args.end, friday_exit=False)
        timings["collect_weekly"] = time.perf_counter() - t0
        _book_block("HA-EMA-MACD weekly unconstrained", w_trades, outdir, "weekly")

    timings["total"] = time.perf_counter() - t_all
    LOG.info("Wrote %s (total %.1fs)", outdir, timings["total"])
    LOG.info("timings=%s", {k: round(v, 1) for k, v in timings.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
