#!/usr/bin/env python3
"""Backtest choppy-regime RSI-2 MOC and 3-day pullback vs SPY.

2026-09-13 confirm: RSI cross-back (arm <5, buy reclaim 5 or 10) and pullback
min_close_loc 0.7. SMA50 / no VIX sit-out / 0.10% friction stay. Not the
threshold refine book and not Phase 5 rsi2_stock.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\run_choppy_regime.py --confirm-only --outdir reports\\choppy_regime\\confirm
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
from utils.research.choppy_regime import (  # noqa: E402
    DEFAULT_ATR_MULT,
    DEFAULT_PULLBACK_TIME_STOP,
    DEFAULT_RSI_ENTRY,
    DEFAULT_RSI_EXIT,
    DEFAULT_SMA_TREND_RSI,
    DEFAULT_TIME_STOP,
    collect_pullback_trades,
    collect_rsi2_moc_trades,
    occupancy_ranked,
    sized_gain_pct,
)
from utils.research.cta_trend import load_macro_closes  # noqa: E402
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
LOG = logging.getLogger("choppy_regime")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

EVAL_START = "2018-01-01"
EVAL_END = "2026-08-21"
PANEL_START = "2017-01-03"
FRICTION_PCTS = (0.10, 0.05)
FIELDS = ("open", "high", "low", "close", "volume")


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


def _book_block(label: str, trades: pd.DataFrame, outdir: Path, stem: str) -> dict:
    years = summarize_by_year(trades, gain_col="gain_pct", buckets=YEAR_BUCKETS) if not trades.empty else pd.DataFrame()
    stats = _print_book(label, trades, years)
    if not trades.empty:
        trades.to_csv(outdir / f"{stem}_trades.csv", index=False)
        LOG.info("%s exits: %s", label, trades["exit_reason"].value_counts().to_dict())
        LOG.info(
            "%s hold_bars median=%.1f mean=%.1f max_concurrent_est=%s",
            label,
            float(trades["hold_bars"].median()),
            float(trades["hold_bars"].mean()),
            _max_concurrent(trades),
        )
    if not years.empty:
        years.to_csv(outdir / f"{stem}_years.csv", index=False)
        LOG.info("%s year buckets:\n%s", label, years.to_string(index=False))
    if not trades.empty and len(trades) >= 4:
        dropped = _drop_top_n(trades, 3)
        d_years = summarize_by_year(dropped, gain_col="gain_pct", buckets=YEAR_BUCKETS)
        _print_book(label + " drop-top-3", dropped, d_years)
    if not trades.empty and "size_mult" in trades.columns:
        sized = trades.copy()
        sized["gain_pct"] = sized_gain_pct(trades)
        s_years = summarize_by_year(sized, gain_col="gain_pct", buckets=YEAR_BUCKETS)
        _print_book(label + " VIX-size", sized, s_years)
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
    cal_2022 = (
        trades.loc[pd.to_datetime(trades["buy_date"]).dt.year == 2022]
        if not trades.empty
        else trades
    )
    if cal_2022 is not None and not cal_2022.empty:
        y2022 = summarize_by_year(
            cal_2022,
            gain_col="gain_pct",
            buckets=(("2022", "2022-01-01", "2022-12-31"),),
        )
        _print_book(label + " calendar 2022", cal_2022, y2022)
    cal_2023 = (
        trades.loc[pd.to_datetime(trades["buy_date"]).dt.year == 2023]
        if not trades.empty
        else trades
    )
    if cal_2023 is not None and not cal_2023.empty:
        y2023 = summarize_by_year(
            cal_2023,
            gain_col="gain_pct",
            buckets=(("2023", "2023-01-01", "2023-12-31"),),
        )
        _print_book(label + " calendar 2023", cal_2023, y2023)
    stats["years"] = years
    return stats


def _max_concurrent(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    events: Dict[pd.Timestamp, int] = {}
    for _, row in trades.iterrows():
        b = pd.Timestamp(row["buy_date"]).normalize()
        s = pd.Timestamp(row["sell_date"]).normalize()
        events[b] = events.get(b, 0) + 1
        events[s] = events.get(s, 0) - 1
    cur = 0
    peak = 0
    for dt in sorted(events):
        cur += events[dt]
        if cur > peak:
            peak = cur
    return int(peak)


def main() -> int:
    ap = argparse.ArgumentParser(description="Choppy-regime RSI-2 MOC and 3-day pullback")
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    ap.add_argument("--panel-start", default=PANEL_START)
    ap.add_argument("--strategy", choices=("both", "rsi2", "pullback"), default="both")
    ap.add_argument("--n-symbols", type=int, default=0, help="0 = full ALPACA 1d universe")
    ap.add_argument("--liquid-n", type=int, default=500)
    ap.add_argument("--min-price", type=float, default=10.0)
    ap.add_argument("--min-adv", type=float, default=1_000_000.0)
    ap.add_argument("--max-pos", type=int, default=15)
    ap.add_argument("--rsi-entry", type=float, default=DEFAULT_RSI_ENTRY)
    ap.add_argument("--rsi-exit", type=float, default=DEFAULT_RSI_EXIT)
    ap.add_argument("--time-stop", type=int, default=DEFAULT_TIME_STOP)
    ap.add_argument("--sma-trend", type=int, default=DEFAULT_SMA_TREND_RSI)
    ap.add_argument("--pullback-time-stop", type=int, default=DEFAULT_PULLBACK_TIME_STOP)
    ap.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_MULT)
    ap.add_argument("--vix-max", type=float, default=None, help="Sit-out threshold; default off")
    ap.add_argument("--vix-skip", action="store_true", help="Restore VIX>30 sit-out")
    ap.add_argument("--no-cpi", action="store_true")
    ap.add_argument("--no-extreme", action="store_true", help="Skip RSI<2 / no-MA book")
    ap.add_argument("--confirm-only", action="store_true", help="RSI cross-back + pullback close-loc only")
    ap.add_argument("--no-confirm", action="store_true", help="Skip cross-back / close-loc books")
    ap.add_argument("--allow-more-lows", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "choppy_regime")
    args = ap.parse_args()

    timings: Dict[str, float] = {}
    t_all = time.perf_counter()
    vix_max = 30.0 if args.vix_skip else args.vix_max
    skip_cpi = not args.no_cpi
    liq = dict(liquid_n=args.liquid_n, min_price=args.min_price, min_adv=args.min_adv)

    t0 = time.perf_counter()
    symbols = list_daily_symbols("ALPACA", "1d")
    timings["list_symbols"] = time.perf_counter() - t0
    LOG.info("ALPACA 1d symbols: %d", len(symbols))
    if args.n_symbols and args.n_symbols > 0:
        symbols = symbols[: int(args.n_symbols)]
        LOG.info("Truncated to --n-symbols %d", len(symbols))
    if len(symbols) < 20:
        LOG.error("Universe too small (%d)", len(symbols))
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

    t0 = time.perf_counter()
    macros = load_macro_closes(["VIX"], args.panel_start, args.end)
    timings["load_vix"] = time.perf_counter() - t0
    vix = macros.get("VIX")
    if vix is None:
        LOG.warning("VIX daily close missing; size-scale is 1.0")
    else:
        LOG.info("VIX n=%d %s->%s (skip=%s)", len(vix), vix.index[0].date(), vix.index[-1].date(), vix_max)

    outdir = dated_outdir(args.outdir)

    if args.strategy in ("both", "rsi2") and not args.confirm_only:
        t0 = time.perf_counter()
        rsi_trades = collect_rsi2_moc_trades(
            close,
            panels["volume"],
            vix=vix,
            eval_start=args.start,
            eval_end=args.end,
            rsi_entry=args.rsi_entry,
            rsi_exit=args.rsi_exit,
            time_stop=args.time_stop,
            vix_max=vix_max,
            skip_cpi=skip_cpi,
            sma_trend=args.sma_trend,
            **liq,
        )
        timings["collect_rsi2"] = time.perf_counter() - t0
        _book_block("RSI2 SMA%d unconstrained" % args.sma_trend, rsi_trades, outdir, "rsi2_moc")
        occ = occupancy_ranked(rsi_trades, max_open=args.max_pos, score_ascending=True)
        _book_block("RSI2 SMA%d occ%s" % (args.sma_trend, args.max_pos), occ, outdir, "rsi2_moc_occ")
        if not args.no_extreme:
            t0 = time.perf_counter()
            ext = collect_rsi2_moc_trades(
                close,
                panels["volume"],
                vix=vix,
                eval_start=args.start,
                eval_end=args.end,
                rsi_entry=2.0,
                rsi_exit=args.rsi_exit,
                time_stop=args.time_stop,
                vix_max=vix_max,
                skip_cpi=skip_cpi,
                sma_trend=0,
                **liq,
            )
            timings["collect_rsi2_extreme"] = time.perf_counter() - t0
            _book_block("RSI2 extreme<2 noMA unconstrained", ext, outdir, "rsi2_extreme")
            occ_e = occupancy_ranked(ext, max_open=args.max_pos, score_ascending=True)
            _book_block("RSI2 extreme<2 noMA occ%s" % args.max_pos, occ_e, outdir, "rsi2_extreme_occ")

    if args.strategy in ("both", "rsi2") and not args.no_confirm:
        for xb, stem in ((5.0, "rsi2_xback5"), (10.0, "rsi2_xback10")):
            t0 = time.perf_counter()
            xb_trades = collect_rsi2_moc_trades(
                close,
                panels["volume"],
                vix=vix,
                eval_start=args.start,
                eval_end=args.end,
                rsi_entry=args.rsi_entry,
                rsi_exit=args.rsi_exit,
                time_stop=args.time_stop,
                vix_max=vix_max,
                skip_cpi=skip_cpi,
                sma_trend=args.sma_trend,
                rsi_cross_back=xb,
                **liq,
            )
            timings["collect_%s" % stem] = time.perf_counter() - t0
            label = "RSI2 SMA%d xback%.0f" % (args.sma_trend, xb)
            _book_block("%s unconstrained" % label, xb_trades, outdir, stem)
            # Higher RSI at reclaim = stronger bounce; do not take the still-crashing tail.
            occ_x = occupancy_ranked(xb_trades, max_open=args.max_pos, score_ascending=False)
            _book_block("%s occ%s" % (label, args.max_pos), occ_x, outdir, stem + "_occ")

    if args.strategy in ("both", "pullback") and not args.confirm_only:
        t0 = time.perf_counter()
        pb_trades = collect_pullback_trades(
            panels["open"],
            panels["high"],
            panels["low"],
            close,
            panels["volume"],
            vix=vix,
            eval_start=args.start,
            eval_end=args.end,
            atr_mult=args.atr_mult,
            exactly_three_lows=not args.allow_more_lows,
            vix_max=vix_max,
            skip_cpi=skip_cpi,
            time_stop=args.pullback_time_stop,
            **liq,
        )
        timings["collect_pullback"] = time.perf_counter() - t0
        _book_block("Pullback3 unconstrained", pb_trades, outdir, "pullback3")
        occ = occupancy_ranked(pb_trades, max_open=args.max_pos, score_ascending=False)
        _book_block("Pullback3 occ%s" % args.max_pos, occ, outdir, "pullback3_occ")

    if args.strategy in ("both", "pullback") and not args.no_confirm:
        t0 = time.perf_counter()
        pb_loc = collect_pullback_trades(
            panels["open"],
            panels["high"],
            panels["low"],
            close,
            panels["volume"],
            vix=vix,
            eval_start=args.start,
            eval_end=args.end,
            atr_mult=args.atr_mult,
            exactly_three_lows=not args.allow_more_lows,
            vix_max=vix_max,
            skip_cpi=skip_cpi,
            time_stop=args.pullback_time_stop,
            min_close_loc=0.7,
            **liq,
        )
        timings["collect_pullback_loc"] = time.perf_counter() - t0
        _book_block("Pullback3 close-loc0.7 unconstrained", pb_loc, outdir, "pullback3_loc07")
        occ_l = occupancy_ranked(pb_loc, max_open=args.max_pos, score_ascending=False)
        _book_block("Pullback3 close-loc0.7 occ%s" % args.max_pos, occ_l, outdir, "pullback3_loc07_occ")

    timings["total"] = time.perf_counter() - t_all
    LOG.info("Wrote %s (total %.1fs)", outdir, timings["total"])
    LOG.info("timings=%s", {k: round(v, 1) for k, v in timings.items()})
    LOG.info("SPY annual %s", {str(k): round(float(v) * 100, 2) for k, v in annual_returns(spy_eq).items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
