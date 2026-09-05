#!/usr/bin/env python3
"""A/B: after an H2 resist-break, rebuy a later close-above-resistance (same rails).

Distinct from L3 --shakeout-rebuy-bars (close through support then reclaim).
SXI 2020-08-11 filled, 2020-08-25 6% stop, later Oct breakout was skipped because
the finder stopped after the first close above resistance.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_shakeout_breakout.py --all-symbols
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_shakeout_breakout.py --complete-history
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_h2_break import _daily_base  # noqa: E402
from backtest_channel_touch_trades import (  # noqa: E402
    YEAR_BUCKETS,
    _scan_trades,
    _summarize,
    apply_friction,
    enrich_rs,
    filter_trades,
    keep_one_per_symbol_day,
    summarize_by_year,
)
from find_ascending_channels import list_symbols_fast  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("channel_touch_shakeout_breakout")

RAW_TRADES = resolve_artifact("channel_touch_trades_raw_20260828_194314.csv")
SPAN_CAP = 365.0
FRICTION = 0.25
GAIN_COL = "gain_pct_net"
YEAR_BUCKETS_FULL: list = [
    ("pre-2018", "1990-01-01", "2017-12-31"),
    ("2018-2019", "2018-01-01", "2019-12-31"),
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2022-2023", "2022-01-01", "2023-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
]


def _fmt(s: dict) -> str:
    return "n=%s E=%s PF=%s WR=%s med=%s hard_stop=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct"),
        s.get("hard_stop_exits"),
    )


def _net(df: pd.DataFrame, friction: float) -> pd.DataFrame:
    if df.empty:
        return df
    return apply_friction(df, friction) if friction else df


def _is_sbo(df: pd.DataFrame) -> pd.Series:
    if df.empty or "shakeout_breakout" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["shakeout_breakout"].fillna(False).astype(bool)


def _book(df: pd.DataFrame, *, extra_only: bool, hard_stop: bool) -> pd.DataFrame:
    if df.empty:
        return df
    sbo = _is_sbo(df)
    if extra_only:
        out = df.loc[sbo].copy()
        if hard_stop and "parent_exit_reason" in out.columns:
            out = out.loc[out["parent_exit_reason"].astype(str) == "hard_stop"].copy()
        return out
    return df.loc[~sbo].copy()


def _combo(parent: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if parent.empty and extra.empty:
        return parent
    if extra.empty:
        return parent
    if parent.empty:
        return extra
    return pd.concat([parent, extra], ignore_index=True, sort=False)


def _print_block(title: str, trades: pd.DataFrame, buckets=YEAR_BUCKETS) -> dict:
    s = _summarize(trades, gain_col=GAIN_COL)
    print("=== %s ===" % title)
    print(_fmt(s))
    if not trades.empty:
        print(summarize_by_year(trades, gain_col=GAIN_COL, buckets=buckets).to_string(index=False))
    return s


def _print_sxi(df: pd.DataFrame, label: str) -> None:
    if df.empty or "stock" not in df.columns:
        print("SXI %s: none" % label)
        return
    rows = df.loc[df["stock"].astype(str).str.upper() == "SXI"]
    if rows.empty:
        print("SXI %s: none" % label)
        return
    cols = [
        c
        for c in (
            "buy_date",
            "sell_date",
            "buy_price",
            "sell_price",
            "gain_pct_net",
            "exit_reason",
            "shakeout_breakout",
            "shakeout_inside_bars",
            "parent_exit_reason",
            "channel_start",
            "channel_end",
        )
        if c in rows.columns
    ]
    print("SXI %s (%d rows):" % (label, len(rows)))
    print(rows[cols].to_string(index=False))


def _universe_symbols(*, all_symbols: bool, include_ib: bool, raw_path: Path) -> list:
    if not all_symbols:
        raw = pd.read_csv(raw_path)
        symbols = sorted(set(raw["stock"].astype(str).str.upper()) | {"SPY"})
        logger.info("Raw-CSV universe n=%d unique=%d", len(raw), len(symbols) - 1)
        return symbols
    alpaca = {s.upper() for s in list_symbols_fast("ALPACA", "1d")}
    ib = {s.upper() for s in list_symbols_fast("IB", "1d")} if include_ib else set()
    symbols = sorted(alpaca | ib | {"SPY"})
    logger.info(
        "Full universe ALPACA 1d=%d IB 1d=%d union=%d (incl SPY)",
        len(alpaca),
        len(ib),
        len(symbols),
    )
    return symbols


def _log_panel_span(panels: dict) -> None:
    firsts = []
    lasts = []
    n_ok = 0
    for sym, df in panels.items():
        if sym == "SPY" or df is None or df.empty:
            continue
        n_ok += 1
        firsts.append(pd.Timestamp(df.index.min()))
        lasts.append(pd.Timestamp(df.index.max()))
    if not firsts:
        logger.warning("No loaded non-SPY panels")
        return
    spy = panels.get("SPY")
    spy_span = ""
    if spy is not None and not spy.empty:
        spy_span = " SPY %s -> %s" % (
            pd.Timestamp(spy.index.min()).date(),
            pd.Timestamp(spy.index.max()).date(),
        )
    logger.info(
        "Loaded %d names; panel firsts %s .. %s; lasts %s .. %s;%s",
        n_ok,
        min(firsts).date(),
        max(firsts).date(),
        min(lasts).date(),
        max(lasts).date(),
        spy_span,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="H2 shakeout then second resist-break A/B")
    ap.add_argument("--raw", type=Path, default=None)
    ap.add_argument(
        "--start",
        default="2018-11-01",
        help="YYYY-MM-DD. Empty with --complete-history loads all stored bars.",
    )
    ap.add_argument("--end", default="2026-08-27")
    ap.add_argument("--friction-pct", type=float, default=FRICTION)
    ap.add_argument(
        "--all-symbols",
        action="store_true",
        help="Scan full ALPACA 1d universe (IB prefix). Default uses the frozen raw-trades names.",
    )
    ap.add_argument(
        "--complete-history",
        action="store_true",
        help="All ALPACA+IB 1d names, IB prefix from 2006-01-01 through last stored bar, windowed 504/252.",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--min-inside", default="1,5", help="Comma list of min inside closes")
    args = ap.parse_args()
    t0 = time.perf_counter()
    friction = float(args.friction_pct)
    raw_path = args.raw or RAW_TRADES
    complete = bool(args.complete_history)
    all_symbols = bool(args.all_symbols) or complete
    buckets = YEAR_BUCKETS_FULL if complete else YEAR_BUCKETS
    # start=None skips IB prefix (only empty ALPACA names get IB). Clip start
    # early so prefix stitches bars before the first Alpaca print.
    if complete:
        start_s = "2006-01-01"
        end_s = ""
    else:
        start_s = args.start.strip() or "2018-11-01"
        end_s = args.end.strip() or "2026-08-27"

    symbols = _universe_symbols(
        all_symbols=all_symbols, include_ib=complete, raw_path=raw_path
    )

    start = datetime.strptime(start_s, "%Y-%m-%d") if start_s else None
    end = datetime.strptime(end_s, "%Y-%m-%d") if end_s else None
    logger.info("Load window start=%s end=%s complete_history=%s", start, end, complete)
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=start,
        end=end,
        fallback_provider="IB",
        merge_mode="prefix",
        workers=int(args.load_workers),
    )
    spy_df = panels.get("SPY")
    if spy_df is None or spy_df.empty:
        logger.error("No SPY panel")
        return 1
    logger.info("OHLCV loaded in %.1fs", time.perf_counter() - t0)
    _log_panel_span(panels)

    mins = [int(x.strip()) for x in str(args.min_inside).split(",") if x.strip()]
    if not mins:
        mins = [1]
    base = _daily_base()
    base["shakeout_breakout"] = True
    base["shakeout_breakout_hard_stop"] = False
    base["entry_features"] = True

    outdir = dated_outdir()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows = []

    for min_inside in mins:
        t_scan = time.perf_counter()
        scan_base = dict(base)
        scan_base["shakeout_breakout_min_inside"] = int(min_inside)
        scanned = _scan_trades(
            panels, symbols=symbols, workers=int(args.workers), base=scan_base
        )
        logger.info(
            "Scan min_inside=%d n=%d in %.1fs",
            min_inside,
            len(scanned),
            time.perf_counter() - t_scan,
        )
        if scanned.empty:
            logger.error("No trades for min_inside=%d", min_inside)
            continue
        scanned = enrich_rs(scanned, panels, spy_df, lookbacks=(63, 126), bars_per_session=1)
        spanned = filter_trades(scanned, max_channel_span_days=SPAN_CAP)
        spanned = _net(spanned, friction)

        parent = keep_one_per_symbol_day(_book(spanned, extra_only=False, hard_stop=False))
        extra_any = _book(spanned, extra_only=True, hard_stop=False)
        extra_hs = _book(spanned, extra_only=True, hard_stop=True)
        extra_any_u = keep_one_per_symbol_day(extra_any)
        extra_hs_u = keep_one_per_symbol_day(extra_hs)
        combo_any = keep_one_per_symbol_day(_combo(parent, extra_any))
        combo_hs = keep_one_per_symbol_day(_combo(parent, extra_hs))

        print("")
        print("----- min_inside=%d -----" % min_inside)
        parent_s = _print_block(
            "parent unique-symbol H2 span365 (no extras)", parent, buckets=buckets
        )
        any_s = _print_block("sleeve any-closed unique-symbol", extra_any_u, buckets=buckets)
        hs_s = _print_block("sleeve hard-stop unique-symbol", extra_hs_u, buckets=buckets)
        combo_any_s = _print_block(
            "combined unique-symbol any-closed", combo_any, buckets=buckets
        )
        combo_hs_s = _print_block(
            "combined unique-symbol hard-stop", combo_hs, buckets=buckets
        )
        _print_sxi(parent, "parent min_inside=%d" % min_inside)
        _print_sxi(extra_any, "extras any-closed min_inside=%d" % min_inside)
        _print_sxi(extra_hs, "extras hard-stop min_inside=%d" % min_inside)

        tag = "fullhist_" if complete else ""
        csv_any = outdir / (
            "channel_touch_shakeout_breakout_%sany_min%d_%s.csv" % (tag, min_inside, stamp)
        )
        csv_hs = outdir / (
            "channel_touch_shakeout_breakout_%shs_min%d_%s.csv" % (tag, min_inside, stamp)
        )
        extra_any.to_csv(csv_any, index=False)
        extra_hs.to_csv(csv_hs, index=False)
        logger.info("Wrote %s n=%d", csv_any, len(extra_any))
        logger.info("Wrote %s n=%d", csv_hs, len(extra_hs))

        for name, stats, n_raw in (
            ("parent", parent_s, len(parent)),
            ("sleeve_any", any_s, len(extra_any_u)),
            ("sleeve_hard_stop", hs_s, len(extra_hs_u)),
            ("combo_any", combo_any_s, len(combo_any)),
            ("combo_hard_stop", combo_hs_s, len(combo_hs)),
        ):
            row = dict(stats)
            row["variant"] = name
            row["min_inside"] = int(min_inside)
            row["n_raw"] = int(n_raw)
            row["complete_history"] = bool(complete)
            summary_rows.append(row)

    elapsed = time.perf_counter() - t0
    print("")
    print("elapsed_sec=%.1f" % elapsed)
    print("Nightly stays off until a sleeve beats parent E/PF without wrecking 2020-21.")
    if summary_rows:
        summary_csv = outdir / (
            "channel_touch_shakeout_breakout_%ssummary_%s.csv"
            % ("fullhist_" if complete else "", stamp)
        )
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
        logger.info("Summary -> %s", summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
