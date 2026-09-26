"""BS short-put overlay on the current_best last-15m 1d book.

Sell ATM (or delta) cash-secured puts at the stock fill; TP at 50% credit;
stop when underlying tags ATR k=2 hard-stop (constant-IV BS mark).

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_short_put_bs.py --workers 8
  python scripts\\research\\overlay_last_15m_short_put_bs.py --smoke --limit 20
  python scripts\\research\\overlay_last_15m_short_put_bs.py --strike-mode delta --dte 45
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_short_put_bs import (  # noqa: E402
    DEFAULT_DTE,
    DEFAULT_R,
    DEFAULT_STRIKE_MODE,
    DEFAULT_TARGET_DELTA,
    DEFAULT_TP_FRAC,
    apply_short_put_book,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_short_put_bs")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"
)
SMOKE_DEFAULT = ("AMPL", "VST", "TARS", "ACHC", "RDWR")


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    g = (
        pd.to_numeric(df[gain_col], errors="coerce")
        if (not df.empty and gain_col in df.columns)
        else pd.Series(dtype=float)
    )
    years = (
        summarize_by_year(df.dropna(subset=[gain_col]), gain_col=gain_col, buckets=YEAR_BUCKETS)
        if not df.empty and gain_col in df.columns
        else pd.DataFrame()
    )
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")] if not df.empty else df
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")] if not df.empty else df
    s22 = book_stats(pd.to_numeric(y22[gain_col], errors="coerce")) if gain_col in y22.columns else {}
    s223 = book_stats(pd.to_numeric(both[gain_col], errors="coerce")) if gain_col in both.columns else {}
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s | 2022 n=%s E=%s PF=%s | 2022-23 E=%s PF=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s22.get("n"),
        s22.get("expectancy_pct"),
        s22.get("profit_factor"),
        s223.get("expectancy_pct"),
        s223.get("profit_factor"),
    )
    print("---- %s ----" % label)
    if not years.empty:
        print(years.to_string(index=False))
    return {
        "label": label,
        "n": s.get("n"),
        "expectancy_pct": s.get("expectancy_pct"),
        "profit_factor": s.get("profit_factor"),
        "geo_year_pf": geo,
        "median_pct": s.get("median_pct"),
        "win_rate_pct": s.get("win_rate_pct"),
        "n_2022": s22.get("n"),
        "e_2022": s22.get("expectancy_pct"),
        "pf_2022": s22.get("profit_factor"),
        "e_2022_23": s223.get("expectancy_pct"),
        "pf_2022_23": s223.get("profit_factor"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dte", type=int, default=DEFAULT_DTE)
    ap.add_argument("--tp-frac", type=float, default=DEFAULT_TP_FRAC)
    ap.add_argument("--r", type=float, default=DEFAULT_R)
    ap.add_argument(
        "--strike-mode",
        choices=("atm", "delta"),
        default=DEFAULT_STRIKE_MODE,
    )
    ap.add_argument("--target-delta", type=float, default=DEFAULT_TARGET_DELTA)
    ap.add_argument("--iv", type=float, default=None, help="Override IV (e.g. 0.35)")
    ap.add_argument(
        "--opt-friction",
        type=float,
        default=0.0,
        help="Per-share option friction each side (dollars)",
    )
    ap.add_argument(
        "--stop-iv-mult",
        type=float,
        default=1.0,
        help="Multiply IV only on hard-stop marks (stress crash IV)",
    )
    ap.add_argument("--pad-before", type=int, default=5)
    ap.add_argument(
        "--smoke",
        nargs="*",
        default=None,
        help="Limit to symbols (default list if flag present with no args)",
    )
    ap.add_argument("--limit", type=int, default=0, help="First N trades only (0=all)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trades = pd.read_csv(args.trades)
    if args.smoke is not None:
        want = {s.upper() for s in (args.smoke or SMOKE_DEFAULT)}
        trades = trades[trades["stock"].astype(str).str.upper().isin(want)].copy()
        LOG.info("Smoke symbols=%s n=%d", sorted(want), len(trades))
    if int(args.limit) > 0:
        trades = trades.head(int(args.limit)).copy()

    symbols = sorted(trades["stock"].astype(str).str.upper().unique())
    buy_min = pd.to_datetime(trades["buy_date"], errors="coerce").min() - timedelta(
        days=int(args.pad_before)
    )
    buy_max = pd.to_datetime(trades["buy_date"], errors="coerce").max() + timedelta(
        days=int(args.dte) + 14
    )

    t0 = time.perf_counter()
    panels_1d = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=buy_min,
        end=buy_max,
        workers=int(args.workers),
        use_cache=True,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    load_s = time.perf_counter() - t0
    LOG.info(
        "Loaded 1d %d/%d in %.1fs",
        sum(1 for s in symbols if panels_1d.get(s) is not None and not panels_1d[s].empty),
        len(symbols),
        load_s,
    )

    t1 = time.perf_counter()
    put_df = apply_short_put_book(
        trades,
        panels_1d,
        dte=int(args.dte),
        tp_frac=float(args.tp_frac),
        r=float(args.r),
        strike_mode=str(args.strike_mode),
        target_delta=float(args.target_delta),
        iv_override=args.iv,
        opt_friction_per_share=float(args.opt_friction),
        stop_iv_mult=float(args.stop_iv_mult),
    )
    walk_s = time.perf_counter() - t1
    LOG.info("Walked puts in %.1fs", walk_s)

    ok = put_df[put_df["put_ok"] == True].copy()  # noqa: E712
    skip = put_df[put_df["put_ok"] != True]
    if not skip.empty:
        LOG.info("Skips:\n%s", skip["put_reason"].value_counts().to_string())

    stock = trades.copy()
    stock["gain_pct"] = pd.to_numeric(stock["gain_pct"], errors="coerce")
    base = _print_block("stock_baseline", stock, "gain_pct")
    put_cs = _print_block(
        "put_cash_secured dte=%s mode=%s tp=%s" % (args.dte, args.strike_mode, args.tp_frac),
        ok,
        "gain_pct",
    )
    put_rop = _print_block("put_rop (credit basis)", ok, "gain_rop_pct")

    if "put_exit_reason" in ok.columns and not ok.empty:
        print("---- put_exit_reason ----")
        print(ok["put_exit_reason"].value_counts().to_string())
        for reason, sub in ok.groupby("put_exit_reason"):
            g = pd.to_numeric(sub["gain_pct"], errors="coerce")
            print(
                "  %s n=%d E=%.3f med=%.3f"
                % (reason, len(sub), float(g.mean()), float(g.median()))
            )

    outdir = args.outdir or dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stem = "last_15m_short_put_bs_dte%s_%s" % (args.dte, args.strike_mode)
    trades_path = outdir / ("%s_trades_%s.csv" % (stem, stamp))
    summary_path = outdir / ("%s_summary_%s.csv" % (stem, stamp))
    put_df.to_csv(trades_path, index=False)
    pd.DataFrame([base, put_cs, put_rop]).to_csv(summary_path, index=False)
    LOG.info("Wrote %s", trades_path)
    LOG.info("Wrote %s", summary_path)
    LOG.info("Wall load=%.1fs walk=%.1fs", load_s, walk_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
