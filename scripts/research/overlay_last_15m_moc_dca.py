"""3-day MOC DCA on the current_best last-15m 1d book.

Split allocated capital into 3 equal MOC buys (signal day + next 2 sessions).
Pause a tranche when that close is below the projected resist rail. ATR k=2
hard-stop (from first fill) can close during scale-in; remaining shares exit
via 15m N+1 mid. Occupancy not re-walked.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\overlay_last_15m_moc_dca.py --workers 8
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
from compare_1d_last_15m_realistic_sells import build_hold_session_index  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_moc_dca import (  # noqa: E402
    FRICTION_PCT,
    N_TRANCHES_DEFAULT,
    apply_moc_dca,
    friction_gain,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("overlay_last_15m_moc_dca")

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "ascending_channels"
    / "2026-09-11"
    / "channel_touch_h2_last_15m_open_mid_sell_15m_next_mid_span365.csv"
)
SMOKE_DEFAULT = ("AMPL", "VST", "TARS", "ACHC", "RDWR")


def _in_range(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    d = pd.to_datetime(df["buy_date"], errors="coerce")
    return (d >= start) & (d <= end)


def _print_block(label: str, df: pd.DataFrame, gain_col: str = "gain_pct") -> dict:
    g = pd.to_numeric(df[gain_col], errors="coerce") if not df.empty else pd.Series(dtype=float)
    years = (
        summarize_by_year(df, gain_col=gain_col, buckets=YEAR_BUCKETS)
        if not df.empty
        else pd.DataFrame()
    )
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    y22 = df.loc[_in_range(df, "2022-01-01", "2022-12-31")] if not df.empty else df
    y23 = df.loc[_in_range(df, "2023-01-01", "2023-12-31")] if not df.empty else df
    both = df.loc[_in_range(df, "2022-01-01", "2023-12-31")] if not df.empty else df
    s22 = book_stats(pd.to_numeric(y22[gain_col], errors="coerce"))
    s23 = book_stats(pd.to_numeric(y23[gain_col], errors="coerce"))
    s223 = book_stats(pd.to_numeric(both[gain_col], errors="coerce"))
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s | 2022 n=%s E=%s PF=%s | 2023 n=%s E=%s PF=%s | 2022-23 E=%s PF=%s",
        label,
        s.get("n"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        None if geo is None else round(geo, 3),
        s22.get("n"),
        s22.get("expectancy_pct"),
        s22.get("profit_factor"),
        s23.get("n"),
        s23.get("expectancy_pct"),
        s23.get("profit_factor"),
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
        "n_2022": s22.get("n"),
        "e_2022": s22.get("expectancy_pct"),
        "pf_2022": s22.get("profit_factor"),
        "n_2023": s23.get("n"),
        "e_2023": s23.get("expectancy_pct"),
        "pf_2023": s23.get("profit_factor"),
        "e_2022_23": s223.get("expectancy_pct"),
        "pf_2022_23": s223.get("profit_factor"),
    }


def _apply_book(
    trades: pd.DataFrame,
    panels_1d: Dict[str, pd.DataFrame],
    indexed_15m: Dict[str, dict],
    *,
    n_tranches: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for idx, row in trades.iterrows():
        sym = str(row["stock"]).upper()
        got = apply_moc_dca(
            row,
            panel_1d=panels_1d.get(sym),
            by_day_15m=indexed_15m.get(sym),
            n_tranches=n_tranches,
        )
        rec = row.to_dict()
        rec["src_idx"] = idx
        rec["dca_ok"] = bool(got.filled)
        rec["dca_reason"] = got.reason
        rec["dca_n_filled"] = got.n_tranches_filled
        rec["dca_n_paused"] = got.n_paused
        rec["dca_avg_cost"] = got.avg_cost
        rec["dca_first_px"] = got.first_px
        rec["dca_capital_deployed"] = got.capital_deployed
        rec["dca_exit_reason"] = got.exit_reason
        rec["gain_pct_before"] = float(row["gain_pct"]) if pd.notna(row.get("gain_pct")) else None
        if got.filled and got.gain_allocated_pct is not None:
            rec["buy_price"] = got.avg_cost
            rec["sell_price"] = got.sell_px
            rec["gain_pct"] = got.gain_allocated_pct
            rec["gain_deployed_pct"] = got.gain_deployed_pct
            rec["gain_pct_fric"] = friction_gain(got.gain_allocated_pct)
            if got.last_fill_day is not None:
                rec["buy_date"] = got.last_fill_day.isoformat()
            if got.sell_ts is not None:
                rec["sell_time"] = str(got.sell_ts)
            if got.exit_reason:
                rec["exit_reason"] = got.exit_reason
        else:
            rec["gain_pct"] = None
            rec["gain_deployed_pct"] = None
            rec["gain_pct_fric"] = None
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-tranches", type=int, default=N_TRANCHES_DEFAULT)
    ap.add_argument("--pad-before", type=int, default=5)
    ap.add_argument("--pad-after", type=int, default=30)
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
    sell_max = pd.to_datetime(trades["sell_date"], errors="coerce").max() + timedelta(
        days=int(args.pad_after)
    )

    t0 = time.perf_counter()
    panels_1d = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=buy_min,
        end=sell_max,
        workers=int(args.workers),
        use_cache=True,
        fallback_provider="IB",
        merge_mode="prefix",
    )
    LOG.info(
        "Loaded 1d %d/%d in %.1fs",
        sum(1 for s in symbols if panels_1d.get(s) is not None and not panels_1d[s].empty),
        len(symbols),
        time.perf_counter() - t0,
    )

    t1 = time.perf_counter()
    panels_15m = load_ohlcv_many(
        symbols,
        timeframe="15m",
        provider="IB",
        start=buy_min,
        end=sell_max,
        workers=int(args.workers),
        use_cache=True,
    )
    LOG.info(
        "Loaded 15m %d/%d in %.1fs",
        sum(1 for s in symbols if panels_15m.get(s) is not None and not panels_15m[s].empty),
        len(symbols),
        time.perf_counter() - t1,
    )
    indexed = build_hold_session_index(
        panels_15m,
        trades,
        pad_before=int(args.pad_before),
        pad_after=int(args.pad_after),
    )

    outdir = Path(args.outdir) if args.outdir is not None else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = trades.copy()
    rows = [_print_block("baseline_last15m", baseline)]

    t2 = time.perf_counter()
    dca = _apply_book(
        trades,
        panels_1d,
        indexed,
        n_tranches=int(args.n_tranches),
    )
    LOG.info("DCA walk %.1fs", time.perf_counter() - t2)

    ok = dca[dca["dca_ok"]].copy()
    skipped = dca[~dca["dca_ok"]].copy()
    print("dca skip_reason:")
    print(skipped["dca_reason"].value_counts().to_string() if not skipped.empty else "(none)")
    print(
        "tranche fills: mean=%.2f paused_mean=%.2f deployed_mean=%.3f"
        % (
            float(ok["dca_n_filled"].mean()) if not ok.empty else 0.0,
            float(ok["dca_n_paused"].mean()) if not ok.empty else 0.0,
            float(ok["dca_capital_deployed"].mean()) if not ok.empty else 0.0,
        )
    )
    if "dca_exit_reason" in ok.columns:
        print("exit_reason:")
        print(ok["dca_exit_reason"].value_counts().head(10).to_string())

    rows.append(_print_block("moc_dca3_allocated", ok, gain_col="gain_pct"))
    if not ok.empty and "gain_deployed_pct" in ok.columns:
        rows.append(_print_block("moc_dca3_deployed", ok, gain_col="gain_deployed_pct"))
    if not ok.empty and "gain_pct_fric" in ok.columns:
        rows.append(_print_block("moc_dca3_fric%.2f" % FRICTION_PCT, ok, gain_col="gain_pct_fric"))

    # Same-list paired delta where both exist
    if not ok.empty:
        paired = ok.dropna(subset=["gain_pct", "gain_pct_before"]).copy()
        if not paired.empty:
            delta = paired["gain_pct"] - paired["gain_pct_before"]
            print(
                "paired delta (allocated - baseline) n=%d mean=%.4f median=%.4f"
                % (len(paired), float(delta.mean()), float(delta.median()))
            )

    summary = pd.DataFrame(rows)
    trades_path = outdir / "last_15m_moc_dca3_trades.csv"
    dca.to_csv(trades_path, index=False)
    summary_path = outdir / "last_15m_moc_dca3_summary.csv"
    summary.to_csv(summary_path, index=False)
    txt_path = outdir / "last_15m_moc_dca3_summary.txt"
    lines = [
        "last-15m 3x MOC DCA (pause below rail; ATR stop during scale-in)",
        "trades=%s" % args.trades,
        "n_tranches=%d" % int(args.n_tranches),
        "wall_s_1d=%.1f wall_s_15m=%.1f wall_s_dca=%.1f"
        % (time.perf_counter() - t0, time.perf_counter() - t1, time.perf_counter() - t2),
        "",
        summary.to_string(index=False),
        "",
        "Primary gate: allocated-capital E/PF/geo vs baseline; 2022 leftover.",
    ]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("out:", trades_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
