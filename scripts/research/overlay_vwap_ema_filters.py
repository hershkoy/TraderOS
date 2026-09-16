#!/usr/bin/env python3
"""Pre-registered selectivity overlays on the SMB LOD / VWAP reclaim book.

Same-list, occupancy not re-walked except daily top1 (that IS the 100x cut).
Friction 0.10. Grade geo YEAR_BUCKETS PF, 2022, drop-top-3, winner-$ of skips.

Cuts (video + friction economics, not a fitted grid):
  unit>=1.5 / 3 / 5     stop ~0.5% / 1% / 1.7% vs 0.1% RT friction
  morning               signal before 11:00 ET (opening-drive LOD)
  lod_age>=4            at least 1h grind on 15m after LOD
  daily_top1            largest unit_pct each session (the "one trade a day")
  unit3_morning / unit3_age4 / top1_unit2 / top1_morning

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python -u scripts\\research\\overlay_vwap_ema_filters.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.realistic_purchaser import as_et  # noqa: E402
from utils.research.report_paths import dated_outdir  # noqa: E402
from utils.research.vwap_ema_reclaim import (  # noqa: E402
    attach_unit_clock,
    drop_top_n_expectancy,
    index_rth_15m_fast,
    mask_min_lod_age,
    mask_min_unit,
    mask_morning,
    take_daily_topn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("overlay_vwap_ema")
logging.getLogger("utils.db.timescaledb_client").setLevel(logging.WARNING)

DEFAULT_TRADES = (
    ROOT
    / "reports"
    / "vwap_ema_reclaim"
    / "2026-09-13"
    / "vwap_ema_reclaim_next-mid_trades.csv"
)
FRICTION_PCT = 0.10
WINNER_WINDOWS: Tuple[Tuple[str, str, str], ...] = (
    ("2020-2021", "2020-01-01", "2021-12-31"),
    ("2024-2026", "2024-01-01", "2026-12-31"),
)


def _bar_low(bar: dict) -> Optional[float]:
    try:
        val = float(bar["low"])
    except (KeyError, TypeError, ValueError):
        return None
    if val != val:
        return None
    return val


def _clock_hm(ts) -> Optional[tuple]:
    try:
        et = as_et(ts)
    except (TypeError, ValueError):
        return None
    return (int(et.hour), int(et.minute))


def attach_lod_age(trades: pd.DataFrame, indexed: Dict[str, dict]) -> pd.DataFrame:
    """Session LOD age in 15m bars at the stored signal_ts."""
    out = trades.copy()
    age_map = {}
    for sym, g in out.groupby("stock", sort=False):
        by_day = indexed.get(str(sym).upper())
        if not by_day:
            continue
        for i, row in g.iterrows():
            try:
                sess = as_et(row["signal_ts"]).date()
            except (TypeError, ValueError, KeyError):
                continue
            bars = list(by_day.get(sess) or [])
            want = _clock_hm(row.get("signal_ts"))
            if want is None or not bars:
                continue
            lows = []
            sig_i = None
            for j, bar in enumerate(bars):
                if _clock_hm(bar.get("ts")) == want:
                    sig_i = j
                    break
                lo = _bar_low(bar)
                lows.append(lo if lo is not None else float("inf"))
            if sig_i is None or sig_i < 1 or not lows:
                continue
            if min(lows) == float("inf"):
                continue
            lod_i = int(min(range(len(lows)), key=lambda k: lows[k]))
            age_map[i] = sig_i - lod_i
    out["lod_age_bars"] = pd.Series(age_map)
    return out


def _print_block(label: str, df: pd.DataFrame, *, gain_col: str) -> dict:
    g = pd.to_numeric(df[gain_col], errors="coerce") if not df.empty else pd.Series(dtype=float)
    years = summarize_by_year(df, gain_col=gain_col, buckets=YEAR_BUCKETS)
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    drop3 = drop_top_n_expectancy(g.tolist(), 3)
    rec = {
        "label": label,
        "n": s.get("n"),
        "E": s.get("expectancy_pct"),
        "PF": s.get("profit_factor"),
        "geo": None if geo is None else round(geo, 3),
        "WR": s.get("win_rate_pct"),
        "median": s.get("median_pct"),
        "drop3_E": None if drop3 is None else round(drop3, 4),
    }
    LOG.info(
        "%s n=%s E=%s PF=%s geo=%s WR=%s median=%s drop3E=%s",
        label,
        rec["n"],
        rec["E"],
        rec["PF"],
        rec["geo"],
        rec["WR"],
        rec["median"],
        rec["drop3_E"],
    )
    if not years.empty:
        print(years.to_string(index=False))
    if not df.empty:
        bd = pd.to_datetime(df["buy_date"], errors="coerce")
        y22 = df.loc[bd.dt.year == 2022]
        if not y22.empty:
            s22 = book_stats(pd.to_numeric(y22[gain_col], errors="coerce"))
            LOG.info(
                "%s 2022 n=%s E=%s PF=%s",
                label,
                s22.get("n"),
                s22.get("expectancy_pct"),
                s22.get("profit_factor"),
            )
            rec["n_2022"] = s22.get("n")
            rec["E_2022"] = s22.get("expectancy_pct")
            rec["PF_2022"] = s22.get("profit_factor")
        if "exit_reason" in df.columns:
            print(df["exit_reason"].value_counts().to_string())
    return rec


def _winner_windows(base: pd.DataFrame, keep: pd.Series, gain_col: str) -> None:
    g = pd.to_numeric(base[gain_col], errors="coerce")
    cut = winner_cut_skip(g, keep)
    LOG.info(
        "winner-$ dropped=%s win_frac=%s dollar_ratio=%s blunt=%s",
        cut.get("n_dropped"),
        cut.get("winner_frac_of_drops"),
        cut.get("dollar_ratio"),
        cut.get("blunt"),
    )
    d = pd.to_datetime(base["buy_date"], errors="coerce")
    for name, start, end in WINNER_WINDOWS:
        m = (d >= start) & (d <= end)
        if int(m.sum()) < 30:
            continue
        sub = winner_cut_skip(g.loc[m], keep.loc[m])
        LOG.info(
            "winner-$ %s dropped=%s win_frac=%s blunt=%s",
            name,
            sub.get("n_dropped"),
            sub.get("winner_frac_of_drops"),
            sub.get("blunt"),
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="VWAP reclaim selectivity overlays")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--friction-pct", type=float, default=FRICTION_PCT)
    ap.add_argument("--no-15m", action="store_true", help="skip lod_age attach")
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()
    if not args.trades.is_file():
        LOG.error("Missing trades %s", args.trades)
        return 1
    outdir = args.outdir or dated_outdir(ROOT / "reports" / "vwap_ema_reclaim")
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    raw = pd.read_csv(args.trades)
    df = attach_unit_clock(raw)
    net = apply_friction(df, float(args.friction_pct))
    gain_col = "gain_pct_net"
    LOG.info("Loaded %d trades unit_pct median=%.3f", len(net), float(net["unit_pct"].median()))

    if not args.no_15m:
        t1 = time.perf_counter()
        symbols = sorted({str(s).upper() for s in net["stock"].dropna().unique()})
        start = datetime.strptime("2018-11-01", "%Y-%m-%d")
        end = datetime.strptime("2026-08-21", "%Y-%m-%d")
        panels = load_ohlcv_many(
            symbols,
            timeframe="15m",
            provider="IB",
            start=start,
            end=end,
            use_cache=True,
            workers=max(1, int(args.load_workers)),
            chunk_size=int(args.chunk_size),
        )
        indexed: Dict[str, dict] = {}
        for sym, frame in panels.items():
            if frame is None or frame.empty:
                continue
            indexed[str(sym).upper()] = index_rth_15m_fast(frame, naive_tz="UTC")
        net = attach_lod_age(net, indexed)
        LOG.info(
            "lod_age attached %d/%d (%.1fs)",
            int(net["lod_age_bars"].notna().sum()),
            len(net),
            time.perf_counter() - t1,
        )

    rows: List[dict] = []
    rows.append(_print_block("baseline_fric0.10", net, gain_col=gain_col))

    specs: List[tuple] = [
        ("unit>=1.5", mask_min_unit(net, 1.5)),
        ("unit>=3", mask_min_unit(net, 3.0)),
        ("unit>=5", mask_min_unit(net, 5.0)),
        ("morning<11", mask_morning(net, 11.0)),
    ]
    if "lod_age_bars" in net.columns and net["lod_age_bars"].notna().any():
        specs.append(("lod_age>=4", mask_min_lod_age(net, 4)))
        specs.append(
            (
                "unit>=3_age4",
                mask_min_unit(net, 3.0) & mask_min_lod_age(net, 4),
            )
        )
    specs.append(
        ("unit>=3_morning", mask_min_unit(net, 3.0) & mask_morning(net, 11.0))
    )

    for label, keep in specs:
        print("---- %s ----" % label)
        _winner_windows(net, keep, gain_col)
        sub = net.loc[keep].copy()
        rec = _print_block(label, sub, gain_col=gain_col)
        rec["n_factor"] = None if not rec["n"] else round(len(net) / rec["n"], 1)
        rows.append(rec)

    top_specs = [
        ("daily_top1", net),
        ("top1_unit>=2", net.loc[mask_min_unit(net, 2.0)]),
        ("top1_morning", net.loc[mask_morning(net, 11.0)]),
    ]
    if "lod_age_bars" in net.columns and net["lod_age_bars"].notna().any():
        top_specs.append(("top1_age>=4", net.loc[mask_min_lod_age(net, 4)]))
    for label, pool in top_specs:
        print("---- %s ----" % label)
        kept = take_daily_topn(pool, n=1)
        keep = net.index.isin(kept.index)
        keep_s = pd.Series(keep, index=net.index)
        _winner_windows(net, keep_s, gain_col)
        rec = _print_block(label, kept, gain_col=gain_col)
        rec["n_factor"] = None if not rec["n"] else round(len(net) / rec["n"], 1)
        rows.append(rec)

    summary = pd.DataFrame(rows)
    path = outdir / "vwap_ema_reclaim_filter_overlay.csv"
    summary.to_csv(path, index=False)
    LOG.info("Wrote %s wall=%.1fs", path, time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
