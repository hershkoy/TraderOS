"""Outside-channel area ratio on the current_best 1d last-15m book.

Yellow region on the 3-touch chart = daily lows below the support rail from
L1 through the buy day. Ratio = that area / (channel width * n_bars).
Then correlate with losing trades. Occupancy is not re-walked.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\analyze_channel_outside_area.py --workers 8
  python scripts\\research\\analyze_channel_outside_area.py --symbols ACHC,TFSL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.data.ohlcv_loader import load_ohlcv_many  # noqa: E402
from utils.research.channel_outside_area import (  # noqa: E402
    areas_for_trade,
    date_to_first_index,
)
from utils.research.last_15m_volume_geometry import (  # noqa: E402
    book_stats,
    geom_mean_year_pf,
    winner_cut_skip,
)
from utils.research.report_paths import dated_outdir  # noqa: E402

LOG = logging.getLogger("analyze_channel_outside_area")

DEFAULT_TRADES = ROOT / "reports" / "ascending_channels" / "channel_touch_1d_last_15m.csv"

RATIO_COL = "outside_below_low_ratio"
PEAK_COL = "max_undershoot_width"
GAIN_COL = "gain_pct"


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _corr(x: pd.Series, y: pd.Series) -> Optional[float]:
    m = x.notna() & y.notna()
    if int(m.sum()) < 20:
        return None
    v = float(x[m].corr(y[m]))
    return None if not np.isfinite(v) else round(v, 4)


def _spearman(x: pd.Series, y: pd.Series) -> Optional[float]:
    m = x.notna() & y.notna()
    if int(m.sum()) < 20:
        return None
    v = float(x[m].corr(y[m], method="spearman"))
    return None if not np.isfinite(v) else round(v, 4)


def _quintile_rows(df: pd.DataFrame, col: str, n_bins: int = 5) -> pd.DataFrame:
    work = df.loc[_num(df[col]).notna() & _num(df[GAIN_COL]).notna()].copy()
    if work.empty:
        return pd.DataFrame()
    x = _num(work[col])
    rows: List[dict] = []
    zero = x <= 0
    if int(zero.sum()) > 0 and int((~zero).sum()) > 0:
        z = work.loc[zero]
        s = book_stats(_num(z[GAIN_COL]))
        rows.append(
            {
                "feature": col,
                "bucket": "0 (no undershoot)",
                "n": s["n"],
                "loser_rate_pct": round(float((_num(z[GAIN_COL]) <= 0).mean() * 100.0), 2),
                "win_rate_pct": s["win_rate_pct"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "median_pct": s["median_pct"],
                "mean_ratio": round(float(x.loc[zero].mean()), 4),
            }
        )
        pos = work.loc[~zero]
        try:
            labels = pd.qcut(_num(pos[col]), q=int(n_bins), duplicates="drop")
        except (ValueError, TypeError):
            labels = pd.cut(_num(pos[col]), bins=min(int(n_bins), max(2, pos[col].nunique())))
        pos = pos.assign(_bucket=labels)
        for key, part in pos.groupby("_bucket", observed=False):
            s = book_stats(_num(part[GAIN_COL]))
            rows.append(
                {
                    "feature": col,
                    "bucket": str(key),
                    "n": s["n"],
                    "loser_rate_pct": round(
                        float((_num(part[GAIN_COL]) <= 0).mean() * 100.0), 2
                    ),
                    "win_rate_pct": s["win_rate_pct"],
                    "expectancy_pct": s["expectancy_pct"],
                    "profit_factor": s["profit_factor"],
                    "median_pct": s["median_pct"],
                    "mean_ratio": round(float(_num(part[col]).mean()), 4),
                }
            )
        return pd.DataFrame(rows)
    try:
        work["_bucket"] = pd.qcut(x, q=int(n_bins), duplicates="drop")
    except (ValueError, TypeError):
        work["_bucket"] = pd.cut(x, bins=min(int(n_bins), max(2, x.nunique())))
    for key, part in work.groupby("_bucket", observed=False):
        s = book_stats(_num(part[GAIN_COL]))
        rows.append(
            {
                "feature": col,
                "bucket": str(key),
                "n": s["n"],
                "loser_rate_pct": round(float((_num(part[GAIN_COL]) <= 0).mean() * 100.0), 2),
                "win_rate_pct": s["win_rate_pct"],
                "expectancy_pct": s["expectancy_pct"],
                "profit_factor": s["profit_factor"],
                "median_pct": s["median_pct"],
                "mean_ratio": round(float(_num(part[col]).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def _print_block(title: str, df: pd.DataFrame) -> dict:
    g = _num(df[GAIN_COL])
    years = summarize_by_year(df, gain_col=GAIN_COL, buckets=YEAR_BUCKETS)
    s = book_stats(g)
    geo = geom_mean_year_pf(years)
    LOG.info(
        "%s n=%d E=%s PF=%s geo=%s WR=%s",
        title,
        s["n"],
        "n/a" if s["expectancy_pct"] is None else "%.3f" % s["expectancy_pct"],
        "n/a" if s["profit_factor"] is None else "%.3f" % s["profit_factor"],
        "n/a" if geo is None else "%.3f" % geo,
        "n/a" if s["win_rate_pct"] is None else "%.1f" % s["win_rate_pct"],
    )
    print("---- %s ----" % title)
    print(years.to_string(index=False))
    out = dict(s)
    out["label"] = title
    out["geo_year_pf"] = geo
    return out


def _winner_loser_ratio_stats(df: pd.DataFrame, col: str) -> dict:
    x = _num(df[col])
    g = _num(df[GAIN_COL])
    m = x.notna() & g.notna()
    x = x[m]
    g = g[m]
    win = x[g > 0]
    lose = x[g <= 0]
    return {
        "n": int(m.sum()),
        "n_winners": int(len(win)),
        "n_losers": int(len(lose)),
        "mean_all": None if x.empty else round(float(x.mean()), 4),
        "median_all": None if x.empty else round(float(x.median()), 4),
        "mean_winners": None if win.empty else round(float(win.mean()), 4),
        "median_winners": None if win.empty else round(float(win.median()), 4),
        "mean_losers": None if lose.empty else round(float(lose.mean()), 4),
        "median_losers": None if lose.empty else round(float(lose.median()), 4),
        "loser_minus_winner_mean": (
            None
            if win.empty or lose.empty
            else round(float(lose.mean() - win.mean()), 4)
        ),
        "spearman_vs_gain": _spearman(x, g),
        "pearson_vs_gain": _corr(x, g),
        "point_biserial_vs_loser": _corr(x, (g <= 0).astype(float)),
        "frac_zero": None if x.empty else round(float((x <= 0).mean()), 4),
    }


def annotate_trades(trades: pd.DataFrame, panels: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = trades.copy()
    skip_col = []
    ratio_rows: List[dict] = []
    index_cache: Dict[str, dict] = {}
    for _, row in out.iterrows():
        sym = str(row["stock"]).upper()
        df = panels.get(sym)
        if df is None or df.empty:
            skip_col.append("no_panel")
            ratio_rows.append({})
            continue
        if sym not in index_cache:
            index_cache[sym] = date_to_first_index(df.index)
        stats, skip = areas_for_trade(row, df, date_index=index_cache[sym])
        skip_col.append(skip or "")
        ratio_rows.append(stats or {})
    extra = pd.DataFrame(ratio_rows, index=out.index)
    for col in extra.columns:
        out[col] = extra[col]
    out["outside_area_skip"] = skip_col
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--symbols", type=str, default="")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_all = time.perf_counter()
    trades = pd.read_csv(args.trades)
    trades["stock"] = trades["stock"].astype(str).str.upper()
    smoke = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if smoke:
        trades = trades.loc[trades["stock"].isin(smoke)].copy()
        LOG.info("Smoke symbols %s n=%d", ",".join(smoke), len(trades))
    if trades.empty:
        LOG.error("No trades")
        return 1

    symbols = sorted({str(s).upper() for s in trades["stock"].tolist()})
    start = pd.to_datetime(trades["channel_start"], errors="coerce").min()
    if pd.isna(start):
        start = pd.Timestamp("2006-01-01")
    start = min(pd.Timestamp("2006-01-01"), pd.Timestamp(start) - pd.Timedelta(days=5))
    end = pd.to_datetime(trades["buy_date"], errors="coerce").max()
    if pd.isna(end):
        end = pd.Timestamp("2026-09-12")
    end = pd.Timestamp(end) + pd.Timedelta(days=5)
    LOG.info(
        "Trades n=%d symbols=%d daily %s -> %s",
        len(trades),
        len(symbols),
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    t0 = time.perf_counter()
    panels = load_ohlcv_many(
        symbols,
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2006, 1, 1),
        end=end.to_pydatetime(),
        fallback_provider="IB",
        merge_mode="prefix",
        workers=max(1, int(args.workers)),
        use_cache=True,
    )
    n_have = sum(1 for s in symbols if panels.get(s) is not None and not panels[s].empty)
    load_s = time.perf_counter() - t0
    LOG.info("Loaded ALPACA+IB 1d %d/%d in %.1fs", n_have, len(symbols), load_s)

    t1 = time.perf_counter()
    out = annotate_trades(trades, panels)
    LOG.info("Ratios in %.1fs", time.perf_counter() - t1)

    skip_counts = out["outside_area_skip"].fillna("").replace("", "ok").value_counts()
    print("---- coverage ----")
    print(skip_counts.to_string())

    ok = out.loc[out["outside_area_skip"].fillna("") == ""].copy()
    if ok.empty:
        LOG.error("No scored trades")
        return 1

    if smoke:
        cols = [
            "stock",
            "buy_date",
            GAIN_COL,
            RATIO_COL,
            PEAK_COL,
            "outside_below_low_area",
            "channel_area",
            "n_bars",
            "n_bars_below_low",
        ]
        print(ok[cols].to_string(index=False))
        return 0

    baseline = _print_block("baseline_scored", ok)
    wl = _winner_loser_ratio_stats(ok, RATIO_COL)
    peak = _winner_loser_ratio_stats(ok, PEAK_COL)
    beyond = (
        _winner_loser_ratio_stats(ok, "max_beyond_width")
        if "max_beyond_width" in ok.columns
        else {}
    )
    print("---- winner vs loser (below-support AREA ratio) ----")
    print(json.dumps(wl, indent=2))
    print("---- winner vs loser (peak undershoot / width) ----")
    print(json.dumps(peak, indent=2))
    if beyond:
        print("---- winner vs loser (existing max_beyond_width, overshoot) ----")
        print(json.dumps(beyond, indent=2))

    q_area = _quintile_rows(ok, RATIO_COL)
    q_peak = _quintile_rows(ok, PEAK_COL)
    print("---- quintiles AREA ratio ----")
    print(q_area.to_string(index=False))
    print("---- quintiles peak undershoot ----")
    print(q_peak.to_string(index=False))

    skip_rows: List[dict] = []
    ratio = _num(ok[RATIO_COL])
    overlays = [
        ("any_undershoot", ratio <= 0),
        ("ratio_le_median", ratio <= float(ratio.median())),
    ]
    pos = ratio[ratio > 0]
    if len(pos) >= 20:
        overlays.append(("drop_top_pos_quintile", ratio <= float(pos.quantile(0.80))))
    peak_s = _num(ok[PEAK_COL])
    overlays.append(("max_undershoot_le_0.25", peak_s <= 0.25))
    overlays.append(("max_undershoot_le_0.50", peak_s <= 0.50))

    print("---- skip overlays (occupancy not re-walked) ----")
    for label, keep in overlays:
        kept = ok.loc[keep.fillna(False)]
        cut = winner_cut_skip(_num(ok[GAIN_COL]), keep.fillna(False))
        block = _print_block("skip_%s" % label, kept)
        block.update({"overlay": label, **cut})
        skip_rows.append(block)
        print("  cut winners_dropped=%s losers_dropped=%s winner$=%s loser$=%s blunt=%s" % (
            cut.get("winners_dropped"),
            cut.get("losers_dropped"),
            cut.get("winner_drop_gain_sum"),
            cut.get("loser_drop_gain_sum"),
            cut.get("blunt"),
        ))

    outdir = dated_outdir(args.outdir)
    scored_path = outdir / "channel_outside_area_trades.csv"
    out.to_csv(scored_path, index=False)
    q_area.to_csv(outdir / "channel_outside_area_quintiles.csv", index=False)
    q_peak.to_csv(outdir / "channel_outside_area_peak_quintiles.csv", index=False)
    pd.DataFrame(skip_rows).to_csv(outdir / "channel_outside_area_skip_overlays.csv", index=False)
    summary = {
        "n_trades": int(len(trades)),
        "n_scored": int(len(ok)),
        "load_s": round(load_s, 1),
        "wall_s": round(time.perf_counter() - t_all, 1),
        "baseline": baseline,
        "area_ratio": wl,
        "peak_undershoot": peak,
        "max_beyond_width": beyond,
        "skip_counts": {str(k): int(v) for k, v in skip_counts.items()},
    }
    (outdir / "channel_outside_area_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    LOG.info("Wrote %s (wall %.1fs)", scored_path, time.perf_counter() - t_all)
    print("Wrote %s" % scored_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
