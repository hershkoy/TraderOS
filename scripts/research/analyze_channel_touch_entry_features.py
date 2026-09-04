#!/usr/bin/env python3
"""Univariate entry-feature mining for channel-touch trades.

Quintile (or category) expectancy/PF, Spearman vs gain, pairwise feature corr.
No ML. Does not promote a filter by itself.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\analyze_channel_touch_entry_features.py --trades reports\\ascending_channels\\channel_touch_trades_raw_STAMP.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import apply_friction
from utils.research.channel_touch_entry_features import CATEGORICAL_FEATURES, FEATURE_COLS
from utils.research.report_paths import dated_outdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyze_channel_touch_entry_features")

GAIN_CANDIDATES = ("gain_pct_net", "gain_pct")
ALWAYS_FEATURES = (
    "atr_pct",
    "adv_20",
    "channel_pos",
    "channel_width_pct",
    "slope_pct_per_bar",
    "room_to_resist_pct",
    "rs_spy_63d",
    "rs_spy_126d",
    "touch_num",
)


def _gain_col(df: pd.DataFrame) -> str:
    for c in GAIN_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("trades CSV needs gain_pct or gain_pct_net")


def _feature_list(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in list(FEATURE_COLS) + list(ALWAYS_FEATURES):
        if c in df.columns and c not in cols:
            cols.append(c)
    return cols


def _bucket_stats(g: pd.Series) -> dict:
    g = g.astype(float)
    n = int(len(g))
    if n == 0:
        return {
            "n": 0,
            "win_rate_pct": None,
            "expectancy_pct": None,
            "median_pct": None,
            "profit_factor": None,
        }
    wins = g[g > 0]
    losses = g[g <= 0]
    wr = float((g > 0).mean() * 100.0)
    exp = float(g.mean())
    med = float(g.median())
    gp = float(wins.sum()) if len(wins) else 0.0
    gl = float((-losses).sum()) if len(losses) else 0.0
    if gl > 1e-12:
        pf = gp / gl
    elif gp > 0:
        pf = float("inf")
    else:
        pf = 0.0
    return {
        "n": n,
        "win_rate_pct": round(wr, 2),
        "expectancy_pct": round(exp, 3),
        "median_pct": round(med, 3),
        "profit_factor": None if not np.isfinite(pf) else round(float(pf), 3),
    }


def _quintile_table(df: pd.DataFrame, col: str, gain_col: str, n_bins: int = 5) -> pd.DataFrame:
    s = pd.to_numeric(df[col], errors="coerce")
    g = pd.to_numeric(df[gain_col], errors="coerce")
    ok = s.notna() & g.notna()
    work = pd.DataFrame({"x": s[ok], "gain": g[ok]})
    rows = []
    if work.empty:
        return pd.DataFrame()
    is_cat = col in CATEGORICAL_FEATURES or work["x"].nunique(dropna=True) <= 8
    if is_cat:
        grouped = work.groupby(work["x"], dropna=False)
        for key, part in grouped:
            stats = _bucket_stats(part["gain"])
            rows.append({"feature": col, "bucket": str(key), "bucket_kind": "value", **stats})
        return pd.DataFrame(rows)
    try:
        work["bucket"] = pd.qcut(work["x"], q=int(n_bins), duplicates="drop")
    except (ValueError, TypeError):
        work["bucket"] = pd.cut(work["x"], bins=min(int(n_bins), max(2, work["x"].nunique())))
    for key, part in work.groupby("bucket", observed=False):
        stats = _bucket_stats(part["gain"])
        rows.append({"feature": col, "bucket": str(key), "bucket_kind": "quantile", **stats})
    return pd.DataFrame(rows)


def _spearman_vs_gain(df: pd.DataFrame, cols: Sequence[str], gain_col: str) -> pd.DataFrame:
    g = pd.to_numeric(df[gain_col], errors="coerce")
    rows = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & g.notna()
        n = int(mask.sum())
        if n < 20:
            rho = None
        else:
            rho = float(x[mask].corr(g[mask], method="spearman"))
            if not np.isfinite(rho):
                rho = None
        rows.append({"feature": c, "n": n, "spearman_vs_gain": None if rho is None else round(rho, 4)})
    out = pd.DataFrame(rows)
    if not out.empty:
        rho_num = pd.to_numeric(out["spearman_vs_gain"], errors="coerce")
        out = out.assign(_abs=rho_num.abs())
        out = out.sort_values("_abs", ascending=False, na_position="last").drop(columns="_abs")
    return out


def _pairwise_corr(df: pd.DataFrame, cols: Sequence[str], thresh: float = 0.7) -> pd.DataFrame:
    num = df[list(cols)].apply(pd.to_numeric, errors="coerce")
    corr = num.corr(method="spearman")
    rows = []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            v = corr.loc[a, b]
            if pd.isna(v):
                continue
            fv = float(v)
            if abs(fv) >= float(thresh):
                rows.append({"feature_a": a, "feature_b": b, "spearman": round(fv, 4)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["_abs"] = out["spearman"].abs()
        out = out.sort_values("_abs", ascending=False).drop(columns="_abs")
    return out


def analyze_entry_features(
    trades: pd.DataFrame,
    *,
    friction_pct: float = 0.0,
    n_bins: int = 5,
    corr_thresh: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = trades
    if friction_pct and friction_pct > 0 and "gain_pct_net" not in out.columns and "gain_pct" in out.columns:
        out = apply_friction(out, friction_pct)
    gain_col = _gain_col(out)
    cols = _feature_list(out)
    buckets = []
    for c in cols:
        part = _quintile_table(out, c, gain_col, n_bins=n_bins)
        if not part.empty:
            buckets.append(part)
    bucket_df = pd.concat(buckets, ignore_index=True) if buckets else pd.DataFrame()
    spearman = _spearman_vs_gain(out, cols, gain_col)
    clones = _pairwise_corr(out, cols, thresh=corr_thresh)
    return bucket_df, spearman, clones


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel-touch entry-feature univariate analysis")
    ap.add_argument("--trades", required=True, type=Path, help="Raw or filtered trades CSV")
    ap.add_argument("--friction-pct", type=float, default=0.25)
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--corr-thresh", type=float, default=0.7)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)
    if not args.trades.exists():
        logger.error("Trades file not found: %s", args.trades)
        return 1
    df = pd.read_csv(args.trades)
    if df.empty:
        logger.warning("Empty trades file")
        return 0
    logger.info("Loaded %d trades from %s", len(df), args.trades)
    buckets, spearman, clones = analyze_entry_features(
        df,
        friction_pct=float(args.friction_pct),
        n_bins=int(args.n_bins),
        corr_thresh=float(args.corr_thresh),
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    b_csv = args.outdir / f"channel_touch_entry_features_buckets_{stamp}.csv"
    s_csv = args.outdir / f"channel_touch_entry_features_spearman_{stamp}.csv"
    c_csv = args.outdir / f"channel_touch_entry_features_corr_{stamp}.csv"
    if not buckets.empty:
        buckets.to_csv(b_csv, index=False)
    if not spearman.empty:
        spearman.to_csv(s_csv, index=False)
    if not clones.empty:
        clones.to_csv(c_csv, index=False)
    print("Spearman vs gain (top 15):")
    print(spearman.head(15).to_string(index=False) if not spearman.empty else "(none)")
    print("\nHigh pairwise |corr| >= %.2f:" % args.corr_thresh)
    print(clones.to_string(index=False) if not clones.empty else "(none)")
    print("\nWrote:")
    print(f"  {b_csv}")
    print(f"  {s_csv}")
    print(f"  {c_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
