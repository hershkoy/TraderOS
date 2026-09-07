#!/usr/bin/env python3
"""Year-split ridge: predict close-cross MAE from confirm-bar features.

Train years pick L2 and a pad on predicted MAE; later years are the test.
Stop path: if realized MAE < clipped predicted stop, take trail_only_gain_pct;
else stop at -predicted. Gate is E/PF (not win rate). Not a promote.

Usage (Windows CMD):
  venv\\Scripts\\activate
  set PYTHONPATH=.
  python scripts\\research\\fit_close_cross_mae_regression.py --csv reports\\ascending_channels\\2026-09-07\\channel_touch_close_cross_mae_features_STAMP.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.research.channel_touch_entry_model import time_split, trade_metrics
from utils.research.close_cross_mae_regression import (
    DEFAULT_CUTOFF,
    MAE_FEATURE_COLS,
    filled_mae_rows,
    fit_mae_ridge,
    grid_select,
    present_features,
    stop_path_pnl,
    year_fold_table,
)
from utils.research.report_paths import dated_outdir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fit_close_cross_mae")


def _median_stop_pnl(train: pd.DataFrame, test: pd.DataFrame, pad: float) -> dict:
    med = float(pd.to_numeric(train["mae_pct"], errors="coerce").median())
    stop = max(1.5, min(8.0, med * float(pad)))
    mae = pd.to_numeric(test["mae_pct"], errors="coerce").to_numpy(dtype=float)
    trail = pd.to_numeric(test["trail_only_gain_pct"], errors="coerce").to_numpy(dtype=float)
    pnl = stop_path_pnl(mae, trail, stop)
    out = trade_metrics(pnl)
    out["median_mae"] = round(med, 4)
    out["stop_pct"] = round(stop, 4)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Year-split ridge on close-cross MAE features")
    ap.add_argument("--csv", required=True, help="MAE feature CSV from --trail-mae close-cross")
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF, help="Train strictly before this date")
    ap.add_argument(
        "--inner-cutoff",
        default="2022-01-01",
        help="Inner val start inside train (l2/pad grid)",
    )
    ap.add_argument("--l2", type=float, default=None, help="Freeze L2 (skip inner grid)")
    ap.add_argument("--pad", type=float, default=None, help="Freeze MAE pad (skip inner grid)")
    ap.add_argument("--outdir", default="", help="Override dated reports folder")
    args = ap.parse_args()
    src = Path(args.csv)
    if not src.is_file():
        logger.error("CSV not found: %s", src)
        return 1
    raw = pd.read_csv(src)
    df = filled_mae_rows(raw)
    cols = present_features(df, MAE_FEATURE_COLS)
    n_sym = int(df["stock"].nunique()) if "stock" in df.columns else 0
    logger.info(
        "rows=%d fills=%d symbols=%d features=%d cutoff=%s",
        len(raw),
        len(df),
        n_sym,
        len(cols),
        args.cutoff,
    )
    if df.empty or len(cols) < 2:
        logger.error("Need filled MAE rows and at least two feature columns")
        return 1
    train, test = time_split(df, args.cutoff)
    logger.info("train n=%d test n=%d", len(train), len(test))
    if train.empty or test.empty:
        logger.error("Empty train or test after year split")
        return 1
    grid = None
    if args.l2 is not None and args.pad is not None:
        l2, pad = float(args.l2), float(args.pad)
        logger.info("frozen l2=%s pad=%s", l2, pad)
    else:
        l2, pad, grid = grid_select(train, feature_cols=cols, inner_cutoff=str(args.inner_cutoff))
        logger.info("selected l2=%s pad=%s (inner max test-stop E then PF)", l2, pad)
    fit = fit_mae_ridge(train, test, cols, l2=l2, pad=pad)
    folds = year_fold_table(df, feature_cols=cols, l2=l2, pad=pad)
    med_base = _median_stop_pnl(train, test, pad)
    mae3 = pd.to_numeric(test["mae_pct"], errors="coerce").to_numpy(dtype=float)
    trail3 = pd.to_numeric(test["trail_only_gain_pct"], errors="coerce").to_numpy(dtype=float)
    fixed3 = trade_metrics(stop_path_pnl(mae3, trail3, 3.0))
    fixed3["stop_pct"] = 3.0

    outdir = Path(args.outdir) if str(args.outdir or "").strip() else dated_outdir()
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = outdir / ("close_cross_mae_ridge_%s" % stamp)
    scored = test.copy()
    scored["pred_mae_pct"] = fit["pred_test"]
    scored["pred_stop_pct"] = fit["stop_test"]
    scored["stop_path_pnl_pct"] = fit["pnl_test"]
    scored_path = Path(str(stem) + "_holdout_scored.csv")
    grid_path = Path(str(stem) + "_grid.csv")
    folds_path = Path(str(stem) + "_year_folds.csv")
    summary_path = Path(str(stem) + "_summary.txt")
    scored.to_csv(scored_path, index=False)
    if grid is not None and not grid.empty:
        grid.to_csv(grid_path, index=False)
    folds.to_csv(folds_path, index=False)

    def _m(label: str, m: dict) -> str:
        return "%s n=%s E=%s PF=%s WR=%s med=%s" % (
            label,
            m.get("n_trades"),
            m.get("expectancy_pct"),
            m.get("profit_factor"),
            m.get("win_rate_pct"),
            m.get("median_pct"),
        )

    lines = [
        "close-cross MAE ridge (year split). Not a promote.",
        "csv=%s" % src,
        "fills=%d symbols=%d train<%s n=%d test n=%d" % (len(df), n_sym, args.cutoff, len(train), len(test)),
        "l2=%s pad=%s rmse_train=%s rmse_test=%s intercept=%s"
        % (l2, pad, fit["rmse_train"], fit["rmse_test"], fit["intercept"]),
        _m("holdout trail-only (no hard stop)", fit["test_trail"]),
        _m("holdout ridge stop", fit["test_stop"]),
        _m("holdout train-median MAE * pad stop", med_base),
        _m("holdout fixed 3pct stop", fixed3),
        "coefs (z-scored features, |w| desc):",
    ]
    for row in fit["coefs"][:12]:
        lines.append("  %s %s" % (row["feature"], row["weight"]))
    lines.append("year folds (frozen l2/pad):")
    lines.append(folds.to_string(index=False) if not folds.empty else "(none)")
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print(text)
    print("wrote %s" % scored_path)
    print("wrote %s" % summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
