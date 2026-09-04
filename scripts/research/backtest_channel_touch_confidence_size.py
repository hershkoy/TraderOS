#!/usr/bin/env python3
"""Ridge P&L confidence sizing for channel-touch (not a hard entry gate).

Expanding walk-forward with purge+embargo. Train on quality-filtered fills;
evaluate stitched OOS equal-dollar vs scale-all vs skip-negative, plus RS-top1.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_confidence_size.py --stack 1d
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\backtest_channel_touch_confidence_size.py --stack 15m
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    apply_friction,
    filter_trades,
    select_same_day_rs,
)
from utils.research.channel_touch_entry_model import (  # noqa: E402
    DEFAULT_EMBARGO_DAYS,
    DEFAULT_RIDGE_L2,
    RIDGE_FEATURES_15M,
    RIDGE_FEATURES_1D,
    confidence_size_verdict,
    expanding_ridge_walk_forward,
    pred_quintile_table,
    purged_kfold_ridge,
    stitched_oos_metrics,
)
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backtest_channel_touch_confidence_size")

STACKS: Dict[str, Dict[str, Any]] = {
    "1d": {
        "raw": resolve_artifact("channel_touch_trades_raw_20260828_194314.csv"),
        "friction_pct": 0.25,
        "feature_cols": RIDGE_FEATURES_1D,
        "min_wait_bars": 6,
        "filter_kwargs": {
            "require_in_channel": True,
            "max_channel_span_days": 365.0,
            "max_beyond_width": 0.25,
            "max_rsi": 50.0,
        },
    },
    "15m": {
        "raw": resolve_artifact("channel_touch_15m_trades_raw_20260829_105511.csv"),
        "friction_pct": 0.10,
        "feature_cols": RIDGE_FEATURES_15M,
        "min_wait_bars": 12,
        "filter_kwargs": {
            "require_in_channel": True,
            "max_channel_span_days": 10.0,
            "max_beyond_width": None,
            "max_rsi": None,
        },
    },
}


def _load_quality(stack: str, trades_path: Optional[Path]) -> pd.DataFrame:
    cfg = STACKS[stack]
    path = Path(trades_path) if trades_path is not None else Path(cfg["raw"])
    if not path.exists():
        raise FileNotFoundError(f"trades CSV not found: {path}")
    df = pd.read_csv(path)
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    kw = {k: v for k, v in cfg["filter_kwargs"].items() if v is not None}
    out = filter_trades(df, **kw)
    min_wait = int(cfg["min_wait_bars"])
    if min_wait and "wait_bars" in out.columns:
        wb = pd.to_numeric(out["wait_bars"], errors="coerce")
        out = out.loc[wb >= min_wait].copy()
    friction = float(cfg["friction_pct"])
    if "gain_pct_net" not in out.columns:
        out = apply_friction(out, friction)
    logger.info(
        "%s quality-filtered n=%d from %s (friction=%.2f)",
        stack,
        len(out),
        path.name,
        friction,
    )
    return out.reset_index(drop=True)


def _fmt_metrics(label: str, m: dict) -> str:
    pf = m.get("profit_factor")
    pf_s = "inf" if pf is None else f"{pf:.3f}"
    e = m.get("expectancy")
    e_s = "n/a" if e is None else f"{e:+.3f}"
    return (
        f"{label}: n={m.get('n_trades')} active={m.get('n_active')} "
        f"E={e_s} PF={pf_s} MDD={m.get('max_dd')} sum={m.get('sum_pnl')} "
        f"mean_size={m.get('mean_size')} WR={m.get('win_rate_pct')}"
    )


def _print_block(title: str, stitched: dict) -> None:
    print(f"\n==== {title} ====")
    print(_fmt_metrics("equal-dollar", stitched["equal"]))
    print(_fmt_metrics("scale-all", stitched["scale_all"]))
    print(_fmt_metrics("skip-neg", stitched["skip_neg"]))
    print(_fmt_metrics("scale-all mean-norm (diagnostic)", stitched["scale_all_norm"]))
    print(_fmt_metrics("scale-all drop-top-3", stitched["scale_drop3"]))
    print(f"spearman pred vs actual: {stitched['spearman']}")


def _summary_rows(view: str, stitched: dict, verdict: str) -> list:
    rows = []
    for kind in ("equal", "scale_all", "skip_neg", "scale_all_norm", "scale_drop3"):
        m = stitched[kind]
        rows.append(
            {
                "view": view,
                "variant": kind,
                "verdict": verdict,
                "spearman": stitched["spearman"],
                **m,
            }
        )
    return rows


def run_stack(
    stack: str,
    *,
    trades_path: Optional[Path],
    outdir: Path,
    embargo_days: int,
    l2: float,
    kfold: int,
) -> int:
    cfg = STACKS[stack]
    qf = _load_quality(stack, trades_path)
    if qf.empty:
        logger.error("no quality-filtered trades")
        return 1
    feature_cols = cfg["feature_cols"]
    folds, oos = expanding_ridge_walk_forward(
        qf,
        feature_cols,
        embargo_days=embargo_days,
        l2=l2,
    )
    if oos.empty:
        logger.error("walk-forward produced no OOS folds (n_train/n_test floors)")
        return 1

    stitched_qf = stitched_oos_metrics(oos)
    verdict_qf = confidence_size_verdict(stitched_qf, folds)
    rs = select_same_day_rs(oos, rs_col="rs_spy_126d", max_per_day=1)
    stitched_rs = stitched_oos_metrics(rs)
    verdict_rs = confidence_size_verdict(stitched_rs, folds)

    kfold_df = purged_kfold_ridge(
        qf, feature_cols, n_splits=int(kfold), embargo_days=embargo_days, l2=l2
    )
    quint = pred_quintile_table(
        oos["pred_pnl"].to_numpy(),
        oos["gain_pct_net"].to_numpy() if "gain_pct_net" in oos.columns else oos["gain_pct"].to_numpy(),
    )

    _print_block(f"{stack} quality-filtered stitched OOS", stitched_qf)
    print(f"verdict (quality-filtered): {verdict_qf}")
    _print_block(f"{stack} RS-top1 stitched OOS", stitched_rs)
    print(f"verdict (RS-top1): {verdict_rs}")
    print("\n==== expanding WF folds (gate) ====")
    print(folds.to_string(index=False) if not folds.empty else "(none)")
    print("\n==== purged 5-fold diagnostic (not the gate; may use future regimes) ====")
    print(kfold_df.to_string(index=False) if not kfold_df.empty else "(none)")
    if not quint.empty:
        print("\n==== OOS pred quintiles vs realized E (diagnostic) ====")
        print(quint.to_string(index=False))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / f"channel_touch_confidence_size_{stack}_{stamp}"
    folds.to_csv(Path(str(prefix) + "_folds.csv"), index=False)
    oos.to_csv(Path(str(prefix) + "_oos.csv"), index=False)
    rs.to_csv(Path(str(prefix) + "_oos_rstop1.csv"), index=False)
    if not kfold_df.empty:
        kfold_df.to_csv(Path(str(prefix) + "_kfold.csv"), index=False)
    if not quint.empty:
        quint.to_csv(Path(str(prefix) + "_quintiles.csv"), index=False)
    summary = pd.DataFrame(
        _summary_rows("quality_filtered", stitched_qf, verdict_qf)
        + _summary_rows("rs_top1", stitched_rs, verdict_rs)
    )
    summary.to_csv(Path(str(prefix) + "_summary.csv"), index=False)
    print(f"\nwrote {prefix}_*.csv")
    print("Nightly scanner unchanged.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Channel-touch ridge confidence sizing (walk-forward)")
    ap.add_argument("--stack", choices=("1d", "15m", "both"), default="both")
    ap.add_argument("--trades", type=Path, default=None, help="Override raw CSV (single stack only)")
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    ap.add_argument("--embargo-days", type=int, default=DEFAULT_EMBARGO_DAYS)
    ap.add_argument("--l2", type=float, default=DEFAULT_RIDGE_L2)
    ap.add_argument("--kfold", type=int, default=5)
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)
    stacks = ("1d", "15m") if args.stack == "both" else (args.stack,)
    if args.trades is not None and len(stacks) != 1:
        logger.error("--trades requires a single --stack (1d or 15m)")
        return 2
    rc = 0
    for st in stacks:
        rc = max(rc, run_stack(
            st,
            trades_path=args.trades,
            outdir=args.outdir,
            embargo_days=int(args.embargo_days),
            l2=float(args.l2),
            kfold=int(args.kfold),
        ))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
