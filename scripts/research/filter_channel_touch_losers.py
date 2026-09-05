#!/usr/bin/env python3
"""Regression keep/skip filter on the current_best 1d H2 unique-symbol book.

Entry features are already on the trades CSV. Fill-day daily close (RSI, %B,
close_loc, SMA distance, same-session volume) is a leak for next-mid fills;
honest features are rails/wait/calendar/extra plus prior-session SPY.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\filter_channel_touch_losers.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import YEAR_BUCKETS, summarize_by_year  # noqa: E402
from utils.research.channel_touch_entry_features import enrich_spy_entry_features  # noqa: E402
from utils.research.channel_touch_entry_model import (  # noqa: E402
    DEFAULT_CUTOFF,
    HONEST_1D_GEOM_FEATURES,
    HONEST_1D_NEXTMID_FEATURES,
    LEAKY_1D_CLOSE_FEATURES,
    add_loser_filter_columns,
    expanding_keep_walk_forward,
    keep_filter_verdict,
    stitched_keep_metrics,
    time_split,
    trade_metrics,
)
from utils.research.report_paths import dated_outdir, resolve_artifact  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("filter_channel_touch_losers")

DEFAULT_TRADES = "channel_touch_full_h2_break_span365_unique_20260905_135126.csv"

FEATURE_SETS: Dict[str, Sequence[str]] = {
    "honest_geom": HONEST_1D_GEOM_FEATURES,
    "honest_full": HONEST_1D_NEXTMID_FEATURES,
    "leaky_close": LEAKY_1D_CLOSE_FEATURES,
}

MODELS = (
    ("ridge", "skip0"),
    ("ridge", "thr"),
    ("logistic", "skip0"),
    ("logistic", "thr"),
)


def _fmt(m: dict, *, prefix: str = "") -> str:
    pf = m.get("profit_factor")
    pf_s = "n/a" if pf is None else f"{pf:.3f}"
    e = m.get("expectancy_pct", m.get("expectancy"))
    e_s = "n/a" if e is None else f"{e:+.3f}"
    wr = m.get("win_rate_pct")
    wr_s = "n/a" if wr is None else f"{wr:.1f}"
    n = m.get("n_trades", m.get("n_active"))
    return f"{prefix}n={n} E={e_s} PF={pf_s} WR={wr_s}"


def _year_block(df: pd.DataFrame, title: str) -> None:
    print(f"\n==== {title} years ====")
    if df.empty:
        print("(empty)")
        return
    print(summarize_by_year(df, gain_col="gain_pct_net", buckets=YEAR_BUCKETS).to_string(index=False))


def _attach_prior_session_spy(trades: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """SPY regime as-of the prior calendar day (next-mid does not wait for that session's close)."""
    work = trades.copy()
    bd = pd.to_datetime(work["buy_date"], errors="coerce")
    work["feature_asof"] = (bd - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    spy_part = enrich_spy_entry_features(work, spy_df)
    out = trades.copy()
    for col in ("spy_ret_20d", "spy_above_sma50", "spy_atr_pct"):
        out[col] = spy_part[col]
    return out


def _load_spy() -> Optional[pd.DataFrame]:
    try:
        from utils.data.ohlcv_loader import load_ohlcv_many
    except Exception as exc:
        logger.warning("Cannot import ohlcv_loader: %s", exc)
        return None
    panels = load_ohlcv_many(
        ["SPY"],
        timeframe="1d",
        provider="ALPACA",
        start=datetime(2018, 1, 1),
        end=datetime(2026, 9, 3),
        fallback_provider="IB",
        merge_mode="prefix",
        workers=1,
        use_cache=True,
    )
    spy = panels.get("SPY")
    if spy is None or spy.empty:
        logger.warning("No SPY panel; honest_full will miss spy_* columns")
        return None
    logger.info("SPY %s -> %s n=%d", spy.index.min().date(), spy.index.max().date(), len(spy))
    return spy


def _spearman_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    g = pd.to_numeric(df["gain_pct_net"], errors="coerce")
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({"feature": c, "n": 0, "spearman_vs_gain": None, "present": False})
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & g.notna()
        n = int(mask.sum())
        rho = None
        if n >= 20:
            raw_rho = float(x[mask].corr(g[mask], method="spearman"))
            if np.isfinite(raw_rho):
                rho = round(raw_rho, 4)
        rows.append(
            {
                "feature": c,
                "n": n,
                "spearman_vs_gain": rho,
                "present": True,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_abs"] = pd.to_numeric(out["spearman_vs_gain"], errors="coerce").abs()
    return out.sort_values("_abs", ascending=False, na_position="last").drop(columns="_abs")


def _holdout_row(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    kind: str,
    rule: str,
    cutoff: str,
) -> dict:
    train_df, test_df = time_split(df, cutoff)
    from utils.research.channel_touch_entry_model import fit_keep_fold

    scores, keep0, keep_thr, thr, n_feat = fit_keep_fold(
        train_df, test_df, feature_cols, kind=kind
    )
    keep = keep0 if rule == "skip0" else keep_thr
    g = pd.to_numeric(test_df["gain_pct_net"], errors="coerce").to_numpy(dtype=float)
    return {
        "name": f"holdout {kind} {rule} {cutoff}",
        "rule": rule,
        "kind": kind,
        "cutoff": cutoff,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_kept": int(keep.sum()) if len(keep) else 0,
        "n_features": n_feat,
        "threshold": round(float(thr), 4),
        **{f"all_{k}": v for k, v in trade_metrics(g).items()},
        **{f"kept_{k}": v for k, v in trade_metrics(g[keep] if len(g) else g).items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter losing 1d H2 trades with a time-split regression")
    ap.add_argument("--trades", type=Path, default=None)
    ap.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    ap.add_argument("--no-spy", action="store_true")
    ap.add_argument("--outdir", type=Path, default=ROOT / "reports" / "ascending_channels")
    args = ap.parse_args()
    args.outdir = dated_outdir(args.outdir)

    path = Path(args.trades) if args.trades is not None else resolve_artifact(DEFAULT_TRADES)
    if not path.exists():
        logger.error("Trades CSV not found: %s", path)
        return 1
    df = pd.read_csv(path)
    df["buy_date"] = pd.to_datetime(df["buy_date"], errors="coerce")
    if "gain_pct_net" not in df.columns:
        logger.error("need gain_pct_net")
        return 1
    df = add_loser_filter_columns(df)
    n_extra = int((df["is_extra"] > 0).sum()) if "is_extra" in df.columns else 0
    logger.info("Loaded n=%d extras=%d from %s", len(df), n_extra, path.name)
    logger.info(
        "feature_asof sample=%s buy_time=%s",
        df["feature_asof"].head(1).tolist() if "feature_asof" in df.columns else None,
        "buy_time" in df.columns,
    )

    if not args.no_spy:
        spy = _load_spy()
        if spy is not None:
            df = _attach_prior_session_spy(df, spy)

    base = trade_metrics(pd.to_numeric(df["gain_pct_net"], errors="coerce").to_numpy(dtype=float))
    print("\n==== current_best unique 1d (all) ====")
    print(_fmt(base))
    _year_block(df, "all")

    spear_rows: List[pd.DataFrame] = []
    for set_name, cols in FEATURE_SETS.items():
        table = _spearman_table(df, cols)
        table.insert(0, "feature_set", set_name)
        spear_rows.append(table)
        print(f"\n==== Spearman vs gain ({set_name}) ====")
        print(table.to_string(index=False))

    fold_parts: List[pd.DataFrame] = []
    holdout_rows: List[dict] = []
    verdict_rows: List[dict] = []
    oos_by_key: Dict[str, pd.DataFrame] = {}

    for set_name, cols in FEATURE_SETS.items():
        present = [c for c in cols if c in df.columns]
        logger.info("%s features present %d/%d: %s", set_name, len(present), len(cols), present)
        for kind, rule in MODELS:
            folds, oos = expanding_keep_walk_forward(df, present, kind=kind, rule=rule)
            key = f"{set_name}|{kind}|{rule}"
            if not folds.empty:
                folds = folds.copy()
                folds.insert(0, "feature_set", set_name)
                folds.insert(1, "kind", kind)
                fold_parts.append(folds)
            oos_by_key[key] = oos
            stitched = stitched_keep_metrics(oos)
            verdict = keep_filter_verdict(stitched, folds)
            verdict_rows.append(
                {
                    "key": key,
                    "feature_set": set_name,
                    "kind": kind,
                    "rule": rule,
                    "verdict": verdict,
                    "oos_n": stitched["all"].get("n_trades"),
                    "oos_E": stitched["all"].get("expectancy_pct"),
                    "oos_PF": stitched["all"].get("profit_factor"),
                    "kept_n": stitched["kept"].get("n_trades"),
                    "kept_E": stitched["kept"].get("expectancy_pct"),
                    "kept_PF": stitched["kept"].get("profit_factor"),
                    "dropped_n": stitched["dropped"].get("n_trades"),
                    "dropped_E": stitched["dropped"].get("expectancy_pct"),
                    "keep_frac": stitched.get("keep_frac"),
                    "spearman": stitched.get("spearman"),
                    "kept_drop3_E": stitched["kept_drop3"].get("expectancy"),
                    "kept_drop3_PF": stitched["kept_drop3"].get("profit_factor"),
                }
            )
            print(f"\n==== expanding WF {key} verdict={verdict} ====")
            print("OOS all  ", _fmt(stitched["all"]))
            print("OOS kept ", _fmt(stitched["kept"]), f"frac={stitched.get('keep_frac')}")
            print("OOS drop ", _fmt(stitched["dropped"]))
            print("drop-top3", _fmt(stitched["kept_drop3"]))
            print("spearman ", stitched.get("spearman"))
            if not folds.empty:
                print(folds.to_string(index=False))
            if not oos.empty and "keep" in oos.columns:
                _year_block(oos.loc[oos["keep"]], f"{key} OOS kept")

            holdout_rows.append(
                {
                    "feature_set": set_name,
                    **_holdout_row(df, present, kind=kind, rule=rule, cutoff=args.cutoff),
                }
            )

    print("\n==== holdout", args.cutoff, "====")
    hold_df = pd.DataFrame(holdout_rows)
    if not hold_df.empty:
        show = [
            c
            for c in (
                "feature_set",
                "name",
                "n_train",
                "n_test",
                "n_kept",
                "all_expectancy_pct",
                "all_profit_factor",
                "kept_expectancy_pct",
                "kept_profit_factor",
                "kept_n_trades",
            )
            if c in hold_df.columns
        ]
        print(hold_df[show].to_string(index=False))

    print("\n==== verdicts ====")
    verd_df = pd.DataFrame(verdict_rows)
    print(verd_df.to_string(index=False))

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    spear_path = args.outdir / f"channel_touch_1d_loser_filter_spearman_{stamp}.csv"
    fold_path = args.outdir / f"channel_touch_1d_loser_filter_folds_{stamp}.csv"
    verd_path = args.outdir / f"channel_touch_1d_loser_filter_verdict_{stamp}.csv"
    hold_path = args.outdir / f"channel_touch_1d_loser_filter_holdout_{stamp}.csv"
    pd.concat(spear_rows, ignore_index=True).to_csv(spear_path, index=False)
    if fold_parts:
        pd.concat(fold_parts, ignore_index=True).to_csv(fold_path, index=False)
    verd_df.to_csv(verd_path, index=False)
    hold_df.to_csv(hold_path, index=False)
    best_key = None
    research = verd_df.loc[verd_df["verdict"] == "research_only"] if not verd_df.empty else verd_df
    if research is not None and not research.empty:
        best_key = str(research.iloc[0]["key"])
        oos = oos_by_key.get(best_key)
        if oos is not None and not oos.empty:
            oos_path = args.outdir / f"channel_touch_1d_loser_filter_oos_{stamp}.csv"
            oos.to_csv(oos_path, index=False)
            print("wrote", oos_path)
    print("wrote", spear_path)
    print("wrote", fold_path)
    print("wrote", verd_path)
    print("wrote", hold_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
