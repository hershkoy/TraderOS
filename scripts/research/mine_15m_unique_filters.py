#!/usr/bin/env python3
"""Five new 15m unique-symbol filters + expanding-year logistic/ridge.

Uses the full-universe H2 span10 book after unique-symbol/day (not RS top1).
Derived features only; no rescan. Prior-bar snapshots already on the CSV.
Does not promote a gate by itself.

Usage (Windows CMD):
  venv\\Scripts\\activate && set PYTHONPATH=. && python scripts\\research\\mine_15m_unique_filters.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "research"))

from backtest_channel_touch_trades import (  # noqa: E402
    _summarize,
    apply_friction,
    keep_one_per_symbol_day,
)
from utils.research.channel_touch_entry_model import (  # noqa: E402
    LogisticScorer,
    feature_matrix,
    trade_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mine_15m_unique_filters")

DEFAULT_TRADES = (
    ROOT / "reports" / "ascending_channels" / "channel_touch_15m_full_h2_break_span10.csv"
)
FRICTION = 0.10
GAIN_COL = "gain_pct_net"

# Prior-bar / geometry only. No fill-bar close_loc. No calendar. No RS rank.
REG_FEATURES: Sequence[str] = (
    "rsi_14",
    "dist_sma50_pct",
    "overshoot",
    "atr_pct",
    "volume_rel_20",
    "squeeze_mom",
    "squeeze_mom_rising",
    "max_beyond_width",
    "rs_spy_126d",
    "wait_bars",
    "day_n_names",
    "buy_hour",
    "channel_width_pct",
    "range_pct",
    "sma50_gt_sma200",
    "wait_sessions",
)

NEW_UNIVARIATE: Sequence[str] = (
    "overshoot",
    "wait_bars",
    "wait_sessions",
    "buy_hour",
    "day_n_names",
    "rs_spy_126d",
    "atr_pct",
    "volume_rel_20",
    "squeeze_mom",
    "channel_pos",
    "channel_width_pct",
    "rsi_14",
    "max_beyond_width",
    "range_pct",
    "dist_sma50_pct",
)


def _fmt(s: dict) -> str:
    return "n=%s E=%s PF=%s WR=%s med=%s" % (
        s.get("n_trades"),
        s.get("expectancy_pct"),
        s.get("profit_factor"),
        s.get("win_rate_pct"),
        s.get("median_gain_pct") or s.get("median_pct"),
    )


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["buy_dt"] = pd.to_datetime(out["buy_date"], errors="coerce")
    out["_day"] = out["buy_dt"].dt.normalize()
    out["year"] = out["buy_dt"].dt.year
    pos = pd.to_numeric(out.get("channel_pos"), errors="coerce")
    out["overshoot"] = pos - 1.0
    wait = pd.to_numeric(out.get("wait_bars"), errors="coerce")
    out["wait_sessions"] = wait / 26.0
    if "buy_time" in out.columns:
        bt = pd.to_datetime(out["buy_time"], errors="coerce")
        out["buy_hour"] = bt.dt.hour + bt.dt.minute / 60.0
    else:
        bt = pd.to_datetime(out["buy_date"], errors="coerce")
        out["buy_hour"] = np.nan
    # Running unique names so far today (no full-day look-ahead on 15m).
    out["_t"] = bt
    out = out.sort_values(["_day", "_t", "stock"], kind="mergesort")
    first = out.drop_duplicates(["_day", "stock"], keep="first")
    first = first.assign(day_n_asof=first.groupby("_day", sort=False).cumcount() + 1)
    out = out.merge(first[["_day", "stock", "day_n_asof"]], on=["_day", "stock"], how="left")
    out["day_n_names"] = pd.to_numeric(out["day_n_asof"], errors="coerce")
    rs = pd.to_numeric(out.get("rs_spy_126d"), errors="coerce")
    out["rs_beat_spy"] = (rs >= 0).astype(float)
    vol = pd.to_numeric(out.get("volume_rel_20"), errors="coerce")
    out["vol_confirm"] = (vol >= 1.0).astype(float)
    sq = pd.to_numeric(out.get("squeeze_mom"), errors="coerce")
    out["squeeze_pos"] = (sq > 0).astype(float)
    return out


def quintiles(df: pd.DataFrame, col: str, n_bins: int = 5) -> pd.DataFrame:
    x = pd.to_numeric(df[col], errors="coerce")
    g = pd.to_numeric(df[GAIN_COL], errors="coerce")
    ok = x.notna() & g.notna()
    work = pd.DataFrame({"x": x[ok], "gain": g[ok]})
    if work.empty or work["x"].nunique() < 2:
        return pd.DataFrame()
    try:
        work["bucket"] = pd.qcut(work["x"], q=n_bins, duplicates="drop")
    except (ValueError, TypeError):
        return pd.DataFrame()
    rows = []
    for key, part in work.groupby("bucket", observed=False):
        m = trade_metrics(part["gain"].to_numpy())
        rows.append({"feature": col, "bucket": str(key), **m})
    return pd.DataFrame(rows)


def spearman_table(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    g = pd.to_numeric(df[GAIN_COL], errors="coerce")
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & g.notna()
        n = int(mask.sum())
        rho = None
        if n >= 50:
            rho = float(x[mask].corr(g[mask], method="spearman"))
            if not np.isfinite(rho):
                rho = None
        rows.append(
            {
                "feature": c,
                "n": n,
                "spearman": None if rho is None else round(rho, 4),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_abs"] = pd.to_numeric(out["spearman"], errors="coerce").abs()
    return out.sort_values("_abs", ascending=False, na_position="last").drop(columns="_abs")


MaskFn = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def expanding_year_apply(df: pd.DataFrame, mask_fn: MaskFn) -> pd.DataFrame:
    years = sorted(int(y) for y in df["year"].dropna().unique())
    parts: List[pd.DataFrame] = []
    for i, y in enumerate(years):
        if i == 0:
            continue
        train = df.loc[df["year"] < y]
        test = df.loc[df["year"] == y]
        if train.empty or test.empty:
            continue
        keep = mask_fn(train, test)
        parts.append(test.loc[keep.fillna(False)])
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def hyp_masks() -> List[Tuple[str, str, MaskFn]]:
    def h1(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(test["rs_spy_126d"], errors="coerce") >= 0

    def h2(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        pos = pd.to_numeric(test["channel_pos"], errors="coerce")
        return (pos >= 1.0) & (pos <= 1.10)

    def h3(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        atr_tr = pd.to_numeric(train["atr_pct"], errors="coerce")
        cap = float(atr_tr.quantile(0.60))
        return pd.to_numeric(test["atr_pct"], errors="coerce") <= cap

    def h4(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        crowd = pd.to_numeric(train["day_n_names"], errors="coerce")
        cap = float(crowd.quantile(0.75))
        return pd.to_numeric(test["day_n_names"], errors="coerce") <= cap

    def h5(_train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(test["volume_rel_20"], errors="coerce") >= 1.0

    def h2_flip(train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
        ov = pd.to_numeric(train["overshoot"], errors="coerce")
        cap = float(ov.quantile(0.80))
        return pd.to_numeric(test["overshoot"], errors="coerce") >= cap

    return [
        ("H1_rs_floor", "rs_spy_126d >= 0 (beat SPY, not rank)", h1),
        ("H2_tight_break", "1.00 <= channel_pos <= 1.10", h2),
        ("H3_atr_ceiling", "atr_pct <= train p60", h3),
        ("H4_quiet_so_far", "day_n_asof <= train p75 (names already filled today)", h4),
        ("H5_vol_confirm", "volume_rel_20 >= 1 (prior bar)", h5),
        (
            "POST_wide_break",
            "overshoot >= train p80 (H2 flipped after univariate)",
            h2_flip,
        ),
    ]


def _fit_ridge_gain(
    x_tr: np.ndarray, y_tr: np.ndarray, l2: float = 1.0
) -> Tuple[np.ndarray, float]:
    n, f = x_tr.shape
    xtx = x_tr.T @ x_tr + l2 * np.eye(f)
    xty = x_tr.T @ y_tr
    w = np.linalg.solve(xtx, xty)
    b = float(y_tr.mean() - (x_tr.mean(axis=0) @ w))
    return w, b


def expanding_model_keep(
    df: pd.DataFrame,
    *,
    kind: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in df["year"].dropna().unique())
    kept_parts: List[pd.DataFrame] = []
    coef_rows: List[dict] = []
    for i, y in enumerate(years):
        if i == 0:
            continue
        train = df.loc[df["year"] < y]
        test = df.loc[df["year"] == y]
        if len(train) < 200 or test.empty:
            continue
        x_tr, med = feature_matrix(train, REG_FEATURES)
        x_te, _ = feature_matrix(test, REG_FEATURES, medians=med)
        mu = np.nanmean(x_tr, axis=0)
        sd = np.nanstd(x_tr, axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        z_tr = (x_tr - mu) / sd
        z_te = (x_te - mu) / sd
        y_gain = pd.to_numeric(train[GAIN_COL], errors="coerce").to_numpy(dtype=float)
        y_gain = np.where(np.isfinite(y_gain), y_gain, 0.0)
        if kind == "logistic":
            y_bin = (y_gain > 0).astype(float)
            model = LogisticScorer(l2=0.5, lr=0.05, epochs=350)
            model.fit(z_tr, y_bin)
            p = model.score(z_te)
            thresh = float(np.median(model.score(z_tr)))
            keep = p >= thresh
            w = model.w if model.w is not None else np.zeros(len(REG_FEATURES))
        else:
            w, b = _fit_ridge_gain(z_tr, y_gain, l2=1.0)
            pred = z_te @ w + b
            keep = pred > 0.0
        part = test.loc[keep].copy()
        kept_parts.append(part)
        row = {"year": y, "kind": kind, "n_test": int(len(test)), "n_keep": int(keep.sum())}
        for name, coef in zip(REG_FEATURES, w):
            row[name] = round(float(coef), 4)
        coef_rows.append(row)
    kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[0:0].copy()
    return kept, pd.DataFrame(coef_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Mine 15m unique-symbol filter hypotheses")
    ap.add_argument("--trades", type=Path, default=DEFAULT_TRADES)
    ap.add_argument("--friction-pct", type=float, default=FRICTION)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "reports" / "ascending_channels",
    )
    args = ap.parse_args()
    if not args.trades.exists():
        logger.error("Missing %s", args.trades)
        return 1

    raw = pd.read_csv(args.trades)
    uniq = keep_one_per_symbol_day(raw)
    book = apply_friction(uniq, float(args.friction_pct))
    book = add_derived(book)
    logger.info("unique-symbol n=%d from %s", len(book), args.trades.name)

    base = _summarize(book, gain_col=GAIN_COL)
    print("=== baseline unique-symbol/day ===")
    print(_fmt(base))

    spear = spearman_table(book, NEW_UNIVARIATE)
    print("\n=== Spearman vs net gain (new + key features) ===")
    print(spear.to_string(index=False))

    q_parts = [quintiles(book, c) for c in NEW_UNIVARIATE]
    q_df = pd.concat([p for p in q_parts if not p.empty], ignore_index=True)
    print("\n=== Quintiles (selected) ===")
    for feat in ("rs_spy_126d", "overshoot", "atr_pct", "day_n_names", "volume_rel_20", "wait_bars", "buy_hour"):
        part = q_df.loc[q_df["feature"] == feat] if not q_df.empty else pd.DataFrame()
        if part.empty:
            continue
        print(feat)
        print(part.to_string(index=False))

    rows = []
    rows.append({"book": "baseline_unique_symbol", "sample": "full", **base})

    print("\n=== Hypotheses: full sample (in-sample, not the gate) ===")
    dummy_train = book
    for name, desc, fn in hyp_masks():
        keep = fn(dummy_train, book)
        sub = book.loc[keep.fillna(False)]
        s = _summarize(sub, gain_col=GAIN_COL)
        print("%s | %s | %s" % (name, desc, _fmt(s)))
        rows.append({"book": name, "sample": "full", "desc": desc, **s})

    print("\n=== Hypotheses: expanding-year OOS (train years < Y, apply to Y) ===")
    for name, desc, fn in hyp_masks():
        oos = expanding_year_apply(book, fn)
        s = _summarize(oos, gain_col=GAIN_COL)
        print("%s OOS | %s" % (name, _fmt(s)))
        rows.append({"book": name, "sample": "oos_year", "desc": desc, **s})

    print("\n=== Expanding-year logistic keep (p >= train median) ===")
    log_oos, log_coef = expanding_model_keep(book, kind="logistic")
    s_log = _summarize(log_oos, gain_col=GAIN_COL)
    print(_fmt(s_log))
    rows.append({"book": "logistic_p_ge_train_median", "sample": "oos_year", **s_log})

    print("\n=== Expanding-year ridge keep (pred gain > 0) ===")
    ridge_oos, ridge_coef = expanding_model_keep(book, kind="ridge")
    s_ridge = _summarize(ridge_oos, gain_col=GAIN_COL)
    print(_fmt(s_ridge))
    rows.append({"book": "ridge_pred_gt_0", "sample": "oos_year", **s_ridge})

    if not log_coef.empty:
        mean_abs = log_coef[list(REG_FEATURES)].abs().mean().sort_values(ascending=False)
        print("\n=== Logistic |coef| mean across OOS years ===")
        print(mean_abs.to_string())
        sign = np.sign(log_coef[list(REG_FEATURES)].mean())
        print("mean sign:", {k: int(v) for k, v in sign.items()})

    args.outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = pd.DataFrame(rows)
    p_sum = args.outdir / ("channel_touch_15m_unique_filter_hypotheses_%s.csv" % stamp)
    p_sp = args.outdir / ("channel_touch_15m_unique_filter_spearman_%s.csv" % stamp)
    p_q = args.outdir / ("channel_touch_15m_unique_filter_quintiles_%s.csv" % stamp)
    p_log = args.outdir / ("channel_touch_15m_unique_filter_logistic_coef_%s.csv" % stamp)
    p_ridge = args.outdir / ("channel_touch_15m_unique_filter_ridge_coef_%s.csv" % stamp)
    summary.to_csv(p_sum, index=False)
    spear.to_csv(p_sp, index=False)
    if not q_df.empty:
        q_df.to_csv(p_q, index=False)
    if not log_coef.empty:
        log_coef.to_csv(p_log, index=False)
    if not ridge_coef.empty:
        ridge_coef.to_csv(p_ridge, index=False)
    print("\nWrote %s" % p_sum)
    print("Wrote %s" % p_sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
