"""Year-split ridge: predict close-cross MAE (stop to reach trail MFE) from confirm features.

Train only on fills with buy_date before the test cutoff. Features are the
confirm-bar snapshot (known when the 15m closes above the daily rail). Labels
are that trade's realized mae_pct (known only after the trail path).

Stop policy on a predicted MAE ``s_hat`` (optional pad): if actual mae < s_hat
the trail path is assumed to survive and the P&L is ``trail_only_gain_pct``;
otherwise the trade is stopped at ``-s_hat``. Compare E/PF on held-out years.
Not a promote.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.research.channel_touch_entry_model import (
    RidgePnlScorer,
    _standardize,
    feature_matrix,
    time_split,
    trade_metrics,
)

MAE_FEATURE_COLS: Sequence[str] = (
    "volume_rel_20",
    "volume_rel_tod",
    "range_pct",
    "range_atr",
    "range_vs_med20",
    "body_frac",
    "close_loc",
    "confirm_green",
    "gap_15m",
    "open_vs_rail_pct",
    "close_over_rail_pct",
    "close_over_rail_atr",
    "high_over_rail_pct",
    "session_bar_i",
    "session_failed_closes",
    "minutes_from_open",
    "slope_pct_per_bar",
    "rail_rise_since_h2_pct",
    "channel_width_pct",
    "wait_bars",
    "cc_rsi_14",
    "atr_pct_15m",
)

L2_GRID: Sequence[float] = (0.1, 0.5, 1.0, 5.0, 10.0)
PAD_GRID: Sequence[float] = (1.0, 1.1, 1.25, 1.5, 2.0)
DEFAULT_CUTOFF = "2023-01-01"
YEAR_FOLDS: Sequence[Tuple[str, str, str]] = (
    ("train<=2021 test=2022", "2022-01-01", "2022-12-31"),
    ("train<=2022 test=2023", "2023-01-01", "2023-12-31"),
    ("train<=2023 test=2024-26", "2024-01-01", "2026-12-31"),
)
STOP_FLOOR = 1.5
STOP_CEIL = 8.0


def filled_mae_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep filled close-cross rows with a usable MAE label."""
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "skip_reason" in out.columns:
        skip = out["skip_reason"].fillna("").astype(str).str.strip()
        out = out.loc[skip == ""].copy()
    mae = pd.to_numeric(out.get("mae_pct"), errors="coerce")
    trail = pd.to_numeric(out.get("trail_only_gain_pct"), errors="coerce")
    out = out.loc[mae.notna() & trail.notna()].copy()
    return out


def present_features(df: pd.DataFrame, cols: Sequence[str] = MAE_FEATURE_COLS) -> List[str]:
    return [c for c in cols if c in df.columns]


def _clip_stop(raw: np.ndarray, floor: float = STOP_FLOOR, ceil: float = STOP_CEIL) -> np.ndarray:
    s = np.asarray(raw, dtype=float)
    s = np.where(np.isfinite(s), s, float(floor))
    return np.clip(s, float(floor), float(ceil))


def stop_path_pnl(
    mae_pct: np.ndarray,
    trail_gain: np.ndarray,
    stop_pct: np.ndarray,
) -> np.ndarray:
    """Survive to trail P&L if MAE is strictly inside the stop; else -stop."""
    mae = np.asarray(mae_pct, dtype=float)
    trail = np.asarray(trail_gain, dtype=float)
    stop = np.asarray(stop_pct, dtype=float)
    survive = np.isfinite(mae) & np.isfinite(stop) & (mae < stop)
    pnl = np.where(survive, trail, -stop)
    pnl = np.where(np.isfinite(pnl), pnl, np.nan)
    return pnl


def fit_mae_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    l2: float = 1.0,
    pad: float = 1.0,
    y_col: str = "mae_pct",
) -> dict:
    cols = present_features(train, feature_cols)
    x_tr_raw, med = feature_matrix(train, cols)
    x_te_raw, _ = feature_matrix(test, cols, medians=med)
    x_tr, x_te, _, _ = _standardize(x_tr_raw, x_te_raw)
    y_tr = pd.to_numeric(train[y_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y_tr)
    model = RidgePnlScorer(l2=float(l2))
    model.fit(x_tr[ok], y_tr[ok])
    pred_tr = model.predict(x_tr)
    pred_te = model.predict(x_te) if len(test) else np.zeros(0, dtype=float)
    stop_tr = _clip_stop(pred_tr * float(pad))
    stop_te = _clip_stop(pred_te * float(pad))
    mae_tr = pd.to_numeric(train["mae_pct"], errors="coerce").to_numpy(dtype=float)
    mae_te = (
        pd.to_numeric(test["mae_pct"], errors="coerce").to_numpy(dtype=float)
        if len(test)
        else np.zeros(0, dtype=float)
    )
    trail_tr = pd.to_numeric(train["trail_only_gain_pct"], errors="coerce").to_numpy(dtype=float)
    trail_te = (
        pd.to_numeric(test["trail_only_gain_pct"], errors="coerce").to_numpy(dtype=float)
        if len(test)
        else np.zeros(0, dtype=float)
    )
    pnl_tr = stop_path_pnl(mae_tr, trail_tr, stop_tr)
    pnl_te = stop_path_pnl(mae_te, trail_te, stop_te)
    rmse_tr = float(np.sqrt(np.nanmean((pred_tr[ok] - y_tr[ok]) ** 2))) if ok.any() else None
    te_ok = np.isfinite(mae_te) if len(mae_te) else np.zeros(0, dtype=bool)
    rmse_te = (
        float(np.sqrt(np.nanmean((pred_te[te_ok] - mae_te[te_ok]) ** 2))) if te_ok.any() else None
    )
    coefs = []
    if model.w is not None:
        coefs = [
            {"feature": c, "weight": round(float(w), 6)}
            for c, w in sorted(zip(cols, model.w), key=lambda t: abs(t[1]), reverse=True)
        ]
    return {
        "l2": float(l2),
        "pad": float(pad),
        "n_train": int(ok.sum()),
        "n_test": int(te_ok.sum()) if len(te_ok) else 0,
        "rmse_train": None if rmse_tr is None else round(rmse_tr, 4),
        "rmse_test": None if rmse_te is None else round(rmse_te, 4),
        "intercept": round(float(model.b), 4),
        "coefs": coefs,
        "train_trail": trade_metrics(trail_tr),
        "test_trail": trade_metrics(trail_te),
        "train_stop": trade_metrics(pnl_tr),
        "test_stop": trade_metrics(pnl_te),
        "pred_test": pred_te,
        "stop_test": stop_te,
        "pnl_test": pnl_te,
        "model": model,
        "medians": med,
        "cols": cols,
    }


def grid_select(
    train: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = MAE_FEATURE_COLS,
    inner_cutoff: str = "2022-01-01",
    l2_grid: Sequence[float] = L2_GRID,
    pad_grid: Sequence[float] = PAD_GRID,
) -> Tuple[float, float, pd.DataFrame]:
    """Pick l2 and MAE pad on an inner year split of train (max test-stop expectancy, then PF)."""
    inner_tr, inner_te = time_split(train, inner_cutoff)
    rows: List[dict] = []
    best: Optional[Tuple[float, float, float, float]] = None
    best_pair = (1.0, 1.25)
    if inner_tr.empty or inner_te.empty:
        return best_pair[0], best_pair[1], pd.DataFrame(rows)
    for l2 in l2_grid:
        for pad in pad_grid:
            fit = fit_mae_ridge(inner_tr, inner_te, feature_cols, l2=l2, pad=pad)
            e = fit["test_stop"].get("expectancy_pct")
            pf = fit["test_stop"].get("profit_factor")
            e_f = float(e) if e is not None else -1e9
            pf_f = float(pf) if pf is not None and np.isfinite(pf) else 0.0
            rows.append(
                {
                    "l2": l2,
                    "pad": pad,
                    "n_train": fit["n_train"],
                    "n_test": fit["n_test"],
                    "rmse_test": fit["rmse_test"],
                    "E": e,
                    "PF": pf,
                    "WR": fit["test_stop"].get("win_rate_pct"),
                }
            )
            key = (e_f, pf_f)
            if best is None or key > (best[0], best[1]):
                best = (e_f, pf_f, float(l2), float(pad))
                best_pair = (float(l2), float(pad))
    return best_pair[0], best_pair[1], pd.DataFrame(rows)


def year_fold_table(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str] = MAE_FEATURE_COLS,
    l2: float,
    pad: float,
    folds: Sequence[Tuple[str, str, str]] = YEAR_FOLDS,
) -> pd.DataFrame:
    rows: List[dict] = []
    for name, start, end in folds:
        tr, rest = time_split(df, start)
        buy = pd.to_datetime(rest["buy_date"], errors="coerce") if not rest.empty else pd.Series(dtype="datetime64[ns]")
        te = rest.loc[buy <= pd.Timestamp(end)].copy() if not rest.empty else rest
        if tr.empty or te.empty:
            rows.append({"fold": name, "n_train": len(tr), "n_test": len(te)})
            continue
        fit = fit_mae_ridge(tr, te, feature_cols, l2=l2, pad=pad)
        rows.append(
            {
                "fold": name,
                "n_train": fit["n_train"],
                "n_test": fit["n_test"],
                "rmse_test": fit["rmse_test"],
                "trail_E": fit["test_trail"].get("expectancy_pct"),
                "trail_PF": fit["test_trail"].get("profit_factor"),
                "stop_E": fit["test_stop"].get("expectancy_pct"),
                "stop_PF": fit["test_stop"].get("profit_factor"),
                "stop_WR": fit["test_stop"].get("win_rate_pct"),
            }
        )
    return pd.DataFrame(rows)
