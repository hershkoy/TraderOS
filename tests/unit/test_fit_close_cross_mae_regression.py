"""Year-split close-cross MAE ridge: no future-year leak; stop-path P&L."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.channel_touch_entry_model import time_split
from utils.research.close_cross_mae_regression import (
    filled_mae_rows,
    fit_mae_ridge,
    grid_select,
    stop_path_pnl,
    year_fold_table,
)


def _rows() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2019-01-02", periods=1600, freq="B")
    n = len(dates)
    vol = rng.uniform(0.4, 2.5, size=n)
    rng_pct = rng.uniform(0.3, 2.0, size=n)
    mae = np.clip(1.0 + 1.8 * vol + 0.4 * rng_pct + rng.normal(0, 0.15, size=n), 0.4, 7.0)
    trail = np.where(mae < 2.5, 8.0 - 0.5 * vol, -10.0) + rng.normal(0, 0.5, size=n)
    skip = np.array([""] * n, dtype=object)
    skip[0] = "occupancy"
    skip[1] = "wild_low_to_mid"
    return pd.DataFrame(
        {
            "stock": np.where(np.arange(n) < n // 2, "AAA", "BBB"),
            "buy_date": dates,
            "skip_reason": skip,
            "volume_rel_20": vol,
            "range_pct": rng_pct,
            "mae_pct": mae,
            "trail_only_gain_pct": trail,
            "close_over_rail_pct": rng.uniform(0.0, 0.8, size=n),
            "wait_bars": rng.integers(6, 40, size=n),
            "channel_width_pct": rng.uniform(4.0, 12.0, size=n),
        }
    )


def test_filled_mae_rows_drops_skips():
    df = filled_mae_rows(_rows())
    assert (df["skip_reason"].fillna("").astype(str).str.strip() == "").all()
    assert len(df) == 1598
    assert df["mae_pct"].notna().all()


def test_stop_path_pnl_survives_inside_stop():
    mae = np.array([2.0, 4.0, 3.0])
    trail = np.array([10.0, 12.0, 5.0])
    stop = np.array([3.0, 3.0, 3.0])
    pnl = stop_path_pnl(mae, trail, stop)
    np.testing.assert_allclose(pnl, [10.0, -3.0, -3.0])


def test_time_split_train_before_cutoff():
    df = filled_mae_rows(_rows())
    tr, te = time_split(df, "2023-01-01")
    assert tr["buy_date"].max() < pd.Timestamp("2023-01-01")
    assert te["buy_date"].min() >= pd.Timestamp("2023-01-01")


def test_ridge_recovers_volume_weight_on_mae():
    df = filled_mae_rows(_rows())
    tr, te = time_split(df, "2023-01-01")
    fit = fit_mae_ridge(tr, te, ["volume_rel_20", "range_pct"], l2=0.5, pad=1.25)
    weights = {c["feature"]: c["weight"] for c in fit["coefs"]}
    assert weights["volume_rel_20"] > weights["range_pct"]
    assert fit["n_train"] > 50
    assert fit["n_test"] > 20
    assert fit["rmse_test"] is not None and fit["rmse_test"] < 1.5
    assert "expectancy_pct" in fit["test_stop"]


def test_grid_select_and_year_folds_run():
    df = filled_mae_rows(_rows())
    tr, _te = time_split(df, "2023-01-01")
    l2, pad, grid = grid_select(
        tr,
        feature_cols=["volume_rel_20", "range_pct"],
        inner_cutoff="2022-01-01",
        l2_grid=(0.5, 5.0),
        pad_grid=(1.0, 1.5),
    )
    assert l2 in (0.5, 5.0)
    assert pad in (1.0, 1.5)
    assert not grid.empty
    folds = year_fold_table(
        df, feature_cols=["volume_rel_20", "range_pct"], l2=l2, pad=pad
    )
    assert "fold" in folds.columns
    assert len(folds) >= 2
