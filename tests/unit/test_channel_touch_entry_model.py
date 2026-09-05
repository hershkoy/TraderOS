"""Unit tests for time-split entry scorer (no future-row leak)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.channel_touch_entry_model import (  # noqa: E402
    add_loser_filter_columns,
    apply_size_map,
    buy_timestamps,
    expanding_keep_walk_forward,
    expanding_ridge_walk_forward,
    expanding_year_windows,
    fit_and_eval,
    fit_keep_fold,
    fit_ridge_size_fold,
    fit_size_map,
    keep_filter_verdict,
    purged_embargo_split,
    sell_timestamps,
    stitched_keep_metrics,
    time_split,
)


def _fake_trades(n: int = 1600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2019-01-02", periods=n, freq="B")
    rsi = rng.uniform(20, 80, size=n)
    # Early years: low RSI wins; later years: noise. Model must not see later rows in train.
    gain = np.where(rsi < 50, 1.5, -1.0) + rng.normal(0, 0.2, size=n)
    gain = np.where(dates.year >= 2023, rng.normal(0, 1.0, size=n), gain)
    return pd.DataFrame(
        {
            "buy_date": dates,
            "gain_pct_net": gain,
            "rsi_14": rsi,
            "dist_sma50_pct": rng.normal(0, 2, size=n),
            "channel_pos": rng.uniform(0, 0.4, size=n),
            "atr_pct": rng.uniform(0.5, 3.0, size=n),
            "volume_rel_20": rng.uniform(0.5, 2.0, size=n),
            "squeeze_mom": rng.normal(0, 1, size=n),
            "squeeze_mom_rising": rng.integers(0, 2, size=n),
            "max_beyond_width": rng.uniform(0, 0.4, size=n),
            "rs_spy_126d": rng.normal(0, 5, size=n),
            "rs_spy_21d": rng.normal(0, 3, size=n),
            "spy_ret_20d": rng.normal(0, 2, size=n),
            "spy_above_sma50": rng.integers(0, 2, size=n),
            "close_loc": rng.uniform(0, 1, size=n),
            "room_to_resist_pct": rng.uniform(1, 8, size=n),
            "channel_width_pct": rng.uniform(2, 10, size=n),
            "range_pct": rng.uniform(0.5, 3.0, size=n),
        }
    )


def test_time_split_is_strictly_before_cutoff():
    df = _fake_trades()
    tr, te = time_split(df, "2023-01-01")
    assert tr["buy_date"].max() < pd.Timestamp("2023-01-01")
    assert te["buy_date"].min() >= pd.Timestamp("2023-01-01")


def test_time_split_uses_buy_time_when_present():
    df = pd.DataFrame(
        {
            "buy_date": ["2022-12-31", "2023-01-01"],
            "buy_time": ["2022-12-31 15:45", "2023-01-01 09:45"],
            "gain_pct_net": [1.0, -1.0],
            "rsi_14": [40.0, 60.0],
        }
    )
    tr, te = time_split(df, "2023-01-01")
    assert len(tr) == 1
    assert len(te) == 1
    assert str(tr.iloc[0]["buy_date"])[:10] == "2022-12-31"


def test_fit_and_eval_does_not_empty_train():
    df = _fake_trades()
    split = fit_and_eval(df, cutoff="2023-01-01", model_kind="logistic")
    assert split.train["n_trades"] > 0
    assert split.test["n_trades"] > 0
    assert split.n_features > 0


def test_purge_drops_overlapping_hold():
    df = pd.DataFrame(
        {
            "buy_date": pd.to_datetime(["2021-12-01", "2021-12-20", "2022-01-15"]),
            "sell_date": pd.to_datetime(["2021-12-10", "2022-01-05", "2022-01-20"]),
            "gain_pct_net": [1.0, 1.0, -1.0],
            "channel_pos": [0.1, 0.1, 0.1],
            "atr_pct": [2.0, 2.0, 2.0],
        }
    )
    tr, te = purged_embargo_split(df, "2022-01-01", "2022-12-31", embargo_days=0)
    assert len(te) == 1
    assert str(te.iloc[0]["buy_date"])[:10] == "2022-01-15"
    sell = sell_timestamps(tr)
    assert (sell < pd.Timestamp("2022-01-01")).all()
    assert len(tr) == 1
    assert str(tr.iloc[0]["buy_date"])[:10] == "2021-12-01"


def test_embargo_drops_train_selling_inside_buffer():
    df = pd.DataFrame(
        {
            "buy_date": pd.to_datetime(["2021-11-01", "2021-12-01", "2022-02-01"]),
            "sell_date": pd.to_datetime(["2021-11-15", "2021-12-20", "2022-02-10"]),
            "gain_pct_net": [1.0, 1.0, 1.0],
            "channel_pos": [0.1, 0.2, 0.3],
            "atr_pct": [2.0, 2.0, 2.0],
        }
    )
    tr, te = purged_embargo_split(df, "2022-01-01", "2022-12-31", embargo_days=21)
    # train sell must be < 2021-12-11; 2021-12-20 is inside embargo
    assert len(tr) == 1
    assert str(tr.iloc[0]["buy_date"])[:10] == "2021-11-01"
    assert len(te) == 1


def test_size_map_uses_train_percentiles_only():
    pred_tr = np.linspace(-2.0, 2.0, 101)
    smap = fit_size_map(pred_tr, lo_q=0.10, hi_q=0.90, size_min=0.25, size_max=2.0)
    pred_te = np.array([100.0, -100.0, 0.0])
    sizes = apply_size_map(pred_te, smap, skip_negative=False)
    assert sizes[0] == 2.0
    assert sizes[1] == 0.25
    skip = apply_size_map(pred_te, smap, skip_negative=True)
    assert skip[1] == 0.0
    assert skip[0] == 2.0


def test_ridge_fold_train_does_not_see_test_rows():
    rng = np.random.default_rng(1)
    n = 1300
    dates = pd.date_range("2019-01-02", periods=n, freq="B")
    atr = rng.uniform(0.5, 3.0, size=n)
    # Train-era: higher ATR -> higher P&L. Test era would invert if leaked.
    gain = (atr - 1.75) * 2.0 + rng.normal(0, 0.2, size=n)
    gain = np.where(dates.year >= 2023, -(atr - 1.75) * 2.0, gain)
    df = pd.DataFrame(
        {
            "buy_date": dates,
            "sell_date": dates + pd.Timedelta(days=10),
            "gain_pct_net": gain,
            "channel_pos": rng.uniform(0, 0.4, size=n),
            "atr_pct": atr,
            "volume_rel_20": rng.uniform(0.5, 2.0, size=n),
            "squeeze_mom": rng.normal(0, 1, size=n),
            "squeeze_mom_rising": rng.integers(0, 2, size=n),
            "max_beyond_width": rng.uniform(0, 0.4, size=n),
            "rs_spy_126d": rng.normal(0, 5, size=n),
            "rs_spy_21d": rng.normal(0, 3, size=n),
            "spy_ret_20d": rng.normal(0, 2, size=n),
            "spy_above_sma50": rng.integers(0, 2, size=n),
            "room_to_resist_pct": rng.uniform(1, 8, size=n),
            "channel_width_pct": rng.uniform(2, 10, size=n),
        }
    )
    tr, te = purged_embargo_split(df, "2023-01-01", "2023-12-31", embargo_days=21)
    assert sell_timestamps(tr).max() < pd.Timestamp("2023-01-01") - pd.Timedelta(days=21)
    assert buy_timestamps(te).min() >= pd.Timestamp("2023-01-01")
    pred, size_all, size_skip, smap, n_feat = fit_ridge_size_fold(
        tr, te, ["atr_pct", "channel_pos", "volume_rel_20"]
    )
    assert n_feat == 3
    assert len(pred) == len(te)
    assert size_all.min() >= 0.25 - 1e-9
    assert size_all.max() <= 2.0 + 1e-9
    assert smap.pred_lo <= smap.pred_hi
    # Map bounds come from train preds, not the inverted test regime.
    pred_tr, _, _, smap2, _ = fit_ridge_size_fold(
        tr, tr, ["atr_pct", "channel_pos", "volume_rel_20"]
    )
    assert abs(smap.pred_lo - smap2.pred_lo) < 1e-9
    assert abs(smap.pred_hi - smap2.pred_hi) < 1e-9
    assert len(pred_tr) == len(tr)


def test_expanding_wf_train_sells_before_test_minus_embargo():
    df = _fake_trades(n=800)
    df = df.copy()
    df["sell_date"] = pd.to_datetime(df["buy_date"]) + pd.Timedelta(days=10)
    folds, oos = expanding_ridge_walk_forward(
        df,
        ["atr_pct", "channel_pos", "volume_rel_20"],
        embargo_days=21,
        min_train_frac=0.50,
        min_train_n=40,
        min_test_n=20,
    )
    assert not folds.empty
    assert not oos.empty
    buy_oos = buy_timestamps(oos)
    # Every OOS row's buy is on or after some fold start; no OOS buy in the first 50%.
    b_all = buy_timestamps(df).sort_values()
    t50 = pd.Timestamp(b_all.quantile(0.50))
    assert buy_oos.min() >= t50.normalize() or buy_oos.min() >= t50 - pd.Timedelta(days=1)
    windows = expanding_year_windows(buy_timestamps(df), min_train_frac=0.50)
    assert windows
    first_start = windows[0][0]
    tr, te = purged_embargo_split(
        df, str(first_start), str(windows[0][1]), embargo_days=21
    )
    assert sell_timestamps(tr).max() < first_start - pd.Timedelta(days=21)
    assert buy_timestamps(te).min() >= first_start


def test_add_loser_filter_columns_from_shakeout_flag():
    df = pd.DataFrame(
        {
            "buy_date": pd.to_datetime(["2021-01-04", "2021-02-01"]),
            "gain_pct_net": [1.0, -1.0],
            "shakeout_breakout": [False, True],
            "parent_exit_reason": [None, "hard_stop"],
        }
    )
    out = add_loser_filter_columns(df)
    assert list(out["is_extra"]) == [0.0, 1.0]


def test_expanding_keep_wf_does_not_use_future_rows():
    rng = np.random.default_rng(2)
    n = 900
    dates = pd.date_range("2019-01-02", periods=n, freq="B")
    wait = rng.uniform(5, 80, size=n)
    gain = np.where(wait > 40, 2.0, -1.0) + rng.normal(0, 0.2, size=n)
    df = pd.DataFrame(
        {
            "buy_date": dates,
            "sell_date": dates + pd.Timedelta(days=8),
            "gain_pct_net": gain,
            "wait_bars": wait,
            "channel_pos": rng.uniform(0.9, 1.2, size=n),
            "channel_width_pct": rng.uniform(5, 20, size=n),
            "slope_pct_per_bar": rng.uniform(0.02, 0.2, size=n),
            "channel_span_days": rng.uniform(40, 200, size=n),
            "channel_age_at_buy_days": rng.uniform(50, 250, size=n),
            "room_to_resist_pct": rng.uniform(-1, 1, size=n),
            "dow": rng.integers(0, 5, size=n),
            "month": rng.integers(1, 13, size=n),
        }
    )
    folds, oos = expanding_keep_walk_forward(
        df,
        ["wait_bars", "channel_pos", "channel_width_pct"],
        kind="ridge",
        rule="skip0",
        embargo_days=21,
        min_train_frac=0.50,
        min_train_n=40,
        min_test_n=20,
    )
    assert not folds.empty
    assert not oos.empty
    assert "keep" in oos.columns
    first_start = expanding_year_windows(buy_timestamps(df), min_train_frac=0.50)[0][0]
    tr, te = purged_embargo_split(
        df, str(first_start), str(expanding_year_windows(buy_timestamps(df), min_train_frac=0.50)[0][1]),
        embargo_days=21,
    )
    scores, keep0, keep_thr, thr, n_feat = fit_keep_fold(
        tr, te, ["wait_bars", "channel_pos"], kind="ridge"
    )
    assert n_feat == 2
    assert len(scores) == len(te)
    assert len(keep0) == len(te)
    assert len(keep_thr) == len(te)
    assert sell_timestamps(tr).max() < first_start - pd.Timedelta(days=21)
    stitched = stitched_keep_metrics(oos)
    assert stitched["all"]["n_trades"] == len(oos)
    verdict = keep_filter_verdict(stitched, folds)
    assert verdict in ("research_only", "no_promote")

