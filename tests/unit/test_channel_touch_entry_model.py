"""Unit tests for time-split entry scorer (no future-row leak)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.research.channel_touch_entry_model import (  # noqa: E402
    fit_and_eval,
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
