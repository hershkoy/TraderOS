"""Causal VX-curve slope analog (skip if prior-session slope < 4)."""
from __future__ import annotations

from datetime import datetime
import json

import numpy as np
import pandas as pd

from pathlib import Path

from utils.research.channel_touch_entry_features import completed_asof
from utils.research.vx_futures_curve import (
    USER_SLOPE_MIN,
    attach_prior_session_slope,
    curve_slope_pct,
    interpolate_60d_close,
    keep_unless_slope_below,
    load_tv_dump_close,
    pine_band_label,
    vx60_slope_pct,
)


def test_pine_slope_formula_matches_user_script() -> None:
    idx = pd.bdate_range("2020-01-02", periods=2)
    vx1 = pd.Series([20.0, 20.0], index=idx)
    vx2 = pd.Series([20.8, 19.6], index=idx)
    slope = curve_slope_pct(vx1, vx2)
    assert abs(float(slope.iloc[0]) - 4.0) < 1e-9
    assert float(slope.iloc[1]) < 0.0


def test_keep_skips_below_four_keeps_equal_and_missing() -> None:
    trades = pd.DataFrame({"vx_slope_asof": [3.9, 4.0, 8.0, np.nan]})
    keep = keep_unless_slope_below(trades, thresh=USER_SLOPE_MIN)
    assert list(keep.astype(int)) == [0, 1, 1, 1]


def test_vix60_interpolant_is_between_vix_and_vix3m_in_contango() -> None:
    idx = pd.bdate_range("2020-01-02", periods=3)
    vix = pd.Series([15.0, 15.0, 15.0], index=idx)
    vix3m = pd.Series([18.0, 18.0, 18.0], index=idx)
    cm = interpolate_60d_close(vix, vix3m)
    assert abs(float(cm.iloc[0]) - 16.5) < 1e-9
    slope = vx60_slope_pct(vix, vix3m)
    # (16.5/15 - 1)*100 = 10; calm 3m +20% maps to ~10% on the 60d analog.
    assert abs(float(slope.iloc[0]) - 10.0) < 1e-9


def test_deep_inversion_interpolant_is_risk_off() -> None:
    idx = pd.bdate_range("2020-01-02", periods=2)
    vix = pd.Series([30.0, 30.0], index=idx)
    vix3m = pd.Series([10.0, 10.0], index=idx)
    slope = vx60_slope_pct(vix, vix3m)
    assert float(slope.iloc[0]) < 0.0
    trades = pd.DataFrame({"vx_slope_asof": slope.values})
    keep = keep_unless_slope_below(trades, thresh=4.0)
    assert not bool(keep.iloc[0])


def test_four_pct_gate_on_shallow_60d_contango() -> None:
    idx = pd.bdate_range("2020-01-02", periods=1)
    # CM60/VIX = 1.03 -> 3% < 4 skip; 1.05 -> 5% keep.
    vix = pd.Series([20.0], index=idx)
    vix3m_shallow = pd.Series([21.2], index=idx)  # CM60=20.6, slope=3%
    vix3m_ok = pd.Series([22.0], index=idx)  # CM60=21.0, slope=5%
    shallow = vx60_slope_pct(vix, vix3m_shallow)
    ok = vx60_slope_pct(vix, vix3m_ok)
    assert float(shallow.iloc[0]) < 4.0
    assert float(ok.iloc[0]) > 4.0


def test_last_15m_join_uses_prior_session_not_fill_day() -> None:
    idx = pd.bdate_range("2022-03-14", periods=3)
    slope = pd.Series([8.0, 1.0, 1.0], index=idx)
    trades = pd.DataFrame(
        [
            {
                "stock": "AAA",
                "buy_date": "2022-03-15",
                "buy_time": datetime(2022, 3, 15, 15, 45),
            }
        ]
    )
    asof = completed_asof(trades.iloc[0], series_is_daily=True)
    assert asof is not None
    assert pd.Timestamp(asof).normalize() <= pd.Timestamp("2022-03-14")
    tagged = attach_prior_session_slope(trades, slope)
    assert abs(float(tagged.loc[0, "vx_slope_asof"]) - 8.0) < 1e-9
    keep = keep_unless_slope_below(tagged, thresh=4.0)
    assert bool(keep.iloc[0]) is True


def test_pine_band_labels() -> None:
    s = pd.Series([-1.0, 2.0, 6.0, 9.0, np.nan])
    labels = list(pine_band_label(s))
    assert labels == ["<0", "0-5", "5-8", ">=8", "missing"]


def test_tv_dump_close_uses_chicago_session_date(tmp_path: Path) -> None:
    unix = int(pd.Timestamp("2022-03-14 23:00:00", tz="UTC").timestamp())
    payload = {"meta": {"size": 1}, "bars": [[unix, 20.0, 21.0, 19.0, 20.5, 0]]}
    path = tmp_path / "vx1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    close = load_tv_dump_close(path, name="VX1")
    assert list(close.index.normalize()) == [pd.Timestamp("2022-03-14")]
    assert abs(float(close.iloc[0]) - 20.5) < 1e-9
