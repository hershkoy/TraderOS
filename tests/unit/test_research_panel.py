"""Unit tests for utils.research.panel (synthetic frames, no DB)."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.research.panel import (
    adv_dollar,
    align_spy_regime,
    frames_to_wide,
    month_ends,
    panel_cache_path,
    to_naive_index,
    top_n_liquid_mask,
)


def test_panel_cache_path(tmp_path: Path) -> None:
    p = panel_cache_path(
        tmp_path,
        "ALPACA",
        "1d",
        "close",
        datetime(2017, 11, 29),
        datetime(2025, 11, 26),
    )
    assert p == tmp_path / "alpaca_1d_20171129_20251126_all_close.parquet"


def test_frames_to_wide_aligns_dates() -> None:
    a = pd.DataFrame(
        {"close": [1.0, 2.0], "volume": [10.0, 20.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
    )
    b = pd.DataFrame(
        {"close": [3.0], "volume": [30.0]},
        index=pd.to_datetime(["2020-01-03"]),
    )
    wide = frames_to_wide({"aaa": a, "bbb": b}, "close")
    assert list(wide.columns) == ["AAA", "BBB"]
    assert wide.loc[pd.Timestamp("2020-01-03"), "AAA"] == 2.0
    assert wide.loc[pd.Timestamp("2020-01-03"), "BBB"] == 3.0
    assert pd.isna(wide.loc[pd.Timestamp("2020-01-02"), "BBB"])


def test_to_naive_index_strips_tz() -> None:
    idx = pd.date_range("2020-01-02", periods=2, tz="UTC")
    s = pd.Series([1.0, 2.0], index=idx)
    out = to_naive_index(s)
    assert out.index.tz is None
    assert out.index[0] == pd.Timestamp("2020-01-02")


def test_month_ends() -> None:
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    me = month_ends(idx)
    assert len(me) == 3
    assert me[-1].month == 3


def test_top_n_liquid_mask_is_point_in_time() -> None:
    idx = pd.bdate_range("2020-01-02", periods=40)
    close = pd.DataFrame(
        {
            "A": 10.0,
            "B": 10.0,
            "C": 10.0,
            "D": 10.0,
        },
        index=idx,
    )
    vol = pd.DataFrame(1.0, index=idx, columns=list("ABCD"))
    vol.loc[idx[:30], "A"] = 100.0
    vol.loc[idx[:30], "B"] = 50.0
    vol.loc[idx[:30], "C"] = 20.0
    vol.loc[idx[:30], "D"] = 5.0
    vol.loc[idx[30:], "D"] = 1000.0
    vol.loc[idx[30:], "C"] = 900.0
    adv = adv_dollar(close, vol, window=5)
    mask = top_n_liquid_mask(adv, top_n=2, min_adv=1.0, close=close, min_price=1.0)
    early = mask.loc[idx[20]]
    assert bool(early["A"]) and bool(early["B"])
    assert not bool(early["C"]) and not bool(early["D"])
    late = mask.loc[idx[-1]]
    assert bool(late["D"]) and bool(late["C"])
    assert not bool(late["A"])


def test_align_spy_regime_uses_full_spy_history() -> None:
    spy_idx = pd.bdate_range("2018-01-02", periods=400)
    spy = pd.Series(range(100, 500), index=spy_idx, dtype=float)
    # Gapped panel like ALPACA (one early day then a long hole)
    panel_idx = pd.DatetimeIndex([spy_idx[0]]).append(spy_idx[250:])
    _spy_a, _sma_a, risk_on = align_spy_regime(spy, panel_idx, sma_period=200)
    assert bool(risk_on.iloc[-1])
    # Must be valid on the first post-gap panel day, not after 200 panel rows
    first_panel_real = panel_idx[1]
    assert not pd.isna(risk_on.loc[first_panel_real])

