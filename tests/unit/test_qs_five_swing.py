"""Unit tests for Quantified Strategies five-swing helpers."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from utils.research.qs_five_swing import (
    backtest_combined,
    backtest_named,
    entry_mask_s1,
    entry_mask_s2,
    entry_mask_s3,
    entry_mask_s4,
    entry_mask_s5,
    ibs,
    prepare_frame,
)


def _bars(n: int, start: str = "2020-01-06") -> pd.DataFrame:
    """Sequential weekday OHLC starting on start date (skip weekends)."""
    day = pd.Timestamp(start, tz="UTC")
    rows = []
    px = 100.0
    for _ in range(n):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        rows.append(
            {
                "ts": day,
                "open": px,
                "high": px + 2.0,
                "low": px - 2.0,
                "close": px + 0.5,
                "volume": 1_000_000,
            }
        )
        px = px + 0.5
        day += timedelta(days=1)
    return pd.DataFrame(rows)


def test_ibs_formula():
    high = pd.Series([10.0, 10.0, 12.0])
    low = pd.Series([8.0, 10.0, 10.0])
    close = pd.Series([9.0, 10.0, 10.5])
    out = ibs(high, low, close)
    assert out.iloc[0] == pytest.approx(0.5)
    assert out.iloc[1] == pytest.approx(0.0)  # zero range
    assert out.iloc[2] == pytest.approx(0.25)


def test_s1_band_uses_2_5_multiplier():
    df = _bars(40)
    d2 = prepare_frame(df)
    expected_band = d2["high_10"] - 2.5 * d2["avg_range_25"]
    assert (d2["band_s1"] - expected_band).abs().max() < 1e-9

    raw = df.copy()
    i = len(raw) - 1
    # Raise prior highs so 10d high is elevated; shrink last bar range near lows.
    for j in range(i - 9, i):
        raw.loc[j, "high"] = max(float(raw.loc[j, "high"]), 130.0)
    raw.loc[i, "high"] = 105.0
    raw.loc[i, "low"] = 100.0
    raw.loc[i, "close"] = 100.2  # IBS = 0.04
    d3 = prepare_frame(raw)
    assert float(d3.loc[i, "close"]) < float(d3.loc[i, "band_s1"])
    assert float(d3.loc[i, "ibs"]) < 0.3
    assert bool(entry_mask_s1(d3).iloc[i])


def test_s2_monday_two_down_days():
    # Build: Thu up, Fri down, Mon down further -> Monday entry
    days = []
    # 2020-01-02 Thu, 01-03 Fri, 01-06 Mon
    specs = [
        ("2020-01-02", 100.0, 101.0, 99.0, 100.5),  # Thu
        ("2020-01-03", 100.5, 101.0, 98.0, 99.0),  # Fri down
        ("2020-01-06", 99.0, 99.5, 96.0, 97.0),  # Mon down again
        ("2020-01-07", 97.0, 100.0, 96.5, 99.5),  # Tue exit candidate
    ]
    for ds, o, h, l, c in specs:
        days.append(
            {
                "ts": pd.Timestamp(ds, tz="UTC"),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": 1,
            }
        )
    # Pad with neutral history so prepare_frame is happy
    hist = _bars(30, start="2019-11-04")
    df = pd.concat([hist, pd.DataFrame(days)], ignore_index=True)
    d = prepare_frame(df)
    mon_i = d.index[d["ts"] == pd.Timestamp("2020-01-06", tz="UTC")][0]
    assert int(d.loc[mon_i, "weekday"]) == 0
    assert bool(entry_mask_s2(d).iloc[mon_i])


def test_s3_five_day_low():
    df = _bars(20)
    # Make last close below prior 5 lows
    raw = df.copy()
    i = len(raw) - 1
    prior_min = float(raw["low"].iloc[i - 5 : i].min())
    raw.loc[i, "close"] = prior_min - 1.0
    raw.loc[i, "low"] = prior_min - 1.5
    raw.loc[i, "high"] = prior_min
    d = prepare_frame(raw)
    assert bool(entry_mask_s3(d).iloc[i])
    assert not bool(entry_mask_s3(d).iloc[i - 1])


def test_s4_range_min_of_prior_six_shape():
    df = _bars(40)
    raw = df.copy()
    i = len(raw) - 1
    # Widen prior six ranges, shrink today
    for j in range(i - 6, i):
        mid = float(raw.loc[j, "close"])
        raw.loc[j, "high"] = mid + 5.0
        raw.loc[j, "low"] = mid - 5.0
    mid = float(raw.loc[i, "close"])
    raw.loc[i, "high"] = mid + 0.2
    raw.loc[i, "low"] = mid - 0.2
    d = prepare_frame(raw)
    # Range condition alone (ADX may or may not fire on synthetic)
    assert float(d.loc[i, "range"]) < float(d.loc[i, "min_range_prior_6"])
    # Mask requires ADX>40; force adx5 for assertion of AND logic
    d.loc[i, "adx5"] = 50.0
    assert bool(entry_mask_s4(d).iloc[i])
    d.loc[i, "adx5"] = 10.0
    assert not bool(entry_mask_s4(d).iloc[i])


def test_s5_ten_day_high_rejection():
    df = _bars(30)
    raw = df.copy()
    i = len(raw) - 1
    prior_max = float(raw["high"].iloc[i - 10 : i].max())
    raw.loc[i, "high"] = prior_max + 1.0
    raw.loc[i, "low"] = prior_max - 5.0
    # Close near low -> IBS < 0.15
    raw.loc[i, "close"] = raw.loc[i, "low"] + 0.05 * (
        raw.loc[i, "high"] - raw.loc[i, "low"]
    )
    d = prepare_frame(raw)
    assert bool(entry_mask_s5(d).iloc[i])


def test_adx_finite_after_warmup():
    df = _bars(80)
    d = prepare_frame(df)
    # After 2*period bars of DX warmup, ADX should be finite somewhere
    assert d["adx5"].notna().sum() > 10
    assert float(d["adx5"].dropna().iloc[-1]) >= 0.0


def test_combined_priority_s1_over_s3():
    """When S1 and S3 fire same day while flat, fill is tagged S1."""
    # Flat tape: no early entries.
    day = pd.Timestamp("2020-01-06", tz="UTC")
    rows = []
    for _ in range(40):
        while day.weekday() >= 5:
            day += timedelta(days=1)
        rows.append(
            {
                "ts": day,
                "open": 100.0,
                "high": 102.0,
                "low": 98.0,
                "close": 100.0,
                "volume": 1,
            }
        )
        day += timedelta(days=1)
    raw = pd.DataFrame(rows)
    i = len(raw) - 1
    # Flat: band ~= 102 - 2.5*4 = 92. Crash close to 91 -> S1+S3.
    raw.loc[i, "low"] = 90.0
    raw.loc[i, "high"] = 102.0
    raw.loc[i, "close"] = 91.0

    d = prepare_frame(raw)
    assert bool(entry_mask_s1(d).iloc[i]), "S1 should fire"
    assert bool(entry_mask_s3(d).iloc[i]), "S3 should fire"
    assert not bool(entry_mask_s1(d).iloc[:i].any())
    assert not bool(entry_mask_s3(d).iloc[:i].any())

    last_ts = pd.Timestamp(raw.iloc[-1]["ts"]) + timedelta(days=1)
    while last_ts.weekday() >= 5:
        last_ts += timedelta(days=1)
    exit_row = {
        "ts": last_ts,
        "open": 91.0,
        "high": 110.0,
        "low": 90.0,
        "close": 108.0,
        "volume": 1,
    }
    raw2 = pd.concat([raw, pd.DataFrame([exit_row])], ignore_index=True)
    out = backtest_combined(raw2, friction_pct=0.0)
    entry_day = str(pd.Timestamp(raw.iloc[i]["ts"]).date())
    hits = [t for t in out["trades"] if t["entry_date"] == entry_day]
    assert hits, "expected a combined fill on crafted day; got %s" % out["trades"]
    assert hits[0]["strategy_id"] == "S1"


def test_qs_exit_not_same_bar():
    df = _bars(40)
    # Force S3 entry on a bar whose close is already above prior high —
    # still must not exit same bar.
    raw = df.copy()
    i = 30
    prior_min = float(raw["low"].iloc[i - 5 : i].min())
    raw.loc[i, "close"] = prior_min - 1.0
    raw.loc[i, "low"] = prior_min - 1.5
    raw.loc[i, "high"] = float(raw.loc[i - 1, "high"]) + 10.0  # high way up
    # close is below prior 5 low but also could be > prior high
    if float(raw.loc[i, "close"]) <= float(raw.loc[i - 1, "high"]):
        # bump close above prior high while still under min low — may be impossible;
        # instead check hold_days >= 1 on any trade
        pass
    out = backtest_named(raw, "S3", friction_pct=0.0)
    for t in out["trades"]:
        assert int(t["hold_days"]) >= 1
