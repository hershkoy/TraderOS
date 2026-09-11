"""ATR Anchored Range session overlay (TradeSeekers-style)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from indicators.atr_anchored_range import (
    atr_anchored_range,
    htf_unique_bars_needed,
    normalize_atr_tf,
    normalize_mode,
    overlay_payload,
    rma,
    session_ord,
    wilder_atr,
)


def test_normalize_mode_and_tf():
    assert normalize_mode("Open") == "open"
    assert normalize_mode("Prior Close") == "prior_close"
    assert normalize_mode("prior_close") == "prior_close"
    assert normalize_atr_tf("1D") == "1d"
    assert normalize_atr_tf("1W") == "1w"
    assert normalize_atr_tf("1M") == "1M"


def test_htf_unique_bars_needed_covers_span_plus_atr_warmup():
    first = pd.Timestamp("2025-04-10 13:30:00", tz="UTC")
    last = pd.Timestamp("2025-07-29 19:45:00", tz="UTC")
    need = htf_unique_bars_needed(first, last, 20, "1d")
    span_days = (last.normalize() - first.normalize()).days + 1
    assert need >= 20 + span_days
    short = htf_unique_bars_needed(
        pd.Timestamp("2025-06-26"), pd.Timestamp("2025-07-02"), 20, "1d"
    )
    assert short < need
    assert htf_unique_bars_needed(None, last, 20, "1d") >= 60


def test_session_ord_intraday_uses_ny_date():
    # 13:30 UTC = 09:30 ET in June (EDT)
    ts = pd.Timestamp("2025-06-13 13:30:00")
    assert session_ord(ts, "1d", intraday=True) == 20250613


def test_session_ord_daily_keeps_utc_calendar_date():
    # Midnight UTC must stay 2025-06-13, not the previous NY evening
    ts = pd.Timestamp("2025-06-13 00:00:00")
    assert session_ord(ts, "1d", intraday=False) == 20250613
    assert session_ord(ts, "1d", intraday=True) == 20250612


def test_rma_sma_seed_then_wilder():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rma(s, 3)
    assert pd.isna(out.iloc[1])
    assert abs(out.iloc[2] - 2.0) < 1e-9
    expected = 2.0 + (1.0 / 3.0) * (4.0 - 2.0)
    assert abs(out.iloc[3] - expected) < 1e-9


def _daily_htf(n: int = 45, start="2025-04-15", atr_level: float = 2.0):
    idx = pd.bdate_range(start, periods=n)
    close = pd.Series(100.0, index=idx)
    # Constant true range = atr_level (high-low), so Wilder ATR warms to atr_level
    high = close + atr_level / 2.0
    low = close - atr_level / 2.0
    open_ = close.copy()
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def _intraday_two_days():
    """Two RTH sessions, two 15m bars each (UTC = 09:30 and 09:45 ET)."""
    stamps = [
        "2025-06-12 13:30:00",
        "2025-06-12 13:45:00",
        "2025-06-13 13:30:00",
        "2025-06-13 13:45:00",
    ]
    idx = pd.to_datetime(stamps)
    return pd.DataFrame(
        {
            "open": [10.0, 10.2, 11.0, 11.1],
            "high": [10.3, 10.4, 11.2, 11.3],
            "low": [9.9, 10.1, 10.9, 11.0],
            "close": [10.2, 10.5, 11.1, 11.2],
        },
        index=idx,
    )


def test_open_mode_mid_is_session_open_and_flat():
    chart = _intraday_two_days()
    htf = _daily_htf()
    out = atr_anchored_range(
        chart,
        htf,
        mode="Open",
        atr_timeframe="1d",
        chart_timeframe="15m",
        period=20,
    )
    # Second session mid = 11.0 open, same on both bars
    assert abs(out["mid"].iloc[2] - 11.0) < 1e-9
    assert abs(out["mid"].iloc[3] - 11.0) < 1e-9
    # First session mid = 10.0
    assert abs(out["mid"].iloc[0] - 10.0) < 1e-9
    assert abs(out["mid"].iloc[1] - 10.0) < 1e-9


def test_prior_close_mode_uses_previous_bar_close():
    chart = _intraday_two_days()
    htf = _daily_htf()
    out = atr_anchored_range(
        chart,
        htf,
        mode="Prior Close",
        atr_timeframe="1d",
        chart_timeframe="15m",
        period=20,
    )
    # First bar of June 13: close[1] = 10.5
    assert abs(out["mid"].iloc[2] - 10.5) < 1e-9
    assert abs(out["mid"].iloc[3] - 10.5) < 1e-9


def test_htf_atr_is_previous_completed_day():
    chart = _intraday_two_days()
    htf = _daily_htf(atr_level=2.0)
    atr = wilder_atr(htf["high"], htf["low"], htf["close"], 20)
    # June 13 session should use June 12 daily ATR (calendar date), not June 13
    june12 = pd.Timestamp("2025-06-12")
    june13 = pd.Timestamp("2025-06-13")
    assert june12 in atr.index
    assert june13 in atr.index
    out = atr_anchored_range(
        chart,
        htf,
        mode="Open",
        atr_timeframe="1d",
        chart_timeframe="15m",
        period=20,
    )
    expected = float(atr.loc[june12])
    assert abs(out["atr"].iloc[2] - expected) < 1e-9
    assert abs(out["high"].iloc[2] - (11.0 + expected / 2.0)) < 1e-9
    assert abs(out["low"].iloc[2] - (11.0 - expected / 2.0)) < 1e-9
    assert abs(out["high2"].iloc[2] - (11.0 + expected)) < 1e-9
    # Must not equal same-day ATR if they differ; here TR is constant so they match
    # after warmup. Force a different last day:
    htf2 = htf.copy()
    htf2.loc[june13, ["high", "low"]] = [110.0, 90.0]
    out2 = atr_anchored_range(
        chart,
        htf2,
        mode="Open",
        atr_timeframe="1d",
        chart_timeframe="15m",
        period=20,
    )
    atr2 = wilder_atr(htf2["high"], htf2["low"], htf2["close"], 20)
    assert abs(out2["atr"].iloc[2] - float(atr2.loc[june12])) < 1e-9
    assert abs(out2["atr"].iloc[2] - float(atr2.loc[june13])) > 0.1


def test_same_tf_daily_uses_current_bar_atr():
    htf = _daily_htf(n=25, atr_level=2.0)
    htf.iloc[-1, htf.columns.get_loc("high")] = 103.0
    htf.iloc[-1, htf.columns.get_loc("low")] = 97.0
    out = atr_anchored_range(
        htf,
        htf,
        mode="Open",
        atr_timeframe="1d",
        chart_timeframe="1d",
        period=20,
    )
    atr = wilder_atr(htf["high"], htf["low"], htf["close"], 20)
    assert abs(out["atr"].iloc[-1] - float(atr.iloc[-1])) < 1e-9
    assert abs(out["mid"].iloc[-1] - float(htf["open"].iloc[-1])) < 1e-9


def test_golden_pocket_and_payload():
    chart = _intraday_two_days()
    htf = _daily_htf()
    out = atr_anchored_range(
        chart, htf, mode="Open", atr_timeframe="1d", chart_timeframe="15m", period=20
    )
    i = 2
    width = float(out["high"].iloc[i] - out["low"].iloc[i])
    assert abs(out["gp_high1"].iloc[i] - (out["low"].iloc[i] + width * 0.61)) < 1e-9
    payload = overlay_payload(out, show_gp=True)
    assert payload["show_gp"] is True
    assert payload["mid"][2] == 11.0
    assert payload["gp_high1"][2] is not None
    assert None in payload["mid"] or all(v is None or isinstance(v, float) for v in payload["mid"])


def test_empty_frame():
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    out = atr_anchored_range(df)
    assert out["mid"].empty
