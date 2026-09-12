"""IB historical top-of-book helpers (no Gateway)."""
from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pandas as pd

from utils.research.ib_historical_l1 import (
    BAR_DURATION_CAP,
    canonical_symbol,
    chunk_windows,
    combine_bid_ask_bars,
    flags_for_skip,
    last_15m_window,
    metrics_row,
    rth_session_window,
    spread_bps,
    ticks_to_frame,
    to_utc,
    window_metrics,
    bid_ask_from_combo_bars,
    bars_to_frame,
    fetch_historical_bars,
    fetch_historical_ticks,
)

ET = ZoneInfo("America/New_York")


def _bar(ts, o, h, l, c, v=100.0):
    return SimpleNamespace(date=ts, open=o, high=h, low=l, close=c, volume=v)


def test_aliases_apml_tats():
    assert canonical_symbol("APML") == "AMPL"
    assert canonical_symbol("tats") == "TARS"
    assert canonical_symbol("VST") == "VST"


def test_last_15m_window_is_et_rth():
    start, end = last_15m_window("2024-02-09")
    assert start.tzinfo is not None
    assert start.hour == 15 and start.minute == 45
    assert end.hour == 16 and end.minute == 0
    utc_start = to_utc(start)
    # 2024-02-09 is EST (UTC-5)
    assert utc_start.hour == 20 and utc_start.minute == 45


def test_tars_july_last_15m_is_edt():
    start, _end = last_15m_window("2023-07-20")
    utc_start = to_utc(start)
    assert utc_start.hour == 19 and utc_start.minute == 45


def test_rth_window_length():
    start, end = rth_session_window("2024-02-09")
    assert (end - start) == timedelta(hours=6, minutes=30)


def test_chunk_windows_5s_cap():
    start = datetime(2024, 2, 9, 15, 45, tzinfo=ET)
    end = datetime(2024, 2, 9, 16, 0, tzinfo=ET)
    chunks = chunk_windows(start, end, BAR_DURATION_CAP["5 secs"])
    assert len(chunks) == 1
    assert chunks[0] == (start, end)


def test_spread_bps_and_wide_thin_tags():
    assert abs(spread_bps(24.60, 24.64) - 16.260) < 0.05
    quotes = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-02-09T20:45:00Z"), "bid": 14.10, "ask": 14.18, "mid": 14.14, "spread_bps": spread_bps(14.10, 14.18)},
            {"ts": pd.Timestamp("2024-02-09T20:59:00Z"), "bid": 14.12, "ask": 14.20, "mid": 14.16, "spread_bps": spread_bps(14.12, 14.20)},
        ]
    )
    trades = pd.DataFrame(
        [
            {"ts": pd.Timestamp("2024-02-09T20:59:55Z"), "close": 14.19, "volume": 100.0},
        ]
    )
    m = window_metrics(quotes, trades, rail=13.75, fill_px=14.17)
    assert m["wide"] is True
    assert m["thin"] is True
    assert m["paid_the_ask"] is False
    tags = flags_for_skip(m)
    assert "wide_spread" in tags
    assert "thin_print" in tags


def test_extended_vs_rail_tag_for_tars_like():
    quotes = pd.DataFrame(
        [
            {
                "ts": pd.Timestamp("2023-07-20T19:59:00Z"),
                "bid": 23.22,
                "ask": 23.26,
                "mid": 23.24,
                "spread_bps": spread_bps(23.22, 23.26),
            }
        ]
    )
    trades = pd.DataFrame(
        [{"ts": pd.Timestamp("2023-07-20T19:59:50Z"), "close": 23.24, "volume": 800.0}]
    )
    # rail well below last -> extended
    m = window_metrics(quotes, trades, rail=20.50, fill_px=23.24)
    assert m["last_vs_rail_bps"] is not None and m["last_vs_rail_bps"] > 80
    assert "extended_vs_rail" in flags_for_skip(m)
    assert m["wide"] is False
    assert m["thin"] is False


def test_combine_bid_ask_and_combo_bars():
    ts = pd.Timestamp("2024-02-09T20:45:00Z")
    bid = pd.DataFrame([{"ts": ts, "open": 14.10, "high": 14.12, "low": 14.09, "close": 14.11, "volume": 0}])
    ask = pd.DataFrame([{"ts": ts, "open": 14.16, "high": 14.18, "low": 14.15, "close": 14.17, "volume": 0}])
    q = combine_bid_ask_bars(bid, ask)
    assert len(q) == 1
    assert q.iloc[0]["bid"] == 14.11
    assert q.iloc[0]["ask"] == 14.17
    combo = pd.DataFrame(
        [{"ts": ts, "open": 14.10, "high": 14.18, "low": 14.09, "close": 14.16, "volume": 0}]
    )
    q2 = bid_ask_from_combo_bars(combo)
    assert q2.iloc[0]["bid"] == 14.10
    assert q2.iloc[0]["ask"] == 14.16


def test_bars_and_ticks_to_frame():
    ts = datetime(2024, 2, 9, 20, 45, tzinfo=ZoneInfo("UTC"))
    frame = bars_to_frame([_bar(ts, 14.0, 14.2, 13.9, 14.1, 50)], what="TRADES")
    assert list(frame["close"]) == [14.1]
    ticks = ticks_to_frame(
        [
            {
                "time": ts,
                "priceBid": 14.10,
                "priceAsk": 14.14,
                "sizeBid": 4,
                "sizeAsk": 2,
            }
        ],
        what="BID_ASK",
    )
    assert ticks.iloc[0]["bid"] == 14.10
    assert ticks.iloc[0]["spread_bps"] > 0


def test_fetch_historical_bars_pages_and_utc_end():
    ts = datetime(2024, 2, 9, 20, 45, tzinfo=ZoneInfo("UTC"))
    ib = SimpleNamespace(
        reqHistoricalData=lambda *a, **k: [_bar(ts, 14.0, 14.2, 13.9, 14.1)]
    )
    seen = []

    def fmt(dt):
        seen.append(dt)
        return "20240209 21:00:00 UTC"

    start = datetime(2024, 2, 9, 15, 45, tzinfo=ET)
    end = datetime(2024, 2, 9, 16, 0, tzinfo=ET)
    out = fetch_historical_bars(
        ib,
        object(),
        start=start,
        end=end,
        bar_size="5 secs",
        what="BID",
        sleep_s=0,
        end_dt_fn=fmt,
    )
    assert not out.empty
    assert seen[0].tzinfo is not None
    assert to_utc(seen[0]).hour == 21


def test_fetch_historical_ticks_empty_ok():
    ib = SimpleNamespace(reqHistoricalTicks=lambda *a, **k: [])
    start = datetime(2024, 2, 9, 15, 45, tzinfo=ET)
    end = datetime(2024, 2, 9, 16, 0, tzinfo=ET)
    out = fetch_historical_ticks(ib, object(), start=start, end=end, what="BID_ASK", sleep_s=0)
    assert out.empty


def test_qualify_stock_retries_nasdaq():
    from ib_insync import Stock
    from unittest.mock import patch

    from utils.research.ib_historical_l1 import qualify_stock

    calls = []

    def qualify(contract):
        calls.append(getattr(contract, "primaryExchange", None) or "SMART")
        if getattr(contract, "primaryExchange", None) == "NASDAQ":
            return [contract]
        return []

    ib = SimpleNamespace(qualifyContracts=qualify, reqMatchingSymbols=lambda s: [])
    dummy = Stock("AMPL", "SMART", "USD")
    with patch(
        "utils.data.fetch_data.create_ib_contract_with_primary_exchange",
        return_value=dummy,
    ):
        got = qualify_stock(ib, "AMPL")
    assert got.primaryExchange == "NASDAQ"
    assert "NASDAQ" in calls


def test_metrics_row_tags_join():
    payload = {
        "stock": "AMPL",
        "session": "2024-02-09",
        "rail": 13.75,
        "fill_px": 14.17,
        "metrics_last15m": {
            "n_quote": 2,
            "n_trade": 1,
            "median_spread_bps": 25.0,
            "wide": True,
            "thin": True,
        },
        "metrics_session": {"n_quote": 10, "n_trade": 20, "median_spread_bps": 12.0},
        "tags_last15m": ["wide_spread", "thin_print"],
    }
    row = metrics_row(payload, note="fill")
    assert row["tags"] == "wide_spread|thin_print"
    assert row["note"] == "fill"
    assert row["n_quote_last15m"] == 2
