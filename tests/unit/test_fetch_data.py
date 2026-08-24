"""Unit tests for Alpaca multi-symbol fetch helpers in utils.data.fetch_data."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd

from utils.data.fetch_data import (
    _parse_alpaca_start_date,
    _split_alpaca_multi_symbol_df,
    fetch_many_from_alpaca,
)


def _bars_frame(symbols_rows):
    """Build a MultiIndex-style Alpaca bars frame from (symbol, ts, close) rows."""
    rows = []
    for symbol, ts, close in symbols_rows:
        rows.append(
            {
                "symbol": symbol,
                "timestamp": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 1000.0,
            }
        )
    df = pd.DataFrame(rows)
    return df.set_index(["symbol", "timestamp"])


class TestParseAlpacaStartDate:
    def test_string_naive_becomes_utc(self):
        dt = _parse_alpaca_start_date("2026-08-10")
        assert dt.tzinfo is not None
        assert dt.year == 2026 and dt.month == 8 and dt.day == 10

    def test_none(self):
        assert _parse_alpaca_start_date(None) is None


class TestSplitAlpacaMultiSymbolDf:
    def test_splits_multi_index(self):
        ts = datetime(2026, 8, 20, 4, 0, tzinfo=timezone.utc)
        raw = _bars_frame(
            [
                ("AAPL", ts, 100.0),
                ("MSFT", ts, 200.0),
                ("AAPL", ts.replace(day=21), 101.0),
            ]
        )
        out = _split_alpaca_multi_symbol_df(raw, ["AAPL", "MSFT", "GOOG"], "1d")
        assert set(out) == {"AAPL", "MSFT"}
        assert len(out["AAPL"]) == 2
        assert len(out["MSFT"]) == 1
        assert list(out["AAPL"].columns) == [
            "ts_event",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "instrument_id",
            "venue_id",
            "timeframe",
        ]
        assert out["AAPL"]["instrument_id"].iloc[0] == "AAPL"
        assert out["AAPL"]["venue_id"].iloc[0] == "ALPACA"

    def test_empty_raw(self):
        assert _split_alpaca_multi_symbol_df(pd.DataFrame(), ["AAPL"], "1d") == {}


class TestFetchManyFromAlpaca:
    def test_fetch_many_returns_per_symbol_frames(self):
        ts = datetime(2026, 8, 20, 4, 0, tzinfo=timezone.utc)
        raw_df = _bars_frame(
            [
                ("AAPL", ts, 100.0),
                ("MSFT", ts, 200.0),
            ]
        )
        mock_client = MagicMock()
        mock_client.get_stock_bars.return_value.df = raw_df

        with patch(
            "alpaca.data.historical.StockHistoricalDataClient",
            return_value=mock_client,
        ), patch(
            "alpaca.data.requests.StockBarsRequest",
        ) as mock_req, patch(
            "alpaca.data.timeframe.TimeFrame",
        ), patch(
            "alpaca.data.timeframe.TimeFrameUnit",
        ), patch(
            "alpaca.data.enums.DataFeed",
        ):
            mock_req.return_value = MagicMock()
            out = fetch_many_from_alpaca(
                ["aapl", "msft", "zzz"],
                "1d",
                start_date="2026-08-10",
            )

        assert set(out) == {"AAPL", "MSFT"}
        assert "ZZZ" not in out
        mock_client.get_stock_bars.assert_called_once()
        call_kwargs = mock_req.call_args.kwargs
        assert call_kwargs["symbol_or_symbols"] == ["AAPL", "MSFT", "ZZZ"]
        assert call_kwargs["limit"] >= 3

    def test_empty_symbols(self):
        assert fetch_many_from_alpaca([], "1d", start_date="2026-08-10") == {}
