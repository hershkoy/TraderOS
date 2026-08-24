"""Unit tests for Alpaca multi-symbol universe updates."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from utils.data.update_universe_data import UniverseDataUpdater


def _prepared_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_event": [1_700_000_000_000_000_000],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [1000.0],
            "instrument_id": [symbol],
            "venue_id": ["ALPACA"],
            "timeframe": ["1d"],
        }
    )


class TestUpdateUniverseMultiSymbol:
    def test_multi_symbol_batches_and_queues(self):
        with patch(
            "utils.data.update_universe_data.TickerUniverseManager"
        ) as mock_mgr_cls, patch(
            "utils.data.update_universe_data.fetch_many_from_alpaca"
        ) as mock_fetch_many:
            mock_mgr = MagicMock()
            mock_mgr.get_combined_universe.return_value = [
                "AAPL",
                "MSFT",
                "GOOG",
                "AMZN",
            ]
            mock_mgr_cls.return_value = mock_mgr

            mock_fetch_many.side_effect = [
                {"AAPL": _prepared_frame("AAPL"), "MSFT": _prepared_frame("MSFT")},
                {"GOOG": _prepared_frame("GOOG")},  # AMZN missing -> failed
            ]

            updater = UniverseDataUpdater(provider="alpaca", timeframe="1d")
            updater.db_worker = MagicMock()
            updater.db_worker.get_stats.return_value = {
                "saved": 3,
                "failed": 0,
                "errors": [],
            }

            results = updater.update_universe_data(
                batch_size=2,
                delay_between_batches=0.0,
                delay_between_tickers=0.0,
                multi_symbol=True,
                start_date="2026-08-10",
            )

        assert results["multi_symbol"] is True
        assert results["successful"] == 3
        assert results["failed"] == 1
        assert results["failed_symbols"] == ["AMZN"]
        assert mock_fetch_many.call_count == 2
        assert updater.db_worker.save_data.call_count == 3
        assert updater.db_worker.start.called
        assert updater.db_worker.wait_for_completion.called
        assert updater.db_worker.stop.called

    def test_multi_symbol_falls_back_for_ib(self):
        with patch(
            "utils.data.update_universe_data.TickerUniverseManager"
        ) as mock_mgr_cls, patch.object(
            UniverseDataUpdater, "fetch_ticker_data", return_value=True
        ) as mock_fetch:
            mock_mgr = MagicMock()
            mock_mgr.get_combined_universe.return_value = ["AAPL", "MSFT"]
            mock_mgr_cls.return_value = mock_mgr

            updater = UniverseDataUpdater(provider="ib", timeframe="1d")
            updater.db_worker = MagicMock()
            updater.db_worker.get_stats.return_value = {
                "saved": 0,
                "failed": 0,
                "errors": [],
            }
            updater._ensure_ib_connection = MagicMock(return_value=True)
            updater._cleanup_database_connection = MagicMock()

            results = updater.update_universe_data(
                batch_size=10,
                delay_between_batches=0.0,
                delay_between_tickers=0.0,
                multi_symbol=True,
                max_tickers=2,
            )

        assert results["multi_symbol"] is False
        assert mock_fetch.call_count == 2
        assert results["successful"] == 2
