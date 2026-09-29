"""Unit tests for QS RSI+IBS NDX claim-check helpers."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.research.qs_rsi_ndx import book_stats, collect_symbol_trades


class TestQsRsiNdx(unittest.TestCase):
    def _synth(self, n: int = 260) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
        close = 100 + np.linspace(0, 40, n) + rng.normal(0, 0.5, n)
        dip = n // 2
        close[dip : dip + 4] = close[dip] * np.array([0.97, 0.94, 0.91, 0.90])
        high = close + 1.0
        low = close - 1.5
        low[dip : dip + 4] = close[dip : dip + 4] - 0.1
        high[dip : dip + 4] = close[dip : dip + 4] + 2.0
        open_ = close.copy()
        open_[1:] = close[:-1]
        return pd.DataFrame(
            {
                "ts": idx,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": 1e6,
            }
        )

    def test_collect_returns_dataframe(self):
        df = self._synth()
        trades = collect_symbol_trades(df, symbol="TEST", friction_pct=0.0)
        self.assertIsInstance(trades, pd.DataFrame)
        if not trades.empty:
            self.assertIn("gain_pct", trades.columns)
            stats = book_stats(trades)
            self.assertGreaterEqual(stats["n"], 1)

    def test_book_stats_empty(self):
        stats = book_stats(pd.DataFrame())
        self.assertEqual(stats["n"], 0)


if __name__ == "__main__":
    unittest.main()
