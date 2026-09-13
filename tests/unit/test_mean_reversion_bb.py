"""Unit tests for Bollinger mean-reversion walk (next-open fills)."""
from __future__ import annotations

import unittest

import pandas as pd

from utils.research.mean_reversion_bb import bollinger_bands, collect_bb_trades


class TestMeanReversionBb(unittest.TestCase):
    def test_bollinger_constant_series_zero_width(self):
        idx = pd.bdate_range("2022-01-03", periods=25)
        close = pd.DataFrame({"AAA": 50.0}, index=idx)
        sma, upper, lower = bollinger_bands(close, period=20, devfactor=2.0)
        self.assertAlmostEqual(float(sma.iloc[-1, 0]), 50.0)
        self.assertAlmostEqual(float(upper.iloc[-1, 0]), 50.0)
        self.assertAlmostEqual(float(lower.iloc[-1, 0]), 50.0)

    def test_collect_one_trade_next_open(self):
        idx = pd.bdate_range("2021-12-01", periods=40)
        vals = [100.0 + 0.5 * ((-1) ** i) for i in range(40)]
        close = pd.DataFrame({"AAA": vals}, index=idx)
        close.iloc[22] = 85.0
        close.iloc[23] = 90.0
        close.iloc[24] = 102.0
        close.iloc[25] = 100.0
        open_px = close.copy()
        volume = pd.DataFrame({"AAA": 1_000_000.0}, index=idx)
        trades = collect_bb_trades(
            open_px,
            close,
            volume,
            period=20,
            devfactor=2.0,
            eval_start="2021-12-01",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            cost_bps_rt=0.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        row = trades.iloc[0]
        self.assertEqual(row["symbol"], "AAA")
        self.assertEqual(row["buy_date"], idx[23])
        self.assertAlmostEqual(float(row["buy_price"]), 90.0)
        self.assertEqual(row["exit_reason"], "sma")
        self.assertEqual(row["sell_date"], idx[25])
        self.assertAlmostEqual(float(row["sell_price"]), 100.0)
        self.assertAlmostEqual(float(row["gain_pct"]), (100.0 / 90.0 - 1.0) * 100.0, places=4)

    def test_eval_window_filters_buy_date(self):
        idx = pd.bdate_range("2021-12-01", periods=40)
        vals = [100.0 + 0.5 * ((-1) ** i) for i in range(40)]
        close = pd.DataFrame({"AAA": vals}, index=idx)
        close.iloc[22] = 85.0
        close.iloc[23] = 90.0
        close.iloc[24] = 102.0
        open_px = close.copy()
        volume = pd.DataFrame({"AAA": 1_000_000.0}, index=idx)
        trades = collect_bb_trades(
            open_px,
            close,
            volume,
            period=20,
            eval_start="2022-06-01",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            cost_bps_rt=0.0,
        )
        self.assertEqual(len(trades), 0)


if __name__ == "__main__":
    unittest.main()
