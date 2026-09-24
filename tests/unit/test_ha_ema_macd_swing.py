"""Unit tests for HA + 55 EMA channel + MACD swing research translation."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.research.ha_ema_macd_swing import (
    collect_ha_ema_macd_trades,
    ema_array,
    heikin_ashi_arrays,
    macd_arrays,
    max_concurrent,
    occupancy_equity,
)


def _liq_volume(idx: pd.DatetimeIndex, n: int = 1) -> pd.DataFrame:
    cols = [f"S{i}" for i in range(n)] if n > 1 else ["AAA"]
    return pd.DataFrame({c: 5_000_000.0 for c in cols}, index=idx)


class TestHeikinAshiAndEma(unittest.TestCase):
    def test_ha_close_is_ohlc_average(self):
        o = np.array([10.0, 11.0])
        h = np.array([12.0, 13.0])
        lo = np.array([9.0, 10.0])
        c = np.array([11.0, 12.0])
        ha_o, ha_h, ha_l, ha_c = heikin_ashi_arrays(o, h, lo, c)
        self.assertAlmostEqual(float(ha_c[0]), 10.5)
        self.assertAlmostEqual(float(ha_c[1]), 11.5)
        self.assertAlmostEqual(float(ha_o[0]), 10.5)
        self.assertAlmostEqual(float(ha_o[1]), (10.5 + 10.5) / 2.0)
        self.assertGreaterEqual(float(ha_h[1]), float(h[1]))
        self.assertLessEqual(float(ha_l[1]), float(lo[1]))

    def test_ema_constant_series(self):
        x = np.full(20, 7.0)
        ema = ema_array(x, 5)
        self.assertTrue(np.isnan(ema[3]))
        self.assertAlmostEqual(float(ema[-1]), 7.0)

    def test_macd_zero_on_flat(self):
        x = np.full(40, 50.0)
        line, sig, hist = macd_arrays(x)
        self.assertAlmostEqual(float(line[-1]), 0.0, places=6)
        self.assertAlmostEqual(float(sig[-1]), 0.0, places=6)
        self.assertAlmostEqual(float(hist[-1]), 0.0, places=6)


class TestCollectTrades(unittest.TestCase):
    def _panel_from_ohlc(self, idx, o, h, lo, c):
        open_px = pd.DataFrame({"AAA": o}, index=idx)
        high = pd.DataFrame({"AAA": h}, index=idx)
        low = pd.DataFrame({"AAA": lo}, index=idx)
        close = pd.DataFrame({"AAA": c}, index=idx)
        volume = _liq_volume(idx)
        return open_px, high, low, close, volume

    def _uptrend_pullback_break(self, n: int = 80):
        """Rising series, dip into the band, red HA, then green HA breakout."""
        idx = pd.bdate_range("2021-01-04", periods=n)
        close = np.linspace(50.0, 80.0, n)
        # Pullback into the channel near the end.
        close[n - 8] = 76.0
        close[n - 7] = 74.0
        close[n - 6] = 73.0
        close[n - 5] = 72.5  # red-ish
        close[n - 4] = 79.0  # breakout
        close[n - 3] = 79.2
        close[n - 2] = 90.0  # target path
        close[n - 1] = 90.5
        open_px = np.concatenate(([close[0]], close[:-1]))
        # Force a red HA bar just before the breakout: open above close.
        open_px[n - 5] = 75.0
        high = np.maximum(open_px, close) + 0.4
        low = np.minimum(open_px, close) - 0.4
        high[n - 4] = 79.6
        low[n - 4] = 76.0
        return idx, open_px, high, low, close

    def test_breakout_fills_next_open(self):
        idx, o, h, lo, c = self._uptrend_pullback_break()
        open_px, high, low, close, volume = self._panel_from_ohlc(idx, o, h, lo, c)
        trades = collect_ha_ema_macd_trades(
            open_px,
            high,
            low,
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2021-12-31",
            ema_span=8,
            rr=3.0,
            friday_exit=False,
            require_zero_cross=False,
            require_color_flip=True,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            cost_bps_rt=0.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        row = trades.iloc[0]
        self.assertEqual(row["symbol"], "AAA")
        sig_i = int(idx.get_indexer([row["signal_date"]])[0])
        buy_i = int(idx.get_indexer([row["buy_date"]])[0])
        self.assertEqual(buy_i, sig_i + 1)
        self.assertAlmostEqual(float(row["buy_price"]), float(o[buy_i]))
        self.assertIn(row["exit_reason"], ("target", "stop", "macd", "eod"))

    def test_inside_band_does_not_arm(self):
        idx = pd.bdate_range("2021-01-04", periods=40)
        close = np.full(40, 50.0)
        open_px = np.full(40, 50.0)
        high = np.full(40, 50.4)
        low = np.full(40, 49.6)
        panels = self._panel_from_ohlc(idx, open_px, high, low, close)
        trades = collect_ha_ema_macd_trades(
            *panels,
            eval_start="2021-01-04",
            eval_end="2021-12-31",
            ema_span=5,
            friday_exit=False,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
        )
        self.assertEqual(len(trades), 0)

    def test_stop_beats_target_same_bar(self):
        idx, o, h, lo, c = self._uptrend_pullback_break(n=70)
        n = len(idx)
        # Fill day spans both the stop and a huge high.
        o = o.copy()
        h = h.copy()
        lo = lo.copy()
        c = c.copy()
        # Identify likely fill bar (breakout at n-4, fill n-3).
        fill_i = n - 3
        lo[fill_i] = 1.0
        h[fill_i] = 200.0
        c[fill_i] = 80.0
        open_px, high, low, close, volume = self._panel_from_ohlc(idx, o, h, lo, c)
        trades = collect_ha_ema_macd_trades(
            open_px,
            high,
            low,
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2021-12-31",
            ema_span=8,
            rr=3.0,
            friday_exit=False,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(str(trades.iloc[0]["exit_reason"]), "stop")

    def test_gap_stop_at_open(self):
        idx, o, h, lo, c = self._uptrend_pullback_break(n=70)
        n = len(idx)
        o = o.copy()
        h = h.copy()
        lo = lo.copy()
        c = c.copy()
        fill_i = n - 3
        o[fill_i] = 1.0
        h[fill_i] = 2.0
        lo[fill_i] = 0.5
        c[fill_i] = 1.5
        open_px, high, low, close, volume = self._panel_from_ohlc(idx, o, h, lo, c)
        trades = collect_ha_ema_macd_trades(
            open_px,
            high,
            low,
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2021-12-31",
            ema_span=8,
            friday_exit=False,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(str(trades.iloc[0]["exit_reason"]), "gap_stop")
        self.assertAlmostEqual(float(trades.iloc[0]["buy_price"]), 1.0)
        self.assertAlmostEqual(float(trades.iloc[0]["sell_price"]), 1.0)

    def test_friday_exit_uses_close(self):
        idx, o, h, lo, c = self._uptrend_pullback_break(n=80)
        open_px, high, low, close, volume = self._panel_from_ohlc(idx, o, h, lo, c)
        trades = collect_ha_ema_macd_trades(
            open_px,
            high,
            low,
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2021-12-31",
            ema_span=8,
            rr=50.0,
            friday_exit=True,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        row = trades.iloc[0]
        if row["exit_reason"] == "friday":
            self.assertEqual(int(pd.Timestamp(row["sell_date"]).dayofweek), 4)
            sell_i = int(idx.get_indexer([row["sell_date"]])[0])
            self.assertAlmostEqual(float(row["sell_price"]), float(c[sell_i]))

    def test_eval_window_filters_buy_date(self):
        idx, o, h, lo, c = self._uptrend_pullback_break()
        open_px, high, low, close, volume = self._panel_from_ohlc(idx, o, h, lo, c)
        trades = collect_ha_ema_macd_trades(
            open_px,
            high,
            low,
            close,
            volume,
            eval_start="2022-01-01",
            eval_end="2022-12-31",
            ema_span=8,
            friday_exit=False,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
        )
        self.assertEqual(len(trades), 0)


class TestOccupancyHelpers(unittest.TestCase):
    def test_max_concurrent_two_overlap(self):
        trades = pd.DataFrame(
            {
                "symbol": ["A", "B"],
                "buy_date": [pd.Timestamp("2022-01-03"), pd.Timestamp("2022-01-04")],
                "sell_date": [pd.Timestamp("2022-01-10"), pd.Timestamp("2022-01-06")],
            }
        )
        self.assertEqual(max_concurrent(trades), 2)

    def test_occupancy_equity_one_winner(self):
        idx = pd.bdate_range("2022-01-03", periods=6)
        close = pd.DataFrame({"AAA": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]}, index=idx)
        trades = pd.DataFrame(
            {
                "symbol": ["AAA"],
                "buy_date": [idx[1]],
                "sell_date": [idx[4]],
                "buy_price": [10.5],
                "sell_price": [12.0],
            }
        )
        eq, inv, _notes = occupancy_equity(trades, close, max_open=1)
        self.assertGreater(float(eq.iloc[-1]), 1.0)
        self.assertGreater(float(inv.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
