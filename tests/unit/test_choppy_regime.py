"""Unit tests for choppy-regime RSI-2 MOC and 3-day pullback rules."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.research.choppy_regime import (
    atr_wide,
    collect_pullback_trades,
    collect_rsi2_moc_trades,
    consecutive_lower_lows,
    cpi_eve_mask,
    last_rth_15m_closes,
    occupancy_ranked,
    overlay_developing_close,
    rsi_wilder_wide,
    scan_pullback_limits,
    scan_rsi2_moc,
    sized_gain_pct,
    vix_size_mult,
    vix_skip_mask,
)


def _liq_volume(idx: pd.DatetimeIndex, n: int = 1) -> pd.DataFrame:
    cols = [f"S{i}" for i in range(n)] if n > 1 else ["AAA"]
    return pd.DataFrame({c: 5_000_000.0 for c in cols}, index=idx)


class TestRsiWilderAndAtr(unittest.TestCase):
    def test_rsi2_plunges_on_two_down_days(self):
        idx = pd.bdate_range("2022-01-03", periods=10)
        close = pd.DataFrame({"AAA": [100.0] * 7 + [99.0, 80.0, 80.0]}, index=idx)
        rsi = rsi_wilder_wide(close, 2)
        self.assertTrue(np.isfinite(rsi.iloc[-2, 0]))
        self.assertLess(float(rsi.iloc[-2, 0]), 5.0)

    def test_atr_matches_range_on_no_gap(self):
        idx = pd.bdate_range("2022-01-03", periods=12)
        close = pd.DataFrame({"AAA": 50.0}, index=idx)
        high = pd.DataFrame({"AAA": 51.0}, index=idx)
        low = pd.DataFrame({"AAA": 49.0}, index=idx)
        atr = atr_wide(high, low, close, period=10)
        self.assertAlmostEqual(float(atr.iloc[-1, 0]), 2.0, places=6)


class TestLowerLows(unittest.TestCase):
    def test_exactly_three_not_four(self):
        idx = pd.bdate_range("2022-01-03", periods=8)
        # lows: 10, 10, 9, 8, 7, 6, 6, 6  -> at i=5 (value 6) four consecutive lower lows
        low = pd.DataFrame({"AAA": [10.0, 10.0, 9.0, 8.0, 7.0, 6.0, 6.5, 6.6]}, index=idx)
        exact = consecutive_lower_lows(low, n=3, exactly=True)
        any3 = consecutive_lower_lows(low, n=3, exactly=False)
        # i=4: 7<8<9<10 and 10 is not < prior 10 -> exact 3
        self.assertTrue(bool(exact.iloc[4, 0]))
        # i=5: 6<7<8<9 and 9<10 so the 4th lookback is also lower -> not exact
        self.assertFalse(bool(exact.iloc[5, 0]))
        self.assertTrue(bool(any3.iloc[5, 0]))


class TestCpiAndVix(unittest.TestCase):
    def test_cpi_eve_is_prior_session(self):
        cal = pd.bdate_range("2022-01-10", periods=5)  # 10,11,12,13,14
        mask = cpi_eve_mask(cal, cpi_dates=["2022-01-12"])
        self.assertTrue(bool(mask.loc[pd.Timestamp("2022-01-11")]))
        self.assertFalse(bool(mask.loc[pd.Timestamp("2022-01-12")]))

    def test_vix_uses_prior_session(self):
        cal = pd.bdate_range("2022-01-03", periods=4)
        vix = pd.Series([20.0, 20.0, 35.0, 20.0], index=cal)
        prior = vix_skip_mask(cal, vix, vix_max=30.0, use_prior_session=True)
        same = vix_skip_mask(cal, vix, vix_max=30.0, use_prior_session=False)
        self.assertFalse(bool(prior.iloc[2]))
        self.assertTrue(bool(prior.iloc[3]))
        self.assertTrue(bool(same.iloc[2]))


class TestRsi2Moc(unittest.TestCase):
    def _uptrend_then_crash(self):
        # Rising series so SMA200 lags; two down days still close above SMA200.
        idx = pd.bdate_range("2021-01-04", periods=230)
        vals = np.linspace(80.0, 150.0, 230)
        vals[225] = 149.0
        vals[226] = 125.0  # RSI2 < 5, close still above SMA200 (~120)
        vals[227] = 126.0
        vals[228] = 145.0  # reclaim SMA5
        vals[229] = 146.0
        close = pd.DataFrame({"AAA": vals}, index=idx)
        volume = _liq_volume(idx)
        return idx, close, volume

    def test_moc_fill_same_close_sma5_exit(self):
        idx, close, volume = self._uptrend_then_crash()
        trades = collect_rsi2_moc_trades(
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            cost_bps_rt=0.0,
            time_stop=20,
            sma_trend=200,
        )
        self.assertGreaterEqual(len(trades), 1)
        row = trades.iloc[0]
        self.assertEqual(row["symbol"], "AAA")
        self.assertEqual(pd.Timestamp(row["buy_date"]), idx[226])
        self.assertAlmostEqual(float(row["buy_price"]), 125.0, places=4)
        self.assertEqual(row["exit_reason"], "sma5")
        self.assertEqual(pd.Timestamp(row["sell_date"]), idx[228])
        self.assertAlmostEqual(float(row["sell_price"]), 145.0, places=4)

    def test_vix_skip_blocks_entry(self):
        idx, close, volume = self._uptrend_then_crash()
        vix = pd.Series(20.0, index=idx)
        vix.loc[idx[225]] = 40.0  # prior to signal day 226
        trades = collect_rsi2_moc_trades(
            close,
            volume,
            vix=vix,
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=30.0,
            skip_cpi=False,
            time_stop=20,
            sma_trend=200,
        )
        self.assertEqual(len(trades), 0)

    def test_time_stop(self):
        idx = pd.bdate_range("2021-01-04", periods=232)
        vals = np.linspace(80.0, 150.0, 232)
        vals[225] = 149.0
        vals[226] = 125.0
        for j in range(227, 232):
            vals[j] = 126.0  # never reclaim SMA5 / RSI70
        close = pd.DataFrame({"AAA": vals}, index=idx)
        trades = collect_rsi2_moc_trades(
            close,
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=3,
            sma_trend=200,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "time")
        self.assertEqual(int(trades.iloc[0]["hold_bars"]), 3)

    def test_scan_lists_signal_name(self):
        _idx, close, volume = self._uptrend_then_crash()
        asof = close.index[226]
        hits = scan_rsi2_moc(
            close,
            volume,
            asof=asof,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            sma_trend=200,
        )
        self.assertEqual(list(hits["symbol"]), ["AAA"])


class TestPullback(unittest.TestCase):
    def _setup(self):
        n = 120
        idx = pd.bdate_range("2021-01-04", periods=n)
        close_v = np.linspace(40.0, 60.0, n)
        low_v = close_v - 0.25
        high_v = close_v + 0.25
        open_v = close_v.copy()
        # Rising lows => no 3-LL until we implant an exact 3-day cascade at 110.
        low_v[106] = 58.00
        low_v[107] = 58.00
        low_v[108] = 57.00
        low_v[109] = 56.00
        low_v[110] = 55.00
        close_v[106:110] = [59.0, 59.0, 58.5, 58.0]
        high_v[106:110] = [59.5, 59.5, 59.0, 58.5]
        open_v[106:110] = [58.8, 58.8, 58.2, 57.8]
        close_v[110] = 55.20
        high_v[110] = 55.80
        open_v[110] = 56.00
        # Next day tags the limit; up-close vs 55.20; sell following open.
        open_v[111] = 55.10
        low_v[111] = 54.00
        high_v[111] = 56.50
        close_v[111] = 56.00
        open_v[112] = 56.20
        low_v[112] = 55.80
        high_v[112] = 56.80
        close_v[112] = 56.50
        cols = {
            "open": open_v,
            "high": high_v,
            "low": low_v,
            "close": close_v,
        }
        frames = {k: pd.DataFrame({"AAA": v}, index=idx) for k, v in cols.items()}
        return idx, frames

    def test_limit_fill_and_next_open_after_up_close(self):
        idx, f = self._setup()
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=0,
        )
        self.assertGreaterEqual(len(trades), 1)
        row = trades.iloc[0]
        self.assertEqual(pd.Timestamp(row["buy_date"]), idx[111])
        self.assertEqual(row["exit_reason"], "next_open")
        self.assertEqual(pd.Timestamp(row["sell_date"]), idx[112])
        self.assertAlmostEqual(float(row["sell_price"]), 56.2, places=4)
        # Fill at limit (open 51.3 did not gap through; low 49 tagged).
        self.assertLess(float(row["buy_price"]), float(f["close"].iloc[110, 0]))
        self.assertGreater(float(row["buy_price"]), 49.0)

    def test_gap_through_fills_at_open(self):
        idx, f = self._setup()
        f["open"].iloc[111, 0] = 48.0
        f["low"].iloc[111, 0] = 47.5
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertAlmostEqual(float(trades.iloc[0]["buy_price"]), 48.0, places=4)

    def test_miss_if_low_stays_above_limit(self):
        idx, f = self._setup()
        f["open"].iloc[111, 0] = 60.0
        f["low"].iloc[111, 0] = 59.0
        f["high"].iloc[111, 0] = 61.0
        f["close"].iloc[111, 0] = 60.5
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
        )
        self.assertEqual(len(trades), 0)

    def test_scan_emits_limit(self):
        idx, f = self._setup()
        hits = scan_pullback_limits(
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            asof=idx[110],
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
        )
        self.assertEqual(list(hits["symbol"]), ["AAA"])
        self.assertLess(float(hits.iloc[0]["limit"]), float(hits.iloc[0]["close"]))


class TestOccupancy(unittest.TestCase):
    def test_same_day_keeps_more_oversold(self):
        idx = pd.bdate_range("2022-01-03", periods=3)
        trades = pd.DataFrame(
            {
                "symbol": ["AAA", "BBB"],
                "buy_date": [idx[0], idx[0]],
                "sell_date": [idx[2], idx[2]],
                "buy_price": [10.0, 10.0],
                "sell_price": [11.0, 11.0],
                "gain_pct": [10.0, 10.0],
                "exit_reason": ["sma5", "sma5"],
                "hold_bars": [2, 2],
                "entry_score": [4.0, 1.0],
                "strategy": ["rsi2_moc", "rsi2_moc"],
            }
        )
        kept = occupancy_ranked(trades, max_open=1, score_ascending=True)
        self.assertEqual(list(kept["symbol"]), ["BBB"])


class TestVixSizeAndNoMa(unittest.TestCase):
    def test_vix_size_mult_scales_not_skips(self):
        idx, close, volume = TestRsi2Moc()._uptrend_then_crash()
        vix = pd.Series(20.0, index=idx)
        vix.loc[idx[225]] = 40.0
        trades = collect_rsi2_moc_trades(
            close,
            volume,
            vix=vix,
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=20,
            sma_trend=200,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertAlmostEqual(float(trades.iloc[0]["size_mult"]), 0.5, places=4)
        sized = sized_gain_pct(trades)
        self.assertAlmostEqual(float(sized.iloc[0]), float(trades.iloc[0]["gain_pct"]) * 0.5, places=6)

    def test_no_sma_extreme_rsi_still_buys_below_sma50(self):
        idx, close, volume = TestRsi2Moc()._uptrend_then_crash()
        sma50 = float(close["AAA"].rolling(50).mean().iloc[226])
        self.assertLess(125.0, sma50)
        trades = collect_rsi2_moc_trades(
            close,
            volume,
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=20,
            sma_trend=0,
            rsi_entry=2.0,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0]["buy_date"]), idx[226])

    def test_vix_size_helper_floor(self):
        cal = pd.bdate_range("2022-01-03", periods=4)
        vix = pd.Series([20.0, 20.0, 100.0, 20.0], index=cal)
        mult = vix_size_mult(cal, vix, vix_ref=20.0, min_mult=0.25, use_prior_session=True)
        self.assertAlmostEqual(float(mult.iloc[3]), 0.25, places=6)


class TestPullbackTimeStop(unittest.TestCase):
    def test_time_stop_cuts_falling_knife(self):
        idx, f = TestPullback()._setup()
        for j in range(111, 120):
            px = 55.20 - (j - 110) * 0.4
            f["close"].iloc[j, 0] = px
            f["open"].iloc[j, 0] = px + 0.05
            f["low"].iloc[j, 0] = px - 0.4
            f["high"].iloc[j, 0] = px + 0.2
        f["open"].iloc[111, 0] = 55.10
        f["low"].iloc[111, 0] = 54.00
        f["high"].iloc[111, 0] = 55.30
        f["close"].iloc[111, 0] = 54.80
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=6,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["exit_reason"], "time")
        self.assertEqual(int(trades.iloc[0]["hold_bars"]), 6)


class TestDevelopingClose(unittest.TestCase):
    def test_overlay_writes_session_row(self):
        idx = pd.bdate_range("2022-01-03", periods=5)
        close = pd.DataFrame({"AAA": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=idx)
        session = idx[-1]
        out = overlay_developing_close(close, pd.Series({"AAA": 13.5}), session)
        self.assertAlmostEqual(float(out.loc[session, "AAA"]), 13.5, places=6)

    def test_last_rth_prefers_1545(self):
        # 2022-01-03 is EST; 15:45 ET = 20:45 UTC.
        idx = pd.to_datetime(
            ["2022-01-03 20:30:00", "2022-01-03 20:45:00", "2022-01-03 21:00:00"]
        )
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]}, index=idx)
        s = last_rth_15m_closes({"AAA": df}, session=pd.Timestamp("2022-01-03"))
        self.assertAlmostEqual(float(s["AAA"]), 11.0, places=6)


class TestRsiCrossBack(unittest.TestCase):
    def test_does_not_buy_the_crash_bar(self):
        idx, close, volume = TestRsi2Moc()._uptrend_then_crash()
        kw = dict(
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=20,
            sma_trend=200,
        )
        thresh = collect_rsi2_moc_trades(close, volume, **kw)
        xback = collect_rsi2_moc_trades(close, volume, rsi_cross_back=5.0, **kw)
        self.assertGreaterEqual(len(thresh), 1)
        self.assertEqual(pd.Timestamp(thresh.iloc[0]["buy_date"]), idx[226])
        self.assertGreaterEqual(len(xback), 1)
        self.assertGreater(pd.Timestamp(xback.iloc[0]["buy_date"]), idx[226])
        self.assertGreaterEqual(float(xback.iloc[0]["entry_score"]), 5.0)
        self.assertGreaterEqual(int(xback.iloc[0]["wait_bars"]), 1)
        fire_asof = pd.Timestamp(xback.iloc[0]["buy_date"])
        armed = scan_rsi2_moc(
            close,
            volume,
            asof=idx[226],
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            sma_trend=200,
            rsi_cross_back=5.0,
        )
        self.assertEqual(list(armed["symbol"]), ["AAA"])
        self.assertEqual(armed.iloc[0]["status"], "armed")
        fire = scan_rsi2_moc(
            close,
            volume,
            asof=fire_asof,
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            sma_trend=200,
            rsi_cross_back=5.0,
        )
        self.assertEqual(list(fire["symbol"]), ["AAA"])
        self.assertEqual(fire.iloc[0]["status"], "fire")


class TestPullbackCloseLoc(unittest.TestCase):
    def test_rejects_close_near_low(self):
        idx, f = TestPullback()._setup()
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=0,
            min_close_loc=0.7,
        )
        self.assertEqual(len(trades), 0)

    def test_accepts_close_near_high(self):
        idx, f = TestPullback()._setup()
        f["low"].iloc[110, 0] = 54.00
        f["high"].iloc[110, 0] = 55.30
        f["close"].iloc[110, 0] = 55.20
        trades = collect_pullback_trades(
            f["open"],
            f["high"],
            f["low"],
            f["close"],
            _liq_volume(idx),
            eval_start="2021-01-04",
            eval_end="2022-12-31",
            liquid_n=10,
            min_price=1.0,
            min_adv=1.0,
            vix_max=None,
            skip_cpi=False,
            time_stop=0,
            min_close_loc=0.7,
        )
        self.assertGreaterEqual(len(trades), 1)
        self.assertEqual(pd.Timestamp(trades.iloc[0]["buy_date"]), idx[111])


if __name__ == "__main__":
    unittest.main()
