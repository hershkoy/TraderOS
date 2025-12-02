"""Helper indicators and managers for the Weekly Big Volume + TTM Squeeze strategy."""

import math
from typing import Tuple, List

import backtrader as bt
import numpy as np


class WeightedVolStats(bt.Indicator):
    """Weighted volume statistics (mean and stddev) with exponential decay."""

    lines = ("weighted_mean", "weighted_stddev")
    params = (("period", 52), ("decay_factor", 0.95))

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        if len(self.data) < self.p.period:
            self.lines.weighted_mean[0] = 0.0
            self.lines.weighted_stddev[0] = 0.0
            return

        weights: List[float] = []
        total_weight = 0.0
        for i in range(self.p.period):
            weight = self.p.decay_factor ** (self.p.period - 1 - i)
            weights.append(weight)
            total_weight += weight

        weights = [w / total_weight for w in weights]

        weighted_mean = 0.0
        for i in range(self.p.period):
            value = float(self.data[-self.p.period + i])
            weighted_mean += weights[i] * value

        weighted_variance = 0.0
        for i in range(self.p.period):
            value = float(self.data[-self.p.period + i])
            weighted_variance += weights[i] * (value - weighted_mean) ** 2

        weighted_stddev = math.sqrt(weighted_variance) if weighted_variance > 0 else 0.0
        self.lines.weighted_mean[0] = weighted_mean
        self.lines.weighted_stddev[0] = weighted_stddev


class RobustVolStats(bt.Indicator):
    """Robust volume statistics using Median Absolute Deviation (MAD)."""

    lines = ("median", "mad")
    params = (("period", 12),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        if len(self.data) < self.p.period:
            self.lines.median[0] = 0.0
            self.lines.mad[0] = 1e-9
            return

        window: List[float] = []
        for i in range(1, self.p.period + 1):
            try:
                window.append(float(self.data[-i]))
            except (IndexError, TypeError, ValueError):
                break

        if len(window) < self.p.period:
            self.lines.median[0] = 0.0
            self.lines.mad[0] = 1e-9
            return

        med = float(np.median(window))
        deviations = [abs(v - med) for v in window]
        mad = float(np.median(deviations)) if deviations else 1e-9
        if mad < 1e-9:
            mad = 1e-9

        self.lines.median[0] = med
        self.lines.mad[0] = mad


class LinRegSlope(bt.Indicator):
    """Linear Regression Slope indicator."""

    lines = ("slope",)
    params = (("period", 12),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        if len(self.data) < self.p.period:
            self.lines.slope[0] = 0.0
            return

        n = self.p.period
        sum_x = sum_y = sum_xy = sum_x2 = 0.0
        for i in range(n):
            x = float(i)
            y = float(self.data[-n + i])
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            self.lines.slope[0] = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            self.lines.slope[0] = slope


class VolumeDelta(bt.Indicator):
    """Approximate buy/sell volume split for a bar using candle geometry."""

    lines = ("buy_volume", "sell_volume", "buy_pct", "sell_pct", "delta")
    params = (("min_tick", 1e-4),)

    def __init__(self):
        self.addminperiod(1)

    def next(self):
        try:
            volume = max(float(self.data.volume[0]), 0.0)
            high = float(self.data.high[0])
            low = float(self.data.low[0])
            close = float(self.data.close[0])
        except (IndexError, ValueError, TypeError):
            self.lines.buy_volume[0] = 0.0
            self.lines.sell_volume[0] = 0.0
            self.lines.buy_pct[0] = 0.0
            self.lines.sell_pct[0] = 0.0
            self.lines.delta[0] = 0.0
            return

        rng = max(high - low, self.p.min_tick)
        positional = max(close - low, 0.0)
        buy_est = volume * (positional / rng)
        sell_est = volume - buy_est

        buy_volume = max(buy_est, 0.0)
        sell_volume = -max(sell_est, 0.0)
        total = buy_volume + abs(sell_volume)

        if total > 0:
            buy_pct = (buy_volume / total) * 100.0
            sell_pct = 100.0 - buy_pct
        else:
            buy_pct = 0.0
            sell_pct = 0.0

        delta = buy_volume + sell_volume

        self.lines.buy_volume[0] = buy_volume
        self.lines.sell_volume[0] = sell_volume
        self.lines.buy_pct[0] = buy_pct
        self.lines.sell_pct[0] = sell_pct
        self.lines.delta[0] = delta


class VolumeAnalytics:
    """Manages weekly ignition and daily volume-delta anomaly detection."""

    def __init__(self, strategy):
        self.strategy = strategy
        w = strategy.d_w
        p = strategy.p

        self.vol_weighted_stats = WeightedVolStats(
            w.volume, period=p.vol_lookback, decay_factor=p.vol_weight_decay
        )
        self.vol_weighted_stats_short = WeightedVolStats(
            w.volume, period=p.vol_short_lookback, decay_factor=p.vol_weight_decay
        )
        self.vol_robust_stats = RobustVolStats(w.volume, period=p.vol_short_lookback)
        self.vol_sma = bt.ind.SMA(w.volume, period=p.vol_lookback)
        self.vol_stddev = bt.ind.StdDev(w.volume, period=p.vol_lookback)
        self.vol_highest = bt.ind.Highest(w.volume, period=p.max_volume_lookback)

        d = strategy.d_d
        self.daily_vol_weighted_stats = WeightedVolStats(
            d.volume, period=p.daily_vol_lookback, decay_factor=p.vol_weight_decay
        )
        self.daily_vol_robust_stats = RobustVolStats(d.volume, period=p.daily_vol_short_lookback)
        self.daily_vol_delta = VolumeDelta(d)

    # --- Weekly ignition -----------------------------------------------------
    def _volume_tests(self, idx: int) -> Tuple[List[str], dict, float]:
        strategy = self.strategy
        vol = float(strategy.d_w.volume[idx])
        passed: List[str] = []
        details: dict = {}

        if len(self.vol_robust_stats.median) > 0:
            med = float(self.vol_robust_stats.median[idx])
            mad = float(self.vol_robust_stats.mad[idx])
            if mad > 1e-9 and med > 0:
                robust_z = 0.6745 * (vol - med) / mad
                details["robust_z"] = robust_z
                details["median"] = med
                details["mad"] = mad
                if robust_z >= strategy.p.vol_robust_z_min:
                    passed.append("robust_z")

        if len(self.vol_weighted_stats_short.weighted_mean) > 0:
            mean_short = float(self.vol_weighted_stats_short.weighted_mean[idx])
            std_short = float(self.vol_weighted_stats_short.weighted_stddev[idx])
            if std_short > 0 and mean_short > 0:
                weighted_z = (vol - mean_short) / std_short
                details["weighted_z_short"] = weighted_z
                details["weighted_mean_short"] = mean_short
                details["weighted_stddev_short"] = std_short
                if weighted_z >= strategy.p.vol_zscore_min:
                    passed.append("weighted_z_short")

        if len(self.vol_robust_stats.median) > 0:
            med = float(self.vol_robust_stats.median[idx])
            if med > 0:
                roc = (vol / med) - 1.0
                details["roc"] = roc
                details["roc_pct"] = roc * 100
                if roc >= strategy.p.vol_roc_min:
                    passed.append("roc")

        return passed, details, vol

    def is_ignition_bar(self, idx: int) -> bool:
        strategy = self.strategy
        w = strategy.d_w
        if len(w) == 0:
            strategy.debug_log("CONDITION A (Ignition): No weekly data available")
            return False
        if idx < 0 and len(w) < abs(idx):
            strategy.debug_log(f"CONDITION A (Ignition): Index {idx} out of range (len={len(w)})")
            return False
        if idx >= len(w):
            strategy.debug_log(f"CONDITION A (Ignition): Index {idx} out of range (len={len(w)})")
            return False

        try:
            passed_tests, test_details, vol = self._volume_tests(idx)
            min_passed = 2 if strategy.p.vol_use_multi_threshold else 1
            if len(passed_tests) < min_passed:
                details_str = []
                if test_details.get("robust_z") is not None:
                    details_str.append(
                        f"robust_z={test_details['robust_z']:.2f} "
                        f"(need {strategy.p.vol_robust_z_min:.1f})"
                    )
                if test_details.get("weighted_z_short") is not None:
                    details_str.append(
                        f"weighted_z_short={test_details['weighted_z_short']:.2f} "
                        f"(need {strategy.p.vol_zscore_min:.1f})"
                    )
                if test_details.get("roc") is not None:
                    details_str.append(
                        f"ROC={test_details['roc'] * 100:.1f}% "
                        f"(need {strategy.p.vol_roc_min * 100:.0f}%)"
                    )
                details = ", ".join(details_str) if details_str else "insufficient data"
                strategy.debug_log(
                    f"CONDITION A (Ignition): Volume test FAILED - vol={vol:.0f}, "
                    f"passed={len(passed_tests)}/{min_passed} tests ({details})"
                )
                return False

            passed_str = ", ".join(passed_tests)
            details_str = []
            if "robust_z" in passed_tests:
                details_str.append(
                    f"robust_z={test_details['robust_z']:.2f} "
                    f"(median={test_details['median']:.0f}, MAD={test_details['mad']:.0f})"
                )
            if "weighted_z_short" in passed_tests:
                details_str.append(
                    f"weighted_z_short={test_details['weighted_z_short']:.2f} "
                    f"(mean={test_details['weighted_mean_short']:.0f})"
                )
            if "roc" in passed_tests:
                details_str.append(f"ROC={test_details['roc'] * 100:.1f}%")
            details = " | ".join(details_str)
            strategy.debug_log(
                f"CONDITION A (Ignition): Volume test PASSED - vol={vol:.0f}, "
                f"passed tests: {passed_str} ({details})"
            )

            hi = float(w.high[idx])
            lo = float(w.low[idx])
            cl = float(w.close[idx])
            rng = hi - lo
            if rng <= 0:
                strategy.debug_log(
                    f"CONDITION A (Ignition): Range <= 0 (high={hi:.2f}, low={lo:.2f})"
                )
                return False

            bodypos = (cl - lo) / rng
            if bodypos < strategy.p.body_pos_min:
                strategy.debug_log(
                    f"CONDITION A (Ignition): BodyPos {bodypos:.2f} < "
                    f"{strategy.p.body_pos_min} (close={cl:.2f}, low={lo:.2f}, high={hi:.2f})"
                )
                return False

            ma10_val = float(strategy.ma10[idx]) if len(strategy.ma10) > idx else None
            ma30_val = float(strategy.ma30[idx])
            if ma10_val is not None:
                if cl <= ma10_val and cl <= ma30_val:
                    strategy.debug_log(
                        f"CONDITION A (Ignition): Close {cl:.2f} <= MA10 {ma10_val:.2f} "
                        f"AND <= MA30 {ma30_val:.2f}"
                    )
                    return False
            else:
                if cl <= ma30_val:
                    strategy.debug_log(
                        f"CONDITION A (Ignition): Close {cl:.2f} <= MA30 {ma30_val:.2f}"
                    )
                    return False

            if strategy.p.vol_highest_pct > 0:
                vol_highest = float(self.vol_highest[idx])
                vol_min_threshold = vol_highest * strategy.p.vol_highest_pct
                if vol < vol_min_threshold:
                    pct_str = f"{strategy.p.vol_highest_pct * 100:.0f}%"
                    strategy.debug_log(
                        f"CONDITION A (Ignition): Volume {vol:.0f} < {vol_min_threshold:.0f} "
                        f"({pct_str} of highest {vol_highest:.0f})"
                    )
                    return False

            med_display = test_details.get("median", 0)
            med_ratio = vol / med_display if med_display > 0 else 0.0
            strategy.debug_log(
                f"CONDITION A (Ignition): PASSED - vol={vol:.0f} ({med_ratio:.2f}x median), "
                f"bodypos={bodypos:.2f}, close={cl:.2f} > MA30={ma30_val:.2f}"
            )
            return True
        except (IndexError, ValueError, TypeError) as exc:
            strategy.debug_log(f"[{strategy.symbol}] CONDITION A (Ignition): Error at {idx}: {exc}")
            return False

    # --- Daily anomaly -------------------------------------------------------
    def is_daily_anomaly(self, idx: int) -> bool:
        strategy = self.strategy
        d = strategy.d_d
        if len(d) == 0:
            return False
        if idx < 0 and len(d) < abs(idx):
            return False
        if idx >= len(d):
            return False
        try:
            vol = float(d.volume[idx])
        except (IndexError, ValueError, TypeError):
            return False

        passed_tests: List[str] = []
        details: dict = {}

        if len(self.daily_vol_robust_stats.median) > 0:
            med = float(self.daily_vol_robust_stats.median[idx])
            mad = float(self.daily_vol_robust_stats.mad[idx])
            if mad > 1e-9 and med > 0:
                robust_z = 0.6745 * (vol - med) / mad
                details["robust_z"] = robust_z
                if robust_z >= strategy.p.daily_vol_zscore_min:
                    passed_tests.append("robust_z")

        if len(self.daily_vol_weighted_stats.weighted_mean) > 0:
            mean_short = float(self.daily_vol_weighted_stats.weighted_mean[idx])
            std_short = float(self.daily_vol_weighted_stats.weighted_stddev[idx])
            if std_short > 0 and mean_short > 0:
                weighted_z = (vol - mean_short) / std_short
                details["weighted_z"] = weighted_z
                if weighted_z >= strategy.p.daily_vol_zscore_min:
                    passed_tests.append("weighted_z")

        if len(self.daily_vol_robust_stats.median) > 0:
            med = float(self.daily_vol_robust_stats.median[idx])
            if med > 0:
                roc = (vol / med) - 1.0
                details["roc"] = roc
                if roc >= strategy.p.daily_vol_roc_min:
                    passed_tests.append("roc")

        min_passed = 2 if strategy.p.vol_use_multi_threshold else 1
        if len(passed_tests) >= min_passed:
            detail_str = []
            if "robust_z" in passed_tests and "robust_z" in details:
                detail_str.append(f"robust_z={details['robust_z']:.2f}")
            if "weighted_z" in passed_tests and "weighted_z" in details:
                detail_str.append(f"weighted_z={details['weighted_z']:.2f}")
            if "roc" in passed_tests and "roc" in details:
                detail_str.append(f"ROC={details['roc'] * 100:.1f}%")
            msg = " | ".join(detail_str)
            strategy.debug_log(f"DAILY VOLUME: anomaly detected (vol={vol:.0f}) [{msg}]")
            return True

        strategy.debug_log(
            f"DAILY VOLUME: tests failed (vol={vol:.0f}, passed {len(passed_tests)}/{min_passed})"
        )
        return False

    def handle_new_daily_bar(self):
        strategy = self.strategy
        if not strategy._ready or len(strategy.d_d) == 0:
            return

        idx = -1
        if not self.is_daily_anomaly(idx):
            return

        try:
            buy_pct = float(self.daily_vol_delta.buy_pct[idx])
            buy_volume = float(self.daily_vol_delta.buy_volume[idx])
            sell_volume = float(self.daily_vol_delta.sell_volume[idx])
            total_volume = float(strategy.d_d.volume[idx])
            close_price = float(strategy.d_d.close[idx])
            bar_date = strategy.d_d.datetime.datetime(idx)
        except (IndexError, ValueError, TypeError):
            return

        threshold_pct = strategy.p.daily_buy_pressure_threshold * 100.0
        if buy_pct < threshold_pct:
            strategy.debug_log(
                f"DAILY VOLUME: anomaly but buy% {buy_pct:.1f} < {threshold_pct:.1f}"
            )
            return

        net_delta = buy_volume + sell_volume
        date_str = (
            bar_date.strftime("%Y-%m-%d") if hasattr(bar_date, "strftime") else str(bar_date)
        )
        strategy.log(
            f"[{strategy.symbol}] DAILY BUY VOLUME SPIKE | Date={date_str} | Vol={total_volume:,.0f} "
            f"| Buy={buy_volume:,.0f} ({buy_pct:.1f}%) | Delta={net_delta:,.0f} | Close=${close_price:.2f}"
        )


class TrendFilter:
    """Encapsulates trend-related indicators and checks."""

    def __init__(self, strategy):
        self.strategy = strategy
        w = strategy.d_w
        self.ma10 = bt.ind.SMA(w.close, period=strategy.p.ma10_period)
        self.ma30 = bt.ind.SMA(w.close, period=strategy.p.ma30_period)
        self.atr = bt.ind.ATR(w, period=strategy.p.atr_period)

        strategy.ma10 = self.ma10
        strategy.ma30 = self.ma30
        strategy.atrw = self.atr

    def passes(self, idx: int) -> bool:
        w = self.strategy.d_w
        p = self.strategy.p
        log_debug = self.strategy.debug_log

        if len(w) < 2:
            log_debug(f"CONDITION C (Trend): Not enough weekly data (len={len(w)}, need at least 2)")
            return False

        if idx < 0 and len(w) < abs(idx) + 1:
            log_debug(f"CONDITION C (Trend): Index {idx} out of range (len={len(w)})")
            return False
        if idx >= len(w):
            log_debug(f"CONDITION C (Trend): Index {idx} out of range (len={len(w)})")
            return False

        try:
            cl = float(w.close[idx])
            ma30_curr = float(self.ma30[idx])
            ma30_prev = float(self.ma30[idx - 1])

            if cl <= ma30_curr:
                log_debug(f"CONDITION C (Trend): Close {cl:.2f} <= MA30 {ma30_curr:.2f}")
                return False

            if ma30_curr <= ma30_prev:
                log_debug(
                    f"CONDITION C (Trend): MA30 not rising ({ma30_prev:.2f} -> {ma30_curr:.2f})"
                )
                return False

            max_extended = ma30_curr * (1.0 + p.max_extended_pct)
            if cl > max_extended:
                log_debug(
                    f"CONDITION C (Trend): Close {cl:.2f} > max extended {max_extended:.2f} "
                    f"({p.max_extended_pct * 100:.0f}% above MA30)"
                )
                return False

            log_debug(
                f"CONDITION C (Trend): PASSED - close={cl:.2f} > MA30={ma30_curr:.2f} "
                f"(rising), within {p.max_extended_pct * 100:.0f}% limit"
            )
            return True
        except (IndexError, ValueError, TypeError) as exc:
            log_debug(f"CONDITION C (Trend): Error at {idx}: {exc}")
            return False


class MomentumDetector:
    """Handles TTM squeeze heuristics and confirmation checks."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.mom = LinRegSlope(strategy.d_w.close, period=strategy.p.mom_period)

    def is_squeeze_on(self, idx: int) -> bool:
        mom = self.mom
        if len(mom) == 0:
            return False
        if idx < 0 and len(mom) < abs(idx):
            return False
        if idx >= len(mom):
            return False
        try:
            mom_val = float(mom.slope[idx])
            return abs(mom_val) < 0.5
        except (IndexError, TypeError):
            return False

    def check_confirmation(self, idx: int) -> bool:
        strategy = self.strategy
        p = strategy.p
        log_debug = strategy.debug_log
        mom = self.mom

        if len(mom) < 2:
            log_debug(f"CONDITION B (TTM): Not enough momentum data (len={len(mom)}, need at least 2)")
            return False
        if idx < 0 and len(mom) < abs(idx) + 1:
            log_debug(f"CONDITION B (TTM): Index {idx} out of range (len(mom)={len(mom)})")
            return False
        if idx >= len(mom):
            log_debug(f"CONDITION B (TTM): Index {idx} out of range (len(mom)={len(mom)})")
            return False

        try:
            squeeze_found = False
            squeeze_week = None
            for j in range(1, p.squeeze_lookback + 1):
                lookback_idx = idx - j
                if len(mom) >= abs(lookback_idx) + 1:
                    if self.is_squeeze_on(lookback_idx):
                        squeeze_found = True
                        squeeze_week = lookback_idx
                        break

            if not squeeze_found:
                log_debug(
                    f"CONDITION B (TTM): No squeeze-on found in last {p.squeeze_lookback} weeks"
                )
                return False

            mom_prev = float(mom.slope[idx - 1])
            mom_curr = float(mom.slope[idx])

            if not (mom_prev <= 0 and mom_curr > 0):
                log_debug(
                    f"CONDITION B (TTM): No zero-cross (mom_prev={mom_prev:.4f}, "
                    f"mom_curr={mom_curr:.4f})"
                )
                return False

            if mom_curr <= p.mom_slope_min:
                log_debug(
                    f"CONDITION B (TTM): Momentum {mom_curr:.4f} <= threshold {p.mom_slope_min}"
                )
                return False

            log_debug(
                f"CONDITION B (TTM): PASSED - squeeze found at week {squeeze_week}, "
                f"zero-cross: {mom_prev:.4f} -> {mom_curr:.4f}"
            )
            return True
        except (IndexError, ValueError, TypeError) as exc:
            log_debug(f"CONDITION B (TTM): Error at {idx}: {exc}")
            return False


class EntryPatternEvaluator:
    """Implements weekly entry pattern checks."""

    def __init__(self, strategy):
        self.strategy = strategy

    def retest(self, idx: int) -> bool:
        strategy = self.strategy
        w = strategy.d_w
        if len(w) < 2 or len(strategy.ma10) < 1 or len(strategy.ma30) < 1:
            return False
        try:
            cl = float(w.close[idx])
            op = float(w.open[idx])
            lo = float(w.low[idx])
            ma10_val = float(strategy.ma10[idx])
            ma30_val = float(strategy.ma30[idx])

            if cl <= op:
                strategy.debug_log(
                    f"ENTRY Option A (Retest): Bar closed negative (close={cl:.2f} <= open={op:.2f})"
                )
                return False

            lookback = min(10, len(w) - 1)
            prior_lows = []
            for j in range(1, lookback + 1):
                if idx - j >= 0:
                    prior_lows.append(float(w.low[idx - j]))
            prior_swing_low = min(prior_lows) if prior_lows else None
            ma30_support = lo > ma30_val
            swing_low_support = prior_swing_low is not None and lo > prior_swing_low

            if not (ma30_support or swing_low_support):
                strategy.debug_log(
                    f"ENTRY Option A (Retest): Low {lo:.2f} not above MA30 {ma30_val:.2f} "
                    f"or prior swing low {prior_swing_low:.2f if prior_swing_low else 'N/A'}"
                )
                return False

            near_ma10 = abs(cl - ma10_val) / ma10_val < 0.05
            above_ma30 = cl > ma30_val
            if not (near_ma10 or (cl < ma10_val and above_ma30)):
                strategy.debug_log(
                    f"ENTRY Option A (Retest): Not pulling back to MAs "
                    f"(close={cl:.2f}, MA10={ma10_val:.2f}, MA30={ma30_val:.2f})"
                )
                return False

            if cl > ma30_val * (1.0 + strategy.p.max_extended_pct):
                strategy.debug_log(
                    f"ENTRY Option A (Retest): Too extended above MA30 "
                    f"({cl:.2f} > {ma30_val * (1.0 + strategy.p.max_extended_pct):.2f})"
                )
                return False

            strategy.debug_log("ENTRY Option A (Retest): PASSED - green bar, low above support, retesting MAs")
            return True
        except (IndexError, ValueError, TypeError) as exc:
            strategy.debug_log(f"ENTRY Option A (Retest): Error at {idx}: {exc}")
            return False

    # (Higher-low and breakout methods omitted here for brevity in this snippet,
    #  but they remain identical to the ones currently in your refactored file.)


class IntradayManager:
    """Handles pivot calculations, POC estimation, and 15m entry logic."""

    def __init__(self, strategy):
        self.strategy = strategy

    # methods as in refactored file...


class RiskManager:
    """Encapsulates stop/target management."""

    def __init__(self, strategy):
        self.strategy = strategy

    # manage() as in refactored file...


__all__ = [
    "WeightedVolStats",
    "RobustVolStats",
    "LinRegSlope",
    "VolumeDelta",
    "VolumeAnalytics",
    "TrendFilter",
    "MomentumDetector",
    "EntryPatternEvaluator",
    "IntradayManager",
    "RiskManager",
]


