# strategies/weekly_bigvol_ttm_squeeze.py
# Weekly Big Volume + TTM Squeeze Strategy
# Based on webinar: "Rare but Powerful Weekly Setups"
# 
# Conditions:
# A: Weekly "huge volume" ignition bar
# B: Weekly TTM Squeeze zero-cross (red -> green)
# C: Simple trend filter (10/30-week MAs)
#
# This strategy expects the runner to provide:
#   data0: 15-minute base feed (Cerebro clock)
#   data1: Daily (resampled from 15m)
#   data2: Weekly (resampled from 15m)

"""
Usage:
python backtrader_runner_yaml.py ^
--strategy weekly_bigvol_ttm_squeeze ^
--symbol AEO ^
--provider IB ^
--log-level DEBUG ^
--timeframe 15m ^
--fromdate 2023-06-01 ^
--todate 2025-12-31 ^
--log-to-file
"""

import backtrader as bt
import math

# Import the custom tracking mixin
from utils.custom_tracking import CustomTrackingMixin

# Import squeeze scanner for TTM Squeeze logic
try:
    from utils.squeeze_scanner import squeeze_val, to_weekly
except ImportError:
    # Fallback if import fails
    squeeze_val = None
    to_weekly = None


# Custom Linear Regression Slope Indicator
class LinRegSlope(bt.Indicator):
    """
    Linear Regression Slope indicator.
    Calculates the slope of a linear regression over a period.
    """
    lines = ('slope',)
    params = (('period', 12),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        if len(self.data) < self.p.period:
            self.lines.slope[0] = 0.0
            return

        # Calculate linear regression slope
        # Using least squares: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        n = self.p.period
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        for i in range(n):
            x = float(i)  # Time index (0, 1, 2, ..., n-1)
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


class WeeklyBigVolTTMSqueeze(CustomTrackingMixin, bt.Strategy):
    params = dict(
        # Volume ignition (Condition A)
        vol_lookback=52,              # Weeks for volume SMA (1 year)
        vol_mult=3.0,                 # Volume multiplier threshold
        body_pos_min=0.5,             # Minimum body position (close in top half)
        max_volume_lookback=52,       # Lookback for "highest volume in N weeks"
        
        # TTM Squeeze (Condition B) - using squeeze_scanner logic
        lengthKC=20,                   # Keltner Channel length (matches squeeze_scanner)
        mom_period=12,                 # Momentum period (weeks)
        mom_slope_min=0.0,             # Minimum momentum slope threshold
        squeeze_lookback=10,           # Lookback for squeeze-on detection (weeks)
        max_delay_weeks=26,            # Max weeks between ignition and entry trigger
        max_ignition_to_entry_weeks=26, # Max weeks from ignition to TTM cross (20-26 weeks)
        
        # Trend filter (Condition C) - light weekly filter
        ma10_period=10,                # 10-week MA
        ma30_period=30,                # 30-week MA
        max_extended_pct=0.25,         # Max 25% above 30-week MA
        
        # Intraday entry parameters (Stage 3)
        entry_window_weeks=4,          # Weeks to look for intraday entry after TTM cross
        pivot_lookback_days=20,        # Daily bars used to define higher-timeframe pivot
        poc_lookback_bars=32,          # Number of 15m bars (~1 trading day) for volume profile
        poc_bin_pct=0.003,             # Bin size as % of price (0.3%)
        poc_min_price_step=0.05,       # Minimum bin width in dollars
        poc_buffer_pct=0.0,          # Require POC above pivot by at least 0.1%
        ema_fast_15=8,                 # Fast EMA on 15m for short-term trend
        ema_slow_15=21,                # Slow EMA on 15m for short-term trend
        atr20_period=20,               # Daily ATR20 period (for risk)
        
        # Risk management
        risk_per_trade=0.01,           # 1% of equity per trade
        atr_period=14,                 # Weekly ATR period (for reference)
        atr_mult=1.5,                  # ATR multiplier for stop
        buffer_pct=0.01,               # 1% buffer for structure-based stop
        
        # Backtrader / misc
        size=1,
        commission=0.001,
        printlog=True,
        log_level="INFO",
    )

    def log(self, txt, level='INFO'):
        if self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                data_time = self.datetime.datetime(0)
                print(f'[{timestamp}] {data_time} [{level}] {txt}', flush=True)
            except (IndexError, AttributeError):
                # Data not available yet (e.g., during __init__)
                print(f'[{timestamp}] [INIT] [{level}] {txt}', flush=True)

    def debug_log(self, txt):
        if self.p.log_level.upper() == 'DEBUG' and self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                data_time = self.datetime.datetime(0)
                print(f'[{timestamp}] {data_time} [DEBUG] {txt}', flush=True)
            except (IndexError, AttributeError):
                # Data not available yet (e.g., during __init__)
                print(f'[{timestamp}] [INIT] [DEBUG] {txt}', flush=True)

    @staticmethod
    def get_data_requirements():
        """Runner uses this to attach/resample data feeds."""
        return {
            'base_timeframe': '15m',
            'additional_timeframes': ['daily', 'weekly'],
            'requires_resampling': True,  # runner must resample 15m feed up to daily/weekly
        }

    @staticmethod
    def get_description():
        return "Weekly Big Volume + TTM Squeeze Strategy (Rare but Powerful Weekly Setups)"

    def __init__(self):
        # Initialize tracking mixin
        super().__init__()

        # Feeds (runner guarantees order: 15m base, daily resample, weekly resample)
        self.d_15 = self.datas[0]  # 15-minute base feed
        self.d_d = self.datas[1]   # Daily (resampled)
        self.d_w = self.datas[2]   # Weekly (resampled)

        # Weekly indicators
        w = self.d_w
        
        # Moving averages for trend filter
        self.ma10 = bt.ind.SMA(w.close, period=self.p.ma10_period)
        self.ma30 = bt.ind.SMA(w.close, period=self.p.ma30_period)
        
        # ATR for stops
        self.atrw = bt.ind.ATR(w, period=self.p.atr_period)
        
        # Volume stats
        self.vol_sma = bt.ind.SMA(w.volume, period=self.p.vol_lookback)
        self.vol_highest = bt.ind.Highest(w.volume, period=self.p.max_volume_lookback)
        
        # Momentum (linear regression slope) for TTM Squeeze
        self.mom = LinRegSlope(w.close, period=self.p.mom_period)
        
        # Daily indicators
        d = self.d_d
        self.atr20 = bt.ind.ATR(d, period=self.p.atr20_period)
        self.daily_pivot_high = bt.ind.Highest(d.close, period=self.p.pivot_lookback_days)
        
        # Intraday indicators (15m trend filter)
        intraday = self.d_15
        self.ema_fast_15 = bt.ind.EMA(intraday.close, period=self.p.ema_fast_15)
        self.ema_slow_15 = bt.ind.EMA(intraday.close, period=self.p.ema_slow_15)
        
        # Track ignition weeks and TTM cross
        self.last_ignition_idx = None
        self.last_ignition_week_date = None
        self.last_ttm_cross_idx = None
        self.last_ttm_cross_week_date = None
        
        # Armed state (A + B detected, ready for daily entry)
        self.armed = False
        self.entry_window_end_date = None
        self.pivot_level = None  # Higher-timeframe pivot that intraday must clear
        self.ttm_cross_intraday_start_idx = None
        self.ttm_cross_intraday_start_dt = None
        
        # Position management
        self.order = None
        self.current_stop = None
        self.entry_price = None
        self.entry_week_idx = None
        
        # Track previous weekly bar index and datetime to detect updates
        self._prev_week_idx = -1
        self._prev_week_datetime = None
        
        # Warmup flag
        self._ready = False
        
        # Get symbol name from data feed
        self.symbol = self._get_symbol_name()

        self.debug_log("Initialized Weekly Big Volume + TTM Squeeze strategy (15m base)")
    
    def _get_symbol_name(self):
        """Get symbol name from data feed."""
        try:
            # Try to get name from data feed
            if hasattr(self.d_15, '_name') and self.d_15._name:
                return self.d_15._name
            # Try from parent data if available
            if hasattr(self.d_15, 'p') and hasattr(self.d_15.p, 'dataname'):
                # If dataname is a string, try to extract symbol
                dataname = self.d_15.p.dataname
                if isinstance(dataname, str):
                    # Try to extract symbol from path or name
                    import os
                    basename = os.path.basename(dataname)
                    if basename:
                        return basename.split('.')[0].upper()
            # Fallback: try to get from any data feed
            for data in self.datas:
                if hasattr(data, '_name') and data._name:
                    return data._name
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _calculate_squeeze_val(self):
        """
        Calculate TTM Squeeze 'val' (histogram) using squeeze_scanner logic.
        This is called once per weekly bar to update squeeze values.
        """
        if squeeze_val is None:
            # Fallback: use simple momentum if squeeze_scanner not available
            return float(self.mom.slope[0]) if len(self.mom) > 0 else 0.0
        
        # We need to build a DataFrame from weekly bars for squeeze_scanner
        # This is a bit awkward in Backtrader, so we'll use a simplified approach
        # For now, use the momentum slope as proxy (squeeze_scanner uses linreg too)
        return float(self.mom.slope[0]) if len(self.mom) > 0 else 0.0

    def _is_squeeze_on(self, i):
        """
        Check if squeeze is "on" (BB inside KC) at weekly bar index i.
        Simplified version - in full implementation, would calculate BB and KC.
        For V1, we'll use a heuristic: if momentum is near zero, likely in squeeze.
        In Backtrader, i=0 is the current bar, i=-1 is the previous bar.
        """
        if len(self.mom) == 0:
            return False
        # For negative indices, check if we have enough data
        if i < 0 and len(self.mom) < abs(i):
            return False
        if i >= len(self.mom):
            return False
        # Heuristic: squeeze on when momentum is small (between -0.5 and 0.5 std devs)
        # This is a simplification - full version would calculate BB/KC
        try:
            mom_val = float(self.mom.slope[i])
            return abs(mom_val) < 0.5  # Simplified threshold
        except (IndexError, TypeError):
            return False

    def _is_ignition_bar(self, i):
        """
        Condition A: Check if weekly bar at index i is an "ignition bar".
        Returns True if all ignition conditions are met.
        In Backtrader, i=0 is the current bar, i=-1 is the previous bar.
        """
        if self.p.log_level.upper() == 'DEBUG':
            try:
                week_date = self.d_w.datetime.datetime(i) if len(self.d_w) > 0 else "N/A"
                self.debug_log(f"[{self.symbol}] _is_ignition_bar called for index {i} (week date: {week_date})")
            except:
                self.debug_log(f"[{self.symbol}] _is_ignition_bar called for index {i}")
        
        w = self.d_w
        
        # In Backtrader, [0] is current, [-1] is previous, etc.
        # Check if we have enough data
        if len(w) == 0:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION A (Ignition): No weekly data available")
            return False
        
        # For current bar (i=0), we need at least 1 bar
        # For previous bar (i=-1), we need at least 1 bar
        # For i=-2, we need at least 2 bars, etc.
        if i < 0 and len(w) < abs(i):
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION A (Ignition): Index {i} out of range (len={len(w)})")
            return False
        if i >= len(w):
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION A (Ignition): Index {i} out of range (len={len(w)})")
            return False
        
        try:
            vol = float(w.volume[i])
            volsma = float(self.vol_sma[i])
            vol_threshold = self.p.vol_mult * volsma
            
            if volsma == 0:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION A (Ignition): VolSMA is 0")
                return False
            
            if vol <= vol_threshold:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION A (Ignition): Volume {vol:.0f} <= {vol_threshold:.0f} (need > {self.p.vol_mult}x SMA {volsma:.0f})")
                return False
            
            hi = float(w.high[i])
            lo = float(w.low[i])
            cl = float(w.close[i])
            rng = hi - lo
            
            if rng <= 0:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION A (Ignition): Range <= 0 (high={hi:.2f}, low={lo:.2f})")
                return False
            
            # Body position: close in top half of range
            bodypos = (cl - lo) / rng
            if bodypos < self.p.body_pos_min:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION A (Ignition): BodyPos {bodypos:.2f} < {self.p.body_pos_min} (close={cl:.2f}, low={lo:.2f}, high={hi:.2f})")
                return False
            
            # Close above 10-week MA OR 30-week MA (more generous for early turns)
            ma10_val = float(self.ma10[i]) if len(self.ma10) > i else None
            ma30_val = float(self.ma30[i])
            if ma10_val is not None:
                if cl <= ma10_val and cl <= ma30_val:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"CONDITION A (Ignition): Close {cl:.2f} <= MA10 {ma10_val:.2f} AND <= MA30 {ma30_val:.2f}")
                    return False
            else:
                # Fallback to MA30 only if MA10 not available
                if cl <= ma30_val:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"CONDITION A (Ignition): Close {cl:.2f} <= MA30 {ma30_val:.2f}")
                    return False
            
            # Optional: highest volume in last N weeks
            vol_highest = float(self.vol_highest[i])
            vol_min_threshold = vol_highest * 0.9
            if vol < vol_min_threshold:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION A (Ignition): Volume {vol:.0f} < {vol_min_threshold:.0f} (90% of highest {vol_highest:.0f})")
                return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION A (Ignition): PASSED - vol={vol:.0f} ({vol/volsma:.2f}x SMA), bodypos={bodypos:.2f}, close={cl:.2f} > MA30={ma30_val:.2f}")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"[{self.symbol}] CONDITION A (Ignition): Error at {i}: {e}")
            return False
        except Exception as e:
            # Catch any other unexpected exceptions
            self.debug_log(f"[{self.symbol}] CONDITION A (Ignition): Unexpected error at {i}: {type(e).__name__}: {e}")
            import traceback
            self.debug_log(f"[{self.symbol}] Traceback: {traceback.format_exc()}")
            return False

    def _check_ttm_confirmation(self, i):
        """
        Condition B: Check if TTM Squeeze zero-cross confirmation is met at weekly bar i.
        Returns True if:
        1. There was squeeze-on in last N weeks
        2. Momentum crosses from negative to positive
        3. Momentum is "steep" enough
        In Backtrader, i=0 is the current bar, i=-1 is the previous bar.
        """
        # Need at least 2 bars for zero-cross check (current and previous)
        if len(self.mom) < 2:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION B (TTM): Not enough momentum data (len={len(self.mom)}, need at least 2)")
            return False
        
        # For i=0 (current), we need at least 2 bars (0 and -1)
        # For i=-1 (previous), we need at least 2 bars (-1 and -2)
        if i < 0 and len(self.mom) < abs(i) + 1:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION B (TTM): Index {i} out of range (len(mom)={len(self.mom)})")
            return False
        if i >= len(self.mom):
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION B (TTM): Index {i} out of range (len(mom)={len(self.mom)})")
            return False
        
        try:
            # Check for squeeze-on in last N weeks
            # Look back from current bar (i=0 means look at -1, -2, etc.)
            squeeze_found = False
            squeeze_week = None
            # When i=0, we look back at -1, -2, ..., -squeeze_lookback
            # When i=-1, we look back at -2, -3, ..., -(squeeze_lookback+1)
            for j in range(1, self.p.squeeze_lookback + 1):
                lookback_idx = i - j  # This will be negative for i=0
                # Check if we have enough data for this lookback
                if len(self.mom) >= abs(lookback_idx) + 1:
                    if self._is_squeeze_on(lookback_idx):
                        squeeze_found = True
                        squeeze_week = lookback_idx
                        break
            
            if not squeeze_found:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION B (TTM): No squeeze-on found in last {self.p.squeeze_lookback} weeks")
                return False
            
            # Zero-cross: prev <= 0, curr > 0
            mom_prev = float(self.mom.slope[i - 1])
            mom_curr = float(self.mom.slope[i])
            
            if not (mom_prev <= 0 and mom_curr > 0):
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION B (TTM): No zero-cross (mom_prev={mom_prev:.4f}, mom_curr={mom_curr:.4f})")
                return False
            
            # Steepness check: momentum > minimum threshold
            if mom_curr <= self.p.mom_slope_min:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION B (TTM): Momentum {mom_curr:.4f} <= threshold {self.p.mom_slope_min}")
                return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION B (TTM): PASSED - squeeze found at week {squeeze_week}, zero-cross: {mom_prev:.4f} -> {mom_curr:.4f}")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"CONDITION B (TTM): Error at {i}: {e}")
            return False

    def _check_trend_filter(self, i):
        """
        Condition C: Check trend filter at weekly bar i.
        Returns True if:
        1. Close > 30-week MA
        2. 30-week MA is rising
        3. Optional: not too extended above MA
        In Backtrader, i=0 is the current bar, i=-1 is the previous bar.
        """
        # Need at least 2 bars to check if MA is rising (current and previous)
        if len(self.d_w) < 2:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION C (Trend): Not enough weekly data (len={len(self.d_w)}, need at least 2)")
            return False
        
        # For i=0 (current), we need at least 2 bars (0 and -1)
        # For i=-1 (previous), we need at least 2 bars (-1 and -2)
        if i < 0 and len(self.d_w) < abs(i) + 1:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION C (Trend): Index {i} out of range (len={len(self.d_w)})")
            return False
        if i >= len(self.d_w):
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION C (Trend): Index {i} out of range (len={len(self.d_w)})")
            return False
        
        try:
            w = self.d_w
            cl = float(w.close[i])
            ma30_curr = float(self.ma30[i])
            ma30_prev = float(self.ma30[i - 1])
            
            # Close above 30-week MA
            if cl <= ma30_curr:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION C (Trend): Close {cl:.2f} <= MA30 {ma30_curr:.2f}")
                return False
            
            # 30-week MA is rising
            if ma30_curr <= ma30_prev:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION C (Trend): MA30 not rising ({ma30_prev:.2f} -> {ma30_curr:.2f})")
                return False
            
            # Optional: not too extended (within 25% of MA)
            max_extended = ma30_curr * (1.0 + self.p.max_extended_pct)
            if cl > max_extended:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"CONDITION C (Trend): Close {cl:.2f} > max extended {max_extended:.2f} ({self.p.max_extended_pct*100:.0f}% above MA30)")
                return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"CONDITION C (Trend): PASSED - close={cl:.2f} > MA30={ma30_curr:.2f} (rising), within {self.p.max_extended_pct*100:.0f}% limit")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"CONDITION C (Trend): Error at {i}: {e}")
            return False

    def _check_entry_retest(self, i):
        """
        STAGE 3 - Option A: Retest Entry (Ryley's favorite)
        Weekly bar pulls back toward 10w/30w MAs, low stays above prior swing low or MA30,
        and the bar closes positive.
        """
        w = self.d_w
        
        if len(w) < 2 or len(self.ma10) < 1 or len(self.ma30) < 1:
            return False
        
        try:
            cl = float(w.close[i])
            op = float(w.open[i])
            lo = float(w.low[i])
            hi = float(w.high[i])
            ma10_val = float(self.ma10[i])
            ma30_val = float(self.ma30[i])
            
            # 1. Weekly bar closes positive (green candle)
            if cl <= op:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option A (Retest): Bar closed negative (close={cl:.2f} <= open={op:.2f})")
                return False
            
            # 2. Low stays above MA30 OR above prior swing low
            # Find prior swing low (lowest low in last 5-10 weeks)
            lookback = min(10, len(w) - 1)
            prior_lows = []
            for j in range(1, lookback + 1):
                if i - j >= 0:
                    prior_lows.append(float(w.low[i - j]))
            
            prior_swing_low = min(prior_lows) if prior_lows else None
            ma30_support = lo > ma30_val
            swing_low_support = prior_swing_low is not None and lo > prior_swing_low
            
            if not (ma30_support or swing_low_support):
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option A (Retest): Low {lo:.2f} not above MA30 {ma30_val:.2f} or prior swing low {prior_swing_low:.2f if prior_swing_low else 'N/A'}")
                return False
            
            # 3. Price is pulling back toward MAs (close is near or below MA10, but above MA30)
            # This indicates a retest rather than a breakout
            near_ma10 = abs(cl - ma10_val) / ma10_val < 0.05  # Within 5% of MA10
            above_ma30 = cl > ma30_val
            
            if not (near_ma10 or (cl < ma10_val and above_ma30)):
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option A (Retest): Not pulling back to MAs (close={cl:.2f}, MA10={ma10_val:.2f}, MA30={ma30_val:.2f})")
                return False
            
            # 4. Not too extended above 30-week MA
            if cl > ma30_val * (1.0 + self.p.max_extended_pct):
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option A (Retest): Too extended above MA30 ({cl:.2f} > {ma30_val * (1.0 + self.p.max_extended_pct):.2f})")
                return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option A (Retest): PASSED - green bar, low above support, retesting MAs")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option A (Retest): Error at {i}: {e}")
            return False

    def _check_entry_higher_low(self, i):
        """
        STAGE 3 - Option B: Higher-Low Entry (also Ryley's go-to)
        A higher weekly low forms after the TTM confirmation.
        Price shows clear support and closes > prior week's high OR > that mini pivot.
        """
        w = self.d_w
        
        if len(w) < 3:  # Need at least current, previous, and one more for comparison
            return False
        
        try:
            cl = float(w.close[i])
            lo = float(w.low[i])
            hi = float(w.high[i])
            
            # Find the low point after TTM cross (this is our reference pivot)
            # Look for the lowest low between TTM cross and now
            if self.last_ttm_cross_idx is None:
                return False
            
            # Find lowest low since TTM cross
            ttm_cross_week = self.last_ttm_cross_idx
            if i <= ttm_cross_week:
                return False  # Current bar must be after TTM cross
            
            # Look for the pivot low after TTM cross
            pivot_low = None
            pivot_week = None
            for j in range(ttm_cross_week, i + 1):
                if j >= 0 and j < len(w):
                    low_val = float(w.low[j])
                    if pivot_low is None or low_val < pivot_low:
                        pivot_low = low_val
                        pivot_week = j
            
            if pivot_low is None:
                return False
            
            # 1. Current low is higher than the pivot low (higher low pattern)
            if lo <= pivot_low:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option B (Higher Low): Current low {lo:.2f} not higher than pivot low {pivot_low:.2f}")
                return False
            
            # 2. Close > prior week's high OR close > pivot high
            prev_high = float(w.high[i - 1]) if i > 0 else None
            pivot_high = float(w.high[pivot_week]) if pivot_week is not None else None
            
            close_above_prev_high = prev_high is not None and cl > prev_high
            close_above_pivot = pivot_high is not None and cl > pivot_high
            
            if not (close_above_prev_high or close_above_pivot):
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option B (Higher Low): Close {cl:.2f} not above prev high {prev_high:.2f if prev_high else 'N/A'} or pivot high {pivot_high:.2f if pivot_high else 'N/A'}")
                return False
            
            # 3. Not too extended above 30-week MA
            if len(self.ma30) > i:
                ma30_val = float(self.ma30[i])
                if cl > ma30_val * (1.0 + self.p.max_extended_pct):
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"ENTRY Option B (Higher Low): Too extended above MA30")
                    return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option B (Higher Low): PASSED - higher low formed, close above structure")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option B (Higher Low): Error at {i}: {e}")
            return False

    def _check_entry_breakout(self, i):
        """
        STAGE 3 - Option C: Breakout Entry
        Price breaks above the weekly high of a mini consolidation.
        And not >25% extended above the 30-week MA.
        """
        w = self.d_w
        
        if len(w) < 5:  # Need enough bars to identify consolidation
            return False
        
        try:
            cl = float(w.close[i])
            hi = float(w.high[i])
            
            # Find consolidation high (highest high in last 3-8 weeks, excluding current)
            lookback_start = min(3, len(w) - 1)
            lookback_end = min(8, len(w) - 1)
            consolidation_highs = []
            
            for j in range(lookback_start, lookback_end + 1):
                if i - j > 0:  # Exclude current bar
                    consolidation_highs.append(float(w.high[i - j]))
            
            if not consolidation_highs:
                return False
            
            consolidation_high = max(consolidation_highs)
            
            # 1. Price breaks above consolidation high
            if cl <= consolidation_high:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"ENTRY Option C (Breakout): Close {cl:.2f} not above consolidation high {consolidation_high:.2f}")
                return False
            
            # 2. Not too extended above 30-week MA
            if len(self.ma30) > i:
                ma30_val = float(self.ma30[i])
                if cl > ma30_val * (1.0 + self.p.max_extended_pct):
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"ENTRY Option C (Breakout): Too extended above MA30 ({cl:.2f} > {ma30_val * (1.0 + self.p.max_extended_pct):.2f})")
                    return False
            
            # 3. Ensure we're in an uptrend (close > MA30)
            if len(self.ma30) > i:
                ma30_val = float(self.ma30[i])
                if cl <= ma30_val:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"ENTRY Option C (Breakout): Close {cl:.2f} not above MA30 {ma30_val:.2f}")
                    return False
            
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option C (Breakout): PASSED - broke above consolidation {consolidation_high:.2f}")
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"ENTRY Option C (Breakout): Error at {i}: {e}")
            return False

    def _calculate_pivot_level(self):
        """Define higher-timeframe pivot (highest daily close in lookback window)."""
        if len(self.d_d) == 0:
            return None
        try:
            pivot = float(self.daily_pivot_high[0])
            return pivot if pivot > 0 else None
        except (IndexError, TypeError, ValueError):
            return None
    
    def _compute_poc_15m(self):
        """
        Approximate Point of Control (POC) from recent 15m bars using a simple volume profile.
        Returns price of the volume-weighted bin with the most activity.
        """
        total_bars = len(self.d_15)
        if total_bars == 0:
            return None
        
        # Determine how far back we can look
        start_offset = min(self.p.poc_lookback_bars, total_bars)
        if self.ttm_cross_intraday_start_idx is not None:
            bars_since_cross = total_bars - self.ttm_cross_intraday_start_idx
            if bars_since_cross <= 0:
                return None
            start_offset = min(start_offset, bars_since_cross)
        
        prices = []
        volumes = []
        for i in range(1, start_offset + 1):
            try:
                prices.append(float(self.d_15.close[-i]))
                volumes.append(float(self.d_15.volume[-i]))
            except (IndexError, TypeError, ValueError):
                break
        
        if not prices or not volumes:
            return None
        
        ref_price = prices[0]
        bin_size = max(ref_price * self.p.poc_bin_pct, self.p.poc_min_price_step)
        if bin_size <= 0:
            return None
        
        buckets = {}
        for price, volume in zip(prices, volumes):
            bin_idx = int(price / bin_size)
            buckets[bin_idx] = buckets.get(bin_idx, 0.0) + volume
        
        if not buckets:
            return None
        
        best_bin, _ = max(buckets.items(), key=lambda kv: kv[1])
        poc_price = (best_bin + 0.5) * bin_size
        return poc_price
    
    def _check_intraday_entry(self):
        """Stage 3: trigger entries on 15m bars when volume accepts above the pivot."""
        if not self._ready:
            return
        if not self.armed or self.position or self.order is not None:
            return
        if self.pivot_level is None:
            return
        
        # Entry window handling (calendar days)
        if self.entry_window_end_date is not None:
            try:
                current_dt = self.d_15.datetime.datetime(0)
                if current_dt > self.entry_window_end_date:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"Entry window expired on {self.entry_window_end_date}, current {current_dt}")
                    self._reset_armed_state()
                    self.last_ignition_idx = None
                    self.last_ttm_cross_idx = None
                    return
            except Exception:
                return
        
        poc = self._compute_poc_15m()
        if poc is None:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log("Intraday entry check: POC unavailable")
            return
        
        pivot = self.pivot_level
        if poc <= pivot * (1.0 + self.p.poc_buffer_pct):
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Intraday entry check: POC {poc:.2f} <= pivot threshold {pivot * (1.0 + self.p.poc_buffer_pct):.2f}")
            return
        
        try:
            price = float(self.d_15.close[0])
        except (IndexError, TypeError, ValueError):
            return
        
        if price <= pivot:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Intraday entry check: Price {price:.2f} <= pivot {pivot:.2f}")
            return
        
        try:
            ema_fast = float(self.ema_fast_15[0])
            ema_slow = float(self.ema_slow_15[0])
        except (IndexError, TypeError, ValueError):
            return
        
        if ema_fast <= ema_slow:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Intraday entry check: EMA fast {ema_fast:.2f} <= EMA slow {ema_slow:.2f}")
            return
        
        self.log(f"[{self.symbol}] INTRADAY ENTRY TRIGGERED | Pivot={pivot:.2f} | POC={poc:.2f} | Price={price:.2f}")
        self._enter_long_intraday()
        self._reset_armed_state()
    
    def _reset_armed_state(self):
        """Clear armed state and timers after entry or expiry."""
        self.armed = False
        self.entry_window_end_date = None
        self.pivot_level = None
        self.ttm_cross_intraday_start_idx = None
        self.ttm_cross_intraday_start_dt = None

    def _enter_long_intraday(self):
        """Enter long position from 15m entry signal with risk-based position sizing."""
        if self.order is not None:
            return
        
        try:
            d = self.d_15
            
            # Entry price: current close (or next bar's open in live trading)
            entry_price = float(d.close[0])
            signal_low = float(d.low[0])  # Low of signal bar
            
            # Calculate stop price (tighter of two options)
            # Option 1: Below signal bar low minus 1-2%
            stop_price_signal = signal_low * (1.0 - self.p.buffer_pct)
            
            # Option 2: Entry price - 1.5 * ATR20d
            if len(self.atr20) > 0:
                atr20_val = float(self.atr20[0])
                stop_price_atr = entry_price - self.p.atr_mult * atr20_val
            else:
                stop_price_atr = stop_price_signal
            
            # Use the tighter stop
            stop_price = max(stop_price_signal, stop_price_atr)
            
            # Position sizing based on risk
            risk_per_share = max(entry_price - stop_price, 0.01)
            risk_capital = self.broker.getvalue() * self.p.risk_per_trade
            size = math.floor(risk_capital / risk_per_share)
            
            if size <= 0:
                self.debug_log("Position size is 0 or negative, skipping entry")
                return
            
            # Place buy order
            self.order = self.buy(data=self.d_15, size=size)
            self.current_stop = stop_price
            self.entry_price = entry_price
            self.entry_week_idx = len(self.d_w) - 1
            
            self.log(f"[{self.symbol}] INTRADAY BUY SIGNAL | Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Size: {size}")
            self.track_trade_entry(entry_price, size)
            
            # Place stop order
            self.sell(data=self.d_15, exectype=bt.Order.Stop, price=stop_price, size=size)
            
        except Exception as e:
            self.debug_log(f"Intraday entry error: {e}")

    def _manage_exits(self):
        """Manage position exits based on weekly trend filters."""
        if not self.position or self.order is not None:
            return
        
        try:
            w = self.d_w
            
            # Fast exit: close below 10-week MA
            if len(self.ma10) > 0:
                cl = float(w.close[-1])
                ma10_val = float(self.ma10[-1])
                
                if cl < ma10_val:
                    self.log("EXIT: Close below 10-week MA")
                    self.order = self.close(data=self.d_15)
                    if self.entry_price:
                        self.track_trade_exit(float(self.d_15.close[0]), abs(self.position.size))
                    return
            
            # Slow exit: close below 30-week MA
            if len(self.ma30) > 0:
                cl = float(w.close[-1])
                ma30_val = float(self.ma30[-1])
                
                if cl < ma30_val:
                    self.log("EXIT: Close below 30-week MA")
                    self.order = self.close(data=self.d_15)
                    if self.entry_price:
                        self.track_trade_exit(float(self.d_15.close[0]), abs(self.position.size))
                    return
                    
        except Exception as e:
            self.debug_log(f"Exit management error: {e}")

    def notify_order(self, order):
        """Handle order notifications."""
        try:
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(f'[{self.symbol}] BUY EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                elif order.issell():
                    self.log(f'[{self.symbol}] SELL EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                    if self.position.size == 0:
                        # Position fully closed - reset state to allow new signals
                        self.current_stop = None
                        self.entry_price = None
                        self.entry_week_idx = None
                        # CRITICAL: Clear order reference immediately when position closes
                        # This ensures we don't skip signal checks due to stale order references
                        self.order = None
                        # Reset armed state and ignition/ttm tracking so we can detect NEW signals
                        self._reset_armed_state()
                        # Reset ignition and TTM cross tracking to allow detection of NEW signals
                        # This ensures we don't get stuck looking for signals from the previous trade
                        self.last_ignition_idx = None
                        self.last_ignition_week_date = None
                        self.last_ttm_cross_idx = None
                        self.last_ttm_cross_week_date = None
                        self.debug_log(f"[{self.symbol}] Trade closed - resetting signal tracking to allow new signals")

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"[{self.symbol}] ORDER {order.getstatusname()}")
                # Clear order reference if it was canceled/rejected
                if self.order is order:
                    self.order = None

            # Clear order reference if this was the tracked order
            if self.order is order:
                self.order = None
                
        except Exception as e:
            self.debug_log(f'notify_order error: {e}')

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            pnl = trade.pnl
            # Calculate PnL percentage, avoiding division by zero
            denominator = trade.price * abs(trade.size) if trade.size else 1.0
            pnl_pct = (pnl / denominator) * 100 if denominator > 0 else 0.0
            self.log(f'[{self.symbol}] TRADE CLOSED, pnl={pnl:.2f} ({pnl_pct:.2f}%)')
            # Safety check: clear any stale order references when trade closes
            if not self.position and self.order is not None:
                self.debug_log(f"[{self.symbol}] Clearing order reference after trade close (safety check)")
                self.order = None

    def start(self):
        """Called once at the start of the backtest."""
        try:
            self.broker.setcommission(commission=self.p.commission)
        except Exception:
            pass

    def next(self):
        """Called on each bar."""
        # Track portfolio value
        try:
            self.track_portfolio_value()
        except Exception:
            pass

        # Warmup check
        min_periods = max(self.p.vol_lookback, self.p.ma30_period, self.p.mom_period + 1, self.p.atr_period)
        if not self._ready:
            daily_len = len(self.d_d)
            weekly_len = len(self.d_w)
            if weekly_len >= min_periods:
                self._ready = True
                bars_after_warmup = weekly_len - min_periods
                self.debug_log(f"Warmup complete: daily bars={daily_len}, weekly bars={weekly_len}, min_required={min_periods}")
                if bars_after_warmup == 0:
                    self.debug_log(f"WARNING: No weekly bars available after warmup! Need more data.")
                else:
                    self.debug_log(f"Will process {bars_after_warmup} weekly bars after warmup")
                # Initialize tracking - set to None so first weekly bar after warmup will be processed
                # The datetime check will handle detecting new weekly bars
                self._prev_week_datetime = None
                self._prev_week_idx = -1
                self.debug_log(f"Ready to process weekly bars (starting from warmup completion)")
            else:
                if self.p.log_level.upper() == 'DEBUG' and daily_len % 10 == 0:  # Log every 10th daily bar during warmup
                    self.debug_log(f"Warmup: daily={daily_len}, weekly={weekly_len}, need={min_periods}")
                return

        # INTRADAY LOGIC: Check for entry on every 15m bar while armed
        if self.armed and not self.position and self.order is None:
            self._check_intraday_entry()
        
        # Only act on weekly bar closes
        # In Backtrader, weekly resampled data only updates when a new week completes
        # We check if we're at a new weekly bar by comparing datetime (more reliable than length)
        
        current_week_idx = len(self.d_w) - 1
        current_intraday_idx = len(self.d_15) - 1
        
        # Check if weekly bar has updated by comparing datetime
        try:
            if len(self.d_w) == 0:
                return  # No weekly data yet
            
            current_week_datetime = self.d_w.datetime.datetime(-1)
            
            # Check if this is a new weekly bar
            if self._prev_week_datetime is not None and current_week_datetime == self._prev_week_datetime:
                # Weekly bar hasn't updated yet - this is normal, we're on a daily bar
                # We already checked for daily entry above, so we can return now
                if self.p.log_level.upper() == 'DEBUG' and current_intraday_idx % 40 == 0:  # Log every 40th 15m bar to reduce noise
                    try:
                        intraday_date = self.d_15.datetime.datetime(0) if len(self.d_15) > 0 else "N/A"
                        self.debug_log(f"15m bar {current_intraday_idx} ({intraday_date}): Waiting for weekly bar update (current week: {current_week_datetime})")
                    except Exception:
                        pass
                return  # Weekly bar hasn't updated yet, skip
            
            # Weekly bar has updated!
            self._prev_week_datetime = current_week_datetime
            self._prev_week_idx = current_week_idx
            
        except (IndexError, AttributeError) as e:
            # Weekly data not ready yet
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Weekly data not ready: {e}")
            return
        
        # Debug: Log weekly bar analysis
        w = self.d_w
        try:
            week_date = w.datetime.datetime(-1) if len(w) > 0 else "N/A"
            close = float(w.close[-1]) if len(w) > 0 else 0.0
            volume = float(w.volume[-1]) if len(w) > 0 else 0.0
            intraday_date = self.d_15.datetime.datetime(0) if len(self.d_15) > 0 else "N/A"
            
            # Log daily volumes that make up this weekly bar for debugging
            daily_volumes = []
            if len(self.d_d) > 0:
                try:
                    # Find daily bars that fall within this week
                    # Weekly bar date is the end of the week (typically Friday)
                    week_end = w.datetime.datetime(-1) if len(w) > 0 else None
                    if week_end:
                        # Calculate week start (Monday of the same week)
                        # Get the weekday: 0=Monday, 6=Sunday
                        from datetime import timedelta
                        week_end_weekday = week_end.weekday()
                        # Calculate days to subtract to get to Monday (0=Monday, so subtract weekday)
                        days_to_monday = week_end_weekday
                        week_start = week_end - timedelta(days=days_to_monday)
                        # Normalize to date only for comparison (ignore time component)
                        week_start_date = week_start.date()
                        week_end_date = week_end.date()
                        
                        # Get all daily bars that fall within this week (Monday to Friday)
                        for i in range(min(10, len(self.d_d))):  # Look back up to 10 days to find the week
                            try:
                                daily_date = self.d_d.datetime.datetime(-1 - i)
                                daily_vol = float(self.d_d.volume[-1 - i]) if len(self.d_d.volume) > i else 0.0
                                # Check if this daily bar is within the week (Monday to Friday)
                                # Compare dates only, ignoring time component
                                daily_date_only = daily_date.date() if hasattr(daily_date, 'date') else daily_date
                                if week_start_date <= daily_date_only <= week_end_date:
                                    daily_volumes.append((daily_date.strftime('%Y-%m-%d'), daily_vol))
                            except (IndexError, AttributeError, ValueError):
                                break
                        # Sort by date to show in chronological order
                        daily_volumes.sort(key=lambda x: x[0])
                except Exception as e:
                    self.debug_log(f"Error getting daily volumes: {e}")
                    import traceback
                    self.debug_log(f"Traceback: {traceback.format_exc()}")
            
            self.debug_log(f"")
            self.debug_log(f"{'='*80}")
            self.debug_log(f"[{self.symbol}] WEEKLY BAR #{current_week_idx} COMPLETED | Date: {week_date} | 15m bar: {current_intraday_idx} ({intraday_date})")
            self.debug_log(f"  Close: ${close:.2f} | Volume: {volume:,.0f}")
            if daily_volumes:
                daily_sum = sum(vol for _, vol in daily_volumes)
                self.debug_log(f"  Daily volumes for this week ({len(daily_volumes)} days found, expected 5 for full week):")
                for date_str, vol in daily_volumes:  # Already sorted chronologically
                    self.debug_log(f"    {date_str}: {vol:,.0f}")
                self.debug_log(f"  Daily volume sum: {daily_sum:,.0f} (weekly bar: {volume:,.0f})")
                # Check if there's a discrepancy
                if abs(daily_sum - volume) > volume * 0.1:  # More than 10% difference
                    self.debug_log(f"  WARNING: Daily volume sum ({daily_sum:,.0f}) differs significantly from weekly bar volume ({volume:,.0f})")
                    self.debug_log(f"    Difference: {abs(daily_sum - volume):,.0f} ({abs(daily_sum - volume) / volume * 100:.1f}%)")
                    if len(daily_volumes) < 5:
                        self.debug_log(f"    Only {len(daily_volumes)} daily bars found - may be missing days in daily data")
                    self.debug_log(f"    Note: Weekly bar is calculated from 15m data, daily bars are resampled separately")
                # Warning if daily volumes seem suspiciously low (potential data quality issue)
                if daily_sum > 0 and volume > 0:
                    # Check if any single day has very low volume (might indicate missing data)
                    max_daily_vol = max(vol for _, vol in daily_volumes)
                    if max_daily_vol < 100000 and volume < 1000000:  # Less than 100k per day, less than 1M total
                        self.debug_log(f"  WARNING: Suspiciously low volume detected - may indicate data quality issue")
                        self.debug_log(f"    Max daily volume: {max_daily_vol:,.0f} (expected typically 500k-5M+ for active stocks)")
            self.debug_log(f"{'='*80}")
        except Exception as e:
            self.debug_log(f"Error logging weekly bar info: {e}")
        
        # Manage exits first (if in position)
        if self.position:
            self._manage_exits()
            return

        # If no position, check for entry signals
        # Safety check: if position is closed and order exists, verify it's not stale
        # Only clear if we can confirm the order is completed/canceled (stale reference)
        if not self.position and self.order is not None:
            try:
                # Check if order is actually completed/canceled (stale reference)
                if hasattr(self.order, 'status'):
                    order_status = self.order.status
                    if order_status in [self.order.Completed, self.order.Canceled, self.order.Margin, self.order.Rejected]:
                        self.debug_log(f"[{self.symbol}] Clearing stale order reference (status: {self.order.getstatusname()})")
                        self.order = None
            except Exception:
                # If we can't check, leave it - might be a legitimate pending order
                pass
        
        if self.order is not None:
            self.debug_log(f"[{self.symbol}] Skipping signal checks: pending order")
            return  # Wait for pending order

        # STAGE 1: Detect ignition bars (Weekly Big Volume)
        self.debug_log(f"[{self.symbol}] Checking Condition A (Ignition) for week {current_week_idx}")
        ignition_result = self._is_ignition_bar(-1)  # Evaluate most recent completed week
        if ignition_result:
            self.last_ignition_idx = current_week_idx
            try:
                self.last_ignition_week_date = self.d_w.datetime.datetime(-1)
            except:
                pass
            self.debug_log(f"*** STAGE 1: IGNITION BAR DETECTED at week {current_week_idx} ***")

        # STAGE 2: Check for TTM Confirmation (Zero Cross) - ARM THE TICKER
        if self.last_ignition_idx is not None:
            # Check if TTM crosses from negative to positive
            ttm_cross_result = self._check_ttm_confirmation(-1)
            if ttm_cross_result:
                # TTM cross must occur after ignition
                if self.last_ttm_cross_idx is None or current_week_idx > self.last_ttm_cross_idx:
                    self.last_ttm_cross_idx = current_week_idx
                    try:
                        self.last_ttm_cross_week_date = self.d_w.datetime.datetime(-1)
                        # Calculate entry window end date (TTM cross week + entry_window_weeks)
                        from datetime import timedelta
                        self.entry_window_end_date = self.last_ttm_cross_week_date + timedelta(weeks=self.p.entry_window_weeks)
                    except:
                        pass
                    
                    pivot_candidate = self._calculate_pivot_level()
                    if pivot_candidate is None and len(self.d_d) > 0:
                        try:
                            pivot_candidate = float(self.d_d.close[0])
                        except Exception:
                            pivot_candidate = None
                    self.pivot_level = pivot_candidate
                    
                    # Light weekly trend filter: price above 30-week MA, not too extended
                    w = self.d_w
                    if len(self.ma30) > 0:
                        cl = float(w.close[-1])
                        ma30_val = float(self.ma30[-1])
                        weekly_filter_pass = cl > ma30_val and cl <= ma30_val * (1.0 + self.p.max_extended_pct)
                        if weekly_filter_pass and self.pivot_level is not None:
                            # ARM THE TICKER
                            self.armed = True
                            self.ttm_cross_intraday_start_idx = len(self.d_15) - 1 if len(self.d_15) > 0 else None
                            try:
                                self.ttm_cross_intraday_start_dt = self.d_15.datetime.datetime(0)
                            except Exception:
                                self.ttm_cross_intraday_start_dt = None
                            self.debug_log(f"*** STAGE 2: TTM CROSS DETECTED at week {current_week_idx} | TICKER ARMED FOR 15m ENTRY | Pivot={self.pivot_level:.2f} ***")
                        else:
                            if self.p.log_level.upper() == 'DEBUG':
                                reason = "pivot unavailable" if self.pivot_level is None else "weekly filter failed"
                                self.debug_log(f"TTM cross detected but {reason}: close={cl:.2f}, MA30={ma30_val:.2f}, extended={cl > ma30_val * (1.0 + self.p.max_extended_pct)}")
                                if self.pivot_level is None:
                                    self.debug_log("Need more daily data to set pivot before arming.")
                    else:
                        # If MA30 not ready, still arm (generous)
                        if self.pivot_level is not None:
                            self.armed = True
                            self.ttm_cross_intraday_start_idx = len(self.d_15) - 1 if len(self.d_15) > 0 else None
                            try:
                                self.ttm_cross_intraday_start_dt = self.d_15.datetime.datetime(0)
                            except Exception:
                                self.ttm_cross_intraday_start_dt = None
                            self.debug_log(f"*** STAGE 2: TTM CROSS DETECTED at week {current_week_idx} | TICKER ARMED (MA30 not ready) ***")
                        else:
                            if self.p.log_level.upper() == 'DEBUG':
                                self.debug_log("TTM cross detected but pivot unavailable; waiting for more daily data.")
            
            # Check if ignition expired
            weeks_since_ignition = current_week_idx - self.last_ignition_idx
            if weeks_since_ignition > self.p.max_ignition_to_entry_weeks:
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"Ignition expired: {weeks_since_ignition} weeks > max {self.p.max_ignition_to_entry_weeks} weeks")
                self.last_ignition_idx = None
                self.last_ignition_week_date = None
                self._reset_armed_state()

        # Check if entry window expired (check on every bar, not just weekly)
        if self.armed and self.entry_window_end_date is not None:
            try:
                current_date = self.d_15.datetime.datetime(0)
                if current_date > self.entry_window_end_date:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"Entry window expired: {current_date} > {self.entry_window_end_date}")
                    self._reset_armed_state()
                    self.last_ignition_idx = None
                    self.last_ttm_cross_idx = None
            except:
                pass

    def stop(self):
        """Called at the end of the backtest."""
        try:
            stats = self.get_trade_statistics()
            total_weekly_bars = len(self.d_w)
            total_daily_bars = len(self.d_d)
            self.log(f"SUMMARY: total_trades={stats.get('total_trades')} win_rate={stats.get('win_rate'):.1f}% profit_factor={stats.get('profit_factor'):.2f}", level='INFO')
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Backtest complete: processed {total_weekly_bars} weekly bars from {total_daily_bars} daily bars")
                self.debug_log(f"Last weekly bar index processed: {self._prev_week_idx}")
        except Exception:
            pass

