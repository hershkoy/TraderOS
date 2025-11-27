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
#   0: Daily (base), 1: Weekly (resampled)

import backtrader as bt
import math
import pandas as pd
import numpy as np
from typing import Optional

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
        max_delay_weeks=26,            # Max weeks between ignition and TTM confirmation
        
        # Trend filter (Condition C)
        ma10_period=10,                # 10-week MA
        ma30_period=30,                # 30-week MA
        max_extended_pct=0.25,         # Max 25% above 30-week MA
        
        # Risk management
        risk_per_trade=0.01,           # 1% of equity per trade
        atr_period=14,                 # ATR period for stops
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
                print(f'[{timestamp}] {data_time} [{level}] {txt}')
            except (IndexError, AttributeError):
                # Data not available yet (e.g., during __init__)
                print(f'[{timestamp}] [INIT] [{level}] {txt}')

    def debug_log(self, txt):
        if self.p.log_level.upper() == 'DEBUG' and self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            try:
                data_time = self.datetime.datetime(0)
                print(f'[{timestamp}] {data_time} [DEBUG] {txt}')
            except (IndexError, AttributeError):
                # Data not available yet (e.g., during __init__)
                print(f'[{timestamp}] [INIT] [DEBUG] {txt}')

    @staticmethod
    def get_data_requirements():
        """Runner uses this to attach/resample data feeds."""
        return {
            'base_timeframe': 'daily',
            'additional_timeframes': ['weekly'],
            'requires_resampling': True,
        }

    @staticmethod
    def get_description():
        return "Weekly Big Volume + TTM Squeeze Strategy (Rare but Powerful Weekly Setups)"

    def __init__(self):
        # Initialize tracking mixin
        super().__init__()

        # Feeds (runner guarantees order: daily, weekly)
        self.d_d = self.datas[0]  # Daily
        self.d_w = self.datas[1]  # Weekly (resampled)

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
        
        # Track ignition weeks and squeeze state
        self.last_ignition_idx = None
        self.squeeze_vals = []  # Store squeeze values for zero-cross detection
        self.squeeze_on_flags = []  # Store squeeze-on flags
        
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

        self.debug_log("Initialized Weekly Big Volume + TTM Squeeze strategy")

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
            
            # Close above 30-week MA
            ma30_val = float(self.ma30[i])
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
            self.debug_log(f"CONDITION A (Ignition): Error at {i}: {e}")
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

    def _enter_long(self):
        """Enter long position with risk-based position sizing."""
        if self.order is not None:
            return
        
        try:
            w = self.d_w
            d = self.d_d
            
            # Entry price: next daily bar's open (approximation)
            entry_price = float(d.close[0])
            
            # Calculate stop price
            setup_low = float(w.low[-1])  # Low of signal week
            stop_price_struct = setup_low * (1.0 - self.p.buffer_pct)
            atr_val = float(self.atrw[-1])
            stop_price_atr = entry_price - self.p.atr_mult * atr_val
            stop_price = min(stop_price_struct, stop_price_atr)
            
            # Position sizing based on risk
            risk_per_share = max(entry_price - stop_price, 0.01)
            risk_capital = self.broker.getvalue() * self.p.risk_per_trade
            size = math.floor(risk_capital / risk_per_share)
            
            if size <= 0:
                self.debug_log("Position size is 0 or negative, skipping entry")
                return
            
            # Place buy order
            self.order = self.buy(data=self.d_d, size=size)
            self.current_stop = stop_price
            self.entry_price = entry_price
            self.entry_week_idx = len(self.d_w) - 1
            
            self.log(f"BUY SIGNAL | Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Size: {size}")
            self.track_trade_entry(entry_price, size)
            
            # Place stop order
            self.sell(exectype=bt.Order.Stop, price=stop_price, size=size)
            
        except Exception as e:
            self.debug_log(f"Entry error: {e}")

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
                    self.order = self.close(data=self.d_d)
                    if self.entry_price:
                        self.track_trade_exit(float(self.d_d.close[0]), abs(self.position.size))
                    return
            
            # Slow exit: close below 30-week MA
            if len(self.ma30) > 0:
                cl = float(w.close[-1])
                ma30_val = float(self.ma30[-1])
                
                if cl < ma30_val:
                    self.log("EXIT: Close below 30-week MA")
                    self.order = self.close(data=self.d_d)
                    if self.entry_price:
                        self.track_trade_exit(float(self.d_d.close[0]), abs(self.position.size))
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
                    self.log(f'BUY EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                elif order.issell():
                    self.log(f'SELL EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                    if self.position.size == 0:
                        # Position fully closed
                        self.current_stop = None
                        self.entry_price = None
                        self.entry_week_idx = None

            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"ORDER {order.getstatusname()}")

            if self.order is order:
                self.order = None
                
        except Exception as e:
            self.debug_log(f'notify_order error: {e}')

    def notify_trade(self, trade):
        """Handle trade notifications."""
        if trade.isclosed:
            pnl = trade.pnl
            pnl_pct = (pnl / (trade.price * abs(trade.size))) * 100 if trade.price else 0.0
            self.log(f'TRADE CLOSED, pnl={pnl:.2f} ({pnl_pct:.2f}%)')

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

        # Only act on weekly bar closes
        # In Backtrader, weekly resampled data only updates when a new week completes
        # We check if we're at a new weekly bar by comparing datetime (more reliable than length)
        
        current_week_idx = len(self.d_w) - 1
        current_daily_idx = len(self.d_d) - 1
        
        # Check if weekly bar has updated by comparing datetime
        try:
            if len(self.d_w) == 0:
                return  # No weekly data yet
            
            current_week_datetime = self.d_w.datetime.datetime(-1)
            
            # Check if this is a new weekly bar
            if self._prev_week_datetime is not None and current_week_datetime == self._prev_week_datetime:
                # Weekly bar hasn't updated yet - this is normal, we're on a daily bar
                if self.p.log_level.upper() == 'DEBUG' and current_daily_idx % 20 == 0:  # Log every 20th daily bar to reduce noise
                    try:
                        daily_date = self.d_d.datetime.datetime(0) if len(self.d_d) > 0 else "N/A"
                        self.debug_log(f"Daily bar {current_daily_idx} ({daily_date}): Waiting for weekly bar update (current week: {current_week_datetime})")
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
            daily_date = self.d_d.datetime.datetime(0) if len(self.d_d) > 0 else "N/A"
            
            self.debug_log(f"")
            self.debug_log(f"{'='*80}")
            self.debug_log(f"WEEKLY BAR #{current_week_idx} COMPLETED | Date: {week_date} | Daily bar: {current_daily_idx} ({daily_date})")
            self.debug_log(f"  Close: ${close:.2f} | Volume: {volume:,.0f}")
            self.debug_log(f"{'='*80}")
        except Exception as e:
            self.debug_log(f"Error logging weekly bar info: {e}")
        
        # Manage exits first (if in position)
        if self.position:
            self._manage_exits()
            return

        # If no position, check for entry signals
        if self.order is not None:
            return  # Wait for pending order

        # 1) Detect ignition bars (Condition A)
        # In Backtrader, [0] is the current bar, not [-1]
        ignition_result = self._is_ignition_bar(0)  # Current week
        if ignition_result:
            self.last_ignition_idx = current_week_idx
            self.debug_log(f"*** IGNITION BAR DETECTED at week {current_week_idx} ***")

        # 2) If we have an ignition, check for TTM + trend confirmation
        if self.last_ignition_idx is not None:
            # Check delay window
            weeks_since_ignition = current_week_idx - self.last_ignition_idx
            if weeks_since_ignition > self.p.max_delay_weeks:
                # Too old, reset
                if self.p.log_level.upper() == 'DEBUG':
                    self.debug_log(f"Ignition expired: {weeks_since_ignition} weeks > max {self.p.max_delay_weeks} weeks")
                self.last_ignition_idx = None
                return

            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log(f"Checking confirmation conditions (ignition was {weeks_since_ignition} weeks ago at week {self.last_ignition_idx})")

            # Check TTM confirmation (Condition B)
            ttm_ok = self._check_ttm_confirmation(0)
            
            # Check trend filter (Condition C)
            trend_ok = self._check_trend_filter(0)

            if self.p.log_level.upper() == 'DEBUG':
                status = "PASSED" if ttm_ok and trend_ok else "FAILED"
                self.debug_log(f"CONFIRMATION CHECK: TTM={ttm_ok}, Trend={trend_ok} -> {status}")

            if ttm_ok and trend_ok:
                # All conditions met - enter long
                self.log(f"ALL CONDITIONS MET | Ignition: week {self.last_ignition_idx}, TTM: OK, Trend: OK")
                self._enter_long()
                # Reset ignition tracker after entry
                self.last_ignition_idx = None
        else:
            if self.p.log_level.upper() == 'DEBUG':
                self.debug_log("No active ignition bar - skipping confirmation checks")

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

