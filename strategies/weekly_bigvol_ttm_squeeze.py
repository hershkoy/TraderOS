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
            print(f'[{timestamp}] {self.datetime.datetime(0)} [{level}] {txt}')

    def debug_log(self, txt):
        if self.p.log_level.upper() == 'DEBUG' and self.p.printlog:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f'[{timestamp}] {self.datetime.datetime(0)} [DEBUG] {txt}')

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
        self.mom = bt.ind.LinRegSlope(w.close, period=self.p.mom_period)
        
        # Track ignition weeks and squeeze state
        self.last_ignition_idx = None
        self.squeeze_vals = []  # Store squeeze values for zero-cross detection
        self.squeeze_on_flags = []  # Store squeeze-on flags
        
        # Position management
        self.order = None
        self.current_stop = None
        self.entry_price = None
        self.entry_week_idx = None
        
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
            return float(self.mom[0]) if len(self.mom) > 0 else 0.0
        
        # We need to build a DataFrame from weekly bars for squeeze_scanner
        # This is a bit awkward in Backtrader, so we'll use a simplified approach
        # For now, use the momentum slope as proxy (squeeze_scanner uses linreg too)
        return float(self.mom[0]) if len(self.mom) > 0 else 0.0

    def _is_squeeze_on(self, i):
        """
        Check if squeeze is "on" (BB inside KC) at weekly bar index i.
        Simplified version - in full implementation, would calculate BB and KC.
        For V1, we'll use a heuristic: if momentum is near zero, likely in squeeze.
        """
        if i < 0 or len(self.mom) <= abs(i):
            return False
        # Heuristic: squeeze on when momentum is small (between -0.5 and 0.5 std devs)
        # This is a simplification - full version would calculate BB/KC
        mom_val = float(self.mom[i])
        return abs(mom_val) < 0.5  # Simplified threshold

    def _is_ignition_bar(self, i):
        """
        Condition A: Check if weekly bar at index i is an "ignition bar".
        Returns True if all ignition conditions are met.
        """
        w = self.d_w
        
        if i < 0 or len(w) <= abs(i):
            return False
        
        try:
            vol = float(w.volume[i])
            volsma = float(self.vol_sma[i])
            
            if volsma == 0 or vol <= self.p.vol_mult * volsma:
                return False
            
            hi = float(w.high[i])
            lo = float(w.low[i])
            cl = float(w.close[i])
            rng = hi - lo
            
            if rng <= 0:
                return False
            
            # Body position: close in top half of range
            bodypos = (cl - lo) / rng
            if bodypos < self.p.body_pos_min:
                return False
            
            # Close above 30-week MA
            ma30_val = float(self.ma30[i])
            if cl <= ma30_val:
                return False
            
            # Optional: highest volume in last N weeks
            vol_highest = float(self.vol_highest[i])
            if vol < vol_highest * 0.9:  # At least 90% of highest
                return False
            
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"Ignition check error at {i}: {e}")
            return False

    def _check_ttm_confirmation(self, i):
        """
        Condition B: Check if TTM Squeeze zero-cross confirmation is met at weekly bar i.
        Returns True if:
        1. There was squeeze-on in last N weeks
        2. Momentum crosses from negative to positive
        3. Momentum is "steep" enough
        """
        if i < 0 or len(self.mom) <= abs(i) + 1:
            return False
        
        try:
            # Check for squeeze-on in last N weeks
            squeeze_found = False
            for j in range(1, min(self.p.squeeze_lookback + 1, i + 2)):
                if i - j >= 0 and self._is_squeeze_on(i - j):
                    squeeze_found = True
                    break
            
            if not squeeze_found:
                return False
            
            # Zero-cross: prev <= 0, curr > 0
            mom_prev = float(self.mom[i - 1])
            mom_curr = float(self.mom[i])
            
            if not (mom_prev <= 0 and mom_curr > 0):
                return False
            
            # Steepness check: momentum > minimum threshold
            if mom_curr <= self.p.mom_slope_min:
                return False
            
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"TTM confirmation check error at {i}: {e}")
            return False

    def _check_trend_filter(self, i):
        """
        Condition C: Check trend filter at weekly bar i.
        Returns True if:
        1. Close > 30-week MA
        2. 30-week MA is rising
        3. Optional: not too extended above MA
        """
        if i < 0 or len(self.d_w) <= abs(i) + 1:
            return False
        
        try:
            w = self.d_w
            cl = float(w.close[i])
            ma30_curr = float(self.ma30[i])
            ma30_prev = float(self.ma30[i - 1])
            
            # Close above 30-week MA
            if cl <= ma30_curr:
                return False
            
            # 30-week MA is rising
            if ma30_curr <= ma30_prev:
                return False
            
            # Optional: not too extended (within 25% of MA)
            if cl > ma30_curr * (1.0 + self.p.max_extended_pct):
                return False
            
            return True
            
        except (IndexError, ValueError, TypeError) as e:
            self.debug_log(f"Trend filter check error at {i}: {e}")
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
            if len(self.d_w) >= min_periods:
                self._ready = True
                self.debug_log("Warmup complete")
            else:
                return

        # Only act on weekly bar closes
        # In Backtrader, weekly resampled data only updates when a new week completes
        # We check if we're at a new weekly bar by comparing current length to previous
        
        current_week_idx = len(self.d_w) - 1
        
        # Manage exits first (if in position)
        if self.position:
            self._manage_exits()
            return

        # If no position, check for entry signals
        if self.order is not None:
            return  # Wait for pending order

        # 1) Detect ignition bars (Condition A)
        if self._is_ignition_bar(-1):  # Current week
            self.last_ignition_idx = current_week_idx
            self.debug_log(f"Ignition bar detected at week {current_week_idx}")

        # 2) If we have an ignition, check for TTM + trend confirmation
        if self.last_ignition_idx is not None:
            # Check delay window
            weeks_since_ignition = current_week_idx - self.last_ignition_idx
            if weeks_since_ignition > self.p.max_delay_weeks:
                # Too old, reset
                self.last_ignition_idx = None
                return

            # Check TTM confirmation (Condition B)
            ttm_ok = self._check_ttm_confirmation(-1)
            
            # Check trend filter (Condition C)
            trend_ok = self._check_trend_filter(-1)

            if ttm_ok and trend_ok:
                # All conditions met - enter long
                self.log(f"ALL CONDITIONS MET | Ignition: week {self.last_ignition_idx}, TTM: OK, Trend: OK")
                self._enter_long()
                # Reset ignition tracker after entry
                self.last_ignition_idx = None

    def stop(self):
        """Called at the end of the backtest."""
        try:
            stats = self.get_trade_statistics()
            self.log(f"SUMMARY: total_trades={stats.get('total_trades')} win_rate={stats.get('win_rate'):.1f}% profit_factor={stats.get('profit_factor'):.2f}", level='INFO')
        except Exception:
            pass

