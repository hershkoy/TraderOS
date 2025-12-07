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
import numpy as np

from utils.backtesting.custom_tracking import CustomTrackingMixin
from .weekly_bigvol_components import (
    WeightedVolStats,
    RobustVolStats,
    LinRegSlope,
    VolumeDelta,
    VolumeAnalytics,
    TrendFilter,
    MomentumDetector,
    EntryPatternEvaluator,
    IntradayManager,
    RiskManager,
)


class WeeklyBigVolTTMSqueeze(CustomTrackingMixin, bt.Strategy):
    params = dict(
        # Volume ignition (Condition A) - using robust multi-threshold detection
        vol_lookback=52,              # Weeks for long-term volume statistics (for weighted stats)
        vol_short_lookback=12,        # Weeks for short-term/robust stats (quarterly, less polluted by outliers)
        vol_zscore_min=2.5,           # Minimum weighted z-score for volume anomaly
        vol_robust_z_min=2.0,         # Minimum robust z-score (MAD-based, resistant to outliers)
        vol_roc_min=0.5,              # Minimum rate of change vs median (50% above median)
        vol_weight_decay=0.85,        # Exponential decay factor for volume weights (0.80-0.90 recommended)
        vol_mult=3.0,                  # Fallback: multiplier threshold (deprecated)
        body_pos_min=0.5,             # Minimum body position (close in top half)
        max_volume_lookback=52,       # Lookback for "highest volume in N weeks" (for reference only)
        vol_highest_pct=0.0,          # Disabled: was too restrictive (0.0 = disabled, was 0.75)
        vol_use_multi_threshold=True, # Use 2-of-3 rule: robust_z OR weighted_z OR ROC
        
        # Daily volume delta anomaly detection (info-only logging)
        daily_vol_lookback=63,         # ~3 months of trading days
        daily_vol_short_lookback=21,   # 1 trading month for robust stats
        daily_vol_zscore_min=2.0,      # Min z-score to call daily volume anomalous
        daily_vol_roc_min=0.4,         # Daily volume at least 40% above median
        daily_buy_pressure_threshold=0.92,  # 92% or more buy-side volume required
        
        # TTM Squeeze (Condition B) - using squeeze logic
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
        stop_loss_pct=0.15,            # Stop loss: 7% below entry price
        take_profit_pct=0.30,           # Take profit: 14% above entry price (2:1 risk reward)
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

        # Modular components
        self.trend_filter = TrendFilter(self)
        self.momentum = MomentumDetector(self)
        self.volume_analytics = VolumeAnalytics(self)
        self.entry_patterns = EntryPatternEvaluator(self)
        self.intraday_manager = IntradayManager(self)
        self.risk_manager = RiskManager(self)
        
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
        self.profit_target = None
        self.entry_price = None
        self.entry_week_idx = None
        self._pending_stop_order = None
        self._pending_limit_order = None
        self._last_entry_price = None  # Saved for PnL calculation in notify_trade
        self._last_entry_size = None
        
        # Track previous weekly bar index and datetime to detect updates
        self._prev_week_idx = -1
        self._prev_week_datetime = None
        self._prev_daily_datetime = None
        
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



    def _is_ignition_bar(self, i):
        """Legacy wrapper retained for compatibility."""
        return self.volume_analytics.is_ignition_bar(i)

    def _check_ttm_confirmation(self, i):
        """Legacy wrapper for momentum detector."""
        return self.momentum.check_confirmation(i)

    def _check_trend_filter(self, i):
        """Legacy wrapper for the modular trend filter."""
        return self.trend_filter.passes(i)

    def _is_daily_volume_anomaly(self, i):
        """Legacy wrapper for the daily volume monitor."""
        return self.volume_analytics.is_daily_anomaly(i)

    def _handle_new_daily_bar(self):
        """Legacy wrapper for compatibility."""
        self.volume_analytics.handle_new_daily_bar()


    def notify_order(self, order):
        """Handle order notifications."""
        try:
            if order.status in [order.Submitted, order.Accepted]:
                return

            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(f'[{self.symbol}] BUY EXECUTED, price={order.executed.price:.2f}, size={order.executed.size}')
                    # Place stop and limit orders after buy order is executed
                    if self.current_stop is not None and self.profit_target is not None and self.position.size > 0:
                        size = abs(self.position.size)
                        # Place stop order for stop loss
                        self._pending_stop_order = self.sell(data=self.d_15, exectype=bt.Order.Stop, price=self.current_stop, size=size)
                        # Place limit order for profit target
                        self._pending_limit_order = self.sell(data=self.d_15, exectype=bt.Order.Limit, price=self.profit_target, size=size)
                        self.debug_log(f"[{self.symbol}] Placed stop order at ${self.current_stop:.2f} and limit order at ${self.profit_target:.2f}")
                elif order.issell():
                    exit_price = order.executed.price
                    exit_size = abs(order.executed.size)
                    # Determine exit reason
                    if order.exectype == bt.Order.Stop:
                        self.log(f'[{self.symbol}] SELL EXECUTED (STOP LOSS), price={exit_price:.2f}, size={exit_size}')
                    elif order.exectype == bt.Order.Limit:
                        self.log(f'[{self.symbol}] SELL EXECUTED (PROFIT TARGET), price={exit_price:.2f}, size={exit_size}')
                    else:
                        self.log(f'[{self.symbol}] SELL EXECUTED, price={exit_price:.2f}, size={exit_size}')
                    
                    # Track the exit for statistics
                    if self.entry_price is not None:
                        self.track_trade_exit(exit_price, exit_size)
                    
                    if self.position.size == 0:
                        # Position fully closed - save entry_price for notify_trade before resetting
                        # (notify_trade is called after notify_order, so we need to preserve it)
                        self._last_entry_price = self.entry_price
                        self._last_entry_size = exit_size
                        # Reset state to allow new signals
                        self.current_stop = None
                        self.profit_target = None
                        self.entry_price = None
                        self.entry_week_idx = None
                        self._pending_stop_order = None
                        self._pending_limit_order = None
                        # CRITICAL: Clear order reference immediately when position closes
                        # This ensures we don't skip signal checks due to stale order references
                        self.order = None
                        # Reset armed state and ignition/ttm tracking so we can detect NEW signals
                        self.intraday_manager.reset_armed_state()
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
            # Calculate PnL percentage based on entry cost
            # Use saved entry_price from notify_order (most reliable)
            if self._last_entry_price and self._last_entry_price > 0:
                entry_price = self._last_entry_price
                trade_size = self._last_entry_size if self._last_entry_size else (abs(trade.size) if trade.size else 0)
            elif self.entry_price and self.entry_price > 0:
                # Fallback: entry_price might still be set if notify_trade is called before reset
                entry_price = self.entry_price
                trade_size = abs(trade.size) if trade.size else 0
            elif hasattr(trade, 'price') and trade.price > 0:
                # Last resort: use trade.price (might be average entry price)
                entry_price = trade.price
                trade_size = abs(trade.size) if trade.size else 0
            else:
                # Cannot calculate percentage without entry price
                entry_price = 0
                trade_size = abs(trade.size) if trade.size else 0
                self.debug_log(f"[{self.symbol}] WARNING: Cannot calculate PnL % - entry price unknown")
            
            # Calculate entry cost and PnL percentage
            entry_cost = entry_price * trade_size if entry_price > 0 and trade_size > 0 else 1.0
            pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0.0
            
            # Clear saved values after use
            self._last_entry_price = None
            self._last_entry_size = None
            
            self.log(f'[{self.symbol}] TRADE CLOSED, pnl=${pnl:.2f} ({pnl_pct:.2f}%) | Entry: ${entry_price:.2f}, Size: {trade_size}')
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

        # Detect new daily bars and run daily volume delta checks once per bar
        if len(self.d_d) > 0:
            try:
                current_daily_datetime = self.d_d.datetime.datetime(-1)
                if self._prev_daily_datetime is None or current_daily_datetime != self._prev_daily_datetime:
                    self._prev_daily_datetime = current_daily_datetime
                    self.volume_analytics.handle_new_daily_bar()
            except (IndexError, AttributeError):
                pass

        # INTRADAY LOGIC: Check for entry on every 15m bar while armed
        if self.armed and not self.position and self.order is None:
            # Log periodic status when armed (every 100th 15m bar to reduce noise)
            current_intraday_idx = len(self.d_15) - 1 if len(self.d_15) > 0 else 0
            if self.p.log_level.upper() == 'DEBUG' and current_intraday_idx % 100 == 0:
                try:
                    intraday_date = self.d_15.datetime.datetime(0) if len(self.d_15) > 0 else "N/A"
                    pivot_str = f"${self.pivot_level:.2f}" if self.pivot_level else "N/A"
                    window_str = f"until {self.entry_window_end_date}" if self.entry_window_end_date else "no limit"
                    self.debug_log(f"[{self.symbol}] ARMED - Checking for intraday entry | Pivot: {pivot_str} | Window: {window_str} | 15m bar: {current_intraday_idx} ({intraday_date})")
                except Exception:
                    pass
            self.intraday_manager.intraday_entry_check()
        
        # Check exits on every 15m bar (stop loss and profit target)
        if self.position:
            self.risk_manager.manage()
            # If position was closed by exit management, return early
            if not self.position:
                return
        
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
        ignition_result = self.volume_analytics.is_ignition_bar(-1)  # Evaluate most recent completed week
        
        # Log Condition A status at INFO level
        try:
            w = self.d_w
            vol = float(w.volume[-1]) if len(w.volume) > 0 else 0.0
            close = float(w.close[-1]) if len(w.close) > 0 else 0.0
            ma30_val = float(self.ma30[-1]) if len(self.ma30) > 0 else 0.0
            
            # Get volume stats for context
            vol_median = 0.0
            if hasattr(self.volume_analytics, 'vol_robust_stats') and len(self.volume_analytics.vol_robust_stats.median) > 0:
                vol_median = float(self.volume_analytics.vol_robust_stats.median[-1])
            vol_ratio = vol / vol_median if vol_median > 0 else 0.0
            
            if ignition_result:
                self.log(f"[{self.symbol}] CONDITION A (Ignition): PASSED | Week {current_week_idx} | Vol={vol:,.0f} ({vol_ratio:.2f}x median) | Close=${close:.2f} > MA30=${ma30_val:.2f}", level='INFO')
                self.last_ignition_idx = current_week_idx
                try:
                    self.last_ignition_week_date = self.d_w.datetime.datetime(-1)
                except:
                    pass
                self.debug_log(f"*** STAGE 1: IGNITION BAR DETECTED at week {current_week_idx} ***")
            else:
                self.debug_log(f"[{self.symbol}] CONDITION A (Ignition): FAILED | Week {current_week_idx} | Vol={vol:,.0f} ({vol_ratio:.2f}x median) | Close=${close:.2f} vs MA30=${ma30_val:.2f}")
        except Exception as e:
            self.debug_log(f"Error logging Condition A info: {e}")
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
            ttm_cross_result = self.momentum.check_confirmation(-1)
            
            # Log Condition B status at INFO level
            try:
                mom = self.momentum.mom
                if len(mom) >= 2:
                    mom_prev = float(mom.momentum[-2]) if len(mom.momentum) >= 2 else 0.0
                    mom_curr = float(mom.momentum[-1]) if len(mom.momentum) >= 1 else 0.0
                    weeks_since_ignition = current_week_idx - self.last_ignition_idx
                    
                    if ttm_cross_result:
                        self.log(f"[{self.symbol}] CONDITION B (TTM Cross): PASSED | Week {current_week_idx} | Momentum: {mom_prev:.4f} -> {mom_curr:.4f} | {weeks_since_ignition} weeks after ignition", level='INFO')
                    else:
                        self.debug_log(f"[{self.symbol}] CONDITION B (TTM Cross): FAILED | Week {current_week_idx} | Momentum: {mom_prev:.4f} -> {mom_curr:.4f} | {weeks_since_ignition} weeks after ignition")
                else:
                    self.debug_log(f"[{self.symbol}] CONDITION B (TTM Cross): CHECKING | Week {current_week_idx} | Insufficient momentum data (len={len(mom)})")
            except Exception as e:
                self.debug_log(f"Error logging Condition B info: {e}")
            
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
                    
                    pivot_candidate = self.intraday_manager.calculate_pivot()
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
                                max_extended_price = ma30_val * (1.0 + self.p.max_extended_pct)
                                extended_pct = ((cl - ma30_val) / ma30_val * 100) if ma30_val > 0 else 0
                                self.debug_log(f"TTM cross detected but {reason}: close={cl:.2f}, MA30={ma30_val:.2f}, max_extended={max_extended_price:.2f} ({self.p.max_extended_pct*100:.0f}%), actual_extended={extended_pct:.1f}%")
                                if self.pivot_level is None:
                                    self.debug_log("Need more daily data to set pivot before arming.")
                                else:
                                    self.debug_log(f"TICKER NOT ARMED - No intraday entry checks will be performed until weekly filter passes or conditions change.")
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
                self.intraday_manager.reset_armed_state()
        else:
            # Log Condition B status even when Condition A hasn't passed yet (DEBUG only)
            try:
                mom = self.momentum.mom
                if len(mom) >= 2:
                    mom_prev = float(mom.momentum[-2]) if len(mom.momentum) >= 2 else 0.0
                    mom_curr = float(mom.momentum[-1]) if len(mom.momentum) >= 1 else 0.0
                    self.debug_log(f"[{self.symbol}] CONDITION B (TTM Cross): CHECKING | Week {current_week_idx} | Momentum: {mom_prev:.4f} -> {mom_curr:.4f} | Waiting for Condition A (Ignition)")
            except Exception:
                pass

        # Check if entry window expired (check on every bar, not just weekly)
        if self.armed and self.entry_window_end_date is not None:
            try:
                current_date = self.d_15.datetime.datetime(0)
                if current_date > self.entry_window_end_date:
                    if self.p.log_level.upper() == 'DEBUG':
                        self.debug_log(f"Entry window expired: {current_date} > {self.entry_window_end_date}")
                    self.intraday_manager.reset_armed_state()
                    self.last_ignition_idx = None
                    self.last_ttm_cross_idx = None
            except:
                pass

    def stop(self):
        """Called at the end of the backtest."""
        # Close any open position at the end of simulation for P&L calculation
        try:
            if self.position and self.position.size != 0:
                try:
                    # Get current price from 15m data
                    if len(self.d_15) > 0:
                        exit_price = float(self.d_15.close[0])
                        exit_size = abs(self.position.size)
                        self.log(f"[{self.symbol}] CLOSING OPEN POSITION AT END OF SIMULATION | Price: ${exit_price:.2f}, Size: {exit_size}")
                        # Close the position
                        self.order = self.close(data=self.d_15)
                        # Track the exit for statistics
                        if self.entry_price:
                            self.track_trade_exit(exit_price, exit_size)
                except Exception as e:
                    self.debug_log(f"Error closing position at end of simulation: {e}")
        except Exception:
            pass
        
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

