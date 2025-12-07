"""
TTM Squeeze Momentum Indicator (LazyBear implementation).

This module provides both:
1. Backtrader indicator class for on-line processing (bar-by-bar)
2. Pandas-based functions for batch processing (DataFrame operations)

Matches Pine Script:
val = linreg(close - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC)), lengthKC, 0)

Both implementations calculate the same formula:
- Baseline = avg(avg(highest(high, L), lowest(low, L)), sma(close, L))
- Difference = close - baseline
- Momentum = linreg(difference, L, 0) - linear regression predicted value at current bar
"""

import backtrader as bt
import numpy as np
import pandas as pd
from typing import Optional


class TTMSqueezeMomentum(bt.Indicator):
    """
    TTM Squeeze Momentum Indicator (LazyBear implementation).
    
    Matches Pine Script:
    val = linreg(close - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC)), lengthKC, 0)
    
    The indicator calculates:
    1. Baseline = avg(avg(highest(high, L), lowest(low, L)), sma(close, L))
    2. Difference = close - baseline
    3. Momentum = linreg(difference, L, 0) - linear regression predicted value at current bar
    """

    lines = ("momentum", "slope")
    params = (("lengthKC", 20),)

    def __init__(self):
        # Need high, low, close from the data feed
        self.data_hlc = self.data  # Assume data feed has high, low, close
        self.addminperiod(self.p.lengthKC)

    def next(self):
        if len(self.data) < self.p.lengthKC:
            self.lines.momentum[0] = 0.0
            self.lines.slope[0] = 0.0
            return

        L = self.p.lengthKC
        
        # For each bar in the regression window (last L bars including current),
        # calculate its baseline and then the difference (close - baseline)
        # This matches Pine Script: linreg(close - avg(...), lengthKC, 0)
        # In Backtrader: index 0 = current bar, -1 = previous bar, ..., -(L-1) = L bars ago
        differences = []
        
        for j in range(L):
            # Bar index: -L+1+j gives us bars from -(L-1) down to 0 (current bar)
            # j=0: bar at -(L-1), j=L-1: bar at 0 (current)
            bar_idx = -L + 1 + j
            highest_high = 0.0
            lowest_low = float('inf')
            sum_close = 0.0
            
            # Calculate baseline components for this bar (using last L bars ending at bar_idx)
            # For bar at bar_idx, look at bars from bar_idx-(L-1) to bar_idx
            for i in range(L):
                lookback_idx = bar_idx - (L - 1) + i
                try:
                    h = float(self.data_hlc.high[lookback_idx])
                    l = float(self.data_hlc.low[lookback_idx])
                    c = float(self.data_hlc.close[lookback_idx])
                    highest_high = max(highest_high, h)
                    lowest_low = min(lowest_low, l)
                    sum_close += c
                except (IndexError, TypeError, ValueError):
                    self.lines.momentum[0] = 0.0
                    self.lines.slope[0] = 0.0
                    return
            
            if lowest_low == float('inf'):
                self.lines.momentum[0] = 0.0
                self.lines.slope[0] = 0.0
                return
            
            # Baseline for this bar
            mid_hl = (highest_high + lowest_low) / 2.0
            sma_close = sum_close / L
            baseline = (mid_hl + sma_close) / 2.0
            
            # Calculate difference: close - baseline for this bar
            try:
                c = float(self.data_hlc.close[bar_idx])
                differences.append(c - baseline)
            except (IndexError, TypeError, ValueError):
                self.lines.momentum[0] = 0.0
                self.lines.slope[0] = 0.0
                return
        
        # Linear regression on differences
        # x = 0, 1, 2, ..., L-1 (time indices, where 0 = oldest, L-1 = current)
        # y = differences[0], differences[1], ..., differences[L-1]
        n = L
        sum_x = sum_y = sum_xy = sum_x2 = 0.0
        for i in range(n):
            x = float(i)
            y = differences[i]
            sum_x += x
            sum_y += y
            sum_xy += x * y
            sum_x2 += x * x
        
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            self.lines.momentum[0] = 0.0
            self.lines.slope[0] = 0.0
            return
        
        # Calculate slope and intercept
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # linreg(x, length, 0) returns predicted value at x = length-1 (last point = current bar)
        # This is: slope * (length-1) + intercept
        predicted_value = slope * (L - 1) + intercept
        
        self.lines.momentum[0] = predicted_value
        self.lines.slope[0] = slope


# -----------------------------
# Pandas-based batch processing functions
# Used by scanners and batch analysis tools
# -----------------------------

def rolling_linreg_yhat(x: pd.Series, length: int) -> pd.Series:
    """
    Rolling least-squares fit (degree=1) evaluated at the most-recent x.
    Equivalent to Pine's linreg(x, length, 0).
    
    This is the pandas-based version used for batch processing.
    For on-line processing in Backtrader, see TTMSqueezeMomentum class above.
    
    Args:
        x: Input series to perform rolling linear regression on
        length: Window length for regression
        
    Returns:
        Series with predicted values at the last point of each window
    """
    # index positions 0..length-1 for regression
    idx = np.arange(length, dtype=float)

    def _fit(arr):
        y = np.asarray(arr, dtype=float)
        # polyfit on [0..length-1], return predicted y at last x (length-1)
        m, b = np.polyfit(idx, y, 1)
        return m * (length - 1) + b

    return x.rolling(length, min_periods=length).apply(_fit, raw=False)


def calculate_squeeze_momentum(
    df: pd.DataFrame,
    lengthKC: int = 20,
    use_logging: bool = False
) -> pd.Series:
    """
    Calculate TTM Squeeze momentum values from a pandas DataFrame.
    
    This is the pandas-based batch processing version of the squeeze calculation.
    It matches the same formula as TTMSqueezeMomentum but works on entire DataFrames.
    
    Args:
        df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
            Must be indexed by datetime
        lengthKC: Keltner Channel length (default: 20)
        use_logging: If True, log intermediate calculation values (default: False)
        
    Returns:
        Series of squeeze momentum values (same as 'val' in Pine Script)
        
    Example:
        >>> df = pd.DataFrame({
        ...     'high': [...], 'low': [...], 'close': [...]
        ... }, index=pd.date_range('2024-01-01', periods=100))
        >>> momentum = calculate_squeeze_momentum(df, lengthKC=20)
    """
    import logging
    logger = logging.getLogger(__name__) if use_logging else None
    
    src = df["close"]
    if logger:
        logger.debug(f"Calculating squeeze momentum with lengthKC={lengthKC}")
    
    # component = close - avg( avg(highest(high,L), lowest(low,L)), sma(close,L) )
    highest_h = df["high"].rolling(lengthKC, min_periods=lengthKC).max()
    lowest_l  = df["low"].rolling(lengthKC,  min_periods=lengthKC).min()
    mid_hl    = (highest_h + lowest_l) / 2.0
    sma_c     = src.rolling(lengthKC, min_periods=lengthKC).mean()
    baseline  = (mid_hl + sma_c) / 2.0
    x = src - baseline
    
    # Log key intermediate values for the last few periods
    if logger and len(df) >= 3:
        logger.debug(f"Last 3 periods - Close: {src.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Highest H: {highest_h.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Lowest L: {lowest_l.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Mid HL: {mid_hl.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - SMA C: {sma_c.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - Baseline: {baseline.tail(3).tolist()}")
        logger.debug(f"Last 3 periods - X (close-baseline): {x.tail(3).tolist()}")
    
    val = rolling_linreg_yhat(x, lengthKC)
    
    # Log the final squeeze values
    if logger:
        valid_vals = val.dropna()
        if len(valid_vals) > 0:
            logger.debug(f"Final squeeze momentum range: {valid_vals.min():.4f} to {valid_vals.max():.4f}")
            logger.debug(f"Last 3 squeeze momentum values: {valid_vals.tail(3).tolist()}")
    
    return val

