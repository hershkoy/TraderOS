"""
TTM Squeeze Momentum Indicator (LazyBear implementation for Backtrader).

Matches Pine Script:
val = linreg(close - avg(avg(highest(high, lengthKC), lowest(low, lengthKC)), sma(close, lengthKC)), lengthKC, 0)
"""

import backtrader as bt


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

