# indicators/supertrend.py
import backtrader as bt
import pandas as pd

class SuperTrend(bt.Indicator):
    """
    SuperTrend indicator (final bands + direction).
    
    Lines:
      - trend:     the active Supertrend line (use this to compare with price)
      - dir:       +1 for uptrend, -1 for downtrend
      - upperband: current final upper band
      - lowerband: current final lower band

    Params:
      - period (int): ATR period (default: 10)
      - multiplier (float): ATR multiplier (default: 3.0)
    """
    lines = ('trend', 'dir', 'upperband', 'lowerband')
    params = dict(period=10, multiplier=3.0)

    plotinfo = dict(plot=True, subplot=False)
    plotlines = dict(
        trend=dict(color='green'),
        upperband=dict(_plotskip=True),
        lowerband=dict(_plotskip=True),
        dir=dict(_plotskip=True),
    )

    def __init__(self):
        atr = bt.ind.ATR(self.data, period=self.p.period)
        hl2 = (self.data.high + self.data.low) / 2.0

        # "Basic" bands
        self.basic_ub = hl2 + self.p.multiplier * atr
        self.basic_lb = hl2 - self.p.multiplier * atr

        # We'll compute "final" bands & direction in next()
        self.addminperiod(max(2, self.p.period + 1))

    def next(self):
        i = len(self)

        # previous values (safe fallbacks for the first bar)
        prev_final_ub = self.lines.upperband[-1] if i > 1 else self.basic_ub[0]
        prev_final_lb = self.lines.lowerband[-1] if i > 1 else self.basic_lb[0]
        # Handle NaN values safely
        prev_dir_raw = self.lines.dir[-1] if i > 1 else 1
        prev_dir = int(prev_dir_raw) if not pd.isna(prev_dir_raw) else 1  # start as uptrend

        # ----- Final Upper/Lower Bands (carry-forward rules)
        # tighten (never loosen) the bands until a flip condition occurs
        final_ub = (self.basic_ub[0] if (self.basic_ub[0] < prev_final_ub or self.data.close[-1] > prev_final_ub)
                    else prev_final_ub)
        final_lb = (self.basic_lb[0] if (self.basic_lb[0] > prev_final_lb or self.data.close[-1] < prev_final_lb)
                    else prev_final_lb)

        # ----- Direction flip rules
        # If price closes above previous final upper band -> uptrend
        # If price closes below previous final lower band -> downtrend
        if i > 1 and self.data.close[0] > prev_final_ub:
            direction = 1
        elif i > 1 and self.data.close[0] < prev_final_lb:
            direction = -1
        else:
            direction = prev_dir

        # Active trend line = lower band in uptrend, upper band in downtrend
        trendline = final_lb if direction == 1 else final_ub

        # Set current values
        self.lines.upperband[0] = final_ub
        self.lines.lowerband[0] = final_lb
        self.lines.dir[0]       = direction
        self.lines.trend[0]     = trendline
