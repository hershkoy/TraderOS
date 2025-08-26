# strategies/liquidity_sweep.py
import backtrader as bt
import math
from datetime import datetime, date

class LiquiditySweep(bt.Strategy):
    params = dict(
        atr_period=14,
        vol_lookback=20,
        vol_mult=1.5,
        body_frac=0.4,          # min body as fraction of TR on confirmation bar
        sweep_atr_frac=0.20,    # how far beyond PDH/PDL counts as a sweep
        stop_atr_frac=0.10,
        max_bars_after_sweep=3,
        rr=2.0,
        trend_sma=50,
        use_trend=True,
        rth_only=False,          # if your data includes AH, you can later add a session filter
        size=1,                  # position size (added for compatibility)
        printlog=False,          # enable logging (added for compatibility)
        log_level='INFO'         # log level (added for compatibility)
    )

    def __init__(self):
        d = self.data
        self.atr = bt.ind.ATR(d, period=self.p.atr_period)
        self.sma = bt.ind.SMA(d.close, period=self.p.trend_sma)
        self.median_vol = bt.ind.SMA(d.volume, period=self.p.vol_lookback)  # Use SMA instead of Quantile

        # Track prior day high/low updated once per calendar day
        self.pdh = bt.LineNum(0.0)
        self.pdl = bt.LineNum(0.0)
        self._cur_day = None
        self._day_high = None
        self._day_low = None
        self._yday_high = None
        self._yday_low = None

        self.order = None

    def next(self):
        dt = self.data.datetime.date(0)
        high = float(self.data.high[0])
        low  = float(self.data.low[0])
        close= float(self.data.close[0])
        open_= float(self.data.open[0])
        vol  = float(self.data.volume[0])
        atr  = float(self.atr[0]) if len(self.atr) > 0 else None
        tr   = float(self.data.high[0] - self.data.low[0])

        # Build rolling prior-day H/L
        if self._cur_day is None:
            self._cur_day = dt
            self._day_high = high
            self._day_low  = low
        elif dt != self._cur_day:
            # day changed: yesterday becomes prior day for today
            self._yday_high = self._day_high
            self._yday_low  = self._day_low
            self._cur_day   = dt
            self._day_high  = high
            self._day_low   = low
        else:
            self._day_high = max(self._day_high, high)
            self._day_low  = min(self._day_low,  low)

        if self._yday_high is None or self._yday_low is None or atr is None:
            return

        pdh = self._yday_high
        pdl = self._yday_low

        # Optional trend filter
        if self.p.use_trend and len(self.sma) == 0:
            return
        long_ok  = (not self.p.use_trend) or (close > float(self.sma[0]))
        short_ok = (not self.p.use_trend) or (close < float(self.sma[0]))

        # Volume / body confirmations
        is_vol_spike = vol > self.p.vol_mult * float(self.median_vol[0])
        body = abs(close - open_)
        big_body = tr > 0 and (body / tr) >= self.p.body_frac
        bull_displacement = (close > max(open_, self.data.high[-1] if len(self.data) > 1 else open_))
        bear_displacement = (close < min(open_, self.data.low[-1]  if len(self.data) > 1 else open_))

        # Track sweeps with a small state (look back a few bars after the sweep)
        # Bullish sweep: low below PDL by >= sweep_atr_frac*ATR then close back above PDL within N bars
        sweep_thresh = self.p.sweep_atr_frac * atr

        # --- Long setup ---
        if long_ok and (low <= pdl - sweep_thresh):
            # remember sweep extreme for stop calc
            sweep_low = low
            # confirmation within next N bars (including current)
            confirm = big_body or (is_vol_spike and bull_displacement) or (close > pdl)
            if confirm and not self.position and self.order is None:
                # enter long next bar at market
                size = self._risk_sized_qty(entry=close, stop=sweep_low - self.p.stop_atr_frac * atr)
                if size > 0:
                    self.order = self.buy(size=size)
                    self._attach_orders(entry_price=close,
                                        stop_price=sweep_low - self.p.stop_atr_frac * atr,
                                        target_price=close + self.p.rr * (close - (sweep_low - self.p.stop_atr_frac * atr)))

        # --- Short setup ---
        if short_ok and (high >= pdh + sweep_thresh):
            sweep_high = high
            confirm = big_body or (is_vol_spike and bear_displacement) or (close < pdh)
            if confirm and not self.position and self.order is None:
                size = self._risk_sized_qty(entry=close, stop=sweep_high + self.p.stop_atr_frac * atr, short=True)
                if size > 0:
                    self.order = self.sell(size=size)
                    self._attach_orders(entry_price=close,
                                        stop_price=sweep_high + self.p.stop_atr_frac * atr,
                                        target_price=close - self.p.rr * ((sweep_high + self.p.stop_atr_frac * atr) - close))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

    # Simple R-based sizing with 1% default account risk (you can make it a param if your runner supports it)
    def _risk_sized_qty(self, entry, stop, short=False, account_risk=0.01):
        cash = self.broker.getvalue()
        risk_per_share = abs(entry - stop)
        if risk_per_share <= 0:
            return 0
        dollars_at_risk = cash * account_risk
        q = math.floor(dollars_at_risk / risk_per_share)
        return max(q, 0)

    def _attach_orders(self, entry_price, stop_price, target_price):
        if self.position.size == 0 and self.order:
            # Bracket after fill in notify_trade is safer, but for brevity:
            pass

    @staticmethod
    def get_data_requirements():
        """Return the data requirements for this strategy"""
        return {
            'base_timeframe': '1h',
            'additional_timeframes': [],
            'requires_resampling': False
        }

    @staticmethod
    def get_description():
        """Return strategy description"""
        return "Liquidity Sweep Strategy using ATR and volume analysis on hourly data"
