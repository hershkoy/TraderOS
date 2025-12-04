import backtrader as bt
from utils.backtesting.custom_tracking import CustomTrackingMixin
from indicators.supertrend import SuperTrend

class MeanReversionRSI_BT(CustomTrackingMixin, bt.Strategy):
    params = (
        ("rsi_period", 14),
        ("adx_period", 14),
        ("atr_period", 14),
        ("bb_period", 20),
        ("bb_dev", 2),
        ("higher_tf_idx", 1),  # index of 4h resampled data
        ("size", 1),            # position size (added for compatibility)
        ("printlog", False),    # enable logging (added for compatibility)
        ("log_level", "INFO")   # log level (added for compatibility)
    )

    def __init__(self):
        super().__init__()
        # Base timeframe (1h)
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        self.adx = bt.indicators.ADX(self.data, period=self.p.adx_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.bb = bt.indicators.BollingerBands(self.data, period=self.p.bb_period, devfactor=self.p.bb_dev)

        # Higher timeframe (4h) -> data1
        self.rsi_higher = bt.indicators.RSI(self.datas[self.p.higher_tf_idx], period=self.p.rsi_period)
        self.adx_higher = bt.indicators.ADX(self.datas[self.p.higher_tf_idx], period=self.p.adx_period)
        # Using SuperTrend for trend direction
        self.supertrend = SuperTrend(self.datas[self.p.higher_tf_idx], period=10, multiplier=3.0)

    def next(self):
        self.track_portfolio_value()

        # Long setup
        if not self.position:
            if (
                self.rsi_higher[0] > 70
                and self.supertrend.dir[0] > 0  # uptrend on higher timeframe
                and self.adx[0] > 20
                and self.adx_higher[0] > 40
            ):
                entry = self.bb.lines.bot[0]
                stop = entry - self.atr[0] * 6
                size = int(self.broker.get_cash() * 0.03 / (entry - stop))
                self.buy_bracket(
                    size=size,
                    limitprice=entry,
                    stopprice=stop,
                    price=self.bb.lines.top[0],  # TP at upper band
                )
                self.track_trade_entry(entry, size)

            # Short setup
            elif (
                self.rsi_higher[0] < 30
                and self.supertrend.dir[0] < 0  # downtrend on higher timeframe
                and self.adx_higher[0] > 40
            ):
                entry = self.bb.lines.top[0]
                stop = entry + self.atr[0] * 6
                size = int(self.broker.get_cash() * 0.03 / (stop - entry))
                self.sell_bracket(
                    size=size,
                    limitprice=entry,
                    stopprice=stop,
                    price=self.bb.lines.bot[0],  # TP at lower band
                )
                self.track_trade_entry(entry, size)

    def notify_trade(self, trade):
        if trade.isclosed:
            self.track_trade_exit(trade.price, trade.size)

    @staticmethod
    def get_data_requirements():
        """Return the data requirements for this strategy"""
        return {
            'base_timeframe': 'hourly',
            'additional_timeframes': ['4h'],
            'requires_resampling': False
        }

    @staticmethod
    def get_description():
        """Return strategy description"""
        return "Mean Reversion Strategy using RSI, ADX, and SuperTrend on 1h + 4h timeframes"
