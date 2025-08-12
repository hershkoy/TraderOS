# Bare Mean Reversion Strategy - Minimal Backtrader Interface
import backtrader as bt
from custom_tracking import CustomTrackingMixin

class MeanReversionBareStrategy(CustomTrackingMixin):
    """Bare strategy that implements minimal Backtrader interface"""
    
    params = dict(
        period=20,          # Moving average period
        devfactor=2.0,      # Standard deviation factor for bands
        size=1,             # Position size
        commission=0.001,   # 0.1% commission
        printlog=True,
        log_level='INFO',   # Logging level
    )

    def __init__(self):
        # Initialize the custom tracking mixin first
        CustomTrackingMixin.__init__(self)
        
        # These will be set by Backtrader
        self.datas = None
        self.broker = None
        self.position = None
        self.order = None
        
        self.debug_log("Initializing MeanReversionBareStrategy")

    def log(self, txt, level='INFO'):
        if self.p.printlog:
            dt = self.datas[0].datetime.datetime(0)
            print(f"{dt} [{level}] {txt}")
    
    def debug_log(self, txt):
        if self.p.log_level == 'DEBUG':
            self.log(txt, 'DEBUG')

    def set_data(self, datas):
        """Set data feeds (called by Backtrader)"""
        self.datas = datas
        self.debug_log(f"Number of data feeds: {len(self.datas)}")
        
        # Use the first (and only) data feed
        self.data_close = self.datas[0].close
        self.debug_log(f"Data close line type: {type(self.data_close)}")
        
        # Calculate indicators
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data_close, period=self.p.period
        )
        self.debug_log(f"SMA indicator created: {type(self.sma)}")
        
        # Calculate standard deviation
        self.stddev = bt.indicators.StandardDeviation(
            self.data_close, period=self.p.period
        )
        self.debug_log(f"Standard deviation indicator created: {type(self.stddev)}")
        
        # Calculate Bollinger Bands
        self.upper_band = self.sma + (self.stddev * self.p.devfactor)
        self.lower_band = self.sma - (self.stddev * self.p.devfactor)
        self.debug_log(f"Bollinger bands calculated")

    def set_broker(self, broker):
        """Set broker (called by Backtrader)"""
        self.broker = broker
        self.broker.setcommission(commission=self.p.commission)

    def set_position(self, position):
        """Set position (called by Backtrader)"""
        self.position = position

    def buy(self, size=None, price=None):
        """Place buy order"""
        if size is None:
            size = self.p.size
        # This is a simplified version - in practice you'd need to implement order management
        self.debug_log(f"BUY order placed: size={size}, price={price}")
        return None  # No order object returned

    def sell(self, size=None, price=None):
        """Place sell order"""
        if size is None:
            size = self.p.size
        # This is a simplified version - in practice you'd need to implement order management
        self.debug_log(f"SELL order placed: size={size}, price={price}")
        return None  # No order object returned

    def next(self):
        """Main strategy logic - called by Backtrader"""
        try:
            self.debug_log("Entering next() method")
            
            # Track portfolio value for charting
            self.debug_log("Getting current date from datas[0]")
            current_date = self.datas[0].datetime.datetime(0)
            self.debug_log(f"Current date: {current_date}")
            
            # Use custom tracking mixin for portfolio values
            self.track_portfolio_value()
            portfolio_value = self.broker.getvalue()
            self.debug_log(f"Portfolio value tracked: {portfolio_value}")
            
            # Check if we have an order pending
            if self.order:
                self.debug_log("Order pending, returning")
                return

            # Skip if we don't have enough data for indicators
            data_length = len(self.data_close)
            self.debug_log(f"Data length: {data_length}, required period: {self.p.period}")
            if data_length < self.p.period:
                self.debug_log("Not enough data, returning")
                return

            self.debug_log("Getting current price and indicator values")
            current_price = self.data_close[0]
            upper = self.upper_band[0]
            lower = self.lower_band[0]
            sma = self.sma[0]
            
            self.debug_log(f"Current price: {current_price}, SMA: {sma}, Upper: {upper}, Lower: {lower}")

            if self.p.printlog:
                self.log(f"Price: {current_price:.2f}, SMA: {sma:.2f}, Upper: {upper:.2f}, Lower: {lower:.2f}")

            if not self.position:
                # Not in position - look for entry signals
                self.debug_log("Not in position, checking for entry signals")
                if current_price <= lower:
                    # Price hit lower band - buy (mean reversion up)
                    self.log(f"BUY SIGNAL - Price {current_price:.2f} <= Lower Band {lower:.2f}")
                    # Simulate trade entry for tracking
                    self.track_trade_entry(current_price, self.p.size)
                    self.debug_log("Buy signal tracked")
            else:
                # In position - look for exit signals
                self.debug_log("In position, checking for exit signals")
                if current_price >= upper:
                    # Price hit upper band - sell (mean reversion down)
                    self.log(f"SELL SIGNAL - Price {current_price:.2f} >= Upper Band {upper:.2f}")
                    # Simulate trade exit for tracking
                    self.track_trade_exit(current_price, self.p.size)
                    self.debug_log("Sell signal tracked (upper band)")
                elif current_price >= sma:
                    # Price back to mean - take profit
                    self.log(f"PROFIT TAKING - Price {current_price:.2f} >= SMA {sma:.2f}")
                    # Simulate trade exit for tracking
                    self.track_trade_exit(current_price, self.p.size)
                    self.debug_log("Sell signal tracked (profit taking)")
                    
            self.debug_log("next() method completed successfully")
            
        except Exception as e:
            self.debug_log(f"ERROR in next() method: {e}")
            self.debug_log(f"Error type: {type(e)}")
            import traceback
            self.debug_log(f"Traceback: {traceback.format_exc()}")
            raise

    def stop(self):
        """Called when strategy stops"""
        self.log(f"Final Portfolio Value: {self.broker.getvalue():.2f}")
        
        # Print custom tracking results
        stats = self.get_trade_statistics()
        self.log(f"Custom Tracking Results:")
        self.log(f"  Total Trades: {stats['total_trades']}")
        self.log(f"  Win Rate: {stats['win_rate']:.1f}%")
        self.log(f"  Total P&L: ${stats['total_pnl']:.2f}")
        self.log(f"  Max Drawdown: {stats['max_drawdown_pct']:.2f}%")

    @staticmethod
    def get_data_requirements():
        """Return the data requirements for this strategy"""
        return {
            'base_timeframe': 'daily',
            'additional_timeframes': [],
            'requires_resampling': False
        }

    @staticmethod
    def get_description():
        """Return strategy description"""
        return "Bare Mean Reversion Strategy using Bollinger Bands on daily data (minimal Backtrader interface)"

