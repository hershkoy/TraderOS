# Clean Mean Reversion Strategy - No Backtrader Trade Tracking
import backtrader as bt
from custom_tracking import CustomTrackingMixin

class MeanReversionCleanStrategy(CustomTrackingMixin, bt.Strategy):
    params = dict(
        period=20,          # Moving average period
        devfactor=2.0,      # Standard deviation factor for bands
        size=1,             # Position size
        commission=0.001,   # 0.1% commission
        printlog=True,
        log_level='INFO',   # Logging level
    )

    def log(self, txt, level='INFO'):
        if self.p.printlog:
            dt = self.datas[0].datetime.datetime(0)
            print(f"{dt} [{level}] {txt}")
    
    def debug_log(self, txt):
        if self.p.log_level == 'DEBUG':
            self.log(txt, 'DEBUG')

    def __init__(self):
        # Initialize the custom tracking mixin first
        CustomTrackingMixin.__init__(self)
        
        self.debug_log("Initializing MeanReversionCleanStrategy")
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
        
        # Track order to avoid duplicate orders
        self.order = None
        
        # Broker settings
        self.broker.setcommission(commission=self.p.commission)
        
        self.debug_log("MeanReversionCleanStrategy initialization complete")

    def notify_order(self, order):
        try:
            self.debug_log(f"notify_order called - Status: {order.getstatusname()}")
            
            if order.status in [order.Submitted, order.Accepted]:
                # Order submitted/accepted - nothing to do
                self.debug_log("Order submitted/accepted - returning")
                return

            # Check if an order has been completed
            if order.status in [order.Completed]:
                self.debug_log("Order completed - getting current date")
                current_date = self.datas[0].datetime.datetime(0)
                self.debug_log(f"Current date: {current_date}")
                
                if order.isbuy():
                    self.log(f"BUY EXECUTED @ {order.executed.price:.2f}, Size: {order.executed.size}")
                    # Use custom tracking mixin
                    self.track_trade_entry(order.executed.price, order.executed.size)
                    self.debug_log("Buy order tracking completed")
                    
                elif order.issell():
                    self.log(f"SELL EXECUTED @ {order.executed.price:.2f}, Size: {order.executed.size}")
                    # Use custom tracking mixin
                    self.track_trade_exit(order.executed.price, order.executed.size)
                    self.debug_log("Trade tracking completed")
                        
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log(f"ORDER {order.getstatusname()}")

            # Reset order
            self.order = None
            self.debug_log("Order reset")
            
        except Exception as e:
            self.debug_log(f"ERROR in notify_order: {e}")
            self.debug_log(f"Error type: {type(e)}")
            import traceback
            self.debug_log(f"Traceback: {traceback.format_exc()}")
            raise

    def next(self):
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
                    self.order = self.buy(size=self.p.size)
                    self.debug_log("Buy order placed")
            else:
                # In position - look for exit signals
                self.debug_log("In position, checking for exit signals")
                if current_price >= upper:
                    # Price hit upper band - sell (mean reversion down)
                    self.log(f"SELL SIGNAL - Price {current_price:.2f} >= Upper Band {upper:.2f}")
                    self.order = self.sell(size=self.p.size)
                    self.debug_log("Sell order placed (upper band)")
                elif current_price >= sma:
                    # Price back to mean - take profit
                    self.log(f"PROFIT TAKING - Price {current_price:.2f} >= SMA {sma:.2f}")
                    self.order = self.sell(size=self.p.size)
                    self.debug_log("Sell order placed (profit taking)")
                    
            self.debug_log("next() method completed successfully")
            
        except Exception as e:
            self.debug_log(f"ERROR in next() method: {e}")
            self.debug_log(f"Error type: {type(e)}")
            import traceback
            self.debug_log(f"Traceback: {traceback.format_exc()}")
            raise

    def stop(self):
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
        return "Clean Mean Reversion Strategy using Bollinger Bands on daily data (no Backtrader trade tracking)"

