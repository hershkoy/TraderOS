# Custom tracking mixin for strategies
class CustomTrackingMixin:
    """Mixin to add custom tracking capabilities to strategies
    
    To use this in your strategy:
    1. Inherit from this mixin: class MyStrategy(CustomTrackingMixin, bt.Strategy)
    2. Call super().__init__() in your strategy's __init__ method
    3. Call self.track_trade_entry() when entering positions
    4. Call self.track_trade_exit() when exiting positions  
    5. Call self.track_portfolio_value() in next() to track equity curve
    6. Use self.get_trade_statistics() to get performance metrics
    """
    
    def __init__(self):
        # Trade tracking (using different names to avoid conflicts with Backtrader)
        self._custom_trades = []
        self._custom_portfolio_values = []
        self._custom_signals = []
        self._custom_entry_price = None
        self._custom_entry_date = None
        self._custom_entry_size = 0
        
        # Performance tracking
        self._custom_total_pnl = 0.0
        self._custom_winning_trades = 0
        self._custom_losing_trades = 0
        self._custom_max_drawdown = 0.0
        self._custom_max_drawdown_pct = 0.0
        self._custom_peak_value = 0.0
        
    def track_trade_entry(self, price, size):
        """Track when entering a trade"""
        self._custom_entry_price = price
        self._custom_entry_date = self.data.datetime.datetime(0)
        self._custom_entry_size = size
        
        # Record signal
        self._custom_signals.append({
            'date': self._custom_entry_date,
            'action': 'BUY',
            'price': price,
            'size': size
        })
        
    def track_trade_exit(self, price, size):
        """Track when exiting a trade"""
        if self._custom_entry_price is not None:
            # Calculate trade metrics
            trade_pnl = (price - self._custom_entry_price) * size
            trade_return = (price - self._custom_entry_price) / self._custom_entry_price * 100
            duration = (self.data.datetime.datetime(0) - self._custom_entry_date).days
            
            # Store trade details
            trade = {
                'type': 'LONG',
                'entry_date': self._custom_entry_date,
                'entry_price': self._custom_entry_price,
                'exit_date': self.data.datetime.datetime(0),
                'exit_price': price,
                'size': size,
                'pnl': trade_pnl,
                'pnl_pct': trade_return,
                'duration': f"{duration} days"
            }
            self._custom_trades.append(trade)
            
            # Update performance metrics
            self._custom_total_pnl += trade_pnl
            if trade_pnl > 0:
                self._custom_winning_trades += 1
            else:
                self._custom_losing_trades += 1
            
            # Record signal
            self._custom_signals.append({
                'date': self.data.datetime.datetime(0),
                'action': 'SELL',
                'price': price,
                'size': size
            })
            
            # Reset entry tracking
            self._custom_entry_price = None
            self._custom_entry_date = None
            self._custom_entry_size = 0
            
    def track_portfolio_value(self):
        """Track portfolio value for equity curve"""
        current_value = self.broker.getvalue()
        current_date = self.data.datetime.datetime(0)
        
        # Track peak value for drawdown calculation
        if current_value > self._custom_peak_value:
            self._custom_peak_value = current_value
        
        # Calculate current drawdown
        if self._custom_peak_value > 0:
            drawdown = (self._custom_peak_value - current_value) / self._custom_peak_value * 100
            if drawdown > self._custom_max_drawdown_pct:
                self._custom_max_drawdown_pct = drawdown
                self._custom_max_drawdown = self._custom_peak_value - current_value
        
        self._custom_portfolio_values.append({
            'date': current_date,
            'value': current_value
        })
        
    def get_trade_statistics(self):
        """Calculate comprehensive trade statistics"""
        if not self._custom_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown_pct': 0.0,
                'max_drawdown': 0.0
            }
        
        total_trades = len(self._custom_trades)
        win_rate = (self._custom_winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate average wins and losses
        winning_trades = [t for t in self._custom_trades if t['pnl'] > 0]
        losing_trades = [t for t in self._custom_trades if t['pnl'] <= 0]
        
        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': self._custom_winning_trades,
            'losing_trades': self._custom_losing_trades,
            'win_rate': win_rate,
            'total_pnl': self._custom_total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': self._custom_max_drawdown_pct,
            'max_drawdown': self._custom_max_drawdown
        }
