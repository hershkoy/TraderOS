# Point & Figure Multi-Timeframe Strategy
import backtrader as bt

# -----------------------------
# Point & Figure tracker (fixed box, classic reversal)
# -----------------------------
class PnFTracker:
    def __init__(self, box_size: float = 1.0, reversal: int = 3):
        self.box = float(box_size)
        self.rev = int(reversal)
        self.col = None           # 'X' or 'O'
        self.top = None           # top of current X column
        self.bot = None           # bottom of current O column

    def update(self, price: float):
        """Feed one price (use daily/weekly close). Returns (is_x, flip_to_x, flip_to_o)."""
        flip_x = False
        flip_o = False

        if self.col is None:
            # Start column; pick initial based on first price
            self.col = 'X'
            self.top = price
            self.bot = price
            return (True, False, False)

        if self.col == 'X':
            # Extend X column up by >= box
            if price >= self.top + self.box:
                self.top = price
            # Flip to O on >= reversal boxes down from top
            elif price <= self.top - self.box * self.rev:
                self.col = 'O'
                self.bot = price
                flip_o = True
        else:  # self.col == 'O'
            # Extend O column down by >= box
            if price <= self.bot - self.box:
                self.bot = price
            # Flip to X on >= reversal boxes up from bottom
            elif price >= self.bot + self.box * self.rev:
                self.col = 'X'
                self.top = price
                flip_x = True

        return (self.col == 'X', flip_x, flip_o)

# -----------------------------
# Backtrader Strategy
# -----------------------------
class PnF_MTF_Strategy(bt.Strategy):
    params = dict(
        box_size=1.50,      # dollars per box
        reversal=3,         # boxes to reverse
        size=1,             # shares
        commission=0.001,   # 0.1%
        slippage=0.0005,    # 5 bps model (applied via price adjustment below if desired)
        printlog=True,
        log_level='INFO',   # log level (added for compatibility)
    )

    def log(self, txt):
        if self.p.printlog:
            dt = self.data.datetime.datetime(0)
            print(f"{dt} {txt}")

    def __init__(self):
        # feeds: 0 = 1h base, 1 = Daily (resampled), 2 = Weekly (resampled)
        self.base = self.datas[0]
        self.d = self.datas[1]
        self.w = self.datas[2]

        # P&F trackers for daily and weekly
        self.trk_d = PnFTracker(self.p.box_size, self.p.reversal)
        self.trk_w = PnFTracker(self.p.box_size, self.p.reversal)

        # Latches updated only when those TF bars CLOSE
        self.daily_flip_to_x = False
        self.daily_flip_to_o = False
        self.weekly_is_x = False

        # Track last lengths to detect new closed bars
        self._last_len_d = len(self.d)
        self._last_len_w = len(self.w)

        # Broker settings
        self.broker.setcommission(commission=self.p.commission)

    def next(self):
        # 1) When a new DAILY bar has closed, update daily P&F from daily close
        if len(self.d) > self._last_len_d:
            self._last_len_d = len(self.d)
            price_d = float(self.d.close[0])
            is_x, flip_x, flip_o = self.trk_d.update(price_d)
            self.daily_flip_to_x = flip_x
            self.daily_flip_to_o = flip_o
            if self.p.printlog:
                self.log(f"[D] close={price_d:.4f} PnF col={('X' if is_x else 'O')} flipX={flip_x} flipO={flip_o}")

        # 2) When a new WEEKLY bar has closed, update weekly P&F from weekly close
        if len(self.w) > self._last_len_w:
            self._last_len_w = len(self.w)
            price_w = float(self.w.close[0])
            is_x, _, _ = self.trk_w.update(price_w)
            self.weekly_is_x = is_x
            if self.p.printlog:
                self.log(f"[W] close={price_w:.4f} PnF col={('X' if is_x else 'O')}")

        # 3) Trade on the HOURLY bar, gated by the latest CLOSED D/W info
        if not self.position:
            if self.daily_flip_to_x and self.weekly_is_x:
                # optional simple slippage model: buy a tad above close
                px = float(self.base.close[0]) * (1 + self.p.slippage)
                self.buy(size=self.p.size, price=px)
                self.log(f"BUY size={self.p.size} at ~{px:.4f}")
                # consume the flip so we don't re-trigger until next daily flip event
                self.daily_flip_to_x = False
        else:
            if self.daily_flip_to_o:
                px = float(self.base.close[0]) * (1 - self.p.slippage)
                self.close(price=px)
                self.log(f"SELL (exit) at ~{px:.4f}")
                self.daily_flip_to_o = False

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f"EXECUTED BUY {order.executed.size} @ {order.executed.price:.2f}")
            else:
                self.log(f"EXECUTED SELL {order.executed.size} @ {order.executed.price:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"ORDER {order.getstatusname()}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f"TRADE PnL Gross {trade.pnl:.2f} | Net {trade.pnlcomm:.2f}")

    def stop(self):
        self.log(f"End Value: {self.broker.getvalue():.2f}")
        
        # Save strategy state for reporting even if backtrader fails later
        self._final_value = self.broker.getvalue()
        self._trades_executed = getattr(self, '_trade_count', 0)
        
        # Store this strategy instance globally so we can access it if cerebro.run() fails
        global _strategy_backup
        _strategy_backup = self

    @staticmethod
    def get_data_requirements():
        """Return the data requirements for this strategy"""
        return {
            'base_timeframe': 'hourly',
            'additional_timeframes': ['daily', 'weekly'],
            'requires_resampling': True
        }

    @staticmethod
    def get_description():
        """Return strategy description"""
        return "Point & Figure Multi-Timeframe Strategy using hourly data with daily/weekly P&F signals"
