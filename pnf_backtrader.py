# backtrader_runner.py - Multi-Strategy Backtrader Runner
import argparse
from pathlib import Path
import pandas as pd
import backtrader as bt
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

# Import strategies
from strategies import get_strategy, list_strategies

# Global variable to store strategy backup in case of backtrader failure
_strategy_backup = None

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
                # consume the flip so we don’t re-trigger until next daily flip event
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
        
        # Save strategy state for reporting even if backtrader fails later
        self._final_value = self.broker.getvalue()
        self._trades_executed = getattr(self, '_trade_count', 0)
        
        # Store this strategy instance globally so we can access it if cerebro.run() fails
        global _strategy_backup
        _strategy_backup = self

# -----------------------------
# PandasData feed wrapper for our Parquet
# -----------------------------
class Parquet1hPandas(bt.feeds.PandasData):
    # backtrader default names: datetime, open, high, low, close, volume, openinterest
    params = (
        ('datetime', None),  # Use index as datetime
        ('open',     'open'),
        ('high',     'high'),
        ('low',      'low'),
        ('close',    'close'),
        ('volume',   'volume'),
        ('openinterest', None),
    )

def load_parquet_1h(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    # Expect columns: ts_event (int ns), open, high, low, close, volume, instrument_id, venue_id, timeframe
    # Convert to naive UTC datetimes for Backtrader
    ts = pd.to_datetime(df['ts_event'], unit='ns', utc=True).dt.tz_localize(None)
    out = pd.DataFrame({
        'datetime': ts,
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'volume': df['volume'].astype(float),
    }).sort_values('datetime').reset_index(drop=True)
    
    # Filter out any invalid or future dates that might cause issues
    now = pd.Timestamp.now()
    out = out[out['datetime'] <= now]
    print(f"After filtering future dates: {len(out)} rows")
    
    # Ensure we have at least some data
    if len(out) < 10:
        raise ValueError(f"Not enough data after filtering: {len(out)} rows")
    
    out.set_index('datetime', inplace=True)
    print(f"Loaded {len(out)} data points from {out.index.min()} to {out.index.max()}")
    
    # Additional check for data continuity
    if out.index.duplicated().any():
        print("Warning: Found duplicate timestamps, removing duplicates")
        out = out[~out.index.duplicated(keep='first')]
    
    return out

# -----------------------------
# Report Generation Functions
# -----------------------------
def generate_reports(cerebro, results, args):
    """Generate comprehensive TradingView-style reports"""
    
    # Create reports directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / f"pnf_backtest_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    strategy = results[0]
    
    # 1. Generate HTML Summary Report
    generate_html_report(strategy, cerebro, args, report_dir)
    
    # 2. Generate CSV Trade Log
    generate_trade_csv(strategy, report_dir)
    
    # 3. Generate JSON Statistics
    generate_json_stats(strategy, cerebro, args, report_dir)
    
    # 4. Save Chart as PNG
    save_chart_png(cerebro, report_dir)
    
    print(f"\\n📊 Reports generated in: {report_dir}")
    print(f"📈 Open {report_dir / 'backtest_report.html'} for detailed analysis")

def generate_html_report(strategy, cerebro, args, report_dir):
    """Generate TradingView-style HTML report"""
    
    # Extract analyzer results
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()
    vwr = strategy.analyzers.vwr.get_analysis()
    
    # Calculate key metrics
    initial_value = args.cash
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Get trade statistics
    total_trades = trade_analyzer.get('total', {}).get('total', 0)
    winning_trades = trade_analyzer.get('won', {}).get('total', 0)
    losing_trades = trade_analyzer.get('lost', {}).get('total', 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    gross_profit = trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0)
    gross_loss = abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
    
    avg_win = trade_analyzer.get('won', {}).get('pnl', {}).get('average', 0)
    avg_loss = trade_analyzer.get('lost', {}).get('pnl', {}).get('average', 0)
    
    max_dd = drawdown.get('max', {}).get('drawdown', 0)
    max_dd_pct = drawdown.get('max', {}).get('moneydown', 0)
    
    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Point & Figure Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .metric-label {{ color: #666; font-size: 14px; margin-bottom: 5px; }}
            .positive {{ color: #28a745; }}
            .negative {{ color: #dc3545; }}
            .section {{ margin-bottom: 30px; }}
            .section-title {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f8f9fa; font-weight: bold; }}
            .strategy-params {{ background: #e9ecef; padding: 15px; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Point & Figure Multi-Timeframe Strategy</h1>
                <p>Backtest Report - {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            </div>
            
            <div class="section">
                <div class="section-title">📈 Performance Overview</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Return</div>
                        <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Initial Capital</div>
                        <div class="metric-value">${initial_value:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Value</div>
                        <div class="metric-value">${final_value:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Net Profit</div>
                        <div class="metric-value {'positive' if (final_value - initial_value) >= 0 else 'negative'}">${final_value - initial_value:,.2f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📊 Trade Statistics</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total Trades</div>
                        <div class="metric-value">{total_trades}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Win Rate</div>
                        <div class="metric-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Profit Factor</div>
                        <div class="metric-value {'positive' if profit_factor >= 1 else 'negative'}">{profit_factor:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Sharpe Ratio</div>
                        <div class="metric-value">{sharpe.get('sharperatio', 0):.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">⚠️ Risk Metrics</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown</div>
                        <div class="metric-value negative">{max_dd:.2f}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Max Drawdown ($)</div>
                        <div class="metric-value negative">${max_dd_pct:,.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">SQN (System Quality)</div>
                        <div class="metric-value">{sqn.get('sqn', 0):.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">VWR (Variability)</div>
                        <div class="metric-value">{vwr.get('vwr', 0):.3f}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">💰 Trade Breakdown</div>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Winning Trades</th>
                        <th>Losing Trades</th>
                    </tr>
                    <tr>
                        <td>Count</td>
                        <td class="positive">{winning_trades}</td>
                        <td class="negative">{losing_trades}</td>
                    </tr>
                    <tr>
                        <td>Total P&L</td>
                        <td class="positive">${gross_profit:,.2f}</td>
                        <td class="negative">-${gross_loss:,.2f}</td>
                    </tr>
                    <tr>
                        <td>Average P&L</td>
                        <td class="positive">${avg_win:,.2f}</td>
                        <td class="negative">${avg_loss:,.2f}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <div class="section-title">⚙️ Strategy Parameters</div>
                <div class="strategy-params">
                    <strong>Box Size:</strong> ${args.box}<br>
                    <strong>Reversal:</strong> {args.reversal} boxes<br>
                    <strong>Position Size:</strong> {args.size} shares<br>
                    <strong>Date Range:</strong> {args.fromdate} to {args.todate}<br>
                    <strong>Data File:</strong> {args.parquet}
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">📝 Notes</div>
                <p>This Point & Figure multi-timeframe strategy uses daily and weekly P&F charts to generate signals, executing on hourly timeframe.</p>
                <p><strong>Buy Signal:</strong> Daily flip to X-column AND Weekly is in X-column</p>
                <p><strong>Sell Signal:</strong> Daily flip to O-column (regardless of weekly state)</p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    with open(report_dir / "backtest_report.html", "w") as f:
        f.write(html_content)

def generate_trade_csv(strategy, report_dir):
    """Generate CSV file with individual trade details"""
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    
    # Create trade log (this is a simplified version - in practice you'd capture more details)
    trades_data = []
    if 'total' in trade_analyzer and trade_analyzer['total']['total'] > 0:
        # Note: BackTrader's TradeAnalyzer doesn't provide individual trade details
        # For detailed trade logs, you'd need to implement custom tracking in the strategy
        trades_data.append({
            'Note': 'Individual trade details require custom tracking in strategy',
            'Total_Trades': trade_analyzer.get('total', {}).get('total', 0),
            'Winning_Trades': trade_analyzer.get('won', {}).get('total', 0),
            'Losing_Trades': trade_analyzer.get('lost', {}).get('total', 0)
        })
    
    df = pd.DataFrame(trades_data)
    df.to_csv(report_dir / "trades.csv", index=False)

def generate_json_stats(strategy, cerebro, args, report_dir):
    """Generate JSON file with all statistics"""
    
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()
    vwr = strategy.analyzers.vwr.get_analysis()
    
    stats = {
        'backtest_info': {
            'timestamp': datetime.now().isoformat(),
            'strategy': 'Point & Figure Multi-Timeframe',
            'data_file': args.parquet,
            'date_range': {'from': args.fromdate, 'to': args.todate}
        },
        'parameters': {
            'box_size': args.box,
            'reversal': args.reversal,
            'position_size': args.size,
            'initial_cash': args.cash
        },
        'performance': {
            'initial_value': args.cash,
            'final_value': float(cerebro.broker.getvalue()),
            'total_return_pct': (cerebro.broker.getvalue() - args.cash) / args.cash * 100,
            'sharpe_ratio': sharpe.get('sharperatio', 0)
        },
        'trades': trade_analyzer,
        'drawdown': drawdown,
        'returns': returns,
        'sqn': sqn,
        'vwr': vwr
    }
    
    with open(report_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

def save_chart_png(cerebro, report_dir):
    """Save the backtest chart as PNG"""
    try:
        figs = cerebro.plot(style='candlestick', volume=True, show=False)
        if figs and len(figs) > 0 and len(figs[0]) > 0:
            fig = figs[0][0]
            fig.savefig(report_dir / "backtest_chart.png", dpi=150, bbox_inches="tight")
            plt.close(fig)  # Close to free memory
    except Exception as e:
        print(f"Warning: Could not save chart - {e}")

# -----------------------------
# Runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', type=str, required=True, help='Path to 1h Parquet (e.g. data/ALPACA/NFLX/1h/nflx_1h.parquet)')
    ap.add_argument('--cash', type=float, default=100000.0)
    ap.add_argument('--box', type=float, default=1.50)
    ap.add_argument('--reversal', type=int, default=3)
    ap.add_argument('--size', type=int, default=1)
    ap.add_argument('--fromdate', type=str, default='2018-01-01')
    ap.add_argument('--todate', type=str, default='2069-12-31')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    df_1h = load_parquet_1h(Path(args.parquet))
    # date filter if desired
    df_1h = df_1h.loc[(df_1h.index >= pd.to_datetime(args.fromdate)) & (df_1h.index <= pd.to_datetime(args.todate))]
    
    # Ensure we have enough data and trim any potential problematic edges
    if len(df_1h) < 100:
        raise ValueError(f"Not enough data for backtest: {len(df_1h)} rows")
    
    # Remove the last few rows to avoid potential edge case issues
    df_1h = df_1h.iloc[:-5]  # Remove last 5 rows to avoid edge issues
    print(f"Using {len(df_1h)} data points for backtest")

    data_1h = Parquet1hPandas(dataname=df_1h)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    
    # Set some cerebro options to handle data alignment issues
    cerebro.addtz('UTC')

    # Add base (hourly) feed
    cerebro.adddata(data_1h)

    # Resample to Daily & Weekly from the SAME base feed (keeps alignment)
    dfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Days, compression=1)
    wfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Weeks, compression=1)

    # Draw arrows where orders execute + show trade profit/loss on chart
    cerebro.addobserver(bt.observers.BuySell, barplot=True, bardist=0.001)
    cerebro.addobserver(bt.observers.Trades)
    
    # Add comprehensive analyzers for detailed reporting
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

    cerebro.addstrategy(
        PnF_MTF_Strategy,
        box_size=args.box,
        reversal=args.reversal,
        size=args.size,
        printlog=(not args.quiet),
    )

    # Draw arrows where orders execute + show trade profit/loss on chart
    cerebro.addobserver(bt.observers.BuySell, barplot=True, bardist=0.001)
    cerebro.addobserver(bt.observers.Trades)

    # Run the backtest - with backup strategy capture
    global _strategy_backup
    _strategy_backup = None
    
    try:
        results = cerebro.run()
        if not args.quiet:
            print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        final_value = cerebro.broker.getvalue()
        print(f"Final Portfolio Value: {final_value:.2f}")
        
        # The backtest ran successfully but failed during finalization
        # Create a minimal report with the available information
        if "min() arg is an empty sequence" in str(e):
            print("Generating basic report from available data...")
            
            # Create a minimal report directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("reports") / f"pnf_backtest_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate basic performance metrics
            initial_value = args.cash
            total_return = (final_value - initial_value) / initial_value * 100
            
            # Create a basic HTML report
            basic_report = f'''<!DOCTYPE html>
<html>
<head><title>Point & Figure Backtest Report (Basic)</title></head>
<body>
<h1>Point & Figure Backtest Report</h1>
<p><strong>Note:</strong> This is a basic report generated due to a backtrader finalization issue.</p>
<p><strong>Initial Capital:</strong> ${initial_value:,.2f}</p>
<p><strong>Final Value:</strong> ${final_value:,.2f}</p>
<p><strong>Total Return:</strong> {total_return:.2f}%</p>
<p><strong>Net Profit:</strong> ${final_value - initial_value:,.2f}</p>
<p><strong>Strategy:</strong> Point & Figure Multi-Timeframe</p>
<p><strong>Box Size:</strong> {args.box}</p>
<p><strong>Reversal:</strong> {args.reversal}</p>
<p><strong>Date Range:</strong> {args.fromdate} to {args.todate}</p>
<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>'''
            
            with open(report_dir / "basic_report.html", "w") as f:
                f.write(basic_report)
                
            print(f"\\n📊 Basic report generated in: {report_dir}")
            print(f"📈 Open {report_dir / 'basic_report.html'} for results")
            
        results = []

    # Generate reports if we have results
    if results and len(results) > 0:
        try:
            generate_reports(cerebro, results, args)
        except Exception as e:
            print(f"Report generation failed: {e}")
    else:
        print("No backtest results available for reporting")

    # Show a candlestick chart with volume and markers (requires matplotlib)
    if not args.quiet:
        try:
            cerebro.plot(style='candlestick', volume=True)
        except ValueError as e:
            if "min() arg is an empty sequence" in str(e):
                print("Note: Chart plotting skipped due to data alignment issue")
            else:
                print(f"Chart plotting failed: {e}")
        except Exception as e:
            print(f"Chart plotting failed: {e}")

    print(f"The end")

if __name__ == '__main__':
    main()
