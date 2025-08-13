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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from strategies import get_strategy, list_strategies

# Global variable to store strategy backup in case of backtrader failure
_strategy_backup = None

# -----------------------------
# PandasData feed wrapper for our Parquet
# -----------------------------
class Parquet1hPandas(bt.feeds.PandasData):
    # backtrader default names: datetime, open, high, low, close, volume, openinterest
    lines = ('datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest')
    params = (
        ('datetime',    'datetime'),
        ('open',        'open'),
        ('high',        'high'),
        ('low',         'low'),
        ('close',       'close'),
        ('volume',      'volume'),
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

def load_daily_data(parquet_path: Path):
    """Load and resample hourly data to daily for daily strategies"""
    df_1h = load_parquet_1h(parquet_path)
    
    # Resample to daily OHLCV
    df_daily = df_1h.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    print(f"Resampled to {len(df_daily)} daily bars from {df_daily.index.min()} to {df_daily.index.max()}")
    return df_daily

# -----------------------------
# Report Generation Functions
# -----------------------------
def generate_reports(cerebro, results, args, strategy_name):
    """Generate comprehensive TradingView-style reports"""
    
    # Create reports directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / f"{strategy_name}_backtest_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    strategy = results[0]
    
    # 1. Generate HTML Summary Report
    generate_html_report(strategy, cerebro, args, report_dir, strategy_name)
    
    # 2. Generate CSV Trade Log
    generate_trade_csv(strategy, report_dir)
    
    # 3. Generate JSON Statistics
    generate_json_stats(strategy, cerebro, args, report_dir, strategy_name)
    
    # 4. Save Chart as PNG
    save_chart_png(cerebro, report_dir)
    
    print(f"\\nReports generated in: {report_dir}")
    print(f"Open {report_dir / 'backtest_report.html'} for detailed analysis")

def generate_html_report(strategy, cerebro, args, report_dir, strategy_name):
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
        <title>{strategy_name.title()} Backtest Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
            .metric-box {{ background: #ecf0f1; padding: 15px; border-radius: 8px; border-left: 4px solid #3498db; }}
            .metric-label {{ font-weight: bold; color: #7f8c8d; font-size: 12px; text-transform: uppercase; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; margin-top: 5px; }}
            .positive {{ color: #27ae60; }}
            .negative {{ color: #e74c3c; }}
            .info-section {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{strategy_name.title()} Strategy Backtest Report</h1>
            
            <div class="info-section">
                <h2>Strategy Information</h2>
                <p><strong>Strategy:</strong> {strategy_name.title()}</p>
                <p><strong>Data File:</strong> {args.parquet}</p>
                <p><strong>Date Range:</strong> {args.fromdate} to {args.todate}</p>
                <p><strong>Position Size:</strong> {args.size}</p>
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <h2>Performance Overview</h2>
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-label">Initial Capital</div>
                    <div class="metric-value">${initial_value:,.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Final Value</div>
                    <div class="metric-value">${final_value:,.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {'positive' if (final_value - initial_value) >= 0 else 'negative'}">${final_value - initial_value:,.2f}</div>
                </div>
            </div>
            
            <h2>Trading Statistics</h2>
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{total_trades}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Winning Trades</div>
                    <div class="metric-value positive">{winning_trades}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Losing Trades</div>
                    <div class="metric-value negative">{losing_trades}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{win_rate:.1f}%</div>
                </div>
            </div>
            
            <h2>Risk Metrics</h2>
            <div class="metrics">
                <div class="metric-box">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">{profit_factor:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Average Win</div>
                    <div class="metric-value positive">${avg_win:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Average Loss</div>
                    <div class="metric-value negative">${avg_loss:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{max_dd:.2f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{sharpe.get('sharperatio', 0):.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">SQN</div>
                    <div class="metric-value">{sqn.get('sqn', 0):.3f}</div>
                </div>
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
    
    df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame({'Note': ['No trades executed']})
    df.to_csv(report_dir / "trades.csv", index=False)

def generate_json_stats(strategy, cerebro, args, report_dir, strategy_name):
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
            'strategy': strategy_name,
            'data_file': args.parquet,
            'date_range': {'from': args.fromdate, 'to': args.todate}
        },
        'parameters': {
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

def setup_data_feeds(cerebro, strategy_class, df_data, args):
    """Setup data feeds based on strategy requirements"""
    data_reqs = strategy_class.get_data_requirements()
    
    if data_reqs['base_timeframe'] == 'daily':
        # For daily strategies, use daily data
        data_feed = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_feed)
        return [data_feed]
    
    elif data_reqs['base_timeframe'] == 'hourly' and data_reqs['requires_resampling']:
        # For PnF strategy that needs hourly + resampled daily/weekly
        data_1h = Parquet1hPandas(dataname=df_data)
        cerebro.adddata(data_1h)
        
        # Resample to Daily & Weekly from the SAME base feed (keeps alignment)
        dfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Days, compression=1)
        wfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Weeks, compression=1)
        
        return [data_1h, dfeed, wfeed]
    
    else:
        # Default: single feed
        data_feed = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_feed)
        return [data_feed]

# -----------------------------
# Runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-Strategy Backtrader Runner")
    ap.add_argument('--parquet', type=str, required=True, 
                   help='Path to Parquet file (e.g. data/ALPACA/NFLX/1h/nflx_1h.parquet)')
    ap.add_argument('--strategy', type=str, default='mean_reversion',
                   help=f'Strategy to use. Available: {", ".join(list_strategies())}')
    ap.add_argument('--cash', type=float, default=100000.0, help='Initial cash')
    ap.add_argument('--size', type=int, default=1, help='Position size')
    ap.add_argument('--fromdate', type=str, default='2018-01-01', help='Start date (YYYY-MM-DD)')
    ap.add_argument('--todate', type=str, default='2069-12-31', help='End date (YYYY-MM-DD)')
    ap.add_argument('--quiet', action='store_true', help='Suppress trading logs')
    
    # Strategy-specific parameters
    ap.add_argument('--box', type=float, default=1.50, help='PnF box size (for pnf strategy)')
    ap.add_argument('--reversal', type=int, default=3, help='PnF reversal boxes (for pnf strategy)')
    ap.add_argument('--period', type=int, default=20, help='MA period (for mean_reversion strategy)')
    ap.add_argument('--devfactor', type=float, default=2.0, help='Std dev factor (for mean_reversion strategy)')
    
    args = ap.parse_args()

    # Validate strategy
    try:
        strategy_class = get_strategy(args.strategy)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load data based on strategy requirements
    data_reqs = strategy_class.get_data_requirements()
    
    if data_reqs['base_timeframe'] == 'daily':
        print(f"Loading daily data for {args.strategy} strategy...")
        df_data = load_daily_data(Path(args.parquet))
    else:
        print(f"Loading hourly data for {args.strategy} strategy...")
        df_data = load_parquet_1h(Path(args.parquet))
    
    # Apply date filter
    df_data = df_data.loc[(df_data.index >= pd.to_datetime(args.fromdate)) & 
                         (df_data.index <= pd.to_datetime(args.todate))]
    
    # Ensure we have enough data and trim any potential problematic edges
    if len(df_data) < 30:
        raise ValueError(f"Not enough data for backtest: {len(df_data)} rows")
    
    # Remove the last few rows to avoid potential edge case issues
    df_data = df_data.iloc[:-5]  # Remove last 5 rows to avoid edge issues
    print(f"Using {len(df_data)} data points for backtest")

    # Setup Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(args.cash)
    
    # Set some cerebro options to handle data alignment issues
    cerebro.addtz('UTC')

    # Setup data feeds based on strategy requirements
    data_feeds = setup_data_feeds(cerebro, strategy_class, df_data, args)

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

    # Add strategy with appropriate parameters
    strategy_params = {
        'size': args.size,
        'printlog': not args.quiet,
    }
    
    # Add strategy-specific parameters
    if args.strategy == 'pnf':
        strategy_params.update({
            'box_size': args.box,
            'reversal': args.reversal,
        })
    elif args.strategy == 'mean_reversion':
        strategy_params.update({
            'period': args.period,
            'devfactor': args.devfactor,
        })
    
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Run the backtest - with backup strategy capture
    global _strategy_backup
    _strategy_backup = None
    
    print(f"\\nRunning {args.strategy} strategy...")
    print(f"Strategy: {strategy_class.get_description()}")
    
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
            report_dir = Path("reports") / f"{args.strategy}_backtest_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate basic performance metrics
            initial_value = args.cash
            total_return = (final_value - initial_value) / initial_value * 100
            
            # Create a basic HTML report
            basic_report = f'''<!DOCTYPE html>
<html>
<head><title>{args.strategy.title()} Backtest Report (Basic)</title></head>
<body>
<h1>{args.strategy.title()} Strategy Backtest Report</h1>
<p><strong>Note:</strong> This is a basic report generated due to a backtrader finalization issue.</p>
<p><strong>Strategy:</strong> {args.strategy.title()}</p>
<p><strong>Initial Capital:</strong> ${initial_value:,.2f}</p>
<p><strong>Final Value:</strong> ${final_value:,.2f}</p>
<p><strong>Total Return:</strong> {total_return:.2f}%</p>
<p><strong>Net Profit:</strong> ${final_value - initial_value:,.2f}</p>
<p><strong>Date Range:</strong> {args.fromdate} to {args.todate}</p>
<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>'''
            
            with open(report_dir / "basic_report.html", "w") as f:
                f.write(basic_report)
                
            print(f"\\nBasic report generated in: {report_dir}")
            print(f"Open {report_dir / 'basic_report.html'} for results")
            
        results = []

    # Generate reports if we have results
    if results and len(results) > 0:
        try:
            generate_reports(cerebro, results, args, args.strategy)
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
