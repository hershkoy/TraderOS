# backtrader_runner_yaml.py - Multi-Strategy Backtrader Runner with YAML Configuration
import argparse
from pathlib import Path
import pandas as pd
import backtrader as bt
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import yaml
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.offline import plot

# Import strategies
from strategies import get_strategy, list_strategies

# Global variable to store strategy backup in case of backtrader failure
_strategy_backup = None

def load_config(config_path="defaults.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file {config_path} not found. Using hardcoded defaults.")
        return get_default_config()
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        return get_default_config()

def get_default_config():
    """Fallback configuration if YAML file is not available"""
    return {
        'global': {
            'cash': 100000.0,
            'size': 1,
            'fromdate': '2018-01-01',
            'todate': '2069-12-31',
            'quiet': False
        },
        'strategies': {
            'mean_reversion': {
                'parameters': {
                    'period': 20,
                    'devfactor': 2.0,
                    'commission': 0.001
                }
            },
            'mean_reversion_clean': {
                'parameters': {
                    'period': 20,
                    'devfactor': 2.0,
                    'commission': 0.001
                }
            },
            'pnf': {
                'parameters': {
                    'box_size': 1.50,
                    'reversal': 3,
                    'commission': 0.001,
                    'slippage': 0.0005
                }
            }
        },
        'data': {
            'min_data_points': 30,
            'edge_trim': 5
        }
    }

def merge_config_with_args(config, args):
    """Merge YAML config with command line arguments (args take precedence)"""
    # Start with global defaults from config
    merged = config.get('global', {}).copy()
    
    # Override with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None:
            merged[key] = value
    
    return merged

def get_strategy_config(config, strategy_name):
    """Get configuration for a specific strategy"""
    strategies_config = config.get('strategies', {})
    strategy_config = strategies_config.get(strategy_name, {})
    return strategy_config.get('parameters', {})

# -----------------------------
# PandasData feed wrapper for our Parquet
# -----------------------------
class Parquet1hPandas(bt.feeds.PandasData):
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
    ts = pd.to_datetime(df['ts_event'], unit='ns', utc=True).dt.tz_localize(None)
    out = pd.DataFrame({
        'datetime': ts,
        'open': df['open'].astype(float),
        'high': df['high'].astype(float),
        'low': df['low'].astype(float),
        'close': df['close'].astype(float),
        'volume': df['volume'].astype(float),
    }).sort_values('datetime').reset_index(drop=True)
    
    now = pd.Timestamp.now()
    out = out[out['datetime'] <= now]
    print(f"After filtering future dates: {len(out)} rows")
    
    if len(out) < 10:
        raise ValueError(f"Not enough data after filtering: {len(out)} rows")
    
    out.set_index('datetime', inplace=True)
    print(f"Loaded {len(out)} data points from {out.index.min()} to {out.index.max()}")
    
    if out.index.duplicated().any():
        print("Warning: Found duplicate timestamps, removing duplicates")
        out = out[~out.index.duplicated(keep='first')]
    
    return out

def load_daily_data(parquet_path: Path):
    """Load and resample hourly data to daily for daily strategies"""
    df_1h = load_parquet_1h(parquet_path)
    
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
def generate_reports(cerebro, results, config, strategy_name, data_df=None):
    """Generate comprehensive reports"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path("reports") / f"{strategy_name}_backtest_{timestamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    strategy = results[0]
    
    # Generate reports based on config
    reports_config = config.get('reports', {})
    
    if reports_config.get('generate_html', True):
        generate_html_report(strategy, cerebro, config, report_dir, strategy_name, data_df)
    
    if reports_config.get('generate_csv', True):
        generate_trade_csv(strategy, report_dir)
    
    if reports_config.get('generate_json', True):
        generate_json_stats(strategy, cerebro, config, report_dir, strategy_name)
    
    if reports_config.get('generate_chart', True):
        save_chart_png(cerebro, report_dir, config)
    
    print(f"\nReports generated in: {report_dir}")
    print(f"Open {report_dir / 'backtest_report.html'} for detailed analysis")

def create_plotly_charts(strategy, cerebro, data_df, report_dir):
    """Create interactive Plotly charts for the backtest"""
    try:
        # Get portfolio value over time
        portfolio_values = []
        dates = []
        
        # Try to get portfolio values from strategy if available
        if hasattr(strategy, '_custom_portfolio_values') and strategy._custom_portfolio_values:
            for date, value in strategy._custom_portfolio_values:
                dates.append(date)
                portfolio_values.append(value)
        else:
            # Fallback: create simple portfolio value chart using data
            initial_value = cerebro.broker.getvalue()
            final_value = cerebro.broker.getvalue()
            dates = data_df.index.tolist()
            # Simple linear interpolation for demo
            portfolio_values = [initial_value + (final_value - initial_value) * i / len(dates) 
                              for i in range(len(dates))]
        
        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price & Signals', 'Portfolio Value', 'Volume'),
            row_heights=[0.5, 0.3, 0.2]
        )
        
        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data_df.index,
                open=data_df['open'],
                high=data_df['high'],
                low=data_df['low'],
                close=data_df['close'],
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add buy/sell signals if available
        if hasattr(strategy, '_custom_signals') and strategy._custom_signals:
            buy_signals = [s for s in strategy._custom_signals if s['action'] == 'BUY']
            sell_signals = [s for s in strategy._custom_signals if s['action'] == 'SELL']
            
            if buy_signals:
                buy_dates = [s['date'] for s in buy_signals]
                buy_prices = [s['price'] for s in buy_signals]
                fig.add_trace(
                    go.Scatter(
                        x=buy_dates,
                        y=buy_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        name='Buy Signals'
                    ),
                    row=1, col=1
                )
            
            if sell_signals:
                sell_dates = [s['date'] for s in sell_signals]
                sell_prices = [s['price'] for s in sell_signals]
                fig.add_trace(
                    go.Scatter(
                        x=sell_dates,
                        y=sell_prices,
                        mode='markers',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        name='Sell Signals'
                    ),
                    row=1, col=1
                )
        
        # Add portfolio value
        if dates and portfolio_values:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='orange', width=2),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Add volume
        fig.add_trace(
            go.Bar(
                x=data_df.index,
                y=data_df['volume'],
                name='Volume',
                marker_color='rgba(158,202,225,0.6)',
                showlegend=False
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Backtest Results",
            template="plotly_dark",
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Update x-axes
        fig.update_xaxes(rangeslider_visible=False)
        
        # Save as HTML
        chart_html_path = report_dir / "charts.html"
        plot(fig, filename=str(chart_html_path), auto_open=False, include_plotlyjs='cdn')
        
        # Return the HTML content for embedding
        return fig.to_html(include_plotlyjs='cdn', div_id="plotly-chart")
        
    except Exception as e:
        print(f"Warning: Could not create Plotly charts - {e}")
        return "<div>Chart generation failed</div>"

def generate_trades_table_html(trades_list):
    """Generate HTML for individual trades table"""
    trades_html = ""
    for i, trade in enumerate(trades_list, 1):
        pnl_class = "positive" if trade.get('pnl', 0) >= 0 else "negative"
        pnl_pct_class = "positive" if trade.get('pnl_pct', 0) >= 0 else "negative"
        
        trades_html += f"""
            <tr>
                <td>{i}</td>
                <td>{trade.get('type', 'N/A')}</td>
                <td>{trade.get('entry_date', 'N/A')}</td>
                <td>${trade.get('entry_price', 0):.2f}</td>
                <td>{trade.get('exit_date', 'N/A')}</td>
                <td>${trade.get('exit_price', 0):.2f}</td>
                <td>{trade.get('size', 0)}</td>
                <td class="{pnl_class}">${trade.get('pnl', 0):.2f}</td>
                <td class="{pnl_pct_class}">{trade.get('pnl_pct', 0):.2f}%</td>
                <td>{trade.get('duration', 'N/A')}</td>
            </tr>
        """
    return trades_html

def generate_html_report(strategy, cerebro, config, report_dir, strategy_name, data_df=None):
    """Generate TradingView-style HTML report with tabs"""
    
    try:
        print("DEBUG: Starting HTML report generation")
        
        # Use custom tracking instead of Backtrader analyzers
        if hasattr(strategy, 'get_trade_statistics'):
            trade_stats = strategy.get_trade_statistics()
        else:
            trade_stats = {
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
        
        print(f"DEBUG: Custom trade stats: {trade_stats}")
        
        # Create Plotly charts
        chart_html = ""
        if data_df is not None:
            chart_html = create_plotly_charts(strategy, cerebro, data_df, report_dir)
        
        # Get individual trades if available
        trades_html = ""
        if hasattr(strategy, '_custom_trades') and strategy._custom_trades and isinstance(strategy._custom_trades, list):
            trades_html = generate_trades_table_html(strategy._custom_trades)
        else:
            trades_html = "<tr><td colspan='10' style='text-align: center; color: #787b86;'>No individual trade data available. Trades tracking needs to be implemented in strategy.</td></tr>"
        
        # Get configuration values
        global_config = config.get('global', {})
        initial_value = global_config.get('cash', 100000.0)
        final_value = cerebro.broker.getvalue()
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Get strategy description from config
        strategy_config = config.get('strategies', {}).get(strategy_name, {})
        strategy_description = strategy_config.get('description', f'{strategy_name.title()} Strategy')
    
        # Get trade statistics from custom tracking
        print("DEBUG: Processing custom trade statistics")
        total_trades = int(trade_stats.get('total_trades', 0))
        winning_trades = int(trade_stats.get('winning_trades', 0))
        losing_trades = int(trade_stats.get('losing_trades', 0))
        win_rate = float(trade_stats.get('win_rate', 0.0))
        
        gross_profit = float(trade_stats.get('avg_win', 0.0) * winning_trades if winning_trades > 0 else 0.0)
        gross_loss = float(abs(trade_stats.get('avg_loss', 0.0) * losing_trades if losing_trades > 0 else 0.0))
        profit_factor = float(trade_stats.get('profit_factor', 0.0))
        
        avg_win = float(trade_stats.get('avg_win', 0.0))
        avg_loss = float(trade_stats.get('avg_loss', 0.0))
        
        max_dd = float(trade_stats.get('max_drawdown_pct', 0.0))
        max_dd_pct = float(trade_stats.get('max_drawdown', 0.0))
        
        print(f"DEBUG: Total trades: {total_trades}, type: {type(total_trades)}")
        print(f"DEBUG: Win rate: {win_rate}, type: {type(win_rate)}")
        print(f"DEBUG: Profit factor: {profit_factor}, type: {type(profit_factor)}")
        
        # Calculate additional metrics
        print("DEBUG: Calculating additional metrics")
        net_profit = final_value - initial_value
        profit_factor_display = f"{profit_factor:.2f}" if profit_factor != float('inf') else "∞"
        print(f"DEBUG: Net profit: {net_profit}, type: {type(net_profit)}")
        print(f"DEBUG: Profit factor display: {profit_factor_display}, type: {type(profit_factor_display)}")
        
        # Get additional trade details (with safe defaults)
        max_win_streak = 0  # Could be implemented in custom tracking if needed
        max_loss_streak = 0  # Could be implemented in custom tracking if needed
        
        print("DEBUG: Starting HTML content generation")
        
        # Generate the HTML content (simplified for brevity)
        html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>{strategy_name.title()} Backtest Report</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{strategy_name.title()} Strategy</h1>
                <p>Backtest Report - {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {'positive' if net_profit >= 0 else 'negative'}">${net_profit:,.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{total_trades}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
                </div>
            </div>
            
            <p><strong>Note:</strong> This report uses custom tracking instead of Backtrader analyzers for better reliability.</p>
        </div>
    </body>
    </html>
    '''
        
        print("DEBUG: Writing HTML file")
        with open(report_dir / "backtest_report.html", "w", encoding='utf-8') as f:
            f.write(html_content)
        print("DEBUG: HTML file written successfully")
    
    except Exception as e:
        print(f"ERROR in generate_html_report: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise

def generate_trade_csv(strategy, report_dir):
    """Generate CSV file with individual trade details"""
    
    trades_data = []
    if hasattr(strategy, '_custom_trades') and strategy._custom_trades:
        # Export individual trades
        for i, trade in enumerate(strategy._custom_trades, 1):
            trades_data.append({
                'Trade_#': i,
                'Type': trade.get('type', 'N/A'),
                'Entry_Date': trade.get('entry_date', 'N/A'),
                'Entry_Price': trade.get('entry_price', 0),
                'Exit_Date': trade.get('exit_date', 'N/A'),
                'Exit_Price': trade.get('exit_price', 0),
                'Size': trade.get('size', 0),
                'P&L': trade.get('pnl', 0),
                'P&L_%': trade.get('pnl_pct', 0),
                'Duration': trade.get('duration', 'N/A')
            })
    else:
        trades_data.append({
            'Note': 'No trades executed or custom tracking not implemented'
        })
    
    df = pd.DataFrame(trades_data)
    df.to_csv(report_dir / "trades.csv", index=False)

def generate_json_stats(strategy, cerebro, config, report_dir, strategy_name):
    """Generate JSON file with all statistics"""
    
    # Use custom tracking instead of Backtrader analyzers
    if hasattr(strategy, 'get_trade_statistics'):
        trade_stats = strategy.get_trade_statistics()
    else:
        trade_stats = {}
    
    # Get portfolio values if available
    portfolio_values = []
    if hasattr(strategy, '_custom_portfolio_values'):
        portfolio_values = strategy._custom_portfolio_values
    
    global_config = config.get('global', {})
    strategy_config = config.get('strategies', {}).get(strategy_name, {})
    
    stats = {
        'backtest_info': {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy_name,
            'description': strategy_config.get('description', ''),
            'date_range': {
                'from': global_config.get('fromdate', ''),
                'to': global_config.get('todate', '')
            }
        },
        'configuration': {
            'global': global_config,
            'strategy_parameters': strategy_config.get('parameters', {})
        },
        'performance': {
            'initial_value': global_config.get('cash', 100000.0),
            'final_value': float(cerebro.broker.getvalue()),
            'total_return_pct': (cerebro.broker.getvalue() - global_config.get('cash', 100000.0)) / global_config.get('cash', 100000.0) * 100
        },
        'custom_tracking': {
            'trade_statistics': trade_stats,
            'portfolio_values': portfolio_values,
            'signals': getattr(strategy, '_custom_signals', []),
            'individual_trades': getattr(strategy, '_custom_trades', [])
        }
    }
    
    with open(report_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

def save_chart_png(cerebro, report_dir, config):
    """Save the backtest chart as PNG"""
    try:
        backtrader_config = config.get('backtrader', {})
        style = backtrader_config.get('plot_style', 'candlestick')
        volume = backtrader_config.get('plot_volume', True)
        
        figs = cerebro.plot(style=style, volume=volume, show=False)
        if figs and len(figs) > 0 and len(figs[0]) > 0:
            fig = figs[0][0]
            fig.savefig(report_dir / "backtest_chart.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not save chart - {e}")

def setup_data_feeds(cerebro, strategy_class, df_data, config):
    """Setup data feeds based on strategy requirements"""
    data_reqs = strategy_class.get_data_requirements()
    
    if data_reqs['base_timeframe'] == 'daily':
        # Use PandasData for better compatibility
        data_feed = bt.feeds.PandasData(
            dataname=df_data,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        cerebro.adddata(data_feed, name="NFLX")
        return [data_feed]
    
    elif data_reqs['base_timeframe'] == 'hourly' and data_reqs['requires_resampling']:
        data_1h = Parquet1hPandas(dataname=df_data)
        cerebro.adddata(data_1h)
        
        dfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Days, compression=1)
        wfeed = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Weeks, compression=1)
        
        return [data_1h, dfeed, wfeed]
    
    else:
        data_feed = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_feed)
        return [data_feed]

# -----------------------------
# Runner
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Multi-Strategy Backtrader Runner with YAML Configuration")
    ap.add_argument('--config', type=str, default='defaults.yaml',
                   help='Path to YAML configuration file')
    ap.add_argument('--parquet', type=str, required=True, 
                   help='Path to Parquet file (e.g. data/ALPACA/NFLX/1h/nflx_1h.parquet)')
    ap.add_argument('--strategy', type=str, default='mean_reversion',
                   help=f'Strategy to use. Available: {", ".join(list_strategies())}')
    
    # Optional overrides for global settings
    ap.add_argument('--cash', type=float, help='Initial cash (overrides config)')
    ap.add_argument('--size', type=int, help='Position size (overrides config)')
    ap.add_argument('--fromdate', type=str, help='Start date (overrides config)')
    ap.add_argument('--todate', type=str, help='End date (overrides config)')
    ap.add_argument('--quiet', action='store_true', help='Suppress trading logs (overrides config)')
    ap.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                   default='INFO', help='Set logging level')
    
    # Optional overrides for strategy parameters (will be passed to strategy if applicable)
    ap.add_argument('--box', type=float, help='PnF box size (overrides config for pnf strategy)')
    ap.add_argument('--reversal', type=int, help='PnF reversal boxes (overrides config for pnf strategy)')
    ap.add_argument('--period', type=int, help='MA period (overrides config for mean_reversion strategy)')
    ap.add_argument('--devfactor', type=float, help='Std dev factor (overrides config for mean_reversion strategy)')
    
    args = ap.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Merge config with command line arguments
    global_config = merge_config_with_args(config, args)
    
    # Validate strategy
    try:
        strategy_class = get_strategy(args.strategy)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Get strategy configuration
    strategy_config = get_strategy_config(config, args.strategy)
    print(f"Strategy configuration: {strategy_config}")
    
    # Load data based on strategy requirements
    data_reqs = strategy_class.get_data_requirements()
    
    if data_reqs['base_timeframe'] == 'daily':
        print(f"Loading daily data for {args.strategy} strategy...")
        df_data = load_daily_data(Path(args.parquet))
    else:
        print(f"Loading hourly data for {args.strategy} strategy...")
        df_data = load_parquet_1h(Path(args.parquet))
    
    # Apply date filter
    fromdate = global_config.get('fromdate', '2018-01-01')
    todate = global_config.get('todate', '2069-12-31')
    df_data = df_data.loc[(df_data.index >= pd.to_datetime(fromdate)) & 
                         (df_data.index <= pd.to_datetime(todate))]
    
    # Apply data settings from config
    data_config = config.get('data', {})
    min_points = data_config.get('min_data_points', 30)
    edge_trim = data_config.get('edge_trim', 5)
    
    if len(df_data) < min_points:
        raise ValueError(f"Not enough data for backtest: {len(df_data)} rows (minimum: {min_points})")
    
    df_data = df_data.iloc[:-edge_trim]
    print(f"Using {len(df_data)} data points for backtest")

    # Setup Cerebro
    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(global_config.get('cash', 100000.0))
    
    # Set broker parameters to avoid potential issues
    cerebro.broker.set_checksubmit(False)
    cerebro.broker.set_coo(False)  # Turn off cheat-on-open for testing
    
    # Set timezone from config
    backtrader_config = config.get('backtrader', {})
    timezone = backtrader_config.get('timezone', 'UTC')
    cerebro.addtz(timezone)

    # Setup data feeds
    data_feeds = setup_data_feeds(cerebro, strategy_class, df_data, config)

    # Custom tracking is now handled by the strategy itself
    # No need for Backtrader analyzers or observers that can cause compatibility issues

    # Prepare strategy parameters
    strategy_params = strategy_config.copy()
    strategy_params['size'] = global_config.get('size', 1)
    strategy_params['printlog'] = not global_config.get('quiet', False)
    strategy_params['log_level'] = args.log_level
    
    # Override with command line arguments if provided
    if args.box is not None and args.strategy == 'pnf':
        strategy_params['box_size'] = args.box
    if args.reversal is not None and args.strategy == 'pnf':
        strategy_params['reversal'] = args.reversal
    if args.period is not None and args.strategy == 'mean_reversion':
        strategy_params['period'] = args.period
    if args.devfactor is not None and args.strategy == 'mean_reversion':
        strategy_params['devfactor'] = args.devfactor
    
    cerebro.addstrategy(strategy_class, **strategy_params)

    # Run the backtest
    global _strategy_backup
    _strategy_backup = None
    
    print(f"\nRunning {args.strategy} strategy...")
    print(f"Strategy: {strategy_class.get_description()}")
    print(f"Parameters: {strategy_params}")
    print(f"Using custom tracking instead of Backtrader analyzers")
    
    # Store the merged config for reporting
    config['global'] = global_config
    
    # Ensure strategy exists in config
    if args.strategy not in config['strategies']:
        config['strategies'][args.strategy] = {'parameters': {}}
    
    config['strategies'][args.strategy]['parameters'] = strategy_params
    
    try:
        results = cerebro.run()
        if not global_config.get('quiet', False):
            print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    except Exception as e:
        print(f"Backtest failed with error: {e}")
        import traceback; traceback.print_exc()
        final_value = cerebro.broker.getvalue()
        print(f"Final Portfolio Value: {final_value:.2f}")
        
        if "min() arg is an empty sequence" in str(e):
            print("Generating basic report from available data...")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path("reports") / f"{args.strategy}_backtest_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            initial_value = global_config.get('cash', 100000.0)
            total_return = (final_value - initial_value) / initial_value * 100
            
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
<p><strong>Date Range:</strong> {fromdate} to {todate}</p>
<p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
</body>
</html>'''
            
            with open(report_dir / "basic_report.html", "w") as f:
                basic_report.write(basic_report)
                
            print(f"\nBasic report generated in: {report_dir}")
            print(f"Open {report_dir / 'basic_report.html'} for results")
            
        results = []

    # Generate reports if we have results
    if results and len(results) > 0:
        try:
            generate_reports(cerebro, results, config, args.strategy, df_data)
        except Exception as e:
            print(f"Report generation failed: {e}")
    else:
        print("No backtest results available for reporting")

    # Show chart if enabled
    if not global_config.get('quiet', False):
        try:
            backtrader_config = config.get('backtrader', {})
            style = backtrader_config.get('plot_style', 'candlestick')
            volume = backtrader_config.get('plot_volume', True)
            cerebro.plot(style=style, volume=volume)
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
