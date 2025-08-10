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
    
    print(f"\\nReports generated in: {report_dir}")
    print(f"Open {report_dir / 'backtest_report.html'} for detailed analysis")

def create_plotly_charts(strategy, cerebro, data_df, report_dir):
    """Create interactive Plotly charts for the backtest"""
    try:
        # Get portfolio value over time
        portfolio_values = []
        dates = []
        
        # Try to get portfolio values from strategy if available
        if hasattr(strategy, '_portfolio_values') and strategy._portfolio_values:
            for date, value in strategy._portfolio_values:
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
        if hasattr(strategy, '_signals') and strategy._signals:
            buy_signals = [s for s in strategy._signals if s['action'] == 'BUY']
            sell_signals = [s for s in strategy._signals if s['action'] == 'SELL']
            
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
        
        trade_analyzer = strategy.analyzers.trades.get_analysis() or {}
        sharpe = strategy.analyzers.sharpe.get_analysis() or {}
        drawdown = strategy.analyzers.drawdown.get_analysis() or {}
        returns = strategy.analyzers.returns.get_analysis() or {}
        sqn = strategy.analyzers.sqn.get_analysis() or {}
        vwr = strategy.analyzers.vwr.get_analysis() or {}
        
        print(f"DEBUG: Trade analyzer: {trade_analyzer}")
        print(f"DEBUG: Sharpe: {sharpe}")
        print(f"DEBUG: Drawdown: {drawdown}")
        
        # Create Plotly charts
        chart_html = ""
        if data_df is not None:
            chart_html = create_plotly_charts(strategy, cerebro, data_df, report_dir)
        
        # Get individual trades if available
        trades_html = ""
        if hasattr(strategy, '_trades') and strategy._trades:
            trades_html = generate_trades_table_html(strategy._trades)
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
    
        # Get trade statistics with explicit type conversion
        print("DEBUG: Processing trade statistics")
        total_trades = int(trade_analyzer.get('total', {}).get('total', 0))
        winning_trades = int(trade_analyzer.get('won', {}).get('total', 0))
        losing_trades = int(trade_analyzer.get('lost', {}).get('total', 0))
        win_rate = float((winning_trades / total_trades * 100) if total_trades > 0 else 0)
        
        gross_profit = float(trade_analyzer.get('won', {}).get('pnl', {}).get('total', 0) or 0)
        gross_loss = float(abs(trade_analyzer.get('lost', {}).get('pnl', {}).get('total', 0) or 0))
        profit_factor = float((gross_profit / gross_loss) if gross_loss > 0 else float('inf'))
        
        avg_win = float(trade_analyzer.get('won', {}).get('pnl', {}).get('average', 0) or 0)
        avg_loss = float(trade_analyzer.get('lost', {}).get('pnl', {}).get('average', 0) or 0)
        
        max_dd = float(drawdown.get('max', {}).get('drawdown', 0) or 0)
        max_dd_pct = float(drawdown.get('max', {}).get('moneydown', 0) or 0)
        
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
        max_win_streak = int(trade_analyzer.get('streak', {}).get('won', {}).get('longest', 0) or 0)
        max_loss_streak = int(trade_analyzer.get('streak', {}).get('lost', {}).get('longest', 0) or 0)
        
        print("DEBUG: Starting HTML content generation")
        html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{strategy_name.title()} Backtest Report</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: #0d1421;
                color: #d1d4dc;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1400px;
                margin: 0 auto;
                background: #131722;
                min-height: 100vh;
            }}
            
            .header {{
                background: #1e222d;
                padding: 20px 30px;
                border-bottom: 1px solid #2a2e39;
            }}
            
            .header h1 {{
                color: #f7931a;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 5px;
            }}
            
            .header .subtitle {{
                color: #787b86;
                font-size: 14px;
            }}
            
            .nav-tabs {{
                background: #1e222d;
                border-bottom: 1px solid #2a2e39;
                padding: 0 30px;
                display: flex;
                overflow-x: auto;
            }}
            
            .nav-tab {{
                padding: 15px 20px;
                cursor: pointer;
                color: #787b86;
                border-bottom: 3px solid transparent;
                transition: all 0.3s ease;
                font-weight: 500;
                white-space: nowrap;
            }}
            
            .nav-tab:hover {{
                color: #d1d4dc;
                background: #2a2e39;
            }}
            
            .nav-tab.active {{
                color: #f7931a;
                border-bottom-color: #f7931a;
                background: #131722;
            }}
            
            .tab-content {{
                display: none;
                padding: 30px;
                animation: fadeIn 0.3s ease-in-out;
            }}
            
            .tab-content.active {{
                display: block;
            }}
            
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            
            .overview-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 30px;
            }}
            
            .performance-summary {{
                background: #1e222d;
                border-radius: 8px;
                padding: 25px;
                border: 1px solid #2a2e39;
            }}
            
            .performance-summary h3 {{
                color: #d1d4dc;
                margin-bottom: 20px;
                font-size: 18px;
                font-weight: 600;
            }}
            
            .key-metrics {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            .metric {{
                text-align: center;
            }}
            
            .metric-label {{
                color: #787b86;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }}
            
            .metric-value {{
                font-size: 24px;
                font-weight: 600;
                color: #d1d4dc;
            }}
            
            .metric-value.positive {{
                color: #4caf50;
            }}
            
            .metric-value.negative {{
                color: #f44336;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .stat-card {{
                background: #1e222d;
                border-radius: 8px;
                padding: 20px;
                border: 1px solid #2a2e39;
                text-align: center;
            }}
            
            .stat-card .label {{
                color: #787b86;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 10px;
            }}
            
            .stat-card .value {{
                font-size: 20px;
                font-weight: 600;
                color: #d1d4dc;
            }}
            
            .stat-card .value.positive {{
                color: #4caf50;
            }}
            
            .stat-card .value.negative {{
                color: #f44336;
            }}
            
            .info-table {{
                background: #1e222d;
                border-radius: 8px;
                border: 1px solid #2a2e39;
                overflow: hidden;
            }}
            
            .info-table h3 {{
                background: #2a2e39;
                padding: 15px 20px;
                margin: 0;
                color: #d1d4dc;
                font-size: 16px;
                font-weight: 600;
            }}
            
            .info-table .content {{
                padding: 20px;
            }}
            
            .info-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #2a2e39;
            }}
            
            .info-row:last-child {{
                border-bottom: none;
            }}
            
            .info-label {{
                color: #787b86;
                font-weight: 500;
            }}
            
            .info-value {{
                color: #d1d4dc;
                font-weight: 600;
            }}
            
            .trades-table {{
                background: #1e222d;
                border-radius: 8px;
                border: 1px solid #2a2e39;
                overflow: hidden;
                margin-top: 20px;
            }}
            
            .trades-table table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            .trades-table th {{
                background: #2a2e39;
                padding: 15px;
                text-align: left;
                color: #d1d4dc;
                font-weight: 600;
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .trades-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #2a2e39;
                color: #d1d4dc;
            }}
            
            .trades-table tr:last-child td {{
                border-bottom: none;
            }}
            
            .trades-table tr:hover {{
                background: #2a2e39;
            }}
            
            .config-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            
            .section-title {{
                color: #d1d4dc;
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 2px solid #2a2e39;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{strategy_name.title()} Strategy Backtest</h1>
                <div class="subtitle">{strategy_description} • Generated {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
            </div>
            
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="showTab('overview')">Overview</div>
                <div class="nav-tab" onclick="showTab('charts')">Charts</div>
                <div class="nav-tab" onclick="showTab('performance')">Performance</div>
                <div class="nav-tab" onclick="showTab('trades')">Trades Analysis</div>
                <div class="nav-tab" onclick="showTab('tradeslist')">List of Trades</div>
                <div class="nav-tab" onclick="showTab('risk')">Risk/Performance Ratios</div>
                <div class="nav-tab" onclick="showTab('config')">Configuration</div>
            </div>
            
            <div id="overview" class="tab-content active">
                <div class="overview-grid">
                    <div class="performance-summary">
                        <h3>Performance Summary</h3>
                        <div class="key-metrics">
                            <div class="metric">
                                <div class="metric-label">Total P&L</div>
                                <div class="metric-value {'positive' if net_profit >= 0 else 'negative'}">${net_profit:,.2f}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Total Return</div>
                                <div class="metric-value {'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Max Drawdown</div>
                                <div class="metric-value negative">{max_dd:.2f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Profit Factor</div>
                                <div class="metric-value">{profit_factor_display}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-table">
                        <h3>Backtest Information</h3>
                        <div class="content">
                            <div class="info-row">
                                <span class="info-label">Strategy</span>
                                <span class="info-value">{strategy_name.title()}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Initial Capital</span>
                                <span class="info-value">${initial_value:,.2f}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Final Value</span>
                                <span class="info-value">${final_value:,.2f}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Date Range</span>
                                <span class="info-value">{global_config.get('fromdate', 'N/A')} to {global_config.get('todate', 'N/A')}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Position Size</span>
                                <span class="info-value">{global_config.get('size', 1)}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Trades</div>
                        <div class="value">{total_trades}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Profitable Trades</div>
                        <div class="value positive">{winning_trades} ({win_rate:.1f}%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Gross Profit</div>
                        <div class="value positive">${gross_profit:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Gross Loss</div>
                        <div class="value negative">${gross_loss:.2f}</div>
                    </div>
                </div>
            </div>
            
            <div id="charts" class="tab-content">
                <h2 class="section-title">Interactive Charts</h2>
                <div id="plotly-chart-container">
                    {chart_html}
                </div>
            </div>
            
            <div id="performance" class="tab-content">
                <h2 class="section-title">Performance Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Net Profit</div>
                        <div class="value {'positive' if net_profit >= 0 else 'negative'}">${net_profit:,.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Gross Profit</div>
                        <div class="value positive">${gross_profit:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Gross Loss</div>
                        <div class="value negative">${gross_loss:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Equity Run-up</div>
                        <div class="value">N/A</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Equity Drawdown</div>
                        <div class="value negative">{max_dd:.2f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Buy & Hold Return</div>
                        <div class="value">N/A</div>
                    </div>
                </div>
            </div>
            
            <div id="trades" class="tab-content">
                <h2 class="section-title">Trade Analysis</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Trades</div>
                        <div class="value">{total_trades}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Winning Trades</div>
                        <div class="value positive">{winning_trades}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Losing Trades</div>
                        <div class="value negative">{losing_trades}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Percent Profitable</div>
                        <div class="value">{win_rate:.1f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg P&L</div>
                        <div class="value">${(net_profit/total_trades if total_trades > 0 else 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Winning Trade</div>
                        <div class="value positive">${avg_win:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Losing Trade</div>
                        <div class="value negative">${avg_loss:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Ratio Avg Win / Avg Loss</div>
                        <div class="value">{(abs(avg_win/avg_loss) if avg_loss != 0 else 0):.3f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Largest Winning Trade</div>
                        <div class="value positive">${trade_analyzer.get('won', {}).get('pnl', {}).get('max', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Largest Losing Trade</div>
                        <div class="value negative">${trade_analyzer.get('lost', {}).get('pnl', {}).get('max', 0):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Consecutive Winners</div>
                        <div class="value">{max_win_streak}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Consecutive Losers</div>
                        <div class="value">{max_loss_streak}</div>
                    </div>
                </div>
            </div>
            
            <div id="tradeslist" class="tab-content">
                <h2 class="section-title">Individual Trades</h2>
                <div class="trades-table">
                    <table>
                        <thead>
                            <tr>
                                <th>Trade #</th>
                                <th>Type</th>
                                <th>Entry Date</th>
                                <th>Entry Price</th>
                                <th>Exit Date</th>
                                <th>Exit Price</th>
                                <th>Size</th>
                                <th>P&L</th>
                                <th>P&L %</th>
                                <th>Duration</th>
                            </tr>
                        </thead>
                        <tbody>
                            {trades_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="risk" class="tab-content">
                <h2 class="section-title">Risk & Performance Ratios</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Profit Factor</div>
                        <div class="value">{profit_factor_display}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value">{sharpe.get('sharperatio', 0):.3f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">SQN (System Quality Number)</div>
                        <div class="value">{sqn.get('sqn', 0):.3f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">VWR (Variability-Weighted Return)</div>
                        <div class="value">{vwr.get('vwr', 0):.3f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Drawdown</div>
                        <div class="value negative">{max_dd:.2f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Recovery Factor</div>
                        <div class="value">{(net_profit / abs(max_dd_pct) if max_dd_pct != 0 else 0):.2f}</div>
                    </div>
                </div>
            </div>
            
            <div id="config" class="tab-content">
                <h2 class="section-title">Strategy Configuration</h2>
                <div class="config-grid">
                    <div class="info-table">
                        <h3>Global Settings</h3>
                        <div class="content">
                            <div class="info-row">
                                <span class="info-label">Initial Cash</span>
                                <span class="info-value">${global_config.get('cash', 100000.0):,.2f}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Position Size</span>
                                <span class="info-value">{global_config.get('size', 1)}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">From Date</span>
                                <span class="info-value">{global_config.get('fromdate', 'N/A')}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">To Date</span>
                                <span class="info-value">{global_config.get('todate', 'N/A')}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-table">
                        <h3>Strategy Parameters</h3>
                        <div class="content">'''
    
        # Add strategy parameters
        strategy_params = strategy_config.get('parameters', {})
        for param, value in strategy_params.items():
            html_content += f'''
                                <div class="info-row">
                                    <span class="info-label">{param.replace('_', ' ').title()}</span>
                                    <span class="info-value">{value}</span>
                                </div>'''
        
        html_content += f'''
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            function showTab(tabName) {{
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => {{
                    content.classList.remove('active');
                }});
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.nav-tab');
                tabs.forEach(tab => {{
                    tab.classList.remove('active');
                }});
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                
                // Add active class to clicked tab
                event.target.classList.add('active');
            }}
        </script>
    </body>
    </html>
    '''
        
        print("DEBUG: Writing HTML file")
        with open(report_dir / "backtest_report.html", "w") as f:
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
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    
    trades_data = []
    if 'total' in trade_analyzer and trade_analyzer['total']['total'] > 0:
        trades_data.append({
            'Note': 'Individual trade details require custom tracking in strategy',
            'Total_Trades': trade_analyzer.get('total', {}).get('total', 0),
            'Winning_Trades': trade_analyzer.get('won', {}).get('total', 0),
            'Losing_Trades': trade_analyzer.get('lost', {}).get('total', 0)
        })
    
    df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame({'Note': ['No trades executed']})
    df.to_csv(report_dir / "trades.csv", index=False)

def generate_json_stats(strategy, cerebro, config, report_dir, strategy_name):
    """Generate JSON file with all statistics"""
    
    trade_analyzer = strategy.analyzers.trades.get_analysis()
    sharpe = strategy.analyzers.sharpe.get_analysis()
    drawdown = strategy.analyzers.drawdown.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    sqn = strategy.analyzers.sqn.get_analysis()
    vwr = strategy.analyzers.vwr.get_analysis()
    
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
            'total_return_pct': (cerebro.broker.getvalue() - global_config.get('cash', 100000.0)) / global_config.get('cash', 100000.0) * 100,
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
        data_feed = bt.feeds.PandasData(dataname=df_data)
        cerebro.adddata(data_feed)
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
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(global_config.get('cash', 100000.0))
    
    # Set timezone from config
    backtrader_config = config.get('backtrader', {})
    timezone = backtrader_config.get('timezone', 'UTC')
    cerebro.addtz(timezone)

    # Setup data feeds
    data_feeds = setup_data_feeds(cerebro, strategy_class, df_data, config)

    # Add observers
    cerebro.addobserver(bt.observers.BuySell, barplot=True, bardist=0.001)
    cerebro.addobserver(bt.observers.Trades)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn')

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
    
    print(f"\\nRunning {args.strategy} strategy...")
    print(f"Strategy: {strategy_class.get_description()}")
    print(f"Parameters: {strategy_params}")
    
    # Store the merged config for reporting
    config['global'] = global_config
    config['strategies'][args.strategy]['parameters'] = strategy_params
    
    try:
        results = cerebro.run()
        if not global_config.get('quiet', False):
            print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    except Exception as e:
        print(f"Backtest failed with error: {e}")
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
                f.write(basic_report)
                
            print(f"\\nBasic report generated in: {report_dir}")
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
