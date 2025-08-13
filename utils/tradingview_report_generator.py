#!/usr/bin/env python3
"""
TradingView-style HTML Report Generator for Backtrader
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd

def generate_tradingview_report(strategy, cerebro, config, report_dir, strategy_name, data_df=None):
    """Generate a TradingView-style HTML report"""
    
    print("DEBUG: Starting TradingView report generation...")
    
    # Extract strategy data
    strategy_data = extract_strategy_data(strategy, cerebro, config, strategy_name, data_df)
    
    # Generate the HTML content
    html_content = create_tradingview_html(strategy_data)
    
    # Save the report
    report_path = report_dir / "tradingview_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"TradingView-style report generated: {report_path}")
    return report_path

def extract_strategy_data(strategy, cerebro, config, strategy_name, data_df=None):
    """Extract all necessary data from the strategy and cerebro for the report"""
    
    # Basic strategy info
    initial_cash = config.get('global', {}).get('cash', 100000.0)
    final_value = cerebro.broker.getvalue()
    net_profit = final_value - initial_cash
    total_return = (net_profit / initial_cash * 100) if initial_cash > 0 else 0
    
    # Get trade data
    trades_data = extract_trades_data(strategy)
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(trades_data, initial_cash, final_value)
    
    # Get date range
    fromdate = config.get('global', {}).get('fromdate', 'Unknown')
    todate = config.get('global', {}).get('todate', 'Unknown')
    
    # Get strategy parameters
    strategy_params = config.get('strategies', {}).get(strategy_name, {}).get('parameters', {})
    
    # Prepare chart data
    chart_data = prepare_chart_data(data_df, trades_data, initial_cash)
    
    return {
        'strategy_name': strategy_name,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'net_profit': net_profit,
        'total_return': total_return,
        'fromdate': fromdate,
        'todate': todate,
        'strategy_params': strategy_params,
        'trades_data': trades_data,
        'performance_metrics': performance_metrics,
        'chart_data': chart_data,
        'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def prepare_chart_data(data_df, trades_data, initial_cash):
    """Prepare data for the interactive chart"""
    if data_df is None or data_df.empty:
        print("DEBUG: No data available for chart")
        return None
    
    print(f"DEBUG: Preparing chart data from {len(data_df)} data points")
    
    # Convert DataFrame to chart format
    chart_data = {
        'labels': [],
        'datasets': []
    }
    
    # Prepare price data
    if len(data_df) > 0:
        # Sample data for chart (limit to reasonable size for performance)
        sample_size = min(1000, len(data_df))
        step = len(data_df) // sample_size if len(data_df) > sample_size else 1
        
        sampled_df = data_df.iloc[::step].copy()
        
        # Format dates for chart labels
        chart_data['labels'] = [d.strftime('%Y-%m-%d') for d in sampled_df.index]
        
        # Create candlestick dataset
        candlestick_data = []
        for _, row in sampled_df.iterrows():
            candlestick_data.append({
                'x': row.name.strftime('%Y-%m-%d'),
                'o': float(row['open']),
                'h': float(row['high']),
                'l': float(row['low']),
                'c': float(row['close'])
            })
        
        chart_data['candlestick'] = candlestick_data
        
        # Create equity curve data
        equity_data = []
        cumulative_pnl = 0
        
        # Add initial point
        if len(sampled_df) > 0:
            equity_data.append({
                'x': sampled_df.index[0].strftime('%Y-%m-%d'),
                'y': initial_cash
            })
        
        # Add trade points
        for trade in trades_data:
            if trade['entry_date'] != 'Unknown':
                cumulative_pnl += trade['pnl']
                equity_data.append({
                    'x': trade['entry_date'],
                    'y': initial_cash + cumulative_pnl
                })
        
        # Add final point
        if len(sampled_df) > 0:
            equity_data.append({
                'x': sampled_df.index[-1].strftime('%Y-%m-%d'),
                'y': initial_cash + cumulative_pnl
            })
        
        chart_data['equity'] = equity_data
        
        # Prepare trade markers
        trade_markers = []
        for trade in trades_data:
            if trade['entry_date'] != 'Unknown':
                trade_markers.append({
                    'x': trade['entry_date'],
                    'y': float(trade['entry_price']),
                    'type': trade['type'],
                    'signal': trade['signal']
                })
        
        chart_data['trades'] = trade_markers
    
    print(f"DEBUG: Chart data prepared with {len(chart_data.get('candlestick', []))} price points")
    return chart_data

def extract_trades_data(strategy):
    """Extract trade information from the strategy"""
    trades = []
    
    # Try to get trades from custom tracking
    if hasattr(strategy, '_custom_trades') and strategy._custom_trades:
        print(f"DEBUG: Found {len(strategy._custom_trades)} custom trades")
        for i, trade in enumerate(strategy._custom_trades, 1):
            trades.append({
                'trade_number': i,
                'type': trade.get('type', 'Unknown'),
                'entry_date': trade.get('entry_date', 'Unknown'),
                'exit_date': trade.get('exit_date', 'Unknown'),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'position_size': trade.get('size', 1),
                'pnl': trade.get('pnl', 0),
                'pnl_percent': trade.get('pnl_percent', 0),
                'signal': trade.get('signal', 'Unknown'),
                'status': trade.get('status', 'Closed')
            })
    else:
        print("DEBUG: No custom trades found")
    
    return trades

def calculate_performance_metrics(trades_data, initial_cash, final_value):
    """Calculate comprehensive performance metrics"""
    
    if not trades_data:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'gross_profit': 0,
            'gross_loss': 0
        }
    
    total_trades = len(trades_data)
    winning_trades = len([t for t in trades_data if t['pnl'] > 0])
    losing_trades = len([t for t in trades_data if t['pnl'] < 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    winning_pnls = [t['pnl'] for t in trades_data if t['pnl'] > 0]
    losing_pnls = [t['pnl'] for t in trades_data if t['pnl'] < 0]
    
    avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    
    gross_profit = sum(winning_pnls)
    gross_loss = abs(sum(losing_pnls))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate max drawdown (simplified)
    cumulative_pnl = 0
    max_drawdown = 0
    peak_value = initial_cash
    
    for trade in trades_data:
        cumulative_pnl += trade['pnl']
        current_value = initial_cash + cumulative_pnl
        
        if current_value > peak_value:
            peak_value = current_value
        
        drawdown = (peak_value - current_value) / peak_value * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }

def create_tradingview_html(strategy_data):
    """Create the TradingView-style HTML content with real charts"""
    
    # Extract data
    strategy_name = strategy_data['strategy_name']
    initial_cash = strategy_data['initial_cash']
    final_value = strategy_data['final_value']
    net_profit = strategy_data['net_profit']
    total_return = strategy_data['total_return']
    fromdate = strategy_data['fromdate']
    todate = strategy_data['todate']
    trades_data = strategy_data['trades_data']
    performance_metrics = strategy_data['performance_metrics']
    chart_data = strategy_data['chart_data']
    generated_at = strategy_data['generated_at']
    
    # Calculate additional metrics
    open_trades = len([t for t in trades_data if t['status'] == 'Open'])
    closed_trades = len([t for t in trades_data if t['status'] == 'Closed'])
    
    # Determine if net profit is positive or negative
    net_profit_class = "positive" if net_profit >= 0 else "negative"
    net_profit_sign = "+" if net_profit >= 0 else ""
    
    # Read the template HTML file
    template_path = Path("tradingview_style_report.html")
    if template_path.exists():
        print(f"DEBUG: Template file found at {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            html_template = f.read()
        print(f"DEBUG: Template loaded, size: {len(html_template)} characters")
        
        # Prepare data for template update
        template_data = {
            'pageTitle': f"{strategy_name.title()} Strategy Report - TradingView Style",
            'chartTitle': f"{strategy_name.title()} Strategy - Backtest Results",
            'chartSubtitle': f"Initial: ${initial_cash:,.2f} | Final: ${final_value:,.2f} | P&L: {net_profit_sign}${net_profit:,.2f} ({net_profit_sign}{total_return:.2f}%)",
            'reportTitle': f"{strategy_name.title()} Strategy Report",
            'reportSubtitle': f"{fromdate} - {todate} 📅",
            'indicators': {
                'totalTrades': str(performance_metrics['total_trades']),
                'winRate': f"{performance_metrics['win_rate']:.1f}%",
                'profitFactor': f"{performance_metrics['profit_factor']:.2f}",
                'maxDD': f"{performance_metrics['max_drawdown']:.2f}%",
                'netPNL': f"{net_profit_sign}${net_profit:,.2f}"
            },
            'overview': {
                'totalPNL': f"{net_profit_sign}${net_profit:,.2f}",
                'totalPNLPct': f"{net_profit_sign}{total_return:.2f}%",
                'maxDD': f"{performance_metrics['max_drawdown']:.2f}%",
                'maxDDPct': f"{performance_metrics['max_drawdown']:.2f}%",
                'totalTrades': str(performance_metrics['total_trades']),
                'tradesStatus': 'Completed',
                'winRate': f"{performance_metrics['win_rate']:.1f}%",
                'winRatio': f"{performance_metrics['winning_trades']}/{performance_metrics['total_trades']}"
            },
            'performanceTable': generate_performance_table_html(performance_metrics, net_profit, net_profit_sign),
            'tradesAnalysisTable': generate_trades_analysis_table_html(performance_metrics),
            'riskRatiosTable': generate_risk_ratios_table_html(performance_metrics),
            'tradesListTable': generate_trades_list_table_html(trades_data),
            'chartData': prepare_chart_data_for_js(chart_data, strategy_name)
        }
        
        # Create JavaScript to update the template
        update_script = create_template_update_script(template_data)
        
        # Add the update script to the template
        html_content = html_template.replace('</body>', f'{update_script}\n</body>')
        
        print("DEBUG: HTML content generated successfully")
        return html_content
    else:
        print(f"DEBUG: Template file not found at {template_path}")
        # Fallback to basic HTML if template doesn't exist
        return create_basic_html(strategy_data)

def generate_performance_table_html(performance_metrics, net_profit, net_profit_sign):
    """Generate HTML for performance table"""
    html = '''
        <div class="table-row header">
            <div class="table-cell">Metric</div>
            <div class="table-cell">All</div>
            <div class="table-cell">Long</div>
            <div class="table-cell">Short</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Net profit</div>
            <div class="table-cell positive">{net_profit_sign}${net_profit:,.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Gross profit</div>
            <div class="table-cell">${gross_profit:,.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Gross loss</div>
            <div class="table-cell">${gross_loss:,.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Max equity drawdown</div>
            <div class="table-cell">${max_dd:,.2f}%</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
    '''.format(
        net_profit_sign=net_profit_sign,
        net_profit=net_profit,
        gross_profit=performance_metrics['gross_profit'],
        gross_loss=performance_metrics['gross_loss'],
        max_dd=performance_metrics['max_drawdown']
    )
    return html

def generate_trades_analysis_table_html(performance_metrics):
    """Generate HTML for trades analysis table"""
    html = '''
        <div class="table-row header">
            <div class="table-cell">Metric</div>
            <div class="table-cell">All</div>
            <div class="table-cell">Long</div>
            <div class="table-cell">Short</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Total trades</div>
            <div class="table-cell">{total_trades}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Winning trades</div>
            <div class="table-cell">{winning_trades}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Losing trades</div>
            <div class="table-cell">{losing_trades}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Percent profitable</div>
            <div class="table-cell">{win_rate:.1f}%</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Avg winning trade</div>
            <div class="table-cell">${avg_win:.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Avg losing trade</div>
            <div class="table-cell">${avg_loss:.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Profit factor</div>
            <div class="table-cell">{profit_factor:.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
    '''.format(
        total_trades=performance_metrics['total_trades'],
        winning_trades=performance_metrics['winning_trades'],
        losing_trades=performance_metrics['losing_trades'],
        win_rate=performance_metrics['win_rate'],
        avg_win=performance_metrics['avg_win'],
        avg_loss=performance_metrics['avg_loss'],
        profit_factor=performance_metrics['profit_factor']
    )
    return html

def generate_risk_ratios_table_html(performance_metrics):
    """Generate HTML for risk ratios table"""
    html = '''
        <div class="table-row header">
            <div class="table-cell">Metric</div>
            <div class="table-cell">All</div>
            <div class="table-cell">Long</div>
            <div class="table-cell">Short</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Profit factor</div>
            <div class="table-cell">{profit_factor:.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Max drawdown</div>
            <div class="table-cell">{max_dd:.2f}%</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
        <div class="table-row">
            <div class="table-cell">Win rate</div>
            <div class="table-cell">{win_rate:.1f}%</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
        </div>
    '''.format(
        profit_factor=performance_metrics['profit_factor'],
        max_dd=performance_metrics['max_drawdown'],
        win_rate=performance_metrics['win_rate']
    )
    return html

def generate_trades_list_table_html(trades_data):
    """Generate HTML for trades list table"""
    if not trades_data:
        return '<div class="table-row"><div class="table-cell" colspan="10">No trades available</div></div>'
    
    html = '''
        <div class="table-row header">
            <div class="table-cell">Trade #</div>
            <div class="table-cell">Type</div>
            <div class="table-cell">Date/Time</div>
            <div class="table-cell">Signal</div>
            <div class="table-cell">Price</div>
            <div class="table-cell">Position size</div>
            <div class="table-cell">P&L</div>
            <div class="table-cell">Run-up</div>
            <div class="table-cell">Drawdown</div>
            <div class="table-cell">Cumulative P&L</div>
        </div>
    '''
    
    cumulative_pnl = 0
    for i, trade in enumerate(trades_data, 1):
        cumulative_pnl += trade['pnl']
        pnl_class = "positive" if trade['pnl'] >= 0 else "negative"
        pnl_sign = "+" if trade['pnl'] >= 0 else ""
        
        html += f'''
        <div class="table-row">
            <div class="table-cell">{i}</div>
            <div class="table-cell"><span class="trade-type {trade['type'].lower()}">{trade['type']}</span></div>
            <div class="table-cell">{trade['entry_date']}</div>
            <div class="table-cell">{trade['signal']}</div>
            <div class="table-cell">${trade['entry_price']:.2f}</div>
            <div class="table-cell">{trade['position_size']}</div>
            <div class="table-cell {pnl_class}">{pnl_sign}${trade['pnl']:.2f}</div>
            <div class="table-cell">-</div>
            <div class="table-cell">-</div>
            <div class="table-cell">${cumulative_pnl:.2f}</div>
        </div>
        '''
    
    return html

def prepare_chart_data_for_js(chart_data, strategy_name):
    """Prepare chart data for JavaScript"""
    if not chart_data:
        return None
    
    # Convert chart data to Chart.js format
    chart_js_data = {
        'labels': chart_data.get('labels', []),
        'datasets': []
    }
    
    # Add price dataset
    if 'candlestick' in chart_data and chart_data['candlestick']:
        close_prices = [d['c'] for d in chart_data['candlestick']]
        dates = [d['x'] for d in chart_data['candlestick']]
        
        chart_js_data['datasets'].append({
            'label': f'{strategy_name} Price',
            'data': close_prices,
            'borderColor': '#2962ff',
            'backgroundColor': 'rgba(41, 98, 255, 0.1)',
            'borderWidth': 2,
            'fill': False,
            'tension': 0.1,
            'pointRadius': 0,
            'pointHoverRadius': 4
        })
        
        chart_js_data['labels'] = dates
    
    # Add trade markers
    if 'trades' in chart_data and chart_data['trades']:
        for trade in chart_data['trades']:
            chart_js_data['datasets'].append({
                'label': f"{trade['type']} Trade",
                'data': [{'x': trade['x'], 'y': trade['y']}],
                'pointBackgroundColor': '#4caf50' if trade['type'] == 'Long' else '#f44336',
                'pointBorderColor': '#4caf50' if trade['type'] == 'Long' else '#f44336',
                'pointRadius': 6,
                'pointHoverRadius': 8,
                'showLine': False,
                'type': 'scatter'
            })
    
    return chart_js_data

def create_template_update_script(template_data):
    """Create JavaScript to update the template with data"""
    return f'''
    <script>
        // Update template data when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            const templateData = {json.dumps(template_data)};
            updateTemplateData(templateData);
        }});
    </script>
    '''

def create_basic_html(strategy_data):
    """Create a basic HTML report if template is not available"""
    strategy_name = strategy_data['strategy_name']
    net_profit = strategy_data['net_profit']
    total_return = strategy_data['total_return']
    performance_metrics = strategy_data['performance_metrics']
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{strategy_name.title()} Strategy Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #131722; color: #d1d4dc; }}
        .header {{ background: #1e222d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .metric {{ background: #1e222d; padding: 15px; margin: 10px 0; border-radius: 8px; }}
        .positive {{ color: #4caf50; }}
        .negative {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{strategy_name.title()} Strategy Report</h1>
        <p>Generated: {strategy_data['generated_at']}</p>
    </div>
    
    <div class="metric">
        <h3>Performance Summary</h3>
        <p>Net Profit: <span class="{'positive' if net_profit >= 0 else 'negative'}">${net_profit:,.2f}</span></p>
        <p>Total Return: <span class="{'positive' if total_return >= 0 else 'negative'}">{total_return:.2f}%</span></p>
        <p>Total Trades: {performance_metrics['total_trades']}</p>
        <p>Win Rate: {performance_metrics['win_rate']:.1f}%</p>
        <p>Max Drawdown: {performance_metrics['max_drawdown']:.2f}%</p>
        <p>Profit Factor: {performance_metrics['profit_factor']:.2f}</p>
    </div>
</body>
</html>"""

def generate_trade_rows_html(trades_data):
    """Generate HTML rows for the trades table"""
    if not trades_data:
        return '<div class="table-row"><div class="table-cell" colspan="10">No trades available</div></div>'
    
    rows_html = ""
    cumulative_pnl = 0
    
    for trade in trades_data:
        cumulative_pnl += trade['pnl']
        pnl_class = "positive" if trade['pnl'] >= 0 else "negative"
        pnl_sign = "+" if trade['pnl'] >= 0 else ""
        
        rows_html += f'''
                        <div class="table-row">
                            <div class="table-cell">{trade['trade_number']}</div>
                            <div class="table-cell"><span class="trade-type {trade['type'].lower()}">{trade['type']}</span></div>
                            <div class="table-cell">{trade['entry_date']}</div>
                            <div class="table-cell">{trade['signal']}</div>
                            <div class="table-cell">${trade['entry_price']:.2f}</div>
                            <div class="table-cell">{trade['position_size']}</div>
                            <div class="table-cell {pnl_class}">{pnl_sign}${trade['pnl']:.2f}</div>
                            <div class="table-cell">-</div>
                            <div class="table-cell">-</div>
                            <div class="table-cell">${cumulative_pnl:.2f}</div>
                        </div>'''
    
    return rows_html
