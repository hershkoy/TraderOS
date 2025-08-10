#!/usr/bin/env python3
"""
Create a sample TradingView-style report to demonstrate the new design
"""

from datetime import datetime
from pathlib import Path

def create_sample_report():
    """Create a sample report with mock data"""
    
    # Mock data for demonstration
    strategy_name = "mean_reversion"
    initial_value = 100000.0
    final_value = 105250.0
    net_profit = final_value - initial_value
    total_return = (final_value - initial_value) / initial_value * 100
    
    # Mock trade statistics
    total_trades = 15
    winning_trades = 9
    losing_trades = 6
    win_rate = (winning_trades / total_trades * 100)
    
    gross_profit = 8500.0
    gross_loss = 3250.0
    profit_factor = gross_profit / gross_loss
    
    avg_win = gross_profit / winning_trades
    avg_loss = gross_loss / losing_trades
    
    max_dd = 2.5
    
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
                <div class="subtitle">Simple Mean Reversion Strategy using Bollinger Bands on daily data • Generated {datetime.now().strftime("%B %d, %Y at %H:%M")}</div>
            </div>
            
            <div class="nav-tabs">
                <div class="nav-tab active" onclick="showTab('overview')">Overview</div>
                <div class="nav-tab" onclick="showTab('performance')">Performance</div>
                <div class="nav-tab" onclick="showTab('trades')">Trades Analysis</div>
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
                                <div class="metric-value positive">${net_profit:,.2f}</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Total Return</div>
                                <div class="metric-value positive">{total_return:.2f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Max Drawdown</div>
                                <div class="metric-value negative">{max_dd:.2f}%</div>
                            </div>
                            <div class="metric">
                                <div class="metric-label">Profit Factor</div>
                                <div class="metric-value">{profit_factor:.2f}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-table">
                        <h3>Backtest Information</h3>
                        <div class="content">
                            <div class="info-row">
                                <span class="info-label">Strategy</span>
                                <span class="info-value">Mean Reversion</span>
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
                                <span class="info-value">2024-06-18 to 2024-12-31</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Position Size</span>
                                <span class="info-value">1</span>
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
            
            <div id="performance" class="tab-content">
                <h2 class="section-title">Performance Metrics</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Net Profit</div>
                        <div class="value positive">${net_profit:,.2f}</div>
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
                        <div class="label">Max Equity Drawdown</div>
                        <div class="value negative">{max_dd:.2f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Profit Factor</div>
                        <div class="value">{profit_factor:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Recovery Factor</div>
                        <div class="value">{(net_profit / (max_dd * initial_value / 100)):.2f}</div>
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
                        <div class="value">${(net_profit/total_trades):.2f}</div>
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
                        <div class="value">{(avg_win/avg_loss):.3f}</div>
                    </div>
                </div>
            </div>
            
            <div id="risk" class="tab-content">
                <h2 class="section-title">Risk & Performance Ratios</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Profit Factor</div>
                        <div class="value">{profit_factor:.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Sharpe Ratio</div>
                        <div class="value">1.234</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Max Drawdown</div>
                        <div class="value negative">{max_dd:.2f}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Recovery Factor</div>
                        <div class="value">{(net_profit / (max_dd * initial_value / 100)):.2f}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Calmar Ratio</div>
                        <div class="value">2.10</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Sortino Ratio</div>
                        <div class="value">1.85</div>
                    </div>
                </div>
            </div>
            
            <div id="config" class="tab-content">
                <h2 class="section-title">Strategy Configuration</h2>
                <div class="overview-grid">
                    <div class="info-table">
                        <h3>Global Settings</h3>
                        <div class="content">
                            <div class="info-row">
                                <span class="info-label">Initial Cash</span>
                                <span class="info-value">${initial_value:,.2f}</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Position Size</span>
                                <span class="info-value">1</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">From Date</span>
                                <span class="info-value">2024-06-18</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">To Date</span>
                                <span class="info-value">2024-12-31</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="info-table">
                        <h3>Strategy Parameters</h3>
                        <div class="content">
                            <div class="info-row">
                                <span class="info-label">Period</span>
                                <span class="info-value">10</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Devfactor</span>
                                <span class="info-value">1.2</span>
                            </div>
                            <div class="info-row">
                                <span class="info-label">Commission</span>
                                <span class="info-value">0.001</span>
                            </div>
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
    
    # Create reports directory
    report_dir = Path("reports") / "sample_tradingview_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Write the HTML file
    with open(report_dir / "sample_backtest_report.html", "w") as f:
        f.write(html_content)
    
    print(f"Sample TradingView-style report created at: {report_dir / 'sample_backtest_report.html'}")
    print("Open this file in your browser to see the new tabbed interface!")

if __name__ == "__main__":
    create_sample_report()
