# TradingView-Style Report Usage Guide

## Quick Start

### 1. Test the Report Generation
```bash
python test_tradingview_report.py
```
This will generate a sample TradingView-style report to verify everything is working.

### 2. Run a Real Backtest
```bash
python backtrader_runner_yaml.py ^
  --config default.yaml ^
  --parquet "data\ALPACA\NFLX\1h\nflx_1h.parquet" ^
  --strategy mean_reversion ^
  --log-level DEBUG
```

### 3. View the Reports
After running a backtest, check the `reports/` folder for:
- `tradingview_report.html` - Professional TradingView-style interface
- `backtest_report.html` - Detailed backtesting analysis
- `trades.csv` - Trade data export
- `stats.json` - Performance statistics

## Report Features

### TradingView-Style Interface
The `tradingview_report.html` includes:

1. **Chart Header Section**
   - Strategy name and performance summary
   - Timeframe controls (1D, 5D, 1M, 3M, 6M, YTD, 1Y, All)
   - Toolbar with settings, notifications, screenshot, export

2. **Chart Area**
   - Placeholder for interactive candlestick chart
   - Real-time indicator values
   - Professional dark theme

3. **Five Report Tabs**
   - **Overview**: Key metrics cards and equity curve
   - **Performance**: Detailed performance metrics table
   - **Trades analysis**: Trade statistics breakdown
   - **Risk/performance ratios**: Risk metrics and ratios
   - **List of trades**: Individual trade details with entry/exit information

### Key Metrics Displayed
- Total P&L and return percentage
- Maximum drawdown
- Total trades and win rate
- Profit factor and risk ratios
- Individual trade entries and exits
- Cumulative P&L tracking

## Integration

The TradingView-style reporting is automatically integrated into your existing workflow:

1. **No changes needed** to your existing backtest commands
2. **Automatic generation** when you run any strategy
3. **Fallback support** if the report generator is not available
4. **Error handling** to prevent backtest failures

## Customization

### Modify Report Template
Edit `tradingview_style_report.html` to customize:
- Colors and styling
- Layout and sections
- Additional metrics
- Chart placeholders

### Add Custom Metrics
Modify `utils/tradingview_report_generator.py` to include:
- Additional performance ratios
- Custom trade analysis
- Strategy-specific metrics
- Real chart integration

## Troubleshooting

### Report Not Generated
- Check that `utils/tradingview_report_generator.py` is in the utils directory
- Verify Python imports are working
- Check console output for error messages

### Missing Trade Data
- Ensure your strategy has `_custom_trades` attribute
- Verify trade data format matches expected structure
- Check that trades are being recorded during backtest

### Template Issues
- Verify `tradingview_style_report.html` exists
- Check HTML syntax and structure
- Ensure all placeholders are properly formatted

## Next Steps

1. **Add Real Charts**: Integrate Chart.js or TradingView Lightweight Charts
2. **Real-time Data**: Connect to live data feeds
3. **Interactive Features**: Add zoom, pan, and technical indicators
4. **Export Options**: PDF, PNG, and CSV export functionality
5. **Custom Indicators**: Add strategy-specific visualizations
