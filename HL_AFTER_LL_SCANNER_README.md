# HL After LL Scanner

A Python scanner that detects **LL → HH → HL** patterns in weekly stock data, designed to run on Monday after market open to identify potential reversal setups.

## Overview

This scanner implements the market structure analysis pattern where:
- **LL** = Lower Low (confirms downtrend)
- **HH** = Higher High (breaks structure, potential reversal signal)  
- **HL** = Higher Low (confirms new uptrend)

The scanner also supports extended patterns like **LL → HH → HH → HL**.

## Key Features

- **Weekly Timeframe**: Resamples daily data to weekly bars ending on Friday
- **Confirmation Logic**: Uses 1 left bar + 2 right bar confirmation (matching Pine Script)
- **Monday Check**: Verifies current price is above HL to ensure pattern is still valid
- **TimescaleDB Integration**: Loads data from your existing database
- **Flexible Configuration**: YAML-based configuration system
- **CSV Export**: Saves results for further analysis

## Quick Start

### 1. Test the Scanner
```bash
# Test with sample data
python test_hl_after_ll_scanner.py
```

### 2. Run on Your Universe
```bash
# Run with default configuration
python hl_after_ll_scanner_runner.py

# Or use the batch file (Windows)
run_hl_scanner.bat
```

### 3. Run on Specific Symbols
```bash
# Scan specific symbols
python hl_after_ll_scanner_runner.py --symbols AAPL MSFT GOOGL
```

## Configuration

Edit `hl_after_ll_scanner_config.yaml` to customize:

```yaml
scanner:
  left_bars: 1          # 1 left bar for confirmation
  right_bars: 2         # 2 right bars for confirmation
  patterns:
    - "LL→HH→HL"        # Basic pattern
    - "LL→HH→HH→HL"     # Extended pattern

universe:
  source: "database"    # Use ticker universe from database
  # OR
  source: "manual"      # Use specific symbols
  symbols: ["AAPL", "MSFT", "GOOGL"]

output:
  save_csv: true       # Save results to CSV
  show_details: true   # Show detailed results
```

## How It Works

### 1. Data Processing
- Loads daily data from TimescaleDB
- Resamples to weekly bars (Friday close)
- Applies pivot detection with 1 left + 2 right bar confirmation

### 2. Pattern Detection
- Identifies swing highs and lows using pivot logic
- Classifies each swing as HH, HL, LH, or LL
- Searches for LL → HH → HL sequences

### 3. Monday Check
- Verifies current price is above the HL level
- Ensures pattern hasn't been broken by new week's price action

### 4. Results
- Displays matches with dates and prices
- Saves detailed CSV with all pattern information
- Shows Monday check status for each match

## Example Output

```
✅ Found 3 symbols with LL → HH → HL patterns:
================================================================================

1. AAPL
   LL: 2024-01-15 at $150.25
   HH: 2024-02-12 at $185.30
   HL: 2024-03-05 at $165.40
   Current Price: $170.25
   ✅ Monday Check: Current price ($170.25) >= HL ($165.40)

2. MSFT
   LL: 2024-01-22 at $380.15
   HH1: 2024-02-19 at $415.80
   HH2: 2024-03-12 at $425.50
   HL: 2024-03-26 at $395.20
   Current Price: $400.10
   ✅ Monday Check: Current price ($400.10) >= HL ($395.20)
```

## Files Structure

```
├── hl_after_ll_scanner.py          # Core scanner logic
├── hl_after_ll_scanner_runner.py   # Main runner with config
├── test_hl_after_ll_scanner.py     # Test script
├── hl_after_ll_scanner_config.yaml # Configuration file
├── run_hl_scanner.bat              # Windows batch file
└── reports/                        # Generated reports
    └── hl_after_ll_scan_YYYYMMDD_HHMMSS.csv
```

## Requirements

- Python 3.8+
- pandas, numpy
- TimescaleDB connection
- Ticker universe data

## Usage Tips

### Best Practices
1. **Run on Monday**: Best timing after weekend gap analysis
2. **Weekly Data**: Use daily data that gets resampled to weekly
3. **Volume Confirmation**: Consider adding volume analysis to patterns
4. **Multi-timeframe**: Use with higher timeframe trend context

### Pattern Interpretation
- **LL → HH → HL**: Classic reversal pattern
- **LL → HH → HH → HL**: Stronger reversal with double confirmation
- **Monday Check Pass**: Pattern is still valid
- **Monday Check Fail**: Pattern may be broken, needs re-evaluation

### Integration
- Use with your existing backtrader strategies
- Combine with other technical indicators
- Export to your trading platform for further analysis

## Troubleshooting

### Common Issues
1. **No data loaded**: Check TimescaleDB connection and ticker universe
2. **No patterns found**: Try different time periods or adjust pivot parameters
3. **Database errors**: Verify database credentials and network connection

### Debug Mode
```bash
# Enable debug logging
python hl_after_ll_scanner_runner.py --config hl_after_ll_scanner_config.yaml
```

Edit config file to set logging level to DEBUG for detailed output.

## Advanced Usage

### Custom Patterns
You can extend the scanner to detect other patterns by modifying the `find_ll_hh_hl` function in `hl_after_ll_scanner.py`.

### Performance Optimization
- Use symbol filtering to reduce scan time
- Implement parallel processing for large universes
- Cache frequently accessed data

### Integration with Trading
- Set up automated Monday morning scans
- Export results to your trading platform
- Combine with other screening criteria
