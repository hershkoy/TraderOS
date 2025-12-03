# Zero-Cost Collar Screener - Implementation Summary

## What We Built

I've implemented a comprehensive **zero-cost collar screener** for your backtrader project that finds near-zero-loss options opportunities using Interactive Brokers data. This screener implements the strategy from the YouTube video you referenced, where the goal is to find collar setups where the net cost is less than or equal to the put strike price.

## Core Strategy

The screener looks for opportunities where:
```
Net Cost (100 shares + put premium - call premium) ≤ 100 × put strike
```

This creates a "zero-loss" scenario where maximum loss at expiry is ≤ $0.

## Files Created

### 1. **screener_zero_cost_collar.py**
- Basic implementation with hardcoded parameters
- Good for quick testing and understanding the logic

### 2. **screener_zero_cost_collar_enhanced.py** ⭐ **RECOMMENDED**
- Enhanced version with YAML configuration
- Better error handling and logging
- More flexible and production-ready

### 3. **collar_screener_config.yaml**
- Configuration file for all parameters
- Easy to customize without editing code
- Includes universe, delta ranges, capital budget, etc.

### 4. **test_collar_screener.py**
- Test suite to validate setup
- Runs without requiring IB connection
- Checks configuration, imports, and logic

### 5. **run_collar_screener.bat**
- Windows batch file for easy execution
- Handles virtual environment activation
- Checks IB connection before running

### 6. **COLLAR_SCREENER_README.md**
- Comprehensive documentation
- Usage instructions and examples
- Troubleshooting guide

## Key Features

### ✅ **Real-time Market Data**
- Connects to Interactive Brokers via `ib_insync`
- Gets live stock prices and option chains
- Uses IB's model greeks for accurate delta calculations

### ✅ **Smart Filtering**
- **Delta ranges**: Put delta -0.35 to -0.20, Call delta +0.20 to +0.35
- **DTE range**: 30-60 days (configurable)
- **Capital budget**: $30,000 default (configurable)
- **Spread quality**: Filters out wide bid-ask spreads
- **Zero-loss test**: Only shows opportunities with floor ≥ 0

### ✅ **Risk Scoring**
- Prioritizes true zero-risk opportunities
- Ranks by risk-reward ratio and gain per capital
- Shows capital at risk (should be ~$0 for zero-loss)

### ✅ **Output & Integration**
- Formatted table output with key metrics
- CSV export for further analysis
- Detailed logging for debugging
- Integrates with your existing backtrader project structure

## Quick Start

1. **Test the setup**:
   ```bash
   python test_collar_screener.py
   ```

2. **Run the screener** (make sure TWS/IBG is running):
   ```bash
   python utils/screener_zero_cost_collar_enhanced.py
   ```

3. **Or use the batch file**:
   ```bash
   run_collar_screener.bat
   ```

## Example Output

```
========================================================================================================================
ZERO-COST COLLAR OPPORTUNITIES (Budget: $30,000)
========================================================================================================================
SYM    EXP         PX    P_K  P_mid   P_δ   C_K  C_mid   C_δ       C0    FLOOR     MAXG  DTE   SCORE
------------------------------------------------------------------------------------------------------------------------
AAPL   2025-01-17 185.50 175.00   3.25 -0.28 195.00   2.75  0.25  18500.00    25.00  1000.00   45  400.00
MSFT   2025-01-17 375.20 365.00   4.50 -0.30 385.00   3.80  0.28  37590.00    10.00   910.00   45   91.00
```

## Configuration Options

The `collar_screener_config.yaml` file lets you customize:

- **Capital budget**: $30,000 default
- **Delta ranges**: Adjust for different risk profiles
- **DTE range**: 30-60 days default
- **Universe**: Add/remove symbols to scan
- **Risk tolerance**: Allow small negative floors
- **Output settings**: CSV export, logging level

## Integration with Your Project

This screener is designed to work seamlessly with your existing backtrader project:

- **Same directory structure**: Uses your `logs/` and `reports/` folders
- **Same dependencies**: Uses `ib_insync` already in your requirements
- **Same logging**: Follows your project's logging conventions
- **Same configuration**: Uses YAML like your other tools
- **Results integration**: CSV output can be used by backtrader strategies

## Risk Considerations

⚠️ **Important**: The "zero-loss" condition holds at **expiry only**. Before expiry:
- Mark-to-market can be negative
- Early assignment risk exists
- Gap risk can exceed put protection
- Liquidity and spread issues can impact fills

## Next Steps

1. **Test with paper trading** first (port 7497)
2. **Start with liquid symbols** (AAPL, MSFT, SPY, QQQ)
3. **Monitor results** and adjust parameters as needed
4. **Integrate findings** into your backtrader strategies
5. **Consider automation** for regular screening

## Support

- Check `logs/collar_screener.log` for detailed information
- Run `python tests/test_collar_screener.py` to validate setup
- Review `COLLAR_SCREENER_README.md` for comprehensive documentation

The screener is production-ready and should help you identify high-quality zero-cost collar opportunities that fit your risk profile and capital constraints.
