# Zero-Cost Collar Screener

A comprehensive screener for finding near-zero-cost collar opportunities using Interactive Brokers data. This screener implements the "zero-loss" collar strategy where the net cost of the position is less than or equal to the put strike price, creating a scenario where maximum loss at expiry is ≤ $0.

## What is a Zero-Cost Collar?

A zero-cost collar is an options strategy that involves:
1. **Buying 100 shares** of a stock
2. **Buying a protective put** (delta ~-0.35 to -0.20) 
3. **Selling a covered call** (delta ~+0.20 to +0.35)

The goal is to find setups where:
```
Net Cost (shares + put - call) ≤ 100 × put strike
```

This creates a "zero-loss" scenario where the maximum loss at expiry is ≤ $0.

## Features

- **Real-time market data** from Interactive Brokers
- **Configurable parameters** via YAML configuration file
- **Comprehensive filtering** by delta ranges, DTE, capital budget, and spread quality
- **Risk scoring** that prioritizes zero-risk opportunities
- **CSV export** for further analysis
- **Detailed logging** for debugging and monitoring
- **Integration** with existing backtrader project structure

## Files

- `screener_zero_cost_collar.py` - Basic screener implementation
- `utils/screening/zero_cost_collar.py` - Enhanced version with YAML config support
- `collar_screener_config.yaml` - Configuration file
- `tests/test_collar_screener.py` - Test suite for validation
- `COLLAR_SCREENER_README.md` - This documentation

## Installation & Setup

### Prerequisites

1. **Interactive Brokers Account**: You need an IB account with market data permissions
2. **TWS or IB Gateway**: Must be running and configured for API access
3. **Python Dependencies**: Already included in your `requirements.txt`

### Configuration

1. **Edit the configuration file** `collar_screener_config.yaml`:

```yaml
# Capital and Budget
capital:
  budget: 30000  # Maximum capital per position (USD)
  min_position_size: 5000  # Minimum position size to consider

# Option Parameters
options:
  min_dte: 30  # Minimum days to expiry
  max_dte: 60  # Maximum days to expiry
  put_delta_range: [-0.35, -0.20]  # Target put delta range
  call_delta_range: [0.20, 0.35]   # Target call delta range

# Universe of symbols to scan
universe:
  - "AAPL"
  - "MSFT"
  - "SPY"
  - "QQQ"
  # Add more symbols as needed
```

2. **Configure IB Connection**:
```yaml
ib_connection:
  host: "127.0.0.1"
  port: 7497  # 7497 for paper trading, 7496 for live
  client_id: 19
  readonly: true  # Set to false for live trading
```

## Usage

### 1. Test the Setup

First, run the test suite to ensure everything is configured correctly:

```bash
python test_collar_screener.py
```

This will validate:
- Configuration file loading
- Required imports
- Directory structure
- Collar calculation logic
- Delta range validation

### 2. Run the Screener

Make sure TWS/IB Gateway is running, then execute:

```bash
# Using the enhanced version (recommended)
python utils/screening/zero_cost_collar.py

# Or using the basic version
python screener_zero_cost_collar.py
```

### 3. Interpret Results

The screener outputs a table with the following columns:

- **SYM**: Stock symbol
- **EXP**: Option expiration date
- **PX**: Current stock price
- **P_K/C_K**: Put/Call strike prices
- **P_mid/C_mid**: Put/Call mid prices
- **P_δ/C_δ**: Put/Call deltas
- **C0**: Net cost of the collar
- **FLOOR**: Loss floor (should be ≥ 0 for zero-loss)
- **MAXG**: Maximum gain if called away
- **DTE**: Days to expiry
- **SCORE**: Risk-reward score (higher is better)

## Configuration Options

### Capital Settings

- `budget`: Maximum capital to use per position
- `min_position_size`: Minimum position size to consider

### Option Parameters

- `min_dte`/`max_dte`: Days to expiry range
- `put_delta_range`: Target put delta range (negative values)
- `call_delta_range`: Target call delta range (positive values)

### Risk Tolerance

- `floor_tolerance`: Allow small negative floor (e.g., -$25)
- `max_spread_percent`: Maximum bid-ask spread as % of mid price

### Universe

Add or remove symbols from the `universe` list. Recommended symbols:
- **Liquid stocks**: AAPL, MSFT, NVDA, AMZN, META, GOOGL
- **ETFs**: SPY, QQQ, IWM, DIA, VTI, VOO

### Output Settings

- `max_results`: Maximum number of results to display
- `save_to_csv`: Enable/disable CSV export
- `csv_filename`: Output file path
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Understanding the Results

### Zero-Loss Condition

A collar meets the zero-loss condition when:
```
100 × put_strike - net_cost ≥ 0
```

This means if the stock drops to the put strike at expiry, you break even or better.

### Scoring System

Results are ranked by:
1. **Risk-Reward Score**: `max_gain / capital_at_risk`
2. **Gain per Capital**: `max_gain / capital_used`
3. **Days to Expiry**: Shorter DTE preferred

### Key Metrics

- **Net Cost (C0)**: Total capital required
- **Loss Floor**: Minimum value at expiry
- **Max Gain**: Maximum profit if called away
- **Capital at Risk**: Amount that could be lost (should be ~$0 for true zero-loss)

## Risk Considerations

⚠️ **Important Risk Disclaimers**:

1. **"Zero-loss" holds at expiry only** - mark-to-market can be negative before expiry
2. **Early assignment risk** - calls can be exercised early, especially near ex-dividend dates
3. **Liquidity risk** - wide spreads can impact fills
4. **Dividend risk** - high dividend stocks may have early exercise risk
5. **Gap risk** - overnight gaps can exceed put protection

## Advanced Usage

### Custom Delta Ranges

Adjust delta ranges for different risk profiles:

```yaml
# Conservative (more protection, less upside)
put_delta_range: [-0.40, -0.25]
call_delta_range: [0.15, 0.30]

# Aggressive (less protection, more upside)
put_delta_range: [-0.30, -0.15]
call_delta_range: [0.25, 0.40]
```

### Multiple Timeframes

Run the screener with different DTE ranges:

```yaml
# Short-term (30-45 days)
options:
  min_dte: 30
  max_dte: 45

# Medium-term (45-60 days)
options:
  min_dte: 45
  max_dte: 60
```

### Budget Scaling

Adjust for different account sizes:

```yaml
# Small account
capital:
  budget: 10000
  min_position_size: 2000

# Large account
capital:
  budget: 100000
  min_position_size: 10000
```

## Troubleshooting

### Common Issues

1. **Connection Failed**: Ensure TWS/IB Gateway is running and API is enabled
2. **No Data**: Check market data permissions for options
3. **No Results**: Try widening delta ranges or increasing floor tolerance
4. **Slow Performance**: Reduce universe size or increase sleep between requests

### Log Files

Check `logs/collar_screener.log` for detailed error messages and debugging information.

### Performance Tips

1. **Use paper trading** for testing (port 7497)
2. **Limit universe size** to liquid symbols
3. **Adjust sleep times** if hitting rate limits
4. **Use appropriate DTE ranges** for your strategy

## Integration with Backtrader

The screener is designed to integrate with your existing backtrader project:

1. **Results can be used** to identify trading opportunities
2. **CSV output** can be imported into backtrader strategies
3. **Logging** follows your project's conventions
4. **Configuration** uses the same YAML approach as your other tools

## Example Output

```
========================================================================================================================
ZERO-COST COLLAR OPPORTUNITIES (Budget: $30,000)
========================================================================================================================
SYM    EXP         PX    P_K  P_mid   P_δ   C_K  C_mid   C_δ       C0    FLOOR     MAXG  DTE   SCORE
------------------------------------------------------------------------------------------------------------------------
AAPL   2025-01-17 185.50 175.00   3.25 -0.28 195.00   2.75  0.25  18500.00    25.00  1000.00   45  400.00
MSFT   2025-01-17 375.20 365.00   4.50 -0.30 385.00   3.80  0.28  37590.00    10.00   910.00   45   91.00
SPY    2025-01-17  465.80 455.00   2.80 -0.25 475.00   2.20  0.22  46640.00   -40.00   860.00   45   21.50

Found 3 opportunities, showing top 3
Legend: PX=Stock Price, P_K/C_K=Put/Call Strike, P_mid/C_mid=Put/Call Mid Price
        P_δ/C_δ=Put/Call Delta, C0=Net Cost, FLOOR=Loss Floor, MAXG=Max Gain
        DTE=Days to Expiry, SCORE=Risk-Reward Score
```

## Support

For issues or questions:
1. Check the log files in `logs/`
2. Run the test suite: `python tests/test_collar_screener.py`
3. Verify IB connection and market data permissions
4. Review configuration parameters

## License

This screener is part of your backtrader project and follows the same licensing terms.
