# VCP AVWAP Breakout Strategy

## Overview

The VCP AVWAP Breakout Strategy is a multi-timeframe breakout strategy that implements **Playbook #1** from the trading playbook. It looks for stocks near 52-week highs that have formed valid VCP (Volatility Contraction Pattern) bases and are breaking out above key levels with above-average volume.

## Strategy Logic

### Entry Conditions (ALL must be met):
1. **52-Week High Proximity**: Stock must be within 8% of its 52-week high
2. **Valid VCP Base**: 6-12 week base with contracting pullbacks and volume dry-up
3. **Pivot Breakout**: Price must break above the VCP base pivot point
4. **AVWAP Above**: Price must be above the anchored VWAP from base start
5. **Volume Confirmation**: Relative volume must be 1.5x or higher vs trailing average

### Exit Conditions:
- **Weekly Trend Exit**: Close below 10-week moving average
- **Partial Profit**: Take 25% profit when price extends 25% above AVWAP
- **Stop Loss**: Initial stop using daily ATR, tightens when AVWAP control is lost
- **AVWAP Loss of Control**: Stop tightens when price falls below AVWAP with volume

## Data Requirements

The strategy requires 3 data feeds in this specific order:
- **Feed 0**: 1-hour data (base timeframe)
- **Feed 1**: Daily data (resampled from 1h)
- **Feed 2**: Weekly data (resampled from 1h)

```yaml
data_requirements:
  base_timeframe: "hourly"
  additional_timeframes: ["daily", "weekly"]
  requires_resampling: true
```

## Key Parameters

### VCP Base Detection
- `vcp_min_weeks`: 6 (minimum base length)
- `vcp_max_weeks`: 12 (maximum base length)
- `vcp_max_depth_pct`: 25.0 (maximum base depth)
- `vcp_min_contractions`: 2 (minimum pullback contractions)
- `zigzag_pct`: 3.0 (swing detection threshold)

### Entry Filters
- `near_52w_threshold`: 0.92 (within 8% of 52w high)
- `breakout_rvol_min`: 1.5 (minimum relative volume)
- `rvol_window_days`: 20 (volume comparison window)

### Risk Management
- `atr_mult_stop`: 1.0 (initial stop multiplier)
- `take_profit_ext_mult`: 0.25 (partial profit trigger)
- `weekly_ma_period`: 10 (trend filter period)

## Usage

### 1. Run with Existing Runner

```bash
python backtrader_runner_yaml.py \
  --config defaults.yaml \
  --symbol NFLX \
  --provider ALPACA \
  --timeframe 1h \
  --strategy vcp_avwap_breakout \
  --log-level DEBUG
```

### 2. Configuration in defaults.yaml

The strategy is already configured in your `defaults.yaml`:

```yaml
strategies:
  vcp_avwap_breakout:
    description: "52w-high + VCP base + AVWAP trigger on 1h with daily/weekly context"
    data_requirements:
      base_timeframe: "hourly"
      additional_timeframes: ["daily", "weekly"]
      requires_resampling: true
    parameters:
      # All parameters are set to optimal defaults
      vcp_min_weeks: 6
      vcp_max_weeks: 12
      vcp_max_depth_pct: 25.0
      # ... etc
```

### 3. Strategy Registry

The strategy is automatically registered in `strategies/__init__.py`:

```python
from .vcp_avwap_breakout import VcpAvwapBreakoutStrategy

STRATEGIES = {
    # ... existing strategies
    'vcp_avwap_breakout': VcpAvwapBreakoutStrategy,
}
```

## How It Works

### VCP Base Detection
1. **Swing Analysis**: Uses ZigZag algorithm to identify swing highs/lows
2. **Contraction Check**: Ensures pullbacks get progressively smaller
3. **Volume Analysis**: Confirms volume dry-up across the base period
4. **Depth Validation**: Base depth must be within 25% of the high

### AVWAP Calculation
- **Anchored**: VWAP starts accumulating from the VCP base start date
- **Volume-Weighted**: Uses typical price (H+L+C)/3 × volume
- **Dynamic**: Resets when new VCP bases are detected

### Entry Signal
- **Multi-Condition**: All 5 entry conditions must be satisfied
- **Volume Confirmation**: Relative volume must exceed threshold
- **Price Action**: Must break above both pivot and AVWAP

### Risk Management
- **Initial Stop**: Based on daily ATR for volatility-adjusted stops
- **Dynamic Stops**: Tightens when AVWAP control is lost
- **Partial Profits**: Takes 25% off when price extends significantly
- **Trend Exit**: Full exit on weekly trend breakdown

## Example Trades

### Ideal Setup
1. **Stock**: NFLX near 52-week high
2. **Base**: 8-week VCP with 2 contracting pullbacks
3. **Volume**: 20-day average volume declining
4. **Breakout**: Price breaks above pivot with 2x volume
5. **Entry**: Above AVWAP from base start
6. **Stop**: 1 ATR below entry
7. **Target**: Partial at 25% above AVWAP, full on weekly breakdown

## Customization

### Parameter Tuning
- **VCP Parameters**: Adjust base length and depth requirements
- **Volume Thresholds**: Modify relative volume requirements
- **Risk Management**: Change stop and profit-taking levels
- **Timeframes**: Strategy works with any base timeframe (adjust `bars_per_day`)

### Adding Features
- **Multiple Anchors**: Support for swing low anchoring
- **Sector Filters**: Add sector rotation analysis
- **Market Context**: Include broader market conditions
- **Position Sizing**: Dynamic sizing based on volatility

## Troubleshooting

### Common Issues
1. **No VCP Base Found**: Ensure sufficient historical data (252+ days)
2. **Import Errors**: Check that strategy is in registry
3. **Data Feed Issues**: Verify 3 feeds in correct order
4. **Parameter Errors**: Use defaults.yaml configuration

### Debug Mode
Enable detailed logging:
```bash
--log-level DEBUG
```

This will show:
- VCP base detection details
- Entry condition evaluation
- AVWAP calculations
- Stop management decisions

## Performance Notes

### Expected Behavior
- **Low Frequency**: VCP bases are rare, expect few signals
- **High Quality**: When conditions align, high probability setups
- **Risk-Reward**: Typically 2:1 or better with proper management
- **Drawdown**: Controlled by ATR-based stops and trend exits

### Optimization
- **Backtest Period**: Use at least 2 years of data
- **Warmup**: Strategy needs 252+ days to establish 52w high
- **Resampling**: Ensure proper daily/weekly aggregation
- **Volume Data**: Quality volume data is critical for RVOL calculation

## Integration Notes

The strategy integrates seamlessly with your existing:
- **Custom Tracking Mixin**: Automatic trade and portfolio tracking
- **YAML Configuration**: Parameter management through defaults.yaml
- **Multi-Timeframe Runner**: Automatic data resampling and feed management
- **Reporting System**: Compatible with existing backtest reports

## Support

For questions or issues:
1. Check the debug logs for detailed execution information
2. Verify data feed configuration and resampling
3. Review VCP base detection parameters
4. Ensure sufficient historical data for analysis
