# Weekly Big Volume + TTM Squeeze Strategy

A Backtrader strategy implementing the "Rare but Powerful Weekly Setups" approach, combining weekly volume ignition bars with TTM Squeeze momentum confirmation and trend filters.

## Overview

This strategy targets high-probability weekly setups that occur infrequently but offer significant profit potential. It combines three key conditions:

1. **Condition A: Weekly "Huge Volume" Ignition Bar** - Identifies weeks with exceptional volume that often precede major moves
2. **Condition B: Weekly TTM Squeeze Zero-Cross** - Confirms momentum shift from compression to expansion
3. **Condition C: Simple Trend Filter** - Ensures trades align with the broader trend using 10/30-week moving averages

## Strategy Logic

### Condition A: Volume Ignition Bar

A weekly bar qualifies as an "ignition bar" when ALL of the following are true:

- `Volume > vol_mult * SMA(Volume, 52 weeks)` (default: 3.0x average)
- `Body Position >= 0.5` (close in top half of weekly range, indicating buying pressure)
- `Close > 30-week MA` (trend alignment)
- `Volume >= 90% of highest volume in last 52 weeks` (exceptional volume)

This creates a "watchlist" of symbols showing unusual weekly activity.

### Condition B: TTM Squeeze Zero-Cross

After an ignition bar, the strategy waits for TTM Squeeze confirmation:

- **Squeeze-On Detection**: There was at least one week with squeeze "on" (compression) in the last `squeeze_lookback` weeks (default: 10)
- **Zero-Cross**: Momentum histogram crosses from negative to positive:
  - `momentum[t-1] <= 0` and `momentum[t] > 0`
- **Steepness**: Momentum slope exceeds minimum threshold (default: 0.0)

The momentum histogram uses linear regression slope over `mom_period` weeks (default: 12), matching the TTM Squeeze implementation in `utils/squeeze_scanner.py`.

### Condition C: Trend Filter

On the same weekly bar where TTM confirms:

- `Close > 30-week SMA`
- `30-week SMA is rising` (current > previous)
- `Close < 30-week SMA * 1.25` (not too extended, default: 25% max)

Additionally, the TTM confirmation must occur within `max_delay_weeks` (default: 26 weeks) of the ignition bar.

### Entry Rule

When **all three conditions** are met on a weekly bar:

- Generate a **long entry signal**
- Enter at the **next daily bar's open** (approximation using current close)
- Position size based on **risk per trade** (default: 1% of equity)

### Stop Loss

Initial stop is calculated as the minimum of:

1. **Structure-based**: `setup_low * (1 - buffer_pct)` where `setup_low` is the low of the signal week
2. **Volatility-based**: `entry_price - atr_mult * ATR(14 weeks)`

Default parameters:
- `buffer_pct = 0.01` (1% buffer)
- `atr_mult = 1.5`
- `atr_period = 14` weeks

This typically results in 4-10% risk per trade, depending on volatility.

### Exit Rules

The strategy uses trend-following exits (no fixed take-profit):

1. **Fast Exit (10-week MA)**:
   - If weekly close < 10-week MA, exit at next week's open
   - Catches early trend reversals

2. **Slow Exit (30-week MA)**:
   - If weekly close < 30-week MA, exit at next week's open
   - Final trend filter

3. **Stop Loss**:
   - Hard stop order placed at entry (structure or ATR-based)

## Usage

### Basic Usage

```bash
# Run strategy on a single symbol
python backtrader_runner_yaml.py \
    --strategy weekly_bigvol_ttm_squeeze \
    --symbol AAPL \
    --provider ALPACA \
    --timeframe 1d \
    --fromdate 2014-01-01 \
    --todate 2024-12-31
```

### Configuration File

Add to `defaults.yaml`:

```yaml
strategies:
  weekly_bigvol_ttm_squeeze:
    description: "Weekly Big Volume + TTM Squeeze Strategy"
    parameters:
      # Volume ignition
      vol_lookback: 52
      vol_mult: 3.0
      body_pos_min: 0.5
      max_volume_lookback: 52
      
      # TTM Squeeze
      lengthKC: 20
      mom_period: 12
      mom_slope_min: 0.0
      squeeze_lookback: 10
      max_delay_weeks: 26
      
      # Trend filter
      ma10_period: 10
      ma30_period: 30
      max_extended_pct: 0.25
      
      # Risk management
      risk_per_trade: 0.01
      atr_period: 14
      atr_mult: 1.5
      buffer_pct: 0.01
      
      # Misc
      commission: 0.001
      printlog: true
      log_level: INFO
```

### Universe Backtesting

The runner now supports universe backtesting directly:

```bash
# Run on entire universe
python backtrader_runner_yaml.py \
    --strategy weekly_bigvol_ttm_squeeze \
    --universe \
    --provider ALPACA \
    --timeframe 1d \
    --fromdate 2014-01-01 \
    --todate 2024-12-31

# Limit to first 100 symbols
python backtrader_runner_yaml.py \
    --strategy weekly_bigvol_ttm_squeeze \
    --universe \
    --max-symbols 100 \
    --provider ALPACA \
    --timeframe 1d
```

Universe backtesting will:
1. Load symbols from ticker universe database
2. Run the strategy on each symbol individually
3. Aggregate results into a CSV report
4. Display summary statistics

Results are saved to `reports/{strategy}_universe_backtest_{timestamp}/universe_results.csv`

## Parameters

### Volume Ignition Parameters

- `vol_lookback` (default: 52): Weeks for volume SMA calculation
- `vol_mult` (default: 3.0): Volume multiplier threshold (3x average)
- `body_pos_min` (default: 0.5): Minimum body position (0.5 = top half)
- `max_volume_lookback` (default: 52): Lookback for "highest volume" check

### TTM Squeeze Parameters

- `lengthKC` (default: 20): Keltner Channel length (matches squeeze_scanner)
- `mom_period` (default: 12): Momentum period in weeks
- `mom_slope_min` (default: 0.0): Minimum momentum slope threshold
- `squeeze_lookback` (default: 10): Weeks to look back for squeeze-on detection
- `max_delay_weeks` (default: 26): Max weeks between ignition and TTM confirmation

### Trend Filter Parameters

- `ma10_period` (default: 10): 10-week moving average period
- `ma30_period` (default: 30): 30-week moving average period
- `max_extended_pct` (default: 0.25): Maximum extension above 30-week MA (25%)

### Risk Management Parameters

- `risk_per_trade` (default: 0.01): Risk per trade as fraction of equity (1%)
- `atr_period` (default: 14): ATR period in weeks
- `atr_mult` (default: 1.5): ATR multiplier for stop calculation
- `buffer_pct` (default: 0.01): Buffer percentage for structure-based stop (1%)

## Expected Trade Frequency

This strategy targets **rare but powerful** setups:

- **Per symbol**: 1-2 setups per decade (10 years)
- **Across 2000 symbols × 10 years**: ~1000-2000 total trades
- **Trigger rate**: ~0.1% of weekly bars

The low frequency is by design - the strategy waits for high-conviction setups with multiple confirmations.

## Performance Considerations

### Data Requirements

- **Minimum history**: 52+ weeks (1 year) for volume SMA
- **Recommended**: 10+ years of daily data for robust backtesting
- **Timeframe**: Daily bars (resampled to weekly internally)

### Overfitting Prevention

The strategy uses **coarse parameters** to avoid overfitting:

- Round numbers (3.0x volume, 10/30-week MAs, 1.5x ATR)
- Simple logic (no complex conditionals)
- Robust checks (multiple confirmations required)

### Robustness Testing

Recommended parameter ranges for robustness checks:

- `vol_mult`: [2.5, 3.0, 3.5]
- `mom_period`: [10, 12, 14]
- `max_delay_weeks`: [20, 26, 30]

If performance is only good at a single magic combination, that's a red flag.

## Integration with squeeze.py

The strategy uses the TTM Squeeze logic from `utils/scanning/squeeze.py`:

- **Momentum calculation**: Uses linear regression slope (same as `squeeze_val` in squeeze_scanner)
- **Zero-cross detection**: Matches `_find_latest_zero_cross_up` logic
- **Squeeze-on detection**: Simplified version (full implementation would calculate BB/KC)

For standalone scanning (outside Backtrader), use `utils.scanning.squeeze` directly:

```python
from utils.scanning.squeeze import scan_universe, load_from_timescaledb

symbols = ['AAPL', 'MSFT', 'GOOGL']
ohlcv_data = load_from_timescaledb(symbols, timeframe='1d')
results = scan_universe(ohlcv_data, lengthKC=20, confirm_on_close=True)
```

## Reports

The strategy generates comprehensive reports via the standard Backtrader reporting system:

- **TradingView-style HTML report**: Visual analysis with charts
- **Trade CSV**: Individual trade details
- **JSON statistics**: Performance metrics
- **HTML summary**: Key metrics and trade log

Reports are saved to `reports/weekly_bigvol_ttm_squeeze_backtest_YYYYMMDD_HHMMSS/`

## Limitations & Future Enhancements

### Current Limitations

1. **Simplified TTM Squeeze**: Uses momentum slope as proxy for full BB/KC calculation
2. **Single-symbol focus**: Multi-symbol backtesting requires custom runner
3. **No partial profit-taking**: V1 uses full position exits only

### Potential Enhancements

1. **Full TTM Squeeze**: Implement complete Bollinger/Keltner Channel calculation
2. **Partial profit-taking**: Sell half at 3R, trail rest with MA10/30
3. **Multi-symbol runner**: Built-in universe backtesting support
4. **Walk-forward optimization**: Time-split validation (2014-2018 train, 2019-2024 test)

## References

- Strategy concept based on "Rare but Powerful Weekly Setups" webinar
- TTM Squeeze implementation: `utils/scanning/squeeze.py`
- Backtrader framework: https://www.backtrader.com/
- TimescaleDB integration: `utils/timescaledb_client.py`

## Troubleshooting

### No Trades Generated

- Check data history: Need 52+ weeks minimum
- Verify ignition conditions: Volume multiplier may be too high
- Check TTM confirmation: Momentum may not be crossing zero
- Review trend filter: May be too restrictive

### Too Many Trades

- Increase `vol_mult` (e.g., 3.5 or 4.0)
- Tighten `max_extended_pct` (e.g., 0.15)
- Reduce `max_delay_weeks` (e.g., 20)

### Performance Issues

- Strategy is designed for low frequency (rare setups)
- Expect 1-2 trades per symbol per decade
- Focus on win rate and profit factor, not trade count

## Support

For issues or questions:
1. Check strategy logs (set `log_level: DEBUG`)
2. Review individual trade details in CSV report
3. Verify data quality in TimescaleDB
4. Test with known symbols that have clear setups

