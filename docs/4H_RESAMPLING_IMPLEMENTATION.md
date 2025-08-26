# 4-Hour Resampling Implementation

## Overview

This document describes the implementation of automatic 4-hour resampling functionality in the backtrader runner. The feature automatically detects when a strategy requests higher timeframe data and creates a 4-hour resampled feed alongside the base 1-hour feed.

## What Was Implemented

### 1. Helper Function: `_strategy_requests_4h()`

Added to `backtrader_runner_yaml.py`, this function detects if a strategy needs 4h data by:

- Checking if the strategy has a `higher_tf_idx` parameter in its `params`
- Checking if the strategy's `get_data_requirements()` includes `'4h'` in `additional_timeframes`

### 2. Enhanced Data Feed Setup

Modified `setup_data_feeds()` function to add a new branch that:

- Creates the base 1h feed as usual
- Automatically resamples to 4h (240 minutes) when requested
- Returns both feeds in the correct order: `[data_1h, data_4h]`

### 3. New Strategy: `MeanReversionRSI_BT`

Created a new strategy that demonstrates the 4h functionality:

- **Location**: `strategies/mean_reversion_rsi_bt.py`
- **Features**: RSI, ADX, ATR, Bollinger Bands, and SuperTrend indicator
- **Timeframes**: Uses both 1h (base) and 4h (higher) data
- **Parameters**: Includes `higher_tf_idx=1` to request 4h data

## How It Works

### Automatic Detection

The runner automatically detects 4h requirements through:

```python
def _strategy_requests_4h(strategy_class) -> bool:
    # Method 1: Check params for higher_tf_idx
    if hasattr(strategy_class, 'params'):
        params_str = str(strategy_class.params)
        if 'higher_tf_idx' in params_str:
            return True
    
    # Method 2: Check data requirements
    if hasattr(strategy_class, 'get_data_requirements'):
        dr = strategy_class.get_data_requirements()
        addl = [x.lower() for x in dr.get('additional_timeframes', [])]
        if any(x in addl for x in ('4h', '240m', 'hours_4')):
            return True
    
    return False
```

### Data Feed Creation

When 4h is requested, the runner creates:

```python
elif data_reqs['base_timeframe'] == 'hourly' and _strategy_requests_4h(strategy_class):
    # Base 1h feed
    data_1h = bt.feeds.PandasData(dataname=df_data)
    cerebro.adddata(data_1h)

    # Auto-add 4h resample as data1
    data_4h = cerebro.resampledata(data_1h, timeframe=bt.TimeFrame.Minutes, compression=240)

    return [data_1h, data_4h]
```

## Usage

### Running the Strategy

```bash
python backtrader_runner_yaml.py \
  --config defaults.yaml \
  --symbol NFLX \
  --provider ib \
  --timeframe 1h \
  --strategy mean_reversion_rsi
```

### Strategy Implementation

The strategy accesses the 4h data through:

```python
class MeanReversionRSI_BT(CustomTrackingMixin, bt.Strategy):
    params = (
        # ... other params ...
        ("higher_tf_idx", 1),  # index of 4h resampled data
    )

    def __init__(self):
        # Base timeframe (1h) indicators
        self.rsi = bt.indicators.RSI(self.data, period=self.p.rsi_period)
        
        # Higher timeframe (4h) indicators
        self.rsi_higher = bt.indicators.RSI(self.datas[self.p.higher_tf_idx], period=self.p.rsi_period)
        self.supertrend = SuperTrend(self.datas[self.p.higher_tf_idx], period=10, multiplier=3.0)
```

## Benefits

1. **Automatic**: No manual configuration needed - the runner detects and creates 4h feeds automatically
2. **Flexible**: Works with any strategy that requests 4h data
3. **Efficient**: Only creates 4h feeds when needed
4. **Consistent**: Follows the same pattern as existing daily/weekly resampling

## Future Enhancements

1. **Configurable Timeframes**: Allow strategies to request different higher timeframes (2h, 6h, 8h, etc.)
2. **Multiple Timeframes**: Support strategies that need more than one additional timeframe
3. **Performance Optimization**: Cache resampled data for reuse across multiple strategies

## Testing

The implementation has been tested to ensure:

- ✅ Strategy registration works correctly
- ✅ 4h detection function works
- ✅ Data feed setup creates both 1h and 4h feeds
- ✅ Strategy can access both timeframes correctly

## Files Modified

1. `backtrader_runner_yaml.py` - Added helper function and enhanced data feed setup
2. `strategies/__init__.py` - Added new strategy to registry
3. `strategies/mean_reversion_rsi_bt.py` - New strategy implementation
4. `indicators/supertrend.py` - Custom SuperTrend indicator implementation
5. `indicators/__init__.py` - Added SuperTrend to indicators package
6. `docs/4H_RESAMPLING_IMPLEMENTATION.md` - This documentation

## Notes

- The strategy now uses the custom SuperTrend indicator for trend direction
- The SuperTrend indicator is located in `indicators/supertrend.py` and provides trend direction (+1 for uptrend, -1 for downtrend)
- The 4h resampling uses `TimeFrame.Minutes` with `compression=240` for robustness
- The feature is backward compatible - strategies without 4h requirements work unchanged
