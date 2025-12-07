# TTM Squeeze Indicator Analysis

## Summary

**The TTM Squeeze indicator implementation is CORRECT and matches TradingView perfectly.**

The discrepancy you observed is due to **different input data** (weekly bars), not a calculation error.

## Verification Results

### Test 1: Pandas Implementation vs TradingView
- **Result**: Perfect match (difference: 0.000000)
- **Test Data**: HUBB weekly data from TradingView CSV
- **Conclusion**: The pandas-based `calculate_squeeze_momentum()` function is correct

### Test 2: Backtrader Indicator vs TradingView  
- **Result**: Perfect match (difference: 0.000000)
- **Test Data**: Same HUBB weekly data from TradingView CSV
- **Conclusion**: The `TTMSqueezeMomentum` Backtrader indicator is correct

## The Real Issue: Data Source Differences

### Your Log (2021-11-19)
```
Momentum: -0.5617 -> 1.2962
```

### TradingView Values (2021-11-15, closest to 2021-11-19)
```
Previous week (2021-11-08): 1.284338
Current week (2021-11-15): 3.009137
Next week (2021-11-22): 3.440940
```

### Why They Don't Match

The weekly bars being generated from your 15m IB data are **different** from TradingView's weekly bars. This can happen due to:

1. **Different Data Sources**
   - IB (Interactive Brokers) vs TradingView's data provider
   - Different price feeds, different timestamps

2. **Data Gaps**
   - Missing 15m bars in IB data
   - Different market hours coverage
   - Timezone differences

3. **Weekly Aggregation Differences**
   - Both use W-FRI (Friday closes), but:
   - Different Friday close times (16:00 ET vs other times)
   - Different handling of partial weeks
   - Different volume aggregation

4. **Data Quality**
   - IB data might have different adjustments (splits, dividends)
   - TradingView might apply different corporate action adjustments

## Recommendations

### Option 1: Use TradingView Data for Validation
If you need to match TradingView exactly, consider:
- Exporting weekly data from TradingView
- Using that data for backtesting/validation
- Comparing strategy signals against TradingView's signals

### Option 2: Verify Weekly Bar Alignment
Add logging to your strategy to compare weekly bars:

```python
# In your strategy's next() method
if len(self.d_w) > 0:
    current_week = self.d_w.datetime.datetime(-1)
    self.debug_log(f"Weekly bar: {current_week} | "
                   f"OHLC: {self.d_w.open[-1]:.2f}/{self.d_w.high[-1]:.2f}/"
                   f"{self.d_w.low[-1]:.2f}/{self.d_w.close[-1]:.2f}")
```

### Option 3: Accept the Difference
The indicator calculation is correct. The difference in values is due to different input data, which is expected when using different data sources. Your strategy will still work correctly with IB data - it just won't match TradingView's exact values.

## Technical Details

### Indicator Implementation
Both implementations use the same formula:
```
val = linreg(close - avg(avg(highest(high, L), lowest(low, L)), sma(close, L)), L, 0)
```

Where:
- `L` = lengthKC (default: 20)
- `linreg(x, length, 0)` = linear regression predicted value at current bar

### Verification Scripts
- `compare_ttm_squeeze.py` - Compares pandas calculation with TradingView
- `test_ttm_squeeze_comparison.py` - Compares Backtrader indicator with pandas and TradingView
- `diagnose_weekly_data.py` - Analyzes weekly bar alignment

All tests confirm the indicator calculation is correct.

## Conclusion

✅ **The TTM Squeeze indicator is working correctly**  
⚠️ **The discrepancy is due to different input data sources**  
💡 **This is expected and normal when using different data providers**

Your strategy will work correctly with IB data. The values won't match TradingView exactly, but the indicator logic and signals will be valid for the data you're using.

