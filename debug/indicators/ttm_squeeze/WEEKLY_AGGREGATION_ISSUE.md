# Weekly Aggregation Issue - Root Cause Analysis

## Problem

TTM Squeeze cross detection is showing false positives (e.g., June 2019) that don't match TradingView.

## Root Cause: **INCOMPLETE WEEKLY BARS**

The issue is **NOT** with the TTM Squeeze calculation - it's with **incomplete weekly data aggregation**.

### Evidence

**Example: Week ending 2019-06-14**
- **Expected**: 5 trading days (Mon-Fri)
- **Actual**: Only 4 trading days (Mon-Thu)
- **Missing**: Friday June 14, 2019 data is **not in the database**

### Impact

When a weekly bar is missing Friday data:
1. The weekly OHLC values are incorrect (based on only Mon-Thu)
2. The TTM Squeeze momentum calculation uses incomplete data
3. This causes different momentum values than TradingView
4. False cross detections occur

### Why This Happens

1. **Missing Data**: Friday data may not have been fetched from IB
2. **Data Gaps**: Market holidays or data feed issues
3. **Timezone Issues**: Friday data might be in a different timezone bucket
4. **Data Fetching Window**: The data fetch might have stopped before Friday

### Solution

1. **Data Validation**: Added warning when weekly bars are incomplete (see `resample_to_weekly_pandas`)
2. **Improved Cross Detection**: Made TTM cross detection stricter to reduce false positives
3. **Data Completeness Check**: Verify all weekly bars have 5 trading days before using them

### Recommendations

1. **Check Data Completeness**: Before running strategies, verify weekly bars are complete
2. **Backfill Missing Data**: If Friday data is missing, backfill it from IB
3. **Add Data Validation**: Reject incomplete weekly bars in critical calculations
4. **Log Warnings**: The system now logs warnings when incomplete weeks are detected

### Files Modified

- `backtrader_runner_yaml.py`: Added validation for incomplete weekly bars
- `strategies/weekly_bigvol_ttm_squeeze.py`: Improved TTM cross detection logic

### Diagnostic Scripts

- `check_weekly_aggregation.py`: Analyzes weekly bar completeness
- `check_friday_data.py`: Checks if specific Friday data exists
- `compare_weekly_aggregation.py`: Compares our aggregation with TradingView

## Conclusion

**The TTM Squeeze indicator calculation is CORRECT.**

The discrepancy is due to **incomplete weekly bars** caused by missing Friday data in the database. This is a **data quality issue**, not a calculation issue.

To fix:
1. Ensure all Friday data is fetched and stored
2. Backfill missing historical data
3. Add data validation to catch incomplete weeks early

