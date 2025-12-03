# Failed Symbols Identification and Reprocessing

This document explains how to identify failed symbols from universe updates and reprocess them.

## Overview

When running the universe update script, some symbols may fail due to various reasons (API timeouts, connection issues, etc.). This guide shows you how to:

1. **Identify failed symbols** from the complete universe
2. **Save them to a file** for reprocessing
3. **Reprocess only the failed symbols** using the `--universe-file` parameter

## 1. Identify Failed Symbols

### Run the identification script:

```bash
# For IB provider with 1h timeframe (default)
python scripts/identify_failed_symbols.py

# For specific provider and timeframe
python scripts/identify_failed_symbols.py --provider ib --timeframe 1h

# For custom output file
python scripts/identify_failed_symbols.py --output my_failed_symbols.txt
```

### What it does:

- Compares the complete universe (517 symbols) with symbols that have data in the database
- Identifies symbols that are in the universe but don't have data
- Saves the failed symbols to a text file (default: `failed.txt`)

### Example output:

```
INFO:__main__:Identifying failed symbols for ib @ 1h
INFO:__main__:Retrieved 517 symbols from complete universe
INFO:__main__:Found 445 symbols with existing data
INFO:__main__:Identified 72 failed symbols out of 517 total
INFO:__main__:✅ Successfully identified and saved 72 failed symbols
INFO:__main__:📁 Failed symbols saved to: failed.txt
INFO:__main__:📋 First 10 failed symbols: PDD, PEG, PEP, PFE, PFG, PG, PGR, PH, PHM, PKG
```

## 2. Examine the Failed Symbols

### View the failed symbols file:

```bash
# View all failed symbols
cat failed.txt

# Count failed symbols
wc -l failed.txt

# View first 10 failed symbols
head -10 failed.txt

# View last 10 failed symbols
tail -10 failed.txt
```

### Example `failed.txt` content:

```
PDD
PEG
PEP
PFE
PFG
PG
PGR
PH
PHM
PKG
...
```

## 3. Reprocess Failed Symbols

### Use the `--universe-file` parameter:

```bash
# Reprocess only the failed symbols
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --max-bars \
  --universe-file failed.txt

# With additional options
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --max-bars \
  --universe-file failed.txt \
  --batch-size 5 \
  --delay-batches 10 \
  --delay-tickers 2
```

### What happens:

- The script loads tickers from `failed.txt` instead of the default universe
- Only processes the 72 failed symbols instead of all 517
- Much faster execution since it's only processing failed symbols
- Can resume from a specific index if needed

## 4. Complete Logging

### Enhanced logging features:

The script now outputs complete logs with full stack traces to:

- **Console**: Real-time progress and errors
- **`universe_update.log`**: Main script logs
- **`universe_update_complete.log`**: Complete logs from all modules with full tracebacks

### Debug failed symbols:

```bash
# View complete logs for debugging
tail -f universe_update_complete.log

# Search for specific symbol errors
grep "PDD" universe_update_complete.log

# View all errors with stack traces
grep "ERROR" universe_update_complete.log
```

## 5. Workflow Example

### Complete workflow for handling failed symbols:

```bash
# Step 1: Identify failed symbols
python scripts/identify_failed_symbols.py --provider ib --timeframe 1h

# Step 2: Check how many failed
wc -l failed.txt

# Step 3: Reprocess failed symbols
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --max-bars \
  --universe-file failed.txt

# Step 4: Verify all symbols now have data
python scripts/identify_failed_symbols.py --provider ib --timeframe 1h
```

## 6. Troubleshooting

### Common issues:

1. **No failed symbols found**: All symbols in the universe already have data
2. **File not found error**: Check the path to your `failed.txt` file
3. **Permission errors**: Ensure you have read access to the universe file

### Debug commands:

```bash
# Test the identification script
python test_failed_symbols.py

# Check database connection
python -c "from utils.timescaledb_client import get_timescaledb_client; client = get_timescaledb_client(); print('Connected:', client.ensure_connection())"

# Verify universe file format
head -5 failed.txt
```

## 7. Advanced Usage

### Custom universe files:

You can create custom universe files for specific purposes:

```bash
# Create a file with only specific symbols
echo -e "AAPL\nMSFT\nGOOGL" > custom_universe.txt

# Process only these symbols
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --universe-file custom_universe.txt
```

### Batch processing:

```bash
# Process failed symbols in smaller batches
python utils/update_universe_data.py \
  --provider ib \
  --timeframe 1h \
  --max-bars \
  --universe-file failed.txt \
  --batch-size 5 \
  --delay-batches 10
```

## Summary

This workflow allows you to:

1. ✅ **Efficiently identify** which symbols failed during universe updates
2. ✅ **Save failed symbols** to a file for easy reprocessing
3. ✅ **Reprocess only failed symbols** instead of the entire universe
4. ✅ **Debug issues** with complete logging and stack traces
5. ✅ **Resume processing** from any point if needed

The combination of failed symbol identification and the `--universe-file` parameter makes it much more efficient to handle failed symbols and get your universe update to 100% completion.
