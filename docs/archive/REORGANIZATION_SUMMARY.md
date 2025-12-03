# Project Reorganization Summary

## Quick Overview

**Current State:** 30+ files in root directory, many unused/obsolete files, unclear organization

**Goal:** Clean, organized structure with clear separation of concerns

## Immediate Actions

### 1. Delete Obsolete Files (Safe to Remove)
```bash
# These are clearly marked as old/obsolete
run_hl_after_ll_scanner.py_OLD
run_hl_scanner.bat_OLD
```

### 2. Files to Archive (Review First, Then Archive)
- `screener_zero_cost_collar.py` - Old version, superseded by `utils/screener_zero_cost_collar_enhanced.py`
- `run_eod_prices_migration.py` - One-time migration (if already run)
- `run_snapshot_migration.py` - One-time migration (if already run)

### 3. Files to Move to `scripts/` (Utility Scripts)
- `check_data_freshness.py`
- `check_database_data.py`
- `check_duplicates.py`
- `debug_universe_update.py`
- `filter_otc_stocks.py` ⚠️ Review purpose first
- `identify_failed_symbols.py`
- `import_tickers_to_universe.py`
- `optimize_database.py`
- `quick_db_check.py`

### 4. Files to Move to `tests/` (Test Files)
- `test_auto_update.py`
- `test_hl_after_ll_scanner.py`
- `test_scanner_small.py`

### 5. Files to Move to `examples/` (Example Files)
- `example_hl_scanner_usage.py`
- `scanner_examples.py`

### 6. Files to Move to `crons/` (Batch Files)
- `import_tickers.bat`
- `run_collar_screener.bat`
- `update_and_scan.bat`

### 7. Files to Move to `docs/` (Documentation)
- `COLLAR_SCREENER_README.md`
- `COLLAR_SCREENER_SUMMARY.md`
- `HL_AFTER_LL_SCANNER_README.md`

## Proposed New Structure

### Option A: Minimal Changes (Recommended for Start)
```
backTraderTest/
├── scripts/          # All utility scripts
├── tests/            # All test files
├── examples/         # All example files
├── crons/            # All batch files
├── docs/             # All documentation
├── archive/          # Obsolete files
└── [keep main runners in root for now]
```

### Option B: Full Reorganization (Better Long-term)
```
backTraderTest/
├── configs/          # All YAML configs
├── runners/          # Main entry points
├── scripts/          # Utility scripts
├── setup/            # Setup & migrations
├── archive/          # Obsolete files
└── [existing structure]
```

## Files That Should Stay in Root

✅ **Keep These:**
- `README.md`
- `requirements.txt`
- `docker-compose.yml`
- `Taskfile.yml`
- `postgresql.conf`
- `backtrader_runner_yaml.py` (main entry point)
- `scanner_runner.py` (main entry point)
- `charting_server.py` (main entry point)
- `update_and_scan.py` (main entry point)

## Duplicate Code Identified

1. **Screener Implementations:**
   - `screener_zero_cost_collar.py` (root) - OLD
   - `utils/screener_zero_cost_collar_enhanced.py` - CURRENT
   - **Action:** Delete root version, keep utils version

2. **Scanner Runners:**
   - `run_hl_after_ll_scanner.py_OLD` - OLD
   - `hl_after_ll_scanner_runner.py` - CURRENT
   - **Action:** Already marked OLD, safe to delete

## Unknown/Unclear Purpose Files

⚠️ **Review These:**
- `filter_otc_stocks.py` - No imports found, purpose unclear
- `test_auto_update.py` - Test file but in root, check if still needed

## Recommended Migration Steps

1. **Create `archive/` directory**
2. **Move `_OLD` files to archive**
3. **Move utility scripts to `scripts/`**
4. **Move test files to `tests/`**
5. **Move example files to `examples/`**
6. **Move batch files to `crons/`**
7. **Move docs to `docs/`**
8. **Update all references** (imports, batch files, configs)
9. **Test everything still works**
10. **Delete archive after confirming**

## Quick Win: Clean Root Directory First

If you want to start simple, just move these to appropriate existing directories:

```bash
# Move to scripts/
check_*.py
debug_*.py
optimize_*.py
quick_*.py
import_*.py
identify_*.py
filter_*.py

# Move to tests/
test_*.py

# Move to examples/
example_*.py
scanner_examples.py

# Move to crons/
*.bat (except those in crons/ already)

# Move to docs/
*_README.md
*_SUMMARY.md

# Delete
*_OLD.*
```

This alone will reduce root directory from 30+ files to ~10 essential files.

