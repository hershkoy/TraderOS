# Project Reorganization Plan

## Current Problems

1. **30+ files in root directory** - Hard to navigate
2. **Obsolete files** - `_OLD` files still present
3. **Duplicate code** - Old screener vs enhanced version
4. **Unclear organization** - Test files, examples, utilities all mixed in root
5. **Unused files** - Many utility scripts not imported anywhere

## Quick Wins (Do These First)

### Step 1: Delete Obsolete Files
```bash
# These are clearly marked as old and safe to delete
run_hl_after_ll_scanner.py_OLD
run_hl_scanner.bat_OLD
```

### Step 2: Archive Old Screener
```bash
# Create archive directory
mkdir archive

# Move old screener (superseded by utils/screener_zero_cost_collar_enhanced.py)
mv screener_zero_cost_collar.py archive/
```

### Step 3: Move Files to Existing Directories

**Move to `scripts/` (10 files):**
```bash
mv check_data_freshness.py scripts/
mv check_database_data.py scripts/
mv check_duplicates.py scripts/
mv debug_universe_update.py scripts/
mv filter_otc_stocks.py scripts/  # Review purpose first
mv identify_failed_symbols.py scripts/
mv import_tickers_to_universe.py scripts/
mv optimize_database.py scripts/
mv quick_db_check.py scripts/
mv run_eod_prices_migration.py scripts/
mv run_snapshot_migration.py scripts/
```

**Move to `tests/` (3 files):**
```bash
mv test_auto_update.py tests/
mv test_hl_after_ll_scanner.py tests/
mv test_scanner_small.py tests/
```

**Move to `examples/` (2 files):**
```bash
mv example_hl_scanner_usage.py examples/
mv scanner_examples.py examples/
```

**Move to `crons/` (3 files):**
```bash
mv import_tickers.bat crons/
mv run_collar_screener.bat crons/
mv update_and_scan.bat crons/
```

**Move to `docs/` (3 files):**
```bash
mv COLLAR_SCREENER_README.md docs/
mv COLLAR_SCREENER_SUMMARY.md docs/
mv HL_AFTER_LL_SCANNER_README.md docs/
```

**Move to `setup/` (1 file):**
```bash
mkdir setup
mv setup_timescaledb.py setup/
```

## After Moving Files - Update References

### 1. Update Batch Files
Check and update these batch files that reference moved scripts:
- `crons/daily_scanner.bat`
- `crons/run_scanner.bat`
- `crons/run_credit_spreads.bat`
- `crons/run_vertical_spread_hedged.bat`
- Any other `.bat` files

### 2. Update Python Imports
Search for imports of moved files:
```bash
# Search for imports that need updating
grep -r "from check_duplicates" .
grep -r "from identify_failed_symbols" .
grep -r "from screener_zero_cost_collar import" .
grep -r "import test_hl_after_ll_scanner" .
```

### 3. Update Config File Paths
If any configs reference moved files, update them.

## Files to Keep in Root (Essential Only)

✅ **Keep these 10 files in root:**
1. `README.md` - Main documentation
2. `requirements.txt` - Dependencies
3. `docker-compose.yml` - Docker config
4. `Taskfile.yml` - Task runner
5. `postgresql.conf` - DB config
6. `backtrader_runner_yaml.py` - Main backtest runner
7. `scanner_runner.py` - Unified scanner runner
8. `hl_after_ll_scanner_runner.py` - HL scanner runner
9. `charting_server.py` - Charting server
10. `update_and_scan.py` - Update and scan script

## Optional: Further Organization

If you want even better organization, consider:

### Create `configs/` directory
```bash
mkdir configs
mv defaults.yaml configs/
mv scanner_config.yaml configs/
mv hl_after_ll_scanner_config.yaml configs/
mv collar_screener_config.yaml configs/
```

Then update all references to these configs in code.

### Create `runners/` directory
```bash
mkdir runners
mv backtrader_runner_yaml.py runners/
mv scanner_runner.py runners/
mv hl_after_ll_scanner_runner.py runners/
mv charting_server.py runners/
mv update_and_scan.py runners/
```

This would make root even cleaner, but requires updating all batch files and documentation.

## Files to Review Before Moving

⚠️ **Review these files to understand their purpose:**

1. **`filter_otc_stocks.py`** - No imports found, unclear purpose
   - Check if it's a one-off script or still needed
   - If unused, move to `archive/`

2. **Migration scripts** (`run_eod_prices_migration.py`, `run_snapshot_migration.py`)
   - If migrations are already complete, move to `archive/`
   - If still needed, keep in `scripts/` or create `setup/migrations/`

3. **`test_auto_update.py`** - Check if this test is still relevant
   - If obsolete, delete or move to `archive/`

## Duplicate Code Resolution

### Screener Implementations
- ✅ **Keep:** `utils/screener_zero_cost_collar_enhanced.py` (used by tests)
- ❌ **Archive:** `screener_zero_cost_collar.py` (old version)

### Scanner Runners
- ✅ **Keep:** `scanner_runner.py` (unified runner, supports multiple scanners)
- ✅ **Keep:** `hl_after_ll_scanner_runner.py` (specific runner, may still be used)
- ❌ **Delete:** `run_hl_after_ll_scanner.py_OLD` (obsolete)

## Verification Steps

After reorganization:

1. **Test main entry points:**
   ```bash
   python backtrader_runner_yaml.py --help
   python scanner_runner.py --help
   python hl_after_ll_scanner_runner.py
   ```

2. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

3. **Check batch files:**
   - Verify all `.bat` files still work
   - Update paths if needed

4. **Check imports:**
   - Run a quick grep to find any broken imports
   - Fix any import errors

## Summary

**Before:** 30+ files in root, unclear organization
**After:** ~10 essential files in root, everything else organized

**Files to delete:** 2 (`_OLD` files)
**Files to archive:** 1 (old screener)
**Files to move:** 22 files to appropriate directories

This will make the project much more maintainable and easier to navigate!

