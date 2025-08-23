# Symbol Mapping System

This document explains the intelligent symbol mapping system that automatically discovers and caches the correct Interactive Brokers (IB) symbol formats.

## Overview

The symbol mapping system solves the problem of different symbol formats between data sources:
- **Wikipedia/SP500**: Uses `BF.B` (with dot)
- **Interactive Brokers**: Uses `BF B` (with space)

The system automatically discovers the correct IB symbol format and caches it for future use.

## How It Works

### 1. **Database Table**
The system creates a `symbol_mappings` table with the following structure:

```sql
CREATE TABLE symbol_mappings (
    id SERIAL PRIMARY KEY,
    original_symbol VARCHAR(20) NOT NULL,
    ib_symbol VARCHAR(20) NOT NULL,
    provider VARCHAR(10) DEFAULT 'IB',
    is_valid BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used TIMESTAMPTZ DEFAULT NOW(),
    use_count INTEGER DEFAULT 1,
    UNIQUE(original_symbol, provider)
);
```

### 2. **Symbol Conversion Process**

When a symbol needs to be converted for IB:

1. **Check Cache**: Look up the symbol in the database
2. **Test Original**: Try the original symbol first (it might work)
3. **Discover Variations**: Test different symbol variations
4. **Cache Result**: Save the successful mapping for future use

### 3. **Symbol Variations Tested**

For a symbol like `BF.B`, the system tests:
- `BF.B` (original)
- `BF B` (space - most common for Class B)
- `BF-B` (hyphen)
- `BFB` (no separator)
- `BF/B` (slash)
- `BF_B` (underscore)

## Usage

### Basic Usage

The symbol mapping is automatic when using the IB provider:

```bash
# The system will automatically discover and cache symbol mappings
python utils/update_universe_data.py --provider ib --timeframe 1h
```

### View Symbol Mappings

```bash
# View all symbol mappings
python utils/update_universe_data.py --view-mappings

# View mapping statistics
python utils/update_universe_data.py --mapping-stats

# Clear invalid mappings
python utils/update_universe_data.py --clear-invalid-mappings
```

### Programmatic Usage

```python
from utils.update_universe_data import UniverseDataUpdater

# Initialize updater
updater = UniverseDataUpdater('ib', '1h')

# Convert a symbol (automatic caching)
ib_symbol = updater._convert_symbol_for_ib('BF.B')  # Returns 'BF B'

# View mapping statistics
stats = updater.get_symbol_mapping_stats()
print(f"Success rate: {stats['success_rate']:.1f}%")

# View recent mappings
mappings = updater.view_symbol_mappings(limit=10)
for mapping in mappings:
    print(f"{mapping['original_symbol']} -> {mapping['ib_symbol']}")
```

## Features

### 1. **Automatic Discovery**
- Tests multiple symbol variations
- Validates symbols with IB API
- Confirms symbols have tradeable data

### 2. **Intelligent Caching**
- Caches successful mappings
- Tracks usage statistics
- Avoids repeated API calls

### 3. **Performance Optimization**
- Database indexes for fast lookups
- Usage tracking for optimization
- Automatic cleanup of invalid mappings

### 4. **Error Handling**
- Graceful fallback to original symbol
- Invalid mapping tracking
- Comprehensive logging

## Database Schema

### symbol_mappings Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL | Primary key |
| `original_symbol` | VARCHAR(20) | Original symbol (e.g., 'BF.B') |
| `ib_symbol` | VARCHAR(20) | IB symbol (e.g., 'BF B') |
| `provider` | VARCHAR(10) | Data provider (default: 'IB') |
| `is_valid` | BOOLEAN | Whether the mapping is valid |
| `created_at` | TIMESTAMPTZ | When the mapping was created |
| `last_used` | TIMESTAMPTZ | When the mapping was last used |
| `use_count` | INTEGER | Number of times the mapping was used |

### Indexes

- `idx_symbol_mappings_original_symbol`: Fast lookups by original symbol
- `idx_symbol_mappings_ib_symbol`: Fast lookups by IB symbol

## Examples

### Example 1: First Time Discovery

```python
# First time processing BF.B
updater = UniverseDataUpdater('ib', '1h')
ib_symbol = updater._convert_symbol_for_ib('BF.B')

# Process:
# 1. Check cache: Not found
# 2. Test BF.B: Fails
# 3. Test BF B: Success!
# 4. Cache: BF.B -> BF B
# 5. Return: BF B
```

### Example 2: Cached Lookup

```python
# Second time processing BF.B
ib_symbol = updater._convert_symbol_for_ib('BF.B')

# Process:
# 1. Check cache: Found BF.B -> BF B
# 2. Update usage count
# 3. Return: BF B (no API calls needed)
```

### Example 3: Invalid Symbol

```python
# Processing a symbol that doesn't exist in IB
ib_symbol = updater._convert_symbol_for_ib('INVALID')

# Process:
# 1. Check cache: Not found
# 2. Test variations: All fail
# 3. Cache: INVALID -> INVALID (is_valid: false)
# 4. Return: INVALID (will fail gracefully)
```

## Command Line Options

### View Mappings
```bash
python utils/update_universe_data.py --view-mappings
```

Output:
```
--- Symbol Mappings ---
Original: BF.B, IB: BF B, Valid: True, Use Count: 5, Created: 2025-08-23 14:30:00, Last Used: 2025-08-23 15:45:00
Original: BRK.B, IB: BRK B, Valid: True, Use Count: 3, Created: 2025-08-23 14:35:00, Last Used: 2025-08-23 15:40:00
```

### Mapping Statistics
```bash
python utils/update_universe_data.py --mapping-stats
```

Output:
```
--- Symbol Mapping Statistics ---
Total Mappings: 25
Valid Mappings: 22
Invalid Mappings: 3
Success Rate: 88.0%

Most Used Mappings:
Original: BF.B, IB: BF B, Use Count: 15, Last Used: 2025-08-23 15:45:00
Original: BRK.B, IB: BRK B, Use Count: 12, Last Used: 2025-08-23 15:40:00
```

### Clear Invalid Mappings
```bash
python utils/update_universe_data.py --clear-invalid-mappings
```

Output:
```
Cleared 3 invalid symbol mappings.
```

## Benefits

1. **Improved Success Rate**: Automatically handles symbol format differences
2. **Performance**: Caches mappings to avoid repeated API calls
3. **Learning**: System improves over time as more symbols are processed
4. **Transparency**: Full visibility into symbol mappings and statistics
5. **Maintenance**: Easy cleanup of invalid mappings

## Troubleshooting

### Common Issues

1. **Symbol Not Found**: Check if the symbol exists in IB
2. **Cache Issues**: Clear invalid mappings with `--clear-invalid-mappings`
3. **Performance**: Monitor usage statistics to optimize frequently used symbols

### Debug Commands

```bash
# Test the mapping system
python test_symbol_mapping_system.py

# View detailed mapping information
python utils/update_universe_data.py --mapping-stats

# Check specific symbol mappings
python utils/update_universe_data.py --view-mappings
```

## Future Enhancements

1. **Machine Learning**: Predict symbol variations based on patterns
2. **Batch Discovery**: Discover mappings for multiple symbols at once
3. **Provider Support**: Extend to other data providers beyond IB
4. **Symbol Validation**: Periodic validation of cached mappings
5. **Performance Metrics**: Track conversion success rates and timing
