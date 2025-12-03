# Documentation

This directory contains all project documentation organized by category.

## Quick Navigation

### 📋 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
**Start here!** Complete overview of the project structure, components, and organization.

### 📊 Strategy Documentation
Located in `strategies/`:
- [VCP Strategy](strategies/VCP_STRATEGY_README.md) - VCP AVWAP breakout strategy
- [Weekly Big Volume TTM Squeeze](strategies/WEEKLY_BIGVOL_TTM_SQUEEZE_README.md) - Weekly big volume TTM squeeze strategy

### 🔧 Feature Documentation
Located in `features/`:
- [Collar Screener](features/COLLAR_SCREENER_README.md) - Zero-cost collar screener
- [HL After LL Scanner](features/HL_AFTER_LL_SCANNER_README.md) - Pattern detection scanner
- [Charting Server](features/CHARTING_SERVER_README.md) - Web-based charting server
- [Options Pipeline](features/options_pipeline.md) - Options data pipeline
- [Options Trader](features/options_trader.md) - Options trading system
- [Ticker Universe](features/TICKER_UNIVERSE_README.md) - Ticker universe management
- [Universe Data Updater](features/UNIVERSE_DATA_UPDATER_README.md) - Data update system
- [Symbol Mapping](features/SYMBOL_MAPPING_SYSTEM.md) - Symbol mapping system
- [Failed Symbols](features/FAILED_SYMBOLS_README.md) - Failed symbols identification

### ⚙️ Setup & Configuration
Located in `setup/`:
- [TimescaleDB Migration](setup/TIMESCALEDB_MIGRATION.md) - Database migration guide
- [Polygon Setup](setup/POLYGON_SETUP.md) - Polygon.io API setup
- [Historical Snapshots](setup/HISTORICAL_SNAPSHOTS_README.md) - Historical data snapshots
- [4H Resampling](setup/4H_RESAMPLING_IMPLEMENTATION.md) - 4-hour resampling implementation

### 📦 Archive
Located in `archive/`:
- Reorganization documentation
- Old project descriptions
- Example usage (superseded by examples in codebase)

## Documentation Structure

```
docs/
├── README.md                    # This file
├── PROJECT_STRUCTURE.md          # Project structure overview
├── strategies/                  # Strategy documentation
│   ├── VCP_STRATEGY_README.md
│   └── WEEKLY_BIGVOL_TTM_SQUEEZE_README.md
├── features/                    # Feature documentation
│   ├── COLLAR_SCREENER_README.md
│   ├── HL_AFTER_LL_SCANNER_README.md
│   ├── CHARTING_SERVER_README.md
│   ├── options_pipeline.md
│   ├── options_trader.md
│   ├── TICKER_UNIVERSE_README.md
│   ├── UNIVERSE_DATA_UPDATER_README.md
│   ├── SYMBOL_MAPPING_SYSTEM.md
│   └── FAILED_SYMBOLS_README.md
├── setup/                       # Setup and migration docs
│   ├── TIMESCALEDB_MIGRATION.md
│   ├── POLYGON_SETUP.md
│   ├── HISTORICAL_SNAPSHOTS_README.md
│   └── 4H_RESAMPLING_IMPLEMENTATION.md
└── archive/                     # Archived documentation
    ├── PROJECT_REORGANIZATION_ANALYSIS.md
    ├── REORGANIZATION_COMPLETE.md
    ├── REORGANIZATION_SUMMARY.md
    ├── project_description_251024.md
    └── example_usage.md
```

## Getting Started

1. **New to the project?** Start with [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
2. **Setting up?** Check [setup/TIMESCALEDB_MIGRATION.md](setup/TIMESCALEDB_MIGRATION.md)
3. **Using a strategy?** See [strategies/](strategies/)
4. **Using a feature?** See [features/](features/)

## Contributing

When adding new documentation:
- **Strategy docs** → `strategies/`
- **Feature docs** → `features/`
- **Setup/migration docs** → `setup/`
- **Obsolete docs** → `archive/`

