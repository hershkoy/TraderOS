I'll study the folder structure and files to understand the codebase and get ready for your question. Let me explore the key components of this backtrader trading system.


[4 tools called]

Now let me explore more key files to understand the system better:


[4 tools called]

Let me also check the utils directory and some key strategy files:


[3 tools called]

Now let me check some key utility files to understand the system better:


[3 tools called]

Perfect! Now I have a comprehensive understanding of this backtrader testing framework. Let me summarize what I've learned:

## BackTrader Testing Framework Overview

This is a comprehensive backtesting framework with the following key components:

### **Core Architecture:**
- **BackTrader Integration**: Uses BackTrader as the core backtesting engine
- **TimescaleDB Storage**: PostgreSQL with TimescaleDB extension for time-series data storage
- **Multi-Provider Data**: Supports Alpaca Markets and Interactive Brokers (IBKR) for data fetching
- **YAML Configuration**: Flexible configuration system with strategy-specific parameters
- **Custom Tracking**: Custom performance tracking system to replace problematic BackTrader analyzers

### **Key Features:**

1. **Data Management:**
   - TimescaleDB for efficient time-series data storage
   - Support for both hourly and daily timeframes
   - Automatic data resampling (1h → daily, weekly)
   - Data fetching from Alpaca and IBKR APIs

2. **Strategy System:**
   - Dynamic strategy discovery from `strategies/` directory
   - Multiple built-in strategies:
     - Mean Reversion (Bollinger Bands)
     - Point & Figure (PnF)
     - VCP AVWAP Breakout
     - Liquidity Sweep
     - RSI strategies
   - Custom tracking mixin for reliable performance metrics

3. **Reporting:**
   - TradingView-style HTML reports with interactive charts
   - CSV export of individual trades
   - JSON statistics export
   - Plotly-based interactive visualizations

4. **Options Trading:**
   - Polygon.io integration for options data
   - Historical options contract backfilling
   - LEAPS (Long-term Equity Anticipation Securities) support
   - Options screening and analysis tools

5. **Database Infrastructure:**
   - Docker Compose setup with TimescaleDB and pgAdmin
   - Comprehensive database schema for market data and options
   - Migration scripts and data management utilities

### **Current Focus Areas:**
- **Options Backfilling**: Scripts for historical options data collection
- **Pattern Scanning**: HL (Higher Low) after LL (Lower Low) pattern detection
- **Collar Screening**: Zero-cost collar strategy screening
- **Performance Optimization**: Custom tracking to avoid BackTrader analyzer issues

### **File Structure:**
- `backtrader_runner_yaml.py`: Main backtesting runner
- `strategies/`: Trading strategy implementations
- `utils/`: Utility functions for data, database, and analysis
- `scripts/`: Data fetching and processing scripts
- `tests/`: Test files (recently moved to proper location)
- `reports/`: Generated backtest reports
- `docs/`: Comprehensive documentation

The system is well-architected for systematic backtesting with professional-grade reporting and supports both equity and options trading strategies. It's particularly focused on LEAPS options trading and pattern-based strategies.

I'm now ready to help you with any questions about this framework! What would you like to know or work on?