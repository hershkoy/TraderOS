# IB Flex Query Multi-Leg Strategy Report Generator

A tool for fetching trade history from Interactive Brokers Flex Query Web Service and generating comprehensive reports of multi-leg option strategies grouped by OrderID.

## Purpose

This script automates the process of retrieving and analyzing multi-leg option strategies from your Interactive Brokers account. It:

- Fetches trade data via IB Flex Query Web Service
- Groups trades by OrderID to identify multi-leg strategies (spreads, collars, iron condors, etc.)
- Generates formatted reports in HTML or CSV format
- Filters trades by date range
- Provides summary statistics and detailed leg-by-leg breakdowns

## Features

- **Automated Flex Query Processing**: Handles the two-step process of requesting and downloading Flex Query results
- **Multi-Leg Strategy Grouping**: Automatically groups option trades by OrderID to identify complex strategies
- **Date Filtering**: Filter trades by date range (e.g., all trades since a specific date)
- **Multiple Output Formats**: Generate reports in HTML (with styling) or CSV (for spreadsheet analysis)
- **Comprehensive Trade Details**: Includes symbol, strike, expiry, put/call, buy/sell, quantity, price, net cash, and commissions
- **Summary Statistics**: Total strategies, legs, net cash, and commissions
- **Flexible Column Handling**: Automatically handles various column name formats from IB Flex Queries
- **Error Handling**: Robust error handling with retry logic for statement downloads

## Files

- `scripts/api/ib/ib_flex_multi_leg_report.py` - Main script
- `utils/ib_execution_converter.py` - ExecutionConverter class for converting IB API Fill objects to DataFrame format
- `utils/strategy_detector.py` - StrategyDetector class for detecting and grouping multi-leg strategies
- `tests/unit/test_ib_execution_converter.py` - Unit tests for ExecutionConverter
- `tests/unit/test_strategy_detector.py` - Unit tests for StrategyDetector
- `docs/ib/IB_FLEX_MULTI_LEG_REPORT_README.md` - This documentation

## Setup

### Prerequisites

1. **Interactive Brokers Account**: Active IB account with Flex Query Web Service access (only if using API, not required for `--flex-report`)
2. **Python Dependencies**: 
   - `requests` - For HTTP requests to IB Flex Query Web Service
   - `pandas` - For data processing
   - `python-dotenv` - For environment variable management

Install dependencies:
```bash
pip install requests pandas python-dotenv
```

### Configuration

**Option 1: Using API (requires environment variables)**

The script requires two environment variables to authenticate with IB Flex Query Web Service:

#### 1. Get Flex Query Web Service Token

1. Log in to Interactive Brokers Client Portal
2. Navigate to: **Reports** → **Flex Web Service** → **Token**
3. Copy your Flex Web Service token
4. Set it as an environment variable:

**Windows (Command Prompt):**
```cmd
set IB_FLEX_QUERY_TOKEN=your_token_here
```

**Windows (PowerShell):**
```powershell
$env:IB_FLEX_QUERY_TOKEN="your_token_here"
```

**Using .env file (Recommended):**
Create or edit `.env` file in the project root:
```
IB_FLEX_QUERY_TOKEN=your_token_here
IB_FLEX_QUERY_ID=your_query_id_here
```

#### 2. Create and Get Flex Query ID

1. Log in to Interactive Brokers Client Portal
2. Navigate to: **Reports** → **Flex Queries**
3. Create a new Flex Query or use an existing one that includes:
   - **TradeConfirms** section
   - Fields: OrderID, Symbol, Strike, Expiry, Put/Call, Buy/Sell, Quantity, Price, NetCash, Commission, Date/Time
4. Copy the Flex Query ID (usually a number like `123456`)
5. Set it as an environment variable:

**Windows (Command Prompt):**
```cmd
set IB_FLEX_QUERY_ID=123456
```

**Windows (PowerShell):**
```powershell
$env:IB_FLEX_QUERY_ID="123456"
```

**Using .env file:**
Add to your `.env` file (see above)

**Option 2: Using Manually Downloaded File (no environment variables needed)**

If you're having trouble with the API or want to test with a file you've already downloaded:

1. Download your Flex Query report from IB Client Portal:
   - Go to **Reports** → **Flex Queries**
   - Run your query and download the result (CSV or XML format)
2. Save the file locally
3. Use the `--flex-report` option to point to the file:
   ```bash
   python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report path/to/your/flex_report.csv --type html
   ```

### Flex Query Configuration

Your Flex Query should include the following sections and fields:

**Required Sections:**
- `TradeConfirms` - Contains trade confirmation data

**Recommended Fields:**
- `OrderID` - Groups trades into multi-leg strategies
- `Symbol` - Underlying symbol
- `Strike` - Option strike price
- `Expiry` - Option expiration date
- `Put/Call` - Option type (P or C)
- `Buy/Sell` - Trade direction
- `Quantity` - Number of contracts
- `Price` - Trade price
- `NetCash` - Net cash amount
- `Commission` - Commission charged
- `Date/Time` or `Date` - Trade date/time

**Example Flex Query Setup:**
1. In Client Portal, go to Reports → Flex Queries
2. Create a new query or edit existing
3. Add "TradeConfirms" section
4. Select the fields listed above
5. Save the query and note the Query ID

## Usage

### Basic Usage

**Generate HTML report for all trades:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type html
```

**Generate CSV report for all trades:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type csv
```

### Using Manually Downloaded Flex Query Report

If you're having trouble with the API connection, you can download the Flex Query report manually from IB Client Portal and use it directly:

**Generate report from manually downloaded CSV file:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.csv --type html
```

**Generate report from manually downloaded XML file:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.xml --type html
```

**With date filtering:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.csv --since 2025-01-01 --type html
```

**Note:** When using `--flex-report`, you don't need to set `IB_FLEX_QUERY_TOKEN` or `IB_FLEX_QUERY_ID` environment variables.

### Date Filtering

**Generate report for trades since a specific date:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

**Date format:** YYYY-MM-DD (e.g., `2025-01-01`)

### Custom Output Path

**Specify custom output file:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --output reports/my_strategies.html
```

**Default output location:** If `--output` is not specified, reports are saved to:
- `reports/ib_multi_leg_strategies_YYYYMMDD_HHMMSS.html` (or `.csv`)

### Wait Time Configuration

**For large queries, increase wait time:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --max-wait 180
```

The script waits for IB to prepare the Flex Query statement. Large queries may take longer than the default 120 seconds. Use `--max-wait` to increase the timeout.

### Debug Logging

**Enable detailed debug logging:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --log-level DEBUG
```

Debug logging provides detailed information about:
- HTTP request/response details
- XML/CSV parsing steps
- Column detection and mapping
- Data filtering operations
- Strategy grouping process
- Individual leg processing

### Command-Line Options

```
--since DATE          Filter trades on or after this date (YYYY-MM-DD format)
--type {html,csv}     Output format: html or csv (default: html)
--output PATH         Output file path (default: auto-generated in reports/)
--max-wait SECONDS    Maximum seconds to wait for statement (default: 120)
--log-level LEVEL     Set logging level: DEBUG, INFO, WARNING, or ERROR (default: INFO)
--flex-report PATH    Path to manually downloaded Flex Query file (CSV or XML). Skips API download.
```

### Examples

**Generate HTML report for all trades in 2025:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

**Generate CSV report for recent trades (last 30 days):**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-15 --type csv --output reports/recent_strategies.csv
```

**Generate HTML report with custom filename:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --type html --output reports/q1_2025_strategies.html
```

**Generate report with extended wait time for large queries:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --max-wait 300
```

**Generate report with debug logging for troubleshooting:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --log-level DEBUG
```

**Generate report from manually downloaded file:**
```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --flex-report flex_report.csv --type html
```

## Output Formats

### HTML Report

The HTML report provides a visually formatted view with:

- **Summary Section**: Total strategies, total legs, total net cash, total commissions
- **Strategy Cards**: Each multi-leg strategy displayed in a card format showing:
  - Order ID
  - Number of legs
  - Underlying symbol
  - Trade date/time
  - Net cash and commission
  - Detailed leg-by-leg breakdown

**Example HTML Output:**
- Summary statistics at the top
- Each strategy in a styled card
- Legs displayed in a formatted list
- Professional styling with colors and spacing

### CSV Report

The CSV report provides a tabular format suitable for spreadsheet analysis:

**Columns:**
- `OrderID` - Order identifier
- `NumLegs` - Number of legs in the strategy
- `Underlying` - Underlying symbol
- `When` - Trade date/time
- `Legs` - Pipe-separated list of leg descriptions
- `NetCash` - Total net cash for the strategy
- `Commission` - Total commission for the strategy
- `TotalCost` - Net cash + commission

**Example CSV Output:**
```csv
OrderID,NumLegs,Underlying,When,Legs,NetCash,Commission,TotalCost
12345,2,SPY,2025-01-15 10:30:00,"BUY 1 x SPY 2025-02-21 450C @ 5.20 | SELL 1 x SPY 2025-02-21 455C @ 2.10",-310.00,1.00,-311.00
```

## How It Works

### Process Flow

**For TWS API mode (default):**
1. **Connect to TWS**: Establishes connection to Interactive Brokers TWS/IB Gateway
2. **Fetch Executions**: Requests execution history using `reqExecutions()` API
3. **Convert to DataFrame**: Uses `ExecutionConverter` to convert Fill objects to DataFrame format with correct Buy/Sell mapping and NetCash calculations
4. **Group Strategies**: Uses `StrategyDetector` to group executions by combo order (ParentID) to identify multi-leg strategies
5. **Generate Report**: Creates HTML or CSV report with formatted output

**For Flex Query mode (fallback):**
1. **Request Flex Query**: Sends request to IB Flex Query Web Service with token and query ID
2. **Get Reference Code**: Receives reference code for the query execution
3. **Download Statement**: Polls for statement readiness and downloads XML/CSV data
4. **Parse Data**: Parses XML or CSV format into pandas DataFrame
5. **Filter by Date**: Optionally filters trades by date range
6. **Group Strategies**: Groups option trades by OrderID to identify multi-leg strategies
7. **Generate Report**: Creates HTML or CSV report with formatted output

### Strategy Grouping Logic

- Filters to option trades (trades with Put/Call column)
- Groups trades by OrderID
- Only includes strategies with 2+ legs (multi-leg strategies)
- Sorts legs by Symbol, Strike, Put/Call for consistent display
- Calculates totals (NetCash, Commission) across all legs

### Leg Description Format

Each leg is formatted as:
```
{Buy/Sell} {Quantity} x {Symbol} {Expiry} {Strike}{Put/Call} @ {Price}
```

**Example:**
```
BUY 1 x SPY 2025-02-21 450C @ 5.20
SELL 1 x SPY 2025-02-21 455C @ 2.10
```

## Troubleshooting

### Common Issues

**Error: "IB_FLEX_QUERY_TOKEN environment variable not set"**
- **Solution**: Set the `IB_FLEX_QUERY_TOKEN` environment variable or add it to your `.env` file
- **Check**: Verify token is correct in Client Portal → Reports → Flex Web Service → Token

**Error: "IB_FLEX_QUERY_ID environment variable not set"**
- **Solution**: Set the `IB_FLEX_QUERY_ID` environment variable or add it to your `.env` file
- **Check**: Verify query ID exists in Client Portal → Reports → Flex Queries

**Error: "Flex Query request failed"**
- **Solution**: 
  - Verify token and query ID are correct
  - Check that Flex Query is active and accessible
  - Ensure query includes TradeConfirms section

**Error: "Statement not ready after X seconds"**
- **Solution**: 
  - Increase wait time using `--max-wait` option (e.g., `--max-wait 180` or `--max-wait 300`)
  - Large queries with many trades may take longer to process
  - This is usually temporary - try running again with increased wait time
  - Check IB system status if problem persists

**Warning: "No trade data found in Flex Query"**
- **Solution**: 
  - Verify your Flex Query includes TradeConfirms section
  - Check that query date range includes trades
  - Review query configuration in Client Portal

**Warning: "No multi-leg strategies found"**
- **Solution**: 
  - This is normal if you don't have multi-leg option trades
  - Verify OrderID field is included in Flex Query
  - Check that trades have matching OrderIDs

**Error: "No OrderID column found"**
- **Solution**: 
  - Ensure your Flex Query includes OrderID field
  - Check column names in Flex Query output
  - Script handles case-insensitive matching, but field must exist

### Debugging

Enable debug logging using the `--log-level DEBUG` command-line option:

```bash
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html --log-level DEBUG
```

Debug logging provides detailed information about:
- HTTP request URLs and response status codes
- Response content previews
- XML structure and parsing details
- CSV format detection and parsing
- Column names and data types
- Column mapping and normalization
- Data filtering operations
- Strategy grouping by OrderID
- Individual leg processing details
- Retry attempts and wait times
- DataFrame shapes and statistics

### Verification Steps

1. **Test Environment Variables:**
   ```bash
   echo %IB_FLEX_QUERY_TOKEN%
   echo %IB_FLEX_QUERY_ID%
   ```

2. **Test Flex Query in Client Portal:**
   - Manually run the Flex Query in Client Portal
   - Verify it returns trade data
   - Check that OrderID field is present

3. **Test Script with Minimal Date Range:**
   ```bash
   python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type csv
   ```

## Testing

The IB Flex Multi-Leg Report functionality includes comprehensive unit tests to verify execution data conversion, strategy detection, leg direction inference, and report generation logic.

### Running Tests

**Run all execution converter tests:**
```bash
# Activate virtual environment first
venv\Scripts\activate

# Run the execution converter tests
python -m pytest tests/unit/test_ib_execution_converter.py -v
```

**Run all strategy detector tests:**
```bash
# Run the strategy detector tests
python -m pytest tests/unit/test_strategy_detector.py -v
```

**Run both test suites:**
```bash
# Run both execution converter and strategy detector tests
python -m pytest tests/unit/test_ib_execution_converter.py tests/unit/test_strategy_detector.py -v
```

**Run a specific test:**
```bash
# Test execution converter - Buy execution conversion
python -m pytest tests/unit/test_ib_execution_converter.py::TestExecutionConverter::test_convert_buy_execution -v

# Test execution converter - Combo order conversion
python -m pytest tests/unit/test_ib_execution_converter.py::TestExecutionConverter::test_convert_combo_order -v

# Test strategy ID generation
python -m pytest tests/unit/test_strategy_detector.py::TestStrategyDetector::test_generate_strategy_id_bull_put_spread -v

# Test complex multi-leg strategy grouping
python -m pytest tests/unit/test_strategy_detector.py::TestStrategyDetector::test_group_executions_by_combo_complex_multi_leg -v

# Test short strategy string generation
python -m pytest tests/unit/test_strategy_detector.py::TestStrategyDetector::test_generate_short_strategy_string_complex_multi_leg -v
```

**Run all unit tests:**
```bash
python -m pytest tests/unit/ -v
```

### Test Coverage

#### Execution Converter Tests (`tests/unit/test_ib_execution_converter.py`)

The execution converter test suite covers:

- **BAG Execution Handling**: Tests that BAG executions are captured for combo order prices but excluded from DataFrame output
- **Buy Execution Conversion**: Tests conversion of Buy (BOT) executions with correct:
  - Buy/Sell mapping ('Buy')
  - Positive quantity
  - Negative NetCash (money out)
- **Sell Execution Conversion**: Tests conversion of Sell (SLD) executions with correct:
  - Buy/Sell mapping ('SELL')
  - Negative quantity
  - Positive NetCash (money in)
- **Combo Order Conversion**: Tests complete combo orders with multiple legs, including:
  - BAG price capture
  - All legs correctly converted
  - ParentID and BAG_ID extraction
  - Correct Buy/Sell and NetCash for each leg
- **CSV Format Validation**: Tests that DataFrame matches expected CSV format with all required columns
- **Date Filtering**: Tests date-based filtering of executions
- **Non-Option Execution Skipping**: Tests that non-option executions (stocks, etc.) are properly skipped

#### Strategy Detector Tests (`tests/unit/test_strategy_detector.py`)

The strategy detector test suite covers:

- **Strategy ID Generation**: Tests for generating human-readable strategy IDs for various spread types (bull put, bull call, etc.)
- **Short Strategy String Generation**: Tests for generating compact strategy strings (e.g., "RUT + Dec12 2605C - Dec12 2545C")
- **Leg Direction Inference**: Tests for correctly inferring BUY/SELL direction for combo orders based on strike prices and NetCash
- **Vertical Spread Detection**: Tests for detecting and summarizing vertical put and call spreads
- **Complex Multi-Leg Strategies**: Tests for grouping complex strategies with 3+ legs (e.g., RUT 4-leg strategies)
- **BAG Price Handling**: Tests for using BAG execution prices when available
- **NetCash Calculation**: Tests for correct NetCash calculation from execution sides (BOT = Buy = negative, SLD = Sell = positive)

### Test Data

The tests use mock execution data that simulates IB API execution responses, including:
- BAG executions (combo order prices)
- Individual leg executions with various strike prices and expiries
- Different execution sides (BOT/SLD)
- Complex multi-leg strategies with mixed expiries
- Real-world execution examples matching actual IB API responses

### Verifying Test Results

After running tests, you should see output like:
```
tests/unit/test_ib_execution_converter.py::TestExecutionConverter::test_convert_bag_execution PASSED
tests/unit/test_ib_execution_converter.py::TestExecutionConverter::test_convert_buy_execution PASSED
tests/unit/test_ib_execution_converter.py::TestExecutionConverter::test_convert_combo_order PASSED
tests/unit/test_strategy_detector.py::TestStrategyDetector::test_generate_strategy_id_bull_put_spread PASSED
tests/unit/test_strategy_detector.py::TestStrategyDetector::test_group_executions_by_combo_complex_multi_leg PASSED
...
```

All tests should pass. If any test fails, check:
1. That the virtual environment is activated
2. That all dependencies are installed (`pytest`, `pandas`, `ib_insync`, etc.)
3. The test output for specific error messages
4. That mock objects are properly configured

## Integration

### Scheduled Execution

You can schedule this script to run automatically:

**Windows Task Scheduler:**
```cmd
# Create a batch file: crons/ib_flex_report.bat
@echo off
cd /d D:\WORK\Projs\IB\backTraderTest
venv\Scripts\activate
python scripts/api/ib/ib_flex_multi_leg_report.py --since 2025-01-01 --type html
```

### Programmatic Usage

The script can be imported and used programmatically:

```python
from scripts.api.ib.ib_flex_multi_leg_report import (
    request_flex_query,
    download_flex_statement,
    parse_flex_xml,
    group_multi_leg_strategies
)

# Use individual functions as needed
```

## Limitations

- Requires active Flex Query Web Service access
- Flex Query must be pre-configured in Client Portal
- Only processes multi-leg strategies (2+ legs)
- Date filtering depends on date column format in Flex Query
- Statement download may take time for large queries (default 120 seconds wait, configurable with `--max-wait`)

## Future Enhancements

Potential improvements:
- Support for single-leg option trades
- Additional output formats (JSON, Excel)
- Strategy type detection (spread, collar, iron condor, etc.)
- P&L calculation and tracking
- Integration with database storage
- Email report delivery
- Charting and visualization

## Related Documentation

- [IB API Connection Rules](.cursor/rules/ib_api.mdc) - How to connect to IB API
- [Project Structure](../PROJECT_STRUCTURE.md) - Overall project organization

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Flex Query configuration in Client Portal
3. Verify environment variables are set correctly
4. Check script logs for detailed error messages

